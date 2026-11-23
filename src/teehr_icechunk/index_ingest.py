"""Ingest for index sources: references built from the files' indexes (e.g. GRIB2 ``.idx``), on the run model.

An index source lists its files (``list_source_files``), turns them into steps (``steps_from_files``) and
builds a reference table from them (``reference_table``); it also gives an empty ``template`` of its arrays.
The parent prepares ``/references``, runs the units (in-process, or as child runs through a ``Runner``),
then records step metadata for the units that succeeded.
"""
import logging
from dataclasses import dataclass

import icechunk as ic
import numpy as np
import pandas as pd

from teehr_icechunk import grid_utils as gu
from teehr_icechunk import time_grid as tg
from teehr_icechunk.models import IngestGriddedDataInput
from teehr_icechunk.refs import write_virtual_refs
from teehr_icechunk.repo_const import (
    APPEND_DIM_ATTR,
    CONFIGURATION_NAME_ATTR,
    FIXED_DIMS_ATTR,
    FORECAST_LAYOUT_VERSION,
    LAYOUT_VERSION_ATTR,
    REFERENCES_GROUP_PATH,
    STATUS_COORD,
    STATUS_MEANINGS,
    TIME_ORIGIN_ATTR,
    TIME_STEP_ATTR,
    TIMESERIES_TYPE_ATTR,
)
from teehr_icechunk.units import InProcessRunner, Runner, Unit, UnitResult, commit_unit, plan_units, raise_failed, unit_slots

logger = logging.getLogger(__name__)


def open_repo(args: IngestGriddedDataInput) -> ic.Repository:
    """The run's repository, created with the source's manifest splitting if new."""
    source = args.source
    split = getattr(source, "manifest_split_steps", None)
    return gu.configure_icechunk_s3_repo(
        source.source_bucket,
        args.dest_bucket,
        prefix=f"{args.base_prefix}/{args.configuration_name}",
        vc_credentials_kwargs=source.credentials(),
        vc_store_kwargs=source.virtual_store_kwargs,
        manifest_split=(args.append_dim, split) if split else None,
        **args.s3_storage_kwargs,
    )


def prepare(repo: ic.Repository, args: IngestGriddedDataInput, end: pd.Timestamp) -> None:
    """Create ``/references`` (or extend its axis to ``end``) and mark slots the source never publishes, in one commit.

    Every array and axis exists before any unit runs, so units only write chunks.
    """
    source, dim = args.source, args.append_dim
    fixed = source.fixed_axes()
    session = repo.writable_session("main")
    if tg.read_step_metadata(session.store, REFERENCES_GROUP_PATH, dim) is None:
        origin, time_step = source.time_origin(), source.time_step
        attrs = {
            APPEND_DIM_ATTR: dim, TIME_ORIGIN_ATTR: origin.isoformat(), TIME_STEP_ATTR: pd.Timedelta(time_step).isoformat(),
            CONFIGURATION_NAME_ATTR: args.configuration_name, TIMESERIES_TYPE_ATTR: source.timeseries_type(),
        }
        if fixed:
            attrs.update({FIXED_DIMS_ATTR: list(fixed), LAYOUT_VERSION_ATTR: FORECAST_LAYOUT_VERSION})
        tg.write_grid_attrs(session.store, attrs)
        template = gu.standardize_and_inject_geozarr(
            source.template(), fallback_crs=args.fallback_source_crs, x_dim=args.x_dim, y_dim=args.y_dim
        )
        tg.create_group(session, REFERENCES_GROUP_PATH, template, origin, time_step, dim, end, virtual=True, fixed=fixed)
        logger.info(f"Created {REFERENCES_GROUP_PATH} on a {time_step} grid from {origin} to {end}.")
    else:
        if fixed:
            tg.check_fixed_axes(session.store, REFERENCES_GROUP_PATH, fixed)
        added = tg.ensure_axis(session, REFERENCES_GROUP_PATH, end, source.time_step, dim)
        if added:
            logger.info(f"Extended {REFERENCES_GROUP_PATH} by {added} slot(s) to {end}.")
    tg.ensure_grid_name(session.store, source.grid_name())
    # Readers use /references: index sources aren't materialized
    gu.write_data_group(session.store, REFERENCES_GROUP_PATH)
    axis = tg.read_step_metadata(session.store, REFERENCES_GROUP_PATH, dim).index.get_level_values(dim).unique()
    expected = source.expected_steps(axis)
    if expected is not None:
        marked = tg.mark_not_expected(session, REFERENCES_GROUP_PATH, dim, expected)
        if marked:
            logger.info(f"Marked {marked} step(s) the source never publishes as not expected.")
    if session.has_uncommitted_changes:
        session.commit(f"Prepared {REFERENCES_GROUP_PATH} through {end}")


@dataclass
class IndexRefsWork:
    """One unit's references within the run's window: list its files, keep the steps the write mode selects,
    write and commit.

    Runs in the parent (``InProcessRunner``) or in a child run rebuilt from ``parameters``.
    """

    repo: ic.Repository
    args: IngestGriddedDataInput
    start: pd.Timestamp
    end: pd.Timestamp

    def __call__(self, unit: Unit) -> None:
        source, dim = self.args.source, self.args.append_dim
        files = source.list_source_files(max(unit.start, self.start), min(unit.end, self.end))
        stored = tg.read_step_metadata(self.repo.readonly_session("main").store, REFERENCES_GROUP_PATH, dim)
        todo = tg.select_source_steps(source.steps_from_files(files), stored, self.args.write_mode)
        if todo.empty:
            return
        refs = source.reference_table(files)
        refs = refs[refs.index.isin(todo.index)]
        session = self.repo.writable_session("main")
        n = write_virtual_refs(session, REFERENCES_GROUP_PATH, refs)
        commit_unit(session, f"References for {len(todo)} step(s), {unit.start} to {unit.end}")
        logger.info(f"Wrote {n} reference(s) for {len(todo)} step(s) from {unit.start} to {unit.end}.")

    def parameters(self, unit: Unit) -> dict:
        window = {"start": self.start.isoformat(), "end": self.end.isoformat()}
        return {"args": self.args.model_dump(mode="json"), "unit": unit.to_dict(), "window": window}


def ingest_references(
    args: IngestGriddedDataInput,
    start: pd.Timestamp,
    end: pd.Timestamp,
    runner: Runner | None = None,
    unit_steps: int = 365,
    repo: ic.Repository | None = None,
) -> list[UnitResult]:
    """Write ``/references`` for the source's steps from ``start`` to ``end``; returns each unit's outcome.

    Units hold about ``unit_steps`` append-axis slots, aligned to the source's manifest splits. Step metadata
    is recorded for the units that succeeded; a failed unit is retried by the next run.
    """
    source, dim = args.source, args.append_dim
    repo = repo or open_repo(args)
    listing = source.list_files(start, end)
    stored = tg.read_step_metadata(repo.readonly_session("main").store, REFERENCES_GROUP_PATH, dim)
    todo = tg.select_source_steps(listing, stored, args.write_mode)
    logger.info(f"{len(todo)} of {len(listing)} listed step(s) to write in '{args.write_mode}' mode.")
    if todo.empty:
        return []
    times = todo.index.get_level_values(dim)
    prepare(repo, args, times.max())
    split = getattr(source, "manifest_split_steps", None)
    units = plan_units(times, source.time_origin(), source.time_step, unit_slots(unit_steps, *([split] if split else [])))
    logger.info(f"Running {len(units)} unit(s).")
    results = (runner or InProcessRunner()).run(IndexRefsWork(repo, args, pd.Timestamp(start), pd.Timestamp(end)), units)

    done = [r.unit for r in results if r.error is None]
    if done:
        in_done = np.zeros(len(todo), bool)
        for unit in done:
            in_done |= np.asarray((times >= unit.start) & (times <= unit.end))
        steps = todo[in_done]
        session = repo.writable_session("main")
        tg.write_step_metadata(
            session, REFERENCES_GROUP_PATH, dim, steps.index,
            status=steps[STATUS_COORD].map(STATUS_MEANINGS.index).to_numpy(),
            source_last_modified=steps["last_modified"].to_numpy("datetime64[ms]"),
            updated_at=tg.utc_now(),
        )
        session.commit(f"Recorded {len(steps)} step(s) of {len(done)} unit(s)")
        logger.info(f"Recorded {len(steps)} step(s) written by {len(done)} unit(s).")
    raise_failed(results)
    return results
