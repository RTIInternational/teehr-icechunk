"""Map overviews: block means of the data group on its own grid, in ``/overviews/<factor>``, on the run model."""
import logging
import time
from dataclasses import dataclass

import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from teehr_icechunk import grid_utils as gu
from teehr_icechunk import time_grid as tg
from teehr_icechunk.grid_spec import GridSpec, read_grid_spec
from teehr_icechunk.models import BuildOverviewsDataInput, OVERVIEW_CHUNK_SIZE
from teehr_icechunk.packed_encoding import PackedEncoding
from teehr_icechunk.repo_const import OVERVIEW_GROUP_PATH, STATUS_COORD, VALUE_TIME
from teehr_icechunk.units import (
    InProcessRunner, Runner, Unit, UnitResult, commit_unit, plan_units, raise_failed, steps_in, unit_slots,
)

logger = logging.getLogger(__name__)


def open_repo(args: BuildOverviewsDataInput) -> ic.Repository:
    """The run's repository, able to read its data group (``/references`` reads the source with the run's credentials)."""
    storage = gu.build_icechunk_s3_storage(
        bucket=args.dest_bucket,
        prefix=f"{args.base_prefix}/{args.configuration_name}",
        **args.s3_storage_kwargs
    )
    source = getattr(args, "source", None)
    return gu.open_repo_for_reading(storage, source.credentials() if source else None)


def build_overviews(
    args: BuildOverviewsDataInput, runner: Runner | None = None, repo: ic.Repository | None = None
) -> list[UnitResult]:
    """Build map overviews for the steps the data group changed, and write them into their slots.

    Reads the repo's data group (``/data``, or ``/references`` when not materialized), only the
    steps the overviews lack or hold from an older write, and averages blocks of
    ``overview_factors`` pixels on the data's own grid into ``/overviews/<factor>``. The tile
    server uses the data group as full resolution and the overviews as coarser levels, and
    reprojects each tile on request.

    Runs on the run model: the levels are prepared here, units of whole shards build them (in this process, or
    as child runs through ``runner``), and the steps of the units that succeeded are recorded here. Returns
    each unit's outcome; a failed unit is retried by the next run.

    Parameters
    ----------
    args : BuildOverviewsDataInput
        Run parameters; see BuildOverviewsDataInput for field descriptions.
    """
    repo = repo or open_repo(args)
    logger.info(f"Icechunk repo opened at: {args.dest_bucket}/{args.base_prefix}/{args.configuration_name}.")
    store = repo.readonly_session("main").store
    # The repo's own shape, so a standalone run needs no forecast settings
    spec = read_grid_spec(store)
    dim = spec.append_dim
    data_meta = tg.read_step_metadata(store, spec.data_group, dim)
    if data_meta is None:
        logger.info(f"No data found in {spec.data_group}.")
        return []
    stale = _stale_steps(store, spec, data_meta, args)
    if len(stale) == 0:
        logger.info(f"{OVERVIEW_GROUP_PATH} is up to date.")
        return []
    data_ds = _data_dataset(store, spec, args)
    _prepare(repo, spec, data_ds, data_meta, args)

    _, origin, time_step = tg.read_grid(store)
    # Analysis shards span steps along the append axis; a forecast shard holds one cycle, so any cycle bounds one
    blocks = [] if spec.fixed_dims else [args.num_shard_chunks * args.time_chunk_size]
    split = getattr(getattr(args, "source", None), "manifest_split_steps", None)
    times = stale.get_level_values(dim) if isinstance(stale, pd.MultiIndex) else stale
    units = plan_units(times, origin, time_step, unit_slots(args.unit_steps, *blocks, *([split] if split else [])))
    logger.info(f"Building overviews for {len(stale)} step(s) in {len(units)} unit(s).")
    results = (runner or InProcessRunner()).run(OverviewWork(repo, args), units)

    done = steps_in(stale, dim, [r.unit for r in results if r.error is None])
    if len(done):
        meta = data_meta.loc[done]
        session = repo.writable_session("main")
        for factor in args.overview_factors:
            # The data group's status and write time, so a step is current until that group changes again
            tg.write_step_metadata(
                session, f"{OVERVIEW_GROUP_PATH}/{factor}", dim, done,
                status=meta[STATUS_COORD].to_numpy(),
                source_last_modified=meta["source_last_modified"].to_numpy(),
                updated_at=meta["updated_at"].to_numpy(),
            )
        session.commit(f"Recorded {len(done)} overview step(s)")
        logger.info(f"Recorded {len(done)} overview step(s).")
    raise_failed(results)
    return results


@dataclass
class OverviewWork:
    """One unit's overviews: its stale steps, read and averaged in batches of whole shards, written and committed.

    Runs in the parent (``InProcessRunner``) or in a child run rebuilt from ``parameters``.
    """

    repo: ic.Repository
    args: BuildOverviewsDataInput

    def __call__(self, unit: Unit) -> None:
        args = self.args
        store = self.repo.readonly_session("main").store
        spec = read_grid_spec(store)
        dim = spec.append_dim
        data_meta = tg.read_step_metadata(store, spec.data_group, dim)
        stale = unit.select(_stale_steps(store, spec, data_meta, args), dim)
        if len(stale) == 0:
            return
        data_ds = _data_dataset(store, spec, args)
        # Batches of whole shards bound memory by the batch
        fixed = {d: data_ds[d].values for d in spec.fixed_dims}
        if fixed:
            batches = tg.step_batches(stale, fixed, args.num_shard_chunks * args.time_chunk_size, args.time_batch_size)
        else:
            batches = tg.shard_batches(stale, data_meta.index, args.time_batch_size)
        session = self.repo.writable_session("main")
        started = time.perf_counter()
        for i, batch in enumerate(batches, start=1):
            batch_started = time.perf_counter()
            # A forecast batch's stale leads need not be contiguous; each rectangle is written on its own
            for block in tg.step_blocks(batch, fixed) if fixed else [batch]:
                ds_block = data_ds.sel(tg.block_selection(block, dim)).load()
                for factor, level_ds in _overview_levels(ds_block, args.overview_factors).items():
                    level_ds = _clip_to_packed_range(level_ds, args.overview_encoding)
                    tg.write_chunks(session, f"{OVERVIEW_GROUP_PATH}/{factor}", level_ds, dim, fixed_dims=spec.fixed_dims)
            logger.info(
                f"Processed overview batch {i} of {len(batches)} ({len(batch)} step(s)) "
                f"in {time.perf_counter() - batch_started:.1f}s."
            )
        commit_unit(session, f"Overviews for {len(stale)} step(s), {unit.start} to {unit.end}")
        logger.info(f"Built overviews for {len(stale)} step(s) in {time.perf_counter() - started:.1f}s.")

    def parameters(self, unit: Unit) -> dict:
        return {"stage": "overviews", "args": self.args.model_dump(mode="json"), "unit": unit.to_dict()}


def _stale_steps(store: ic.IcechunkStore, spec: GridSpec, data_meta: pd.DataFrame, args: BuildOverviewsDataInput) -> pd.Index:
    """Steps the overviews lack, or hold from an older write of the data group."""
    first_level = f"{OVERVIEW_GROUP_PATH}/{args.overview_factors[0]}"
    return tg.stale_steps(data_meta, tg.read_step_metadata(store, first_level, spec.append_dim))


def _data_dataset(store: ic.IcechunkStore, spec: GridSpec, args: BuildOverviewsDataInput) -> xr.Dataset:
    """The data group's gridded variables, lazily, decoded with NaN for missing values."""
    data_ds = gu.open_zarr_group(store=store, group_path=spec.data_group)
    data_ds = data_ds.drop_vars([n for n in (*tg.STEP_COORDS, VALUE_TIME) if n in data_ds.variables])
    # Scalars such as the grid mapping ride along as coords; only gridded variables get overviews
    data_ds = data_ds.set_coords([v for v in data_ds.data_vars if data_ds[v].ndim == 0])
    y_dim, x_dim = data_ds.attrs["spatial:dimensions"]
    data_ds = data_ds[[v for v in data_ds.data_vars if {y_dim, x_dim} <= set(data_ds[v].dims)]]
    # Values are decoded, with NaN for missing, as in /data; a source's own fill marker (e.g. -9999
    # in /references) would otherwise reach coarsening as a value
    for var in data_ds.data_vars:
        if np.issubdtype(data_ds[var].dtype, np.floating):
            data_ds[var].encoding.pop("missing_value", None)
            data_ds[var].encoding["_FillValue"] = np.nan
    _check_packing_units(data_ds, args.overview_encoding)
    return data_ds


def _prepare(
    repo: ic.Repository, spec: GridSpec, data_ds: xr.Dataset, data_meta: pd.DataFrame, args: BuildOverviewsDataInput
) -> None:
    """Create each overview level (or extend its axis to the data group's), in one commit, so units only write chunks."""
    dim = spec.append_dim
    session = repo.writable_session("main")
    _, origin, time_step = tg.read_grid(session.store)
    axis_end = data_meta.index.get_level_values(dim)[-1]
    fixed = {d: data_ds[d].values for d in spec.fixed_dims}
    # One step's levels give each new level its variables, attrs and encoding
    template = data_ds.isel({dim: [0], **{d: [0] for d in fixed}}).load()
    # The tile server reads the data group as the full-resolution level, so it needs the transform too; repos
    # created before ingest wrote it get it here
    group_attrs = zarr.open_group(session.store, path=spec.data_group.strip("/"), mode="a", zarr_format=3).attrs
    if "spatial:transform" not in group_attrs:
        group_attrs["spatial:transform"] = gu.spatial_transform(template)
        logger.info(f"Recorded spatial:transform on {spec.data_group}.")
    for factor, level_ds in _overview_levels(template, args.overview_factors).items():
        level_ds = _clip_to_packed_range(level_ds, args.overview_encoding)
        group = f"{OVERVIEW_GROUP_PATH}/{factor}"
        if tg.read_step_metadata(session.store, group, dim) is None:
            tg.create_group(session, group, level_ds, origin, time_step, dim, axis_end,
                            encoding=_overview_encoding(level_ds, spec, args), fixed=fixed)
            logger.info(f"Created {group} on a {time_step} grid from {origin} to {axis_end}.")
        else:
            added = tg.ensure_axis(session, group, axis_end, time_step, dim)
            if added:
                logger.info(f"Extended {group} by {added} slot(s) to {axis_end}.")
    if session.has_uncommitted_changes:
        session.commit(f"Prepared {OVERVIEW_GROUP_PATH} through {axis_end}")


def _check_packing_units(ds: xr.Dataset, overview_encoding: dict[str, PackedEncoding]) -> None:
    """Fail if a packing range is given in other units than its variable is stored in."""
    for var, packing in overview_encoding.items():
        stored_units = ds[var].attrs.get("units") if var in ds else None
        if var in ds and stored_units != packing.units:
            raise ValueError(
                f"overview_encoding for '{var}' is in '{packing.units}', but the variable is stored in '{stored_units}'."
            )


def _clip_to_packed_range(ds: xr.Dataset, overview_encoding: dict[str, PackedEncoding]) -> xr.Dataset:
    """Clip packed variables to their packing range, so values cannot wrap."""
    for var, packing in overview_encoding.items():
        if var in ds:
            ds[var] = ds[var].clip(packing.min_value, packing.max_value).assign_attrs(ds[var].attrs)
    return ds


def _overview_encoding(ds: xr.Dataset, spec: GridSpec, args: BuildOverviewsDataInput) -> dict:
    """Chunk/shard encoding for a new overview level: packed where configured, float32 otherwise."""
    encoding = gu.create_encoding_config(
        ds,
        append_dim=spec.append_dim,
        chunk_size=OVERVIEW_CHUNK_SIZE,
        # Tiles read one step (or lead) at a time; shards span as many steps as /data's
        num_shard_chunks=args.num_shard_chunks * args.time_chunk_size,
        time_chunk_size=1,
        fixed_dims=spec.fixed_dims,
    )
    for var in ds.data_vars:
        packing = args.overview_encoding.get(var)
        encoding[var].update(packing.to_encoding() if packing else {"dtype": "float32"})
    return encoding


def _block_sum(values: np.ndarray, k: int) -> np.ndarray:
    """Sum each k x k block of the last two axes, dropping a partial block at the far edges.

    Adds the k*k interleaved slices; several times faster than xarray's coarsen or a reshape-and-sum.
    """
    y, x = values.shape[-2] // k * k, values.shape[-1] // k * k
    return sum(values[..., i:y:k, j:x:k] for i in range(k) for j in range(k))


def _overview_levels(ds: xr.Dataset, factors: list[int]) -> dict[int, xr.Dataset]:
    """Average ``factor`` x ``factor`` blocks of pixels for each factor, skipping missing values.

    Each level is summed from the previous one's block sums and valid-pixel counts, so its means match
    averaging the full-resolution pixels directly.
    """
    y_dim, x_dim = ds.attrs["spatial:dimensions"]
    dims, sums, counts = {}, {}, {}
    for var in ds.data_vars:
        da = ds[var].transpose(..., y_dim, x_dim)
        valid = ~np.isnan(da.values)
        dims[var], sums[var], counts[var] = da.dims, np.where(valid, da.values, 0), valid.astype(np.uint16)
    y, x = ds[y_dim].values, ds[x_dim].values
    coords = {n: c for n, c in ds.coords.items() if y_dim not in c.dims and x_dim not in c.dims}
    levels, previous = {}, 1
    for factor in factors:
        k = factor // previous
        sums = {var: _block_sum(total, k) for var, total in sums.items()}
        counts = {var: _block_sum(count, k) for var, count in counts.items()}
        y, x = y[: len(y) // k * k].reshape(-1, k).mean(axis=1), x[: len(x) // k * k].reshape(-1, k).mean(axis=1)
        with np.errstate(invalid="ignore"):
            data = {var: (dims[var], sums[var] / counts[var], ds[var].attrs) for var in sums}
        level = xr.Dataset(
            data,
            coords={**coords, y_dim: (y_dim, y, ds[y_dim].attrs), x_dim: (x_dim, x, ds[x_dim].attrs)},
            attrs=dict(ds.attrs),
        )
        level.attrs["spatial:transform"] = gu.spatial_transform(level)
        levels[factor], previous = level, factor
    return levels
