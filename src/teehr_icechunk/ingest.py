"""Ingest stages: source files to ``/references``, and ``/references`` to ``/data``."""
import logging
import time
from typing import Callable

import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr

from teehr_icechunk import grid_utils as gu
from teehr_icechunk import time_grid as tg
from teehr_icechunk.grid_spec import read_grid_spec
from teehr_icechunk.models import IngestGriddedDataInput, ParserType
from teehr_icechunk.repo_const import (
    APPEND_DIM_ATTR,
    CONFIGURATION_NAME_ATTR,
    DATA_GROUP_PATH,
    FIXED_DIMS_ATTR,
    FORECAST_LAYOUT_VERSION,
    LAYOUT_VERSION_ATTR,
    LEAD_TIME,
    REFERENCES_GROUP_PATH,
    STATUS_COORD,
    STATUS_MEANINGS,
    TIME_ORIGIN_ATTR,
    TIME_STEP_ATTR,
    TIMESERIES_TYPE_ATTR,
    VALUE_TIME,
)
from teehr_icechunk.sources import GriddedSource

logger = logging.getLogger(__name__)


def _parser(parser_type: ParserType, **kwargs):
    """The VirtualiZarr parser for ``parser_type``; imported here so each format's extra is optional."""
    if parser_type == ParserType.tiff:
        from virtual_tiff import VirtualTIFF

        # IFD 0 is full resolution; the flow builds its own overviews
        return VirtualTIFF(ifd=0, **kwargs)
    import virtualizarr as vz

    if parser_type == ParserType.hdf:
        return vz.parsers.HDFParser(**kwargs)
    return vz.parsers.ZarrParser(**kwargs)


def virtual_blocks(
    todo: pd.DataFrame, source: GriddedSource, credentials: dict, args: IngestGriddedDataInput
) -> list[xr.Dataset]:
    """Virtual references to the files in ``todo``: one dataset for an analysis source, or forecast blocks of steps."""
    obstore_kwargs = {**source.store_kwargs, **credentials, **args.obstore_kwargs}
    if credentials:
        # An unsigned request ignores the credentials
        obstore_kwargs.pop("skip_signature", None)
    registry = gu.create_objectstore_registry(source.source_bucket, **obstore_kwargs)
    parser = _parser(args.parser_type, **source.parser_kwargs)
    fixed = source.fixed_axes()
    if fixed:
        blocks = create_forecast_blocks(todo, fixed, registry, parser, _preprocess(source, args), args)
    else:
        blocks = [gu.create_virtual_xarray_dataset(
            todo["url"].tolist(),
            registry=registry,
            parser=parser,
            concat_dim=args.append_dim,
            ignore_unreadable_file=args.ignore_unreadable_file,
            preprocess=_preprocess(source, args),
            **args.xconcat_kwargs
        )]
    return [_standardize_references(gu.align_virtual_fill_values(block), args) for block in blocks]


def _preprocess(source: GriddedSource, args: IngestGriddedDataInput) -> Callable[[xr.Dataset, str], xr.Dataset]:
    """Per-file preprocessing: GeoTIFF georeferencing for TIFFs, then the source's own."""
    if args.parser_type == ParserType.tiff:
        return lambda ds, url: source.preprocess(
            gu.assign_geotiff_coords(ds, args.fallback_source_crs), url
        )
    return source.preprocess


def _standardize_references(ds: xr.Dataset, args: IngestGriddedDataInput) -> xr.Dataset:
    """Keep the ingested variables as the source names them, with CF and GeoZarr metadata."""
    return gu.standardize_and_inject_geozarr(
        ds[args.variable_names].sortby(args.append_dim),
        fallback_crs=args.fallback_source_crs,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
    )


def record_data_group(repo: ic.Repository, group: str) -> None:
    """Point readers (overviews, EDR, mean areal values) at ``group``."""
    session = repo.writable_session("main")
    if gu.write_data_group(session.store, group):
        session.commit(f"Data group is now {group}")
        logger.info(f"Recorded {group} as the data group.")


def create_forecast_blocks(
    todo: pd.DataFrame,
    fixed: dict,
    registry,
    parser,
    preprocess: Callable[[xr.Dataset, str], xr.Dataset],
    args: IngestGriddedDataInput,
) -> list[xr.Dataset]:
    """Virtual datasets of the forecast files to write, one per rectangular block of steps.

    Blocks are built from the files that opened, so a missing or unreadable file only splits a block.
    """
    datasets = gu.open_virtual_datasets(
        todo["url"].tolist(), registry, parser, ignore_unreadable_file=args.ignore_unreadable_file, preprocess=preprocess
    )
    present = todo.index[todo["url"].isin(list(datasets))]
    blocks = []
    for steps in tg.step_blocks(present, fixed):
        urls = todo.loc[steps, "url"]
        cycles = [
            xr.concat([datasets[url] for url in cycle_urls], dim=LEAD_TIME, **args.xconcat_kwargs)
            for _, cycle_urls in urls.groupby(level=args.append_dim, sort=True)
        ]
        blocks.append(xr.concat(cycles, dim=args.append_dim, **args.xconcat_kwargs))
    return blocks


def write_references(
    repo: ic.Repository,
    blocks: list[xr.Dataset],
    todo: pd.DataFrame,
    origin: pd.Timestamp,
    source: GriddedSource,
    args: IngestGriddedDataInput,
) -> None:
    """Write virtual references into their slots, with their status and source file's last modified time.

    ``blocks`` is one dataset for analysis sources, or forecast blocks of steps; one commit covers them all.
    """
    dim, time_step = args.append_dim, source.time_step
    fixed = source.fixed_axes()
    steps = [tg.block_steps(block, dim, tuple(fixed)) for block in blocks]
    # Files skipped as missing or unreadable aren't in any block
    todo = todo.loc[steps[0].append(steps[1:])]
    session = repo.writable_session("main")
    end = todo.index.get_level_values(dim).max()
    stored = tg.read_step_metadata(session.store, REFERENCES_GROUP_PATH, dim)
    if stored is None:
        attrs = {
            APPEND_DIM_ATTR: dim, TIME_ORIGIN_ATTR: origin.isoformat(), TIME_STEP_ATTR: pd.Timedelta(time_step).isoformat(),
            CONFIGURATION_NAME_ATTR: args.configuration_name, TIMESERIES_TYPE_ATTR: source.timeseries_type(),
        }
        if fixed:
            attrs.update({FIXED_DIMS_ATTR: list(fixed), LAYOUT_VERSION_ATTR: FORECAST_LAYOUT_VERSION})
        tg.write_grid_attrs(session.store, attrs)
        tg.create_group(session, REFERENCES_GROUP_PATH, blocks[0], origin, time_step, dim, end, virtual=True, fixed=fixed)
        logger.info(f"Created {REFERENCES_GROUP_PATH} on a {time_step} grid from {origin} to {end}.")
        stored = tg.read_step_metadata(session.store, REFERENCES_GROUP_PATH, dim)
    else:
        if fixed:
            tg.check_fixed_axes(session.store, REFERENCES_GROUP_PATH, fixed)
        added = tg.ensure_axis(session, REFERENCES_GROUP_PATH, end, time_step, dim)
        if added:
            logger.info(f"Extended {REFERENCES_GROUP_PATH} by {added} slot(s) to {end}.")
    grid_name = source.grid_name()
    if tg.ensure_grid_name(session.store, grid_name):
        logger.info(f"Recorded grid_name '{grid_name}' on the repo.")
    updated_at = tg.utc_now()
    for block, block_steps in zip(blocks, steps):
        meta = todo.loc[block_steps]
        ds = tg.with_step_metadata(
            block,
            dim,
            status=meta[STATUS_COORD].map(STATUS_MEANINGS.index).to_numpy(),
            source_last_modified=meta["last_modified"].to_numpy("datetime64[ms]"),
            updated_at=updated_at,
            existing=stored,
            fixed_dims=tuple(fixed),
        )
        tg.region_write(session, REFERENCES_GROUP_PATH, ds, dim, virtual=True, fixed_dims=tuple(fixed))
    snapshot_id = session.commit(f"Wrote {len(todo)} step(s) of virtual references")
    logger.info(f"Committed {len(todo)} step(s) of virtual references: {snapshot_id}")


def _data_encoding(da: xr.DataArray, fill_value: int | None = None) -> dict:
    """How ``/data`` stores a decoded ``/references`` variable.

    Float sources are stored as plain float32 in the source's units. Integer sources keep their integers,
    losslessly, with the source's own packing; ``fill_value`` marks missing values when the source declares none.
    """
    dtype = np.dtype(da.encoding.get("dtype", da.dtype))
    if not np.issubdtype(dtype, np.integer):
        return {"dtype": "float32"}
    fill = da.encoding.get("_FillValue", fill_value)
    if fill is None:
        raise ValueError(
            f"Integer variable '{da.name}' has no _FillValue; set data_fill_values['{da.name}'] to a value it never uses."
        )
    info = np.iinfo(dtype)
    if not info.min <= fill <= info.max:
        raise ValueError(f"Fill value {fill} for '{da.name}' is outside the {dtype} range [{info.min}, {info.max}].")
    # zarr's own fill must match, or unwritten chunks read as valid data (e.g. 0)
    packing = {"dtype": dtype, "_FillValue": fill, "fill_value": fill}
    # _Unsigned: netCDF-3 style unsigned data, stored signed (fills too) and decoded to the unsigned range
    packing.update({k: da.encoding[k] for k in ("scale_factor", "add_offset", "_Unsigned") if k in da.encoding})
    return packing


# Each shard's worth of steps is committed on its own, so a retry (or the next run) resumes after the last
# committed batch. Callers should retry it: reading source chunks can drop connections.
def materialize_references(repo: ic.Repository, args: IngestGriddedDataInput) -> None:
    """Copy the referenced steps ``/data`` lacks, or holds from an older reference, into their slots."""
    dim = args.append_dim
    store = repo.readonly_session("main").store
    refs_meta = tg.read_step_metadata(store, REFERENCES_GROUP_PATH, dim)
    if refs_meta is None:
        logger.info(f"No {REFERENCES_GROUP_PATH} to materialize.")
        return
    stale = tg.stale_steps(refs_meta, tg.read_step_metadata(store, DATA_GROUP_PATH, dim))
    if len(stale) == 0:
        logger.info(f"{DATA_GROUP_PATH} is up to date.")
        return
    _, origin, time_step = tg.read_grid(store)
    fixed_dims = read_grid_spec(store).fixed_dims
    refs = gu.restore_grid_mapping_attrs(gu.open_zarr_group(store=store, group_path=REFERENCES_GROUP_PATH))
    refs = refs.drop_vars([n for n in (*tg.STEP_COORDS, VALUE_TIME) if n in refs.variables])
    fixed = {d: refs[d].values for d in fixed_dims}
    end = refs_meta.index.get_level_values(dim)[-1]
    shard_steps = args.time_chunk_size * args.num_shard_chunks
    if fixed:
        batches = tg.step_batches(stale, fixed, shard_steps, shard_steps)
    else:
        batches = tg.shard_batches(stale, refs_meta.index, shard_steps)
    logger.info(f"Materializing {len(stale)} step(s) in {len(batches)} shard batch(es).")
    started = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        session = repo.writable_session("main")
        data_meta = tg.read_step_metadata(session.store, DATA_GROUP_PATH, dim)
        if data_meta is None:
            encoding = gu.create_encoding_config(
                refs,
                append_dim=dim,
                chunk_size=args.chunk_size,
                num_shard_chunks=args.num_shard_chunks,
                time_chunk_size=args.time_chunk_size,
                fixed_dims=fixed_dims,
            )
            # Later region writes reuse the stored encoding
            for var in refs.data_vars:
                encoding[var].update(_data_encoding(refs[var], args.data_fill_values.get(var)))
            tg.create_group(session, DATA_GROUP_PATH, refs, origin, time_step, dim, end, encoding=encoding, fixed=fixed)
            logger.info(f"Created {DATA_GROUP_PATH} on a {time_step} grid from {origin} to {end}.")
            data_meta = tg.read_step_metadata(session.store, DATA_GROUP_PATH, dim)
        else:
            added = tg.ensure_axis(session, DATA_GROUP_PATH, end, time_step, dim)
            if added:
                logger.info(f"Extended {DATA_GROUP_PATH} by {added} slot(s) to {end}.")
        # A forecast batch's stale leads need not be contiguous; each rectangle is written on its own
        for block in tg.step_blocks(batch, fixed) if fixed else [batch]:
            meta = refs_meta.loc[block]
            part = tg.with_step_metadata(
                refs.sel(tg.block_selection(block, dim)).load(),
                dim,
                status=meta[STATUS_COORD].to_numpy(),
                source_last_modified=meta["source_last_modified"].to_numpy(),
                # The upstream write time, so the step is current until /references changes again
                updated_at=meta["updated_at"].to_numpy(),
                existing=data_meta,
                fixed_dims=fixed_dims,
            )
            tg.region_write(session, DATA_GROUP_PATH, part, dim, fixed_dims=fixed_dims)
        snapshot_id = session.commit(f"Materialized {len(batch)} step(s) into {DATA_GROUP_PATH}")
        logger.info(
            f"Committed materialized batch {i} of {len(batches)} ({len(batch)} step(s)) "
            f"in {time.perf_counter() - batch_started:.1f}s: {snapshot_id}"
        )
    logger.info(f"Materialized {len(stale)} step(s) in {time.perf_counter() - started:.1f}s.")
