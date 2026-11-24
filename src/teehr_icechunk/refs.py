"""Virtual references written straight through Icechunk, one or more chunks per step and variable.

A reference builder (e.g. ``grib_index``, ``parser_refs``) gives a table with a row per chunk; the writer
places each at its slot on the time grid. Arrays and axes must already exist (the run model's prepare step).
"""
import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import zarr

# Columns of a reference table, indexed by step (times, or forecast steps) like a source listing. A table may
# also have a ``chunk_<dim>`` column for each other dim the arrays are chunked along (e.g. HDF5 tiles)
REF_COLUMNS = ["variable", "url", "offset", "length", "last_modified"]
CHUNK_PREFIX = "chunk_"


def write_virtual_refs(session: ic.Session, group: str, refs: pd.DataFrame) -> int:
    """Set each row's chunk reference in ``group``; returns the number written.

    Every referenced array holds one step per chunk (chunk size 1 along the step dims). Along each other dim
    a row's chunk is its ``chunk_<dim>`` column, or the only chunk when there's no such column (a GRIB message
    is one whole field). ``last_modified`` is recorded with each reference, so a rewritten source file fails to
    read rather than returning other bytes; a file without one (listed after it was published) isn't checked.
    """
    if refs.empty:
        return 0
    g = zarr.open_group(session.store, path=group.strip("/"), mode="r+", zarr_format=3)
    step_dims = list(refs.index.names)
    axes = xr.open_zarr(session.store, group=group, consolidated=False, chunks=None)
    levels = [refs.index] if len(step_dims) == 1 else [refs.index.get_level_values(d) for d in step_dims]
    positions = {d: axes[d].to_index().get_indexer(level) for d, level in zip(step_dims, levels)}
    if any((p < 0).any() for p in positions.values()):
        raise ValueError(f"Some steps are not slots of {group}.")
    written = 0
    for variable, rows in refs.groupby("variable", sort=False):
        arr = g[variable]
        dims = list(arr.metadata.dimension_names)
        chunked = [d for d in dims if f"{CHUNK_PREFIX}{d}" in rows]
        _check_one_step_per_chunk(variable, dims, arr.chunks, arr.shape, step_dims, chunked)
        is_var = refs["variable"].to_numpy() == variable
        pos = np.column_stack([
            positions[d][is_var] if d in step_dims else
            rows[f"{CHUNK_PREFIX}{d}"].to_numpy() if d in chunked else np.zeros(len(rows), int)
            for d in dims
        ])
        # Icechunk keeps whole seconds; rounding up keeps a sub-second time (local files) from reading as changed
        modified = [None if pd.isna(m) else m.to_pydatetime() for m in
                    pd.DatetimeIndex(rows["last_modified"]).ceil("s").tz_localize("UTC")]
        specs = [
            ic.VirtualChunkSpec(index=[int(i) for i in idx], location=url, offset=int(off), length=int(n),
                                last_updated_at_checksum=mod)
            for idx, url, off, n, mod in zip(pos, rows["url"], rows["offset"], rows["length"], modified)
        ]
        failed = session.store.set_virtual_refs(arr.path, specs, validate_containers=True)
        if failed:
            raise ValueError(f"{len(failed)} reference(s) of {variable} fall outside the array, e.g. {failed[0]}.")
        written += len(specs)
    return written


def _check_one_step_per_chunk(
    variable: str, dims: list, chunks: tuple, shape: tuple, step_dims: list, chunked: list
) -> None:
    for d, c, s in zip(dims, chunks, shape):
        if (d in step_dims and c != 1) or (d not in step_dims and d not in chunked and c != s):
            raise ValueError(
                f"{variable} isn't one step per chunk, whole along the dims without a {CHUNK_PREFIX}<dim> column: "
                f"chunks {dict(zip(dims, chunks))}, shape {shape}."
            )
