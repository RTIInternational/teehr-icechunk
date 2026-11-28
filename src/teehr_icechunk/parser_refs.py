"""References of parsed sources (one file per step: HDF5, NetCDF4, GeoTIFF), as reference tables for the run model.

Each file is parsed for its chunks (``gu.open_virtual_datasets``, the format's extra), and each chunk becomes a row:
its step from the listing, its tile along the other dims (``chunk_<dim>``), and where its bytes are.
"""
import logging

import pandas as pd
import xarray as xr
import zarr

from teehr_icechunk import grid_utils as gu
from teehr_icechunk.ingest import source_reader
from teehr_icechunk.models import IngestGriddedDataInput
from teehr_icechunk.refs import CHUNK_PREFIX, REF_COLUMNS
from teehr_icechunk.sources import GriddedSource

logger = logging.getLogger(__name__)

# Metadata a file's array must share with the stored one; shape is compared along the dims other than the steps
LAYOUT_KEYS = ("data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs")
# Files tried for a new repo's template, in case the first ones can't be read
TEMPLATE_TRIES = 5


def _open(source: GriddedSource, urls: list[str], args: IngestGriddedDataInput) -> dict[str, xr.Dataset]:
    """Each file's virtual dataset by URL, preprocessed and with CF fill values; skipped files are left out, so the
    result may be empty."""
    registry, parser, preprocess = source_reader(source, source.credentials(), args)
    try:
        datasets = gu.open_virtual_datasets(
            urls, registry, parser, ignore_unreadable_file=args.ignore_unreadable_file, preprocess=preprocess
        )
    except ValueError as e:
        if str(e).startswith(gu.NO_VIRTUAL_DATASETS):  # every file was skipped
            return {}
        raise
    return {url: gu.align_virtual_fill_values(ds) for url, ds in datasets.items()}


def template(source: GriddedSource, files: pd.DataFrame, args: IngestGriddedDataInput) -> xr.Dataset:
    """The first readable listed file's variables, for creating the repo."""
    urls = files["url"].tolist()
    for start in range(0, min(len(urls), TEMPLATE_TRIES)):
        datasets = _open(source, urls[start:start + 1], args)
        if datasets:
            return next(iter(datasets.values()))[args.variable_names]
    raise ValueError(f"None of the first {TEMPLATE_TRIES} listed file(s) could be parsed, e.g. {urls[0]}.")


def reference_table(
    source: GriddedSource, files: pd.DataFrame, args: IngestGriddedDataInput, stored: zarr.Group
) -> pd.DataFrame:
    """References to every chunk of ``files`` (one per step), indexed by step; files that can't be read are skipped.

    Each file's arrays must be laid out as the ``stored`` ones (shape, chunks, dtype, codecs, fill), so a source
    that changed its layout fails the segment instead of mixing layouts in one array.
    """
    datasets = _open(source, files["url"].tolist(), args)
    step_dims = list(files.index.names)
    rows, steps = [], []
    for i, (url, modified) in enumerate(zip(files["url"], files["last_modified"])):
        ds = datasets.get(url)
        if ds is None:
            continue
        for var in args.variable_names:
            array, stored_array = ds[var], stored[var]
            _check_layout(var, url, array, stored_array, step_dims)
            other = [(axis, d) for axis, d in enumerate(array.dims) if d not in step_dims]
            for index, entry in array.data.manifest.iter_refs():
                rows.append({
                    "variable": var, "url": entry["path"], "offset": entry["offset"], "length": entry["length"],
                    "last_modified": modified, **{f"{CHUNK_PREFIX}{d}": index[axis] for axis, d in other},
                })
                steps.append(i)
    if not rows:
        return pd.DataFrame(columns=REF_COLUMNS, index=files.index[:0])
    return pd.DataFrame(rows, index=files.index.take(steps))


def _check_layout(var: str, url: str, array: xr.DataArray, stored: zarr.Array, step_dims: list[str]) -> None:
    parsed = array.data.metadata.to_dict()
    expected = stored.metadata.to_dict()
    stored_shape = dict(zip(stored.metadata.dimension_names, stored.shape))
    steps = {d: n for d, n in zip(array.dims, array.shape) if d in step_dims}
    if any(n != 1 for n in steps.values()):
        raise ValueError(f"{url} holds {steps} steps of {var}; a parsed source has one step per file.")
    shape = {d: n for d, n in zip(array.dims, array.shape) if d not in step_dims}
    differs = [key for key in LAYOUT_KEYS if parsed.get(key) != expected.get(key)]
    if shape != {d: stored_shape.get(d) for d in shape}:
        differs.append("shape")
    if differs:
        raise ValueError(
            f"{url} stores {var} unlike the repo ({', '.join(differs)} differ): "
            f"{ {k: parsed.get(k) for k in differs if k != 'shape'} } vs { {k: expected.get(k) for k in differs if k != 'shape'} }"
            + (f"; shape {shape} vs {stored_shape}" if "shape" in differs else "")
        )
