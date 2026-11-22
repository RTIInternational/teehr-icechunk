"""Read planning for zonal means: batches of steps that follow a grid's storage chunks and a memory budget."""
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from teehr_icechunk import time_grid as tg
from teehr_icechunk.mean_areal_kernel import chunk_reads
from teehr_icechunk.repo_const import MEMBER, STATUS_COORD

logger = logging.getLogger(__name__)

# Bytes per weighted pixel and step while averaging: weighted_sums_from_matrix's float64 copy, NaN mask,
# zero-filled copy and missing weights
AVERAGING_BYTES_PER_PIXEL = 25
# Steps averaged at once, which caps those copies whatever the batch size
MAX_SLICE_STEPS = 256


def written_steps(da: xr.DataArray, dim: str, start: datetime | None = None, end: datetime | None = None) -> xr.DataArray:
    """Grid steps that hold data (time-grid slots not yet written are skipped), within start..end (naive UTC)."""
    if STATUS_COORD in da.coords:
        da = da.isel({dim: da[STATUS_COORD].values != tg.UNWRITTEN})
    return da.sel({dim: slice(start, end)})


def written_forecast_steps(
    da: xr.DataArray, dim: str, fixed: dict, start: datetime | None = None, end: datetime | None = None
) -> pd.MultiIndex:
    """Forecast steps that hold data, for cycles within start..end (naive UTC).

    Steps not yet written, and those the source never publishes (``NOT_EXPECTED``), are skipped. Cycles are
    selected first, so a long archive costs only the requested window.
    """
    cycles = da[dim].to_index()
    keep = np.ones(len(cycles), bool)
    if start is not None:
        keep &= cycles >= start
    if end is not None:
        keep &= cycles <= end
    da = da.isel({dim: np.flatnonzero(keep)})
    step_dims = (dim, *fixed)
    steps = pd.MultiIndex.from_product([da[d].to_index() for d in step_dims], names=step_dims)
    if STATUS_COORD in da.coords:
        return steps[da[STATUS_COORD].transpose(*step_dims).values.reshape(-1) >= 0]
    return steps


def area_weights(weights_df: pd.DataFrame, da: xr.DataArray, y_dim: str) -> pd.Series:
    """Coverage fractions scaled by pixel area on geographic grids, whose pixels shrink with latitude."""
    if da.rio.crs is None:
        raise ValueError("The grid has no CRS.")
    weights = weights_df["fraction_covered"].astype("float64")
    if da.rio.crs.is_geographic:
        weights = weights * np.cos(np.deg2rad(da[y_dim].values[weights_df["row"].to_numpy()]))
    return weights


@dataclass(frozen=True)
class ReadPlan:
    """Batches of steps, and the grid's (row, column) chunk size, so only chunks with weighted pixels are read."""

    batches: list[pd.Index]
    chunk_shape: tuple[int, int] | None


def chunk_layout(da: xr.DataArray) -> dict[str, tuple[int, int]]:
    """Each dim's stored inner chunk and shard length; empty for a grid without chunk encoding."""
    chunks = da.encoding.get("chunks")
    if chunks is None:
        return {}
    shards = da.encoding.get("shards") or chunks
    return {dim: (int(c), int(s)) for dim, c, s in zip(da.dims, chunks, shards)}


def cycle_batches(steps: pd.MultiIndex, fixed: dict, lead_steps: int) -> list[pd.MultiIndex]:
    """Forecast steps by cycle and block of ``lead_steps`` leads, all members together."""
    dim, *fixed_dims = steps.names
    lead_pos = pd.Index(fixed[fixed_dims[-1]]).get_indexer(steps.get_level_values(fixed_dims[-1]))
    keys = pd.MultiIndex.from_arrays([steps.get_level_values(dim), lead_pos // lead_steps])
    return [steps[group.to_numpy()] for _, group in pd.Series(np.arange(len(steps)), index=keys).groupby(level=[0, 1])]


def plan_reads(
    da: xr.DataArray,
    written: pd.Index,
    fixed: dict,
    rows: np.ndarray,
    cols: np.ndarray,
    n_zones: int,
    *,
    dim: str,
    y_dim: str,
    x_dim: str,
    task_memory_gb: float,
) -> ReadPlan:
    """Batches from the grid's storage layout and ``task_memory_gb``.

    A batch holds whole inner chunks along the append axis (forecasts: the lead axis), so no chunk is decoded
    twice, and whole shards when they fit; members stored in one chunk are read together. Only the chunks
    holding weighted pixels are read, one chunk row at a time (``chunk_reads``), so a batch's size is set by its
    largest read.
    """
    layout = chunk_layout(da)
    lead_dim = list(fixed)[-1] if fixed else dim
    members_together = MEMBER in fixed and layout.get(MEMBER, (1, 1))[0] >= len(fixed[MEMBER])
    members = len(fixed[MEMBER]) if members_together else 1
    unit, shard = layout.get(lead_dim, (1, 1))
    chunk_shape = (layout[y_dim][0], layout[x_dim][0]) if y_dim in layout else None
    reads = [(read_size(r), read_size(c), len(p)) for r, c, p in chunk_reads(rows, cols, chunk_shape)]
    read_pixels = max(r * c for r, c, _ in reads)
    read_weighted = max(p for _, _, p in reads)
    budget = task_memory_gb * 1e9

    def batch_bytes(steps: int) -> int:
        per_step = read_pixels * da.dtype.itemsize * 2 + read_weighted * da.dtype.itemsize + 16 * n_zones
        return int(steps * per_step + min(steps, MAX_SLICE_STEPS) * read_weighted * AVERAGING_BYTES_PER_PIXEL)

    unit_steps = unit * members
    if batch_bytes(unit_steps) > budget:
        logger.warning(
            f"One read unit ({unit_steps} steps) needs {batch_bytes(unit_steps) / 1e9:.1f} GB, "
            f"over task_memory_gb={task_memory_gb}."
        )
    units = max(1, int(budget // batch_bytes(unit_steps)))
    lead_steps = unit * units
    if lead_steps >= shard:
        lead_steps = lead_steps // shard * shard

    if not fixed:
        batches = tg.shard_batches(written, da[dim].to_index(), lead_steps)
    elif members_together:
        batches = cycle_batches(written, fixed, lead_steps)
    else:
        batches = tg.step_batches(written, fixed, lead_steps, lead_steps)
    if chunk_shape is not None:
        used = len(set(zip(rows // chunk_shape[0], cols // chunk_shape[1])))
        logger.info(f"Reading the {used} spatial chunk(s) holding weighted pixels, in {len(reads)} read(s) per batch.")
    steps_per_batch = max(len(batch) for batch in batches)
    logger.info(f"Read plan: {len(batches)} batch(es) of up to {steps_per_batch} step(s).")
    return ReadPlan(batches, chunk_shape)


def read_size(selection: slice | np.ndarray) -> int:
    """Grid rows or columns a read selects."""
    return selection.stop - selection.start if isinstance(selection, slice) else len(selection)


def positions(coord: xr.DataArray, values) -> slice | np.ndarray:
    """``isel`` positions of ``values`` on ``coord``, as a slice when contiguous."""
    positions = coord.to_index().get_indexer(values)
    if (positions < 0).any():
        raise ValueError(f"Values not on the grid's {coord.name} axis.")
    if len(positions) and (np.diff(positions) == 1).all():
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions
