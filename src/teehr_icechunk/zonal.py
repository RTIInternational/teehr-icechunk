"""Zonal statistics (the ``[mean-areal]`` extra): coverage weights of polygons on a grid, and zonal means of a lazily
opened grid run as segments that stage Parquet.

The parent plans segments along the grid's append axis and saves the weights and the output schema to a staging
directory. Each segment (in the parent, or a child run) reads its steps block by block in worker processes, each
writing one Parquet part file, so rows never pass between processes. The client loads finished segments into its
warehouse and removes them. Spark isn't needed here: a segment only reads the grid and writes files.
"""
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rioxarray  # noqa: F401
import xarray as xr
from exactextract import exact_extract
from rasterio.transform import rowcol
from scipy import sparse
from shapely.geometry import box

from teehr_icechunk import time_grid as tg
from teehr_icechunk.concurrency import map_blocking, resolve_budget, run_sync, use_process_pool
from teehr_icechunk.mean_areal_kernel import BlockTask, GridWeights, mean_areal_block
from teehr_icechunk.mean_areal_plan import MAX_SLICE_STEPS, chunk_layout, plan_reads, positions
from teehr_icechunk.repo_const import MEMBER
from teehr_icechunk.segments import Segment, segment_slots
from teehr_icechunk.weights import LOCATION_ID

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "weights.npz"
SCHEMA_FILE = "schema.arrow"
# Equal-area CRS for comparing polygon areas
EQUAL_AREA_CRS = "EPSG:6933"


def weights_for_grid(
    template_da: xr.DataArray,
    polygons: gpd.GeoDataFrame,
    *,
    x_dim: str,
    y_dim: str,
    key: str = LOCATION_ID,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """Fraction of each grid pixel covered by each polygon: ``location_id``, ``row``, ``col``, ``fraction_covered``.

    - ``template_da``: any variable of the grid, with a CRS; only its geometry is used, so its data isn't read.
      ``row``/``col`` index the full stored grid.
    - ``polygons``: polygons in any CRS, keyed by their ``key`` column (written as ``location_id``).
    - ``min_coverage``: polygons with less of their area inside the grid's extent are dropped.

    Only invalid polygons are repaired (once), after reprojecting to the grid's CRS, which can invalidate them.
    """
    missing_dims = {x_dim, y_dim} - set(template_da.dims)
    if missing_dims:
        raise ValueError(f"Grid has no dimension(s) {sorted(missing_dims)}; set x_dim/y_dim. Dims: {template_da.dims}")
    if template_da.rio.crs is None:
        raise ValueError("The grid has no CRS.")
    # Positions are unchanged, so row/col still index the stored grid
    template_da = template_da.rename({x_dim: "x", y_dim: "y"})
    zones = gpd.GeoDataFrame(
        {LOCATION_ID: polygons[key].astype(str).values}, geometry=polygons.geometry.values, crs=polygons.crs
    ).to_crs(template_da.rio.crs)
    if zones[LOCATION_ID].duplicated().any():
        dupes = zones.loc[zones[LOCATION_ID].duplicated(), LOCATION_ID].unique()[:5].tolist()
        raise ValueError(f"Duplicate polygon keys, e.g. {dupes}.")
    invalid = ~zones.is_valid
    if invalid.any():
        # Keep only polygons: exactextract rejects a polygon mixed with collapsed lines, and an empty zone
        zones.loc[invalid, "geometry"] = zones.geometry[invalid].make_valid(method="structure", keep_collapsed=False)
        logger.info(f"Repaired {int(invalid.sum())} invalid polygon(s).")
    zones = zones[~zones.is_empty]
    zones = _filter_by_coverage(zones, template_da, min_coverage)
    weights = _pixel_coverage(template_da, zones)
    if weights.empty:
        raise ValueError("No grid pixels intersect the polygons.")
    logger.info(f"{len(weights)} weight(s) for {weights[LOCATION_ID].nunique()} polygon(s).")
    return weights


def _filter_by_coverage(zones: gpd.GeoDataFrame, grid: xr.DataArray, min_coverage: float) -> gpd.GeoDataFrame:
    """Polygons with at least ``min_coverage`` of their area inside the grid's extent."""
    inside = zones.geometry.intersection(box(*grid.rio.bounds()))
    fraction = inside.to_crs(EQUAL_AREA_CRS).area / zones.geometry.to_crs(EQUAL_AREA_CRS).area
    kept = zones[(fraction >= min_coverage).to_numpy()]
    logger.info(f"Kept {len(kept)} of {len(zones)} polygons with at least {min_coverage:.0%} of their area in the grid.")
    if kept.empty:
        raise ValueError(f"No polygons have at least {min_coverage:.0%} of their area inside the grid.")
    return kept


def _pixel_coverage(grid: xr.DataArray, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """exactextract's coverage of each (zone, pixel) pair, with row/col on the full grid.

    The grid is a zero-filled 2-D copy, so weights depend only on geometry (exactextract skips NaN cells) and its
    window reads don't touch the source.
    """
    grid = grid.isel({d: 0 for d in grid.dims if d not in ("x", "y")}, drop=True)
    grid = grid.copy(data=np.zeros(grid.shape, dtype="float32")).rio.write_crs(grid.rio.crs)
    # A column named "id" is read as the OGR feature id, so the keys go under a private name
    vec = gpd.GeoDataFrame({"_zone_key": zones[LOCATION_ID].values}, geometry=zones.geometry.values, crs=zones.crs)
    result = exact_extract(rast=grid, vec=vec, ops=["cell_id", "coverage"], include_cols=["_zone_key"],
                           output="pandas")
    # One row per zone holding lists of cells; a zone intersecting nothing gives a single null row
    result = result.explode(["cell_id", "coverage"]).dropna(subset=["cell_id", "coverage"])
    cell_id = result["cell_id"].to_numpy(dtype=np.int64)
    width = grid.rio.width
    # cell_id runs row-major from the north-west corner; either axis may be stored the other way, so go through
    # coordinate values and let rowcol map them back
    y_values = grid.y.values
    if y_values[0] < y_values[-1]:
        y_values = y_values[::-1]
    x_values = grid.x.values
    if x_values[0] > x_values[-1]:
        x_values = x_values[::-1]
    rows, cols = rowcol(grid.rio.transform(), x_values[cell_id % width], y_values[cell_id // width])
    return pd.DataFrame({
        LOCATION_ID: result["_zone_key"].to_numpy(dtype=object),
        "row": np.asarray(rows, dtype=np.int64),
        "col": np.asarray(cols, dtype=np.int64),
        "fraction_covered": result["coverage"].to_numpy(dtype=np.float32),
    })


def save_weights(path: str | Path, weights: GridWeights) -> None:
    """Save weights for the segments to load (the matrix as CSC, with each column's pixel and each row's zone)."""
    matrix = sparse.csc_matrix(weights.matrix)
    np.savez(
        path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=np.array(matrix.shape),
        rows=weights.rows, cols=weights.cols, location_ids=weights.location_ids.astype(str),
    )


def load_weights(path: str | Path) -> GridWeights:
    """Weights saved by ``save_weights``."""
    with np.load(path) as f:
        matrix = sparse.csc_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))
        return GridWeights(matrix, f["rows"], f["cols"], f["location_ids"].astype(object))


def save_schema(path: str | Path, schema: pa.Schema) -> None:
    """Save the output schema (e.g. the target table's), so every segment writes the same columns and types."""
    Path(path).write_bytes(schema.serialize().to_pybytes())


def load_schema(path: str | Path) -> pa.Schema:
    """A schema saved by ``save_schema``."""
    return pa.ipc.read_schema(pa.py_buffer(Path(path).read_bytes()))


def step_columns(steps: pd.Index, dim: str, lead_names: dict | None = None) -> dict[str, np.ndarray]:
    """Output columns of each step: its value_time, a forecast's reference_time and member, and with ``lead_names``
    its variable_name by lead."""
    if not isinstance(steps, pd.MultiIndex):
        return {"value_time": steps.values}
    cycles = steps.get_level_values(dim)
    columns = {
        "value_time": (cycles + pd.to_timedelta(steps.get_level_values(steps.names[-1]), unit="h")).values,
        "reference_time": cycles.values,
    }
    if MEMBER in steps.names:
        # teehr's member is a string
        columns["member"] = steps.get_level_values(MEMBER).astype(str).values
    if lead_names is not None:
        columns["variable_name"] = steps.get_level_values(steps.names[-1]).map(lead_names).values.astype(str)
    return columns


def block_tasks(batch: pd.Index, da: xr.DataArray, dim: str, fixed: dict,
                lead_names: dict | None = None) -> list[BlockTask]:
    """The batch as rectangular blocks of ``isel`` positions, with each block's step columns.

    Forecast members sharing their cycles and leads form one block, so a chunk holding every member is read once.
    """
    if not fixed:
        return [BlockTask({dim: positions(da[dim], batch)}, step_columns(batch, dim))]
    lead_dim = list(fixed)[-1]
    merged: dict[tuple, list] = {}
    for block in tg.step_blocks(batch, fixed):
        key = (tuple(block.get_level_values(dim).unique()), tuple(block.get_level_values(lead_dim).unique()))
        merged.setdefault(key, []).extend(block.get_level_values(MEMBER).unique() if MEMBER in fixed else [])
    tasks = []
    for (cycles, leads), block_members in merged.items():
        levels = [list(cycles), *([sorted(set(block_members))] if MEMBER in fixed else []), list(leads)]
        steps = pd.MultiIndex.from_product(levels, names=batch.names)
        selection = {name: positions(da[name], level) for name, level in zip(batch.names, levels)}
        tasks.append(BlockTask(selection, step_columns(steps, dim, lead_names)))
    return tasks


def plan_zonal_segments(da: xr.DataArray, written: pd.Index, dim: str, segment_steps: int) -> list[Segment]:
    """Segments of about ``segment_steps`` append-axis slots holding the written steps, aligned to the grid's
    append-axis chunks and shards so no chunk is read by two segments; in time order."""
    coord = da[dim].to_index()
    times = written.get_level_values(dim) if isinstance(written, pd.MultiIndex) else written
    chunk, shard = chunk_layout(da).get(dim, (1, 1))
    slots = segment_slots(segment_steps, chunk, shard)
    starts = sorted({int(p) // slots * slots for p in coord.get_indexer(times.unique())})
    return [Segment(coord[k], coord[min(k + slots, len(coord)) - 1]) for k in starts]


def segment_dir(staging_dir: str | Path, segment: Segment) -> Path:
    """Where a segment's part files go."""
    return Path(staging_dir) / f"segment-{segment.start:%Y%m%d%H}-{segment.end:%Y%m%d%H}"


@dataclass
class SegmentStats:
    """Rows a segment wrote, and their value range."""

    rows: int = 0
    min: float = np.nan
    max: float = np.nan

    def add(self, rows: int, lo: float, hi: float) -> None:
        self.rows += rows
        self.min, self.max = np.nanmin([self.min, lo]), np.nanmax([self.max, hi])


@dataclass
class ZonalWork:
    """Zonal means of one segment of ``da``'s ``written`` steps, staged as Parquet parts under ``staging_dir``.

    - ``da``: a lazily opened grid variable with ``dim``, the ``fixed`` dims (a forecast's member and lead), ``y_dim``
      and ``x_dim``. Each worker process receives it pickled with its store.
    - ``staging_dir``: holds ``weights.npz`` and ``schema.arrow`` (saved by the parent) and the segments' parts.
    - ``lead_names``: a forecast variable named by the window each lead covers (see ``step_columns``).
    - ``kernel_kwargs``: ``mean_areal_block`` settings (``min_coverage``, ``scale``, ``offset``, ``constants``).
    - ``child_parameters``: a child run's parameters besides its segment (the client's stage and arguments).
    """

    da: xr.DataArray
    written: pd.Index
    dim: str
    fixed: dict
    y_dim: str
    x_dim: str
    staging_dir: str
    task_memory_gb: float
    lead_names: dict | None = None
    kernel_kwargs: dict = field(default_factory=dict)
    child_parameters: dict = field(default_factory=dict)

    def parameters(self, segment: Segment) -> dict:
        return {**self.child_parameters, "segment": segment.to_dict()}

    def __call__(self, segment: Segment) -> SegmentStats:
        steps = segment.select(self.written, self.dim)
        out = segment_dir(self.staging_dir, segment)
        # A retried segment starts over
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        stats = SegmentStats()
        if len(steps) == 0:
            return stats
        weights = load_weights(Path(self.staging_dir) / WEIGHTS_FILE)
        da = self.da.transpose(self.dim, *self.fixed, self.y_dim, self.x_dim)
        plan = plan_reads(
            da, steps, self.fixed, weights.rows, weights.cols, weights.location_ids.size,
            dim=self.dim, y_dim=self.y_dim, x_dim=self.x_dim, task_memory_gb=self.task_memory_gb,
        )
        tasks = [t for batch in plan.batches for t in block_tasks(batch, da, self.dim, self.fixed, self.lead_names)]
        kwargs = dict(
            y_dim=self.y_dim, x_dim=self.x_dim, chunk_shape=plan.chunk_shape, max_slice_steps=MAX_SLICE_STEPS,
            schema=load_schema(Path(self.staging_dir) / SCHEMA_FILE), **self.kernel_kwargs,
        )
        # A worker holds the weights and up to ~1.5x its task budget while averaging
        task_bytes = int(1.5 * self.task_memory_gb * 1e9)
        budget = resolve_budget(memory_per_process=task_bytes + _weights_bytes(weights))
        processes = min(budget.processes, len(tasks)) if use_process_pool(len(tasks), budget.processes, min_items=2) else 0
        initargs = (da, str(Path(self.staging_dir) / WEIGHTS_FILE))
        if not processes:
            _init_worker(*initargs)
        items = [(task, str(out / f"part-{i:05d}.parquet")) for i, task in enumerate(tasks)]
        logger.info(f"Segment {segment.start} to {segment.end}: {len(steps)} step(s) in {len(tasks)} block(s), "
                    f"{processes or 'no'} worker process(es).")
        results = run_sync(map_blocking(
            _run_block, items, workers=max(1, processes or 1), args=(kwargs,), processes=processes,
            initializer=_init_worker if processes else None, initargs=initargs if processes else (),
        ))
        for rows, lo, hi in results:
            stats.add(rows, lo, hi)
        logger.info(f"Segment {segment.start} to {segment.end}: {stats.rows} row(s), values {stats.min} to {stats.max}.")
        return stats


def _weights_bytes(weights: GridWeights) -> int:
    m = weights.matrix
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes + weights.rows.nbytes + weights.cols.nbytes
               + 64 * weights.location_ids.size)


# A worker's grid and weights, set once by _init_worker
_WORKER: dict = {}


def _init_worker(da: xr.DataArray, weights_path: str) -> None:
    _WORKER["da"] = da
    _WORKER["weights"] = load_weights(weights_path)


def _run_block(item: tuple[BlockTask, str], kwargs: dict) -> tuple[int, float, float]:
    """One block's means, written to its part file; returns its row count and value range."""
    task, path = item
    table = mean_areal_block(_WORKER["da"].isel(task.selection), _WORKER["weights"], step_columns=task.step_columns,
                             **kwargs)
    if table.num_rows == 0:
        return 0, np.nan, np.nan
    # Renamed into place, so a part file is never half written
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    values = table.column("value")
    return table.num_rows, float(pc.min(values).as_py()), float(pc.max(values).as_py())


def remove_segment(staging_dir: str | Path, segment: Segment) -> None:
    """Remove a loaded segment's parts."""
    shutil.rmtree(segment_dir(staging_dir, segment), ignore_errors=True)
