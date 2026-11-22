"""Read plans from a grid's storage chunks, and ensemble batches."""
import numpy as np
import pandas as pd
import xarray as xr

from teehr_icechunk import repo_const as rc
from teehr_icechunk.mean_areal_plan import cycle_batches, plan_reads

TIMES = pd.date_range("2020-01-01", periods=20, freq="h")
CYCLES = pd.date_range("2026-10-01", periods=2, freq="6h")
MEMBERS, LEADS = [0, 1, 2], [0, 3, 6, 9, 12]
PIXELS = (np.array([3, 10, 25, 38]), np.array([2, 9, 20, 28]))


def _analysis(chunks, shards=None):
    da = xr.DataArray(
        np.zeros((len(TIMES), 40, 30), "float32"), dims=("time", "y", "x"), coords={"time": TIMES}
    )
    da.encoding = {"chunks": chunks, **({"shards": shards} if shards else {})}
    return da


def _forecast(member_chunk):
    shape = (len(CYCLES), len(MEMBERS), len(LEADS), 8, 8)
    da = xr.DataArray(
        np.zeros(shape, "float32"), dims=(rc.REFERENCE_TIME, rc.MEMBER, rc.LEAD_TIME, "y", "x"),
        coords={rc.REFERENCE_TIME: CYCLES, rc.MEMBER: MEMBERS, rc.LEAD_TIME: LEADS},
    )
    da.encoding = {"chunks": (1, member_chunk, 2, 4, 4)}
    return da


def _plan(da, written, fixed, rows, cols, n_zones, dim="time", task_memory_gb=1.0):
    return plan_reads(
        da, written, fixed, rows, cols, n_zones, dim=dim, y_dim="y", x_dim="x", task_memory_gb=task_memory_gb
    )


def test_analysis_batches_hold_whole_chunks_and_prefer_shards():
    """Batches are whole 6-step chunks from the axis start, rounded to whole 12-step shards when they fit."""
    plan = _plan(_analysis((6, 8, 8), (12, 8, 8)), TIMES, {}, *PIXELS, n_zones=4)
    starts = [TIMES.get_loc(batch[0]) for batch in plan.batches]
    assert all(start % 12 == 0 for start in starts)
    assert plan.chunk_shape == (8, 8)


def test_sparse_pixels_get_bigger_batches():
    """Batches are sized by the largest read, so pixels in a few chunks allow more steps than pixels in every chunk."""
    def steps_per_batch(rows, cols):
        plan = _plan(_analysis((2, 8, 8)), TIMES, {}, rows, cols, n_zones=4, task_memory_gb=2e-5)
        assert all(len(batch) % 2 == 0 or batch[-1] == TIMES[-1] for batch in plan.batches)
        return max(len(batch) for batch in plan.batches)

    dense = np.meshgrid(np.arange(0, 40, 4), np.arange(0, 32, 4), indexing="ij")
    assert steps_per_batch(*PIXELS) > steps_per_batch(dense[0].ravel(), dense[1].ravel())


def test_members_in_one_chunk_are_batched_together():
    """With every member in one chunk, a batch is a cycle's block of leads across all members."""
    da = _forecast(member_chunk=len(MEMBERS))
    fixed = {rc.MEMBER: np.array(MEMBERS), rc.LEAD_TIME: np.array(LEADS)}
    steps = pd.MultiIndex.from_product([CYCLES, MEMBERS, LEADS], names=[rc.REFERENCE_TIME, *fixed])
    batches = cycle_batches(steps, fixed, lead_steps=2)
    assert len(batches) == len(CYCLES) * 3  # lead blocks [0, 3], [6, 9], [12]
    assert all(set(b.get_level_values(rc.MEMBER)) == set(MEMBERS) for b in batches)

    plan = _plan(da, steps, fixed, np.array([1, 6]), np.array([1, 6]), n_zones=2, dim=rc.REFERENCE_TIME)
    assert all(set(b.get_level_values(rc.MEMBER)) == set(MEMBERS) for b in plan.batches)


def test_members_chunked_separately_may_be_split():
    """With one member per chunk, a one-unit budget gives batches of one member's lead chunk."""
    da = _forecast(member_chunk=1)
    fixed = {rc.MEMBER: np.array(MEMBERS), rc.LEAD_TIME: np.array(LEADS)}
    steps = pd.MultiIndex.from_product([CYCLES, MEMBERS, LEADS], names=[rc.REFERENCE_TIME, *fixed])
    plan = _plan(da, steps, fixed, np.array([1, 6]), np.array([1, 6]), n_zones=2, dim=rc.REFERENCE_TIME,
                 task_memory_gb=1e-9)
    assert all(b.get_level_values(rc.MEMBER).nunique() == 1 for b in plan.batches)
    assert max(len(b) for b in plan.batches) == 2  # the lead chunk
