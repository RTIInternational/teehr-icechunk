"""The run model: unit planning, the in-process runner, and units committing side by side."""
import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import zarr

from teehr_icechunk.units import InProcessRunner, Unit, commit_unit, plan_units, raise_failed, unit_slots

ORIGIN, STEP = pd.Timestamp("2000-01-01"), pd.Timedelta("1D")


def test_unit_slots_are_whole_blocks():
    assert unit_slots(100, 30) == 120
    assert unit_slots(10, 30, 365) == 2190  # lcm of a 30-step shard and a 365-step manifest split
    assert unit_slots(5) == 5


def test_units_are_aligned_from_the_origin_and_hold_every_time():
    times = pd.DatetimeIndex(["2000-01-02", "2000-01-29", "2000-03-15", "2000-01-31"])
    units = plan_units(times, ORIGIN, STEP, 30)
    assert [(u.start, u.end) for u in units] == [
        (ORIGIN, ORIGIN + 29 * STEP), (ORIGIN + 30 * STEP, ORIGIN + 59 * STEP), (ORIGIN + 60 * STEP, ORIGIN + 89 * STEP),
    ]
    assert sum(len(u.select(times, "time")) for u in units) == len(times)


def test_unit_selects_forecast_steps_by_cycle_and_round_trips():
    steps = pd.MultiIndex.from_product([pd.date_range("2000-01-01", periods=4), [0, 1]], names=["reference_time", "member"])
    unit = Unit(pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-03"))
    assert len(unit.select(steps, "reference_time")) == 4
    assert Unit.from_dict(unit.to_dict()) == unit


def test_a_failed_unit_does_not_stop_the_others():
    done = []

    def work(unit):
        if unit.start == ORIGIN:
            raise OSError("connection closed")
        done.append(unit)

    units = plan_units(pd.date_range(ORIGIN, periods=3, freq="10D"), ORIGIN, STEP, 10)
    results = InProcessRunner().run(work, units)
    assert [r.error is None for r in results] == [False, True, True] and done == units[1:]
    with pytest.raises(RuntimeError, match="1 of 3 unit"):
        raise_failed(results)


def test_units_commit_disjoint_chunks_from_the_same_snapshot(tmp_path):
    """Two units started from one snapshot both land: each rebases over the other's chunks."""
    repo = ic.Repository.create(ic.local_filesystem_storage(str(tmp_path / "repo")))
    session = repo.writable_session("main")
    zarr.create_array(session.store, name="v", shape=(60, 4), chunks=(30, 4), dtype="float32", fill_value=np.nan)
    session.commit("prepare")
    sessions = {start: repo.writable_session("main") for start in (0, 30)}  # both open before either commits
    for start, s in sessions.items():
        zarr.open_array(s.store, path="v")[start:start + 30] = start
    for start, s in sessions.items():
        commit_unit(s, f"unit {start}")
    stored = zarr.open_array(repo.readonly_session("main").store, path="v", mode="r")[:]
    assert (stored[:30] == 0).all() and (stored[30:] == 30).all()
