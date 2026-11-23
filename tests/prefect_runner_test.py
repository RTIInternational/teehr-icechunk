"""PrefectRunner's bookkeeping, with run_deployment stubbed (no Prefect server needed)."""
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("prefect")
from teehr_icechunk import prefect as ic_prefect  # noqa: E402
from teehr_icechunk.units import plan_units  # noqa: E402

ORIGIN = pd.Timestamp("2000-01-01")
UNITS = plan_units(pd.date_range(ORIGIN, periods=5, freq="10D"), ORIGIN, pd.Timedelta("1D"), 10)


class Work:
    def __call__(self, unit):
        raise AssertionError("a Prefect runner never runs units in the parent")

    def parameters(self, unit):
        return {"unit": unit.to_dict()}


def _state(completed: bool):
    return SimpleNamespace(is_completed=lambda: completed, type=SimpleNamespace(value="COMPLETED" if completed else "FAILED"))


def test_children_get_their_unit_and_failures_are_recorded(monkeypatch):
    calls, running, peak, lock = [], [0], [0], threading.Lock()

    def fake_run_deployment(name, parameters, job_variables, timeout, flow_run_name):
        with lock:
            calls.append((name, parameters["unit"]["start"], job_variables))
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        start = pd.Timestamp(parameters["unit"]["start"])
        if start == UNITS[1].start:
            raise ConnectionError("API unreachable")
        return SimpleNamespace(name=flow_run_name, state=_state(start != UNITS[3].start))

    monkeypatch.setattr(ic_prefect, "run_deployment", fake_run_deployment)
    runner = ic_prefect.PrefectRunner("process-gridded-unit/hub", max_parallel=2, job_variables={"cpu": 4})
    results = runner.run(Work(), UNITS)

    assert [r.unit for r in results] == UNITS
    assert [r.error is None for r in results] == [True, False, True, False, True]
    assert "ConnectionError" in results[1].error and "FAILED" in results[3].error
    assert peak[0] <= 2
    assert sorted(c[1] for c in calls) == sorted(u.start.isoformat() for u in UNITS)
    assert all(c[0] == "process-gridded-unit/hub" and c[2] == {"cpu": 4} for c in calls)
