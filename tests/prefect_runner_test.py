"""PrefectRunner's bookkeeping, with run_deployment stubbed (no Prefect server needed)."""
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("prefect")
from teehr_icechunk import prefect as ic_prefect  # noqa: E402
from teehr_icechunk.segments import plan_segments  # noqa: E402

ORIGIN = pd.Timestamp("2000-01-01")
SEGMENTS = plan_segments(pd.date_range(ORIGIN, periods=5, freq="10D"), ORIGIN, pd.Timedelta("1D"), 10)


class Work:
    def __call__(self, segment):
        raise AssertionError("a Prefect runner never runs segments in the parent")

    def parameters(self, segment):
        return {"segment": segment.to_dict()}


def _state(completed: bool):
    return SimpleNamespace(is_completed=lambda: completed, type=SimpleNamespace(value="COMPLETED" if completed else "FAILED"))


def test_children_get_their_unit_and_failures_are_recorded(monkeypatch):
    calls, running, peak, lock = [], [0], [0], threading.Lock()

    def fake_run_deployment(name, parameters, job_variables, timeout, flow_run_name):
        with lock:
            calls.append((name, parameters["segment"]["start"], job_variables))
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        start = pd.Timestamp(parameters["segment"]["start"])
        if start == SEGMENTS[1].start:
            raise ConnectionError("API unreachable")
        return SimpleNamespace(name=flow_run_name, state=_state(start != SEGMENTS[3].start))

    monkeypatch.setattr(ic_prefect, "run_deployment", fake_run_deployment)
    runner = ic_prefect.PrefectRunner("process-gridded-segment/hub", max_parallel=2, job_variables={"cpu": 4})
    results = runner.run(Work(), SEGMENTS)

    assert [r.segment for r in results] == SEGMENTS
    assert [r.error is None for r in results] == [True, False, True, False, True]
    assert "ConnectionError" in results[1].error and "FAILED" in results[3].error
    assert peak[0] <= 2
    assert sorted(c[1] for c in calls) == sorted(u.start.isoformat() for u in SEGMENTS)
    assert all(c[0] == "process-gridded-segment/hub" and c[2] == {"cpu": 4} for c in calls)


def test_run_names_fit_kubernetes_job_names():
    """The Kubernetes worker cuts job names to 45 characters; the segment's dates must survive."""
    name = ic_prefect.run_name(SEGMENTS[0])
    assert name == f"process-gridded-segment-{SEGMENTS[0].start:%Y%m%d%H}-{SEGMENTS[0].end:%Y%m%d%H}"
    assert len(name) <= 45



class Recorded:
    def __init__(self):
        self.ran = []

    def __call__(self, segment):
        self.ran.append(segment)

    def parameters(self, segment):
        return {}


def test_a_single_unit_runs_in_the_parent(monkeypatch):
    def no_child(*args, **kwargs):
        raise AssertionError("no child run for a single segment")

    monkeypatch.setattr(ic_prefect, "run_deployment", no_child)
    work = Recorded()
    results = ic_prefect.PrefectRunner("d", max_parallel=4).run(work, SEGMENTS[:1])
    assert work.ran == SEGMENTS[:1] and results[0].error is None
