"""Units as Prefect child flow runs, for fan-out across pods (the ``[prefect]`` extra).

The parent flow plans the units and calls ``PrefectRunner.run``, which starts one run of the child-unit
deployment per unit (at most ``max_parallel`` at a time) and waits for them. The child flow, defined in each
client so its parameters validate against the client's sources, calls ``run_unit``.
"""
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from prefect.deployments import run_deployment

from teehr_icechunk import index_ingest
from teehr_icechunk.models import IngestGriddedDataInput
from teehr_icechunk.units import Unit, UnitResult, UnitWork

logger = logging.getLogger(__name__)


@dataclass
class PrefectRunner:
    """Runs each unit as a run of ``deployment`` ("<flow name>/<deployment name>"), as subflows of the caller.

    ``job_variables`` override the child's pod (e.g. its resources); ``timeout_seconds`` bounds the wait for
    one child (None waits until it finishes; the child's own flow timeout still applies).
    """

    deployment: str
    max_parallel: int = 8
    job_variables: dict = field(default_factory=dict)
    timeout_seconds: float | None = None

    def run(self, work: UnitWork, units: Sequence[Unit]) -> list[UnitResult]:
        def run_one(unit: Unit) -> UnitResult:
            try:
                flow_run = run_deployment(
                    self.deployment, parameters=work.parameters(unit), job_variables=self.job_variables or None,
                    timeout=self.timeout_seconds, flow_run_name=f"unit-{unit.start:%Y%m%d%H}-{unit.end:%Y%m%d%H}",
                )
            except Exception as e:  # e.g. the API was unreachable; recorded like a failed run
                return UnitResult(unit, f"{type(e).__name__}: {e}")
            state = flow_run.state
            if state is None or not state.is_completed():
                return UnitResult(unit, f"child run {flow_run.name} ended {state.type.value if state else 'without a state'}")
            return UnitResult(unit)

        logger.info(f"Running {len(units)} unit(s) as runs of {self.deployment}, {self.max_parallel} at a time.")
        with ThreadPoolExecutor(min(self.max_parallel, max(1, len(units)))) as pool:
            # Each thread runs in a copy of this context, so the children are subflows of the calling flow run
            futures = [pool.submit(contextvars.copy_context().run, run_one, unit) for unit in units]
            results = [f.result() for f in futures]
        failed = sum(r.error is not None for r in results)
        logger.info(f"{len(results) - failed} of {len(results)} unit(s) succeeded.")
        return results


def run_unit(args: IngestGriddedDataInput, unit: dict, window: dict) -> None:
    """A child run's work: one unit of an index source's references (``index_ingest.IndexRefsWork``)."""
    repo = index_ingest.open_repo(args)
    work = index_ingest.IndexRefsWork(repo, args, pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
    work(Unit.from_dict(unit))
