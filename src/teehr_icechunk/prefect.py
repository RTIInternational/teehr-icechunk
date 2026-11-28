"""Segments as Prefect child flow runs, for fan-out across pods (the ``[prefect]`` extra).

The parent flow plans the segments and calls ``PrefectRunner.run``, which starts one run of the child-segment
deployment per segment (at most ``max_parallel`` at a time) and waits for them. The child flow, defined in each
client so its parameters validate against the client's sources, calls ``run_segment`` with the segment's stage.
"""
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from prefect.deployments import run_deployment

from teehr_icechunk import overviews, ref_ingest
from teehr_icechunk.models import BuildOverviewsDataInput, FanOutInput
from teehr_icechunk.segments import InProcessRunner, OnDone, Runner, Segment, SegmentResult, SegmentWork

logger = logging.getLogger(__name__)


def run_name(segment: Segment) -> str:
    """The child run's name, also its pod's: within the Kubernetes worker's 45 characters, so the dates stay whole."""
    return f"process-gridded-segment-{segment.start:%Y%m%d%H}-{segment.end:%Y%m%d%H}"


@dataclass
class PrefectRunner:
    """Runs each segment as a run of ``deployment`` ("<flow name>/<deployment name>"), as subflows of the caller.

    ``job_variables`` override the child's pod (e.g. its resources); ``timeout_seconds`` bounds the wait for
    one child (None waits until it finishes; the child's own flow timeout still applies). A run of one segment
    (e.g. a scheduled update) runs it in this process: a child pod would only add its start-up.
    """

    deployment: str
    max_parallel: int = 8
    job_variables: dict = field(default_factory=dict)
    timeout_seconds: float | None = None

    def run(self, work: SegmentWork, segments: Sequence[Segment], on_done: OnDone = None) -> list[SegmentResult]:
        if len(segments) == 1:
            return InProcessRunner().run(work, segments, on_done)

        def run_one(segment: Segment) -> SegmentResult:
            try:
                flow_run = run_deployment(
                    self.deployment, parameters=work.parameters(segment), job_variables=self.job_variables or None,
                    timeout=self.timeout_seconds, flow_run_name=run_name(segment),
                )
            except Exception as e:  # e.g. the API was unreachable; recorded like a failed run
                return SegmentResult(segment, f"{type(e).__name__}: {e}")
            state = flow_run.state
            if state is None or not state.is_completed():
                return SegmentResult(segment, f"child run {flow_run.name} ended {state.type.value if state else 'without a state'}")
            return SegmentResult(segment)

        logger.info(f"Running {len(segments)} segment(s) as runs of {self.deployment}, {self.max_parallel} at a time.")
        with ThreadPoolExecutor(min(self.max_parallel, max(1, len(segments)))) as pool:
            # Each thread runs in a copy of this context, so the children are subflows of the calling flow run
            futures = [pool.submit(contextvars.copy_context().run, run_one, segment) for segment in segments]
            for future in as_completed(futures):
                if on_done:
                    on_done(future.result())
            results = [f.result() for f in futures]
        failed = sum(r.error is not None for r in results)
        logger.info(f"{len(results) - failed} of {len(results)} segment(s) succeeded.")
        return results


def runner(args: FanOutInput, deployment: str) -> Runner:
    """Segments as runs of ``deployment``, ``args.max_child_runs`` at a time, or in this process when that is 0."""
    if args.max_child_runs:
        return PrefectRunner(deployment, args.max_child_runs, args.child_job_variables)
    return InProcessRunner()


def run_segment(args: BuildOverviewsDataInput, segment: dict, window: dict | None = None, stage: str = "references") -> None:
    """A child run's work: one segment of a stage, its references (``args`` an ``IngestGriddedDataInput``, within
    the run's ``window``) or its overviews."""
    if stage == "overviews":
        work = overviews.OverviewWork(overviews.open_repo(args), args)
    elif stage == "references":
        repo = ref_ingest.open_repo(args)
        work = ref_ingest.RefsWork(repo, args, pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
    else:
        raise ValueError(f"Unknown stage {stage!r}; 'references' or 'overviews'.")
    work(Segment.from_dict(segment))
