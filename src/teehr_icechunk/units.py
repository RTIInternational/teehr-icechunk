"""The run model: work planned as units along the append axis, run in-process or elsewhere.

A unit is a contiguous run of append-axis slots, aligned from the origin to whole storage blocks (shards,
manifest splits), so two units never write the same chunk or manifest. Before any unit runs, the caller
prepares every group and axis; a unit then only writes chunks and commits them, rebasing over other units'
commits. The caller records per-step metadata for each unit that succeeds, as it finishes, so that one writer
owns it and a run that stops part-way keeps what its finished units wrote.
"""
import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import icechunk as ic
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Unit:
    """Append-axis slots ``start`` to ``end``, inclusive."""

    start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict:
        """JSON-friendly form, e.g. a child run's parameters."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        return cls(pd.Timestamp(d["start"]), pd.Timestamp(d["end"]))

    def select(self, steps: pd.Index, dim: str) -> pd.Index:
        """The ``steps`` (times, or forecast steps with ``dim`` as a level) inside this unit."""
        times = steps if not isinstance(steps, pd.MultiIndex) else steps.get_level_values(dim)
        return steps[(times >= self.start) & (times <= self.end)]


def unit_slots(target: int, *blocks: int) -> int:
    """Slots per unit: the smallest multiple of every block size (shards, manifest splits) of at least ``target``."""
    align = math.lcm(*blocks) if blocks else 1
    return align * max(1, math.ceil(target / align))


def plan_units(times: pd.DatetimeIndex, time_origin: pd.Timestamp, time_step: pd.Timedelta, slots: int) -> list[Unit]:
    """The units, of ``slots`` slots counted from the origin, that hold ``times``; in time order."""
    index = (pd.DatetimeIndex(times) - time_origin) // time_step
    first = sorted({int(k) // slots * slots for k in index})
    return [Unit(time_origin + k * time_step, time_origin + (k + slots - 1) * time_step) for k in first]


@dataclass(frozen=True)
class UnitResult:
    """A unit's outcome; ``error`` is None when it succeeded."""

    unit: Unit
    error: str | None = None


class UnitWork(Protocol):
    """What runs on each unit: in this process (``__call__``), or in a child run given ``parameters``."""

    def __call__(self, unit: Unit) -> None: ...

    def parameters(self, unit: Unit) -> dict: ...


# Called in the caller's thread with each unit's result as it finishes, e.g. to record the unit's steps
OnDone = Optional[Callable[[UnitResult], None]]


class Runner(Protocol):
    def run(self, work: UnitWork, units: Sequence[Unit], on_done: OnDone = None) -> list[UnitResult]: ...


class InProcessRunner:
    """Runs units one after another in this process; a failed unit doesn't stop the others."""

    def run(self, work: UnitWork, units: Sequence[Unit], on_done: OnDone = None) -> list[UnitResult]:
        results = []
        for i, unit in enumerate(units, start=1):
            try:
                work(unit)
                result = UnitResult(unit)
            except Exception as e:  # recorded; the caller decides whether the run failed
                logger.exception(f"Unit {i} of {len(units)} ({unit.start} to {unit.end}) failed.")
                result = UnitResult(unit, f"{type(e).__name__}: {e}")
            if on_done:
                on_done(result)
            results.append(result)
        return results


def assert_chunks_only(session: ic.Session) -> None:
    """Raise if the session changed anything but chunks: only the parent creates groups and arrays and changes
    their metadata, as concurrent units would otherwise overwrite each other's (Icechunk's rebase doesn't catch it)."""
    diff = session.status()
    changed = {
        name: nodes for name in ("new_groups", "new_arrays", "deleted_groups", "deleted_arrays", "updated_groups",
                                 "updated_arrays", "moved_nodes")
        if (nodes := getattr(diff, name))
    }
    if changed:
        raise RuntimeError(f"A unit may only write chunks, but it changed {changed}.")


def commit_unit(session: ic.Session, message: str) -> str:
    """Commit a unit's chunks, rebasing over units committed since its session started."""
    assert_chunks_only(session)
    return commit_rebasing(session, message)


def commit_rebasing(session: ic.Session, message: str) -> str:
    """Commit, rebasing over commits made since the session started (units' chunks, or the parent's step metadata)."""
    return session.commit(message, rebase_with=ic.ConflictDetector(), rebase_tries=10)


def raise_failed(results: Sequence[UnitResult]) -> None:
    """Raise if any unit failed, after the caller has recorded the ones that succeeded."""
    failed = [r for r in results if r.error is not None]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(results)} unit(s) failed; the next run retries them. First: {failed[0].unit.start} "
            f"to {failed[0].unit.end}: {failed[0].error}"
        )
