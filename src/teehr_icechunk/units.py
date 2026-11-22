"""The run model: work planned as units along the append axis, run in-process or elsewhere.

A unit is a contiguous run of append-axis slots, aligned from the origin to whole storage blocks (shards,
manifest splits), so two units never write the same chunk or manifest. Before any unit runs, the caller
prepares every group and axis; a unit then only writes chunks and commits them, rebasing over other units'
commits. The caller records per-step metadata for the units that succeeded, so that one writer owns it.
"""
import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

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


class Runner(Protocol):
    def run(self, work: UnitWork, units: Sequence[Unit]) -> list[UnitResult]: ...


class InProcessRunner:
    """Runs units one after another in this process; a failed unit doesn't stop the others."""

    def run(self, work: UnitWork, units: Sequence[Unit]) -> list[UnitResult]:
        results = []
        for i, unit in enumerate(units, start=1):
            try:
                work(unit)
                results.append(UnitResult(unit))
            except Exception as e:  # recorded; the caller decides whether the run failed
                logger.exception(f"Unit {i} of {len(units)} ({unit.start} to {unit.end}) failed.")
                results.append(UnitResult(unit, f"{type(e).__name__}: {e}"))
        return results


def commit_unit(session: ic.Session, message: str) -> str:
    """Commit a unit's chunks, rebasing over units committed since its session started."""
    return session.commit(message, rebase_with=ic.ConflictDetector(), rebase_tries=10)


def raise_failed(results: Sequence[UnitResult]) -> None:
    """Raise if any unit failed, after the caller has recorded the ones that succeeded."""
    failed = [r for r in results if r.error is not None]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(results)} unit(s) failed; the next run retries them. First: {failed[0].unit.start} "
            f"to {failed[0].unit.end}: {failed[0].error}"
        )
