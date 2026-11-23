"""GRIB2 message byte ranges from wgrib2-style ``.idx`` files, for virtual references.

One GRIB message is one chunk (the whole field), so a reference is a message's offset and length. Each
message ends where the next one starts, and the last ends at the file's end; an index whose ranges don't
fit the file, or whose last offset doesn't begin a GRIB message that fits, is stale and refused.

Adapted from dynamical.org's reformatters (``noaa/noaa_grib_index.py``, ``noaa/noaa_virtual_region_job.py``),
BSD-3-Clause, Copyright (c) 2026, dynamical-org.
"""
from dataclasses import dataclass

from teehr_icechunk.grid_spec import window_start


class StaleIndexError(ValueError):
    """The index doesn't describe the data file it sits beside."""


@dataclass(frozen=True)
class IndexMessage:
    """One message of an index: where it starts and what it holds."""

    start: int
    date: str
    element: str
    level: str
    window: str
    selectors: tuple[str, ...]


def parse_index(text: str) -> list[IndexMessage]:
    """Messages of an index; lines are ``<n>:<start>:d=<YYYYMMDDHH>:<ELEMENT>:<LEVEL>:<WINDOW>:<selectors...>``.

    The element never contains a colon, so a plain split is safe.
    """
    messages = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(":")
        messages.append(IndexMessage(
            start=int(fields[1]), date=fields[2].removeprefix("d="), element=fields[3], level=fields[4],
            window=fields[5], selectors=tuple(f for f in fields[6:] if f),
        ))
    return messages


def message_ranges(messages: list[IndexMessage], file_size: int) -> list[tuple[IndexMessage, int, int]]:
    """(message, offset, length) for every message; raises ``StaleIndexError`` if a range doesn't fit the file."""
    if not messages:
        raise StaleIndexError("The index is empty.")
    starts = [m.start for m in messages]
    ends = [*starts[1:], file_size]
    if any(end > file_size or end <= start for start, end in zip(starts, ends)):
        raise StaleIndexError(f"The index's byte ranges don't fit the {file_size}-byte data file.")
    return [(m, start, end - start) for m, start, end in zip(messages, starts, ends)]


def check_last_message(section0: bytes, offset: int, file_size: int) -> None:
    """Raise ``StaleIndexError`` unless the 16 bytes at the last offset start a GRIB2 message that fits the file.

    A message that changed size shifts every offset after it, so the last one is where a stale index shows.
    """
    if len(section0) < 16 or section0[:4] != b"GRIB" or section0[7] != 2:
        raise StaleIndexError(f"Offset {offset} doesn't begin a GRIB2 message.")
    length = int.from_bytes(section0[8:16], "big")
    if length > file_size - offset:
        raise StaleIndexError(f"The message at {offset} declares {length} bytes, past the {file_size}-byte file.")


def window(lead_hours: int, step_type: str, reset_hours: int | None = None) -> str:
    """An index's time label for a lead: ``"6 hour fcst"`` (instantaneous) or ``"6-9 hour acc fcst"``.

    Accumulations, averages, maxima and minima (``step_type`` ``acc``, ``ave``, ``max``, ``min``) cover the
    hours since the last multiple of ``reset_hours`` (GEFS: 6), or since the cycle when there's no reset.
    """
    if step_type == "instant":
        return f"{lead_hours} hour fcst"
    since = window_start(lead_hours, reset_hours)
    return f"{since}-{lead_hours} hour {step_type} fcst"
