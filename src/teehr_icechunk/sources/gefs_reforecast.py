"""NOAA GEFS v12 reforecast (2000-2019) on ``s3://noaa-gefs-retrospective``, as virtual references.

Cycles are daily at 00Z with members c00 and p01-p04; once a week (Wednesdays) p05-p10 too, and the run
extends to 35 days. Each GRIB2 file holds one variable at every lead, with a ``.idx`` beside it. Days 1-10
are 3-hourly on a 0.25 degree grid; beyond day 10, 6-hourly on 0.5 degrees. So there are two repos, one per
grid: ``0p25`` (leads 3-240 h) and ``0p50`` (leads 3-840 h, of which 3-240 h hold only the pressure levels
above 700 hPa).

Each message is one chunk, decoded by the gribberish codec (the ``[grib]`` extra) at read time.
Accumulations, averages, maxima and minima are stored as published: their window starts at the last
6-hour boundary (0-3 h, 0-6 h, 6-9 h, ...).
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar, Literal

import numpy as np
import obstore
import pandas as pd
import rioxarray  # noqa: F401 (rio accessor)
import xarray as xr
from obstore.store import S3Store
from pydantic import Field, field_validator

from teehr_icechunk.grib_index import StaleIndexError, check_last_message, message_ranges, parse_index, window
from teehr_icechunk.repo_const import (
    CELL_METHODS_ATTR, LEAD_TIME, MEMBER, REFERENCE_TIME, STATUS_COORD, WINDOW_RESET_HOURS_ATTR,
)
from teehr_icechunk.sources import IndexSource, listing_frame

logger = logging.getLogger(__name__)

BUCKET = "noaa-gefs-retrospective"
PREFIX = "GEFSv12/reforecast"
# Requests in flight while listing and reading indexes; they are small and latency-bound
IO_THREADS = 64
# Back off from 1 s, not obstore's 100 ms: many units reading at once get throttled (503 SlowDown), and
# the retries then spread over tens of seconds within the 3 minute retry timeout
RETRY_CONFIG = {"backoff": {"init_backoff": timedelta(seconds=1)}}
# Messages reset their windows at multiples of this many hours
WINDOW_RESET_HOURS = 6
# The CF cell_methods (along time) of each GRIB step type; arrays carry these, the index labels the step type
CF_TIME_METHODS = {"instant": "point", "acc": "sum", "ave": "mean", "max": "maximum", "min": "minimum"}
WEDNESDAY = 2


@dataclass(frozen=True)
class GefsVariable:
    """A variable as the reforecast publishes it: its file, and how its messages are labelled in the index."""

    file: str  # file name prefix, e.g. "apcp_sfc"
    element: str
    level: str
    step_type: Literal["instant", "acc", "ave", "max", "min"]
    units: str
    long_name: str
    standard_name: str  # CF


# Single-level variables, named <ELEMENT>_<level>; more come with add-variables runs
CATALOG: dict[str, GefsVariable] = {
    "APCP_surface": GefsVariable("apcp_sfc", "APCP", "surface", "acc", "kg m-2", "Total precipitation",
                                 "precipitation_amount"),
    "TMP_2m": GefsVariable("tmp_2m", "TMP", "2 m above ground", "instant", "K", "Temperature at 2 m", "air_temperature"),
    "TMAX_2m": GefsVariable("tmax_2m", "TMAX", "2 m above ground", "max", "K", "Maximum temperature at 2 m",
                            "air_temperature"),
    "TMIN_2m": GefsVariable("tmin_2m", "TMIN", "2 m above ground", "min", "K", "Minimum temperature at 2 m",
                            "air_temperature"),
}

# grid -> (degrees, (n_lat, n_lon))
GRIDS = {"0p25": (0.25, (721, 1440)), "0p50": (0.5, (361, 720))}
_FILE_PATTERN = re.compile(
    rf"^{PREFIX}/(\d{{4}})/(\d{{10}})/(c00|p\d\d)/(Days:1-10|Days:10-16|Days:10-35)/([a-z0-9_]+)_\2_\3\.grib2$"
)


def _member(name: str) -> int:
    return 0 if name == "c00" else int(name[1:])


def folder_leads(folder: str) -> list[int]:
    """Lead hours of the messages in one of a cycle's folders."""
    if folder == "Days:1-10":
        return list(range(3, 241, 3))
    return list(range(246, 385 if folder == "Days:10-16" else 841, 6))


class GEFSv12Reforecast(IndexSource):
    """GEFS v12 reforecast cycles, members 0-10, one repo per grid."""

    type: Literal["gefs-v12-reforecast"] = "gefs-v12-reforecast"
    grid: Literal["0p25", "0p50"] = Field(..., description="0p25: days 1-10 at 0.25 degrees; 0p50: up to 35 days at 0.5")
    variables: list[str] = Field(
        default=["APCP_surface", "TMAX_2m", "TMIN_2m"],
        description="Variables to reference, from CATALOG (more can be added later, e.g. TMP_2m)",
    )

    source_bucket: ClassVar[str] = f"s3://{BUCKET}"
    store_kwargs: ClassVar[dict] = {"region": "us-east-1", "skip_signature": True, "retry_config": RETRY_CONFIG}
    virtual_store_kwargs: ClassVar[dict] = {"region": "us-east-1"}
    materialize_by_default: ClassVar[bool] = False
    time_step: ClassVar[pd.Timedelta] = pd.Timedelta("1D")
    # Cycles per manifest, so a commit rewrites only the manifests of the years it touches
    manifest_split_steps: ClassVar[int] = 365

    @field_validator("variables")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(CATALOG))
        if unknown:
            raise ValueError(f"Unknown GEFS variables {unknown}; known: {sorted(CATALOG)}")
        return v

    def time_origin(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def repository_name(self) -> str:
        return f"gefs_v12_reforecast_{self.grid}"

    def grid_name(self) -> str:
        return f"gefs_{self.grid}_global"

    def ingest_variables(self) -> list[str]:
        return list(self.variables)

    def lead_times(self) -> list[int]:
        if self.grid == "0p25":
            return folder_leads("Days:1-10")
        return folder_leads("Days:1-10") + folder_leads("Days:10-35")

    def members(self) -> list[int]:
        return list(range(11))

    def folders(self, cycle: pd.Timestamp) -> list[str]:
        """The folders of a cycle that hold this grid's variables (above-700 hPa levels aside)."""
        if self.grid == "0p25":
            return ["Days:1-10"]
        return ["Days:10-35" if cycle.weekday() == WEDNESDAY else "Days:10-16"]

    def expected_steps(self, reference_times: pd.DatetimeIndex) -> np.ndarray:
        """Which (cycle, member, lead) steps the reforecast publishes: members 5-10 and leads past 384 h on Wednesdays."""
        leads, members = np.array(self.lead_times()), np.array(self.members())
        wednesday = np.asarray(pd.DatetimeIndex(reference_times).weekday == WEDNESDAY)[:, None, None]
        short = (members[None, :, None] <= 4) & (leads[None, None, :] <= 384)
        # At 0.5 degrees, leads to 240 h hold only the pressure levels above 700 hPa: expected, unwritten until added
        return np.broadcast_to(wednesday | short, (len(reference_times), len(members), len(leads)))

    def _store(self) -> S3Store:
        return S3Store(BUCKET, **self.store_kwargs)

    def list_source_files(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """This repo's GRIB files for cycles in the range, one row each: cycle, member, folder, variable, url, size, last_modified."""
        store = self._store()
        cycles = pd.date_range(pd.Timestamp(start_dt).ceil("D"), pd.Timestamp(end_dt).floor("D"), freq="D")
        files = {CATALOG[v].file: v for v in self.variables}

        def list_cycle(cycle: pd.Timestamp) -> list[dict]:
            rows = []
            for folder in self.folders(cycle):
                for member in range(11 if cycle.weekday() == WEDNESDAY else 5):
                    prefix = f"{PREFIX}/{cycle:%Y}/{cycle:%Y%m%d%H}/{'c00' if member == 0 else f'p{member:02d}'}/{folder}/"
                    try:
                        objects = [obj for batch in obstore.list(store, prefix=prefix) for obj in batch]
                    except Exception as e:  # e.g. malformed keys under a prefix; the step stays unwritten
                        logger.error(f"Couldn't list {prefix}: {e}")
                        continue
                    for obj in objects:
                        match = _FILE_PATTERN.match(obj["path"])
                        if match and match.group(5) in files:
                            rows.append({
                                REFERENCE_TIME: cycle, MEMBER: _member(match.group(3)), "folder": folder,
                                "variable": files[match.group(5)], "url": f"s3://{BUCKET}/{obj['path']}",
                                "size": obj["size"], "last_modified": pd.Timestamp(obj["last_modified"]).tz_convert(None),
                            })
            return rows

        with ThreadPoolExecutor(IO_THREADS) as pool:
            rows = [row for cycle_rows in pool.map(list_cycle, cycles) for row in cycle_rows]
        columns = [REFERENCE_TIME, MEMBER, "folder", "variable", "url", "size", "last_modified"]
        return pd.DataFrame(rows, columns=columns)

    def steps_from_files(self, files: pd.DataFrame) -> pd.DataFrame:
        """One row per (cycle, member, lead) step with files; ``last_modified`` is the newest of the step's files."""
        rows = []
        for (cycle, member, folder), group in files.groupby([REFERENCE_TIME, MEMBER, "folder"]):
            url = group["url"].iloc[0].rsplit("/", 1)[0]
            for lead in folder_leads(folder):
                rows.append({REFERENCE_TIME: cycle, MEMBER: member, LEAD_TIME: lead, "url": url,
                             STATUS_COORD: "stable", "last_modified": group["last_modified"].max()})
        return listing_frame(rows, (REFERENCE_TIME, MEMBER, LEAD_TIME))

    def reference_table(self, files: pd.DataFrame) -> pd.DataFrame:
        """References to every message of ``files`` (rows of ``list_source_files``), indexed by step.

        A file whose index is stale (its ranges or last message don't fit the file) is skipped with an error.
        """
        store = self._store()

        def refs_of(file: pd.Series) -> list[dict]:
            path = file["url"].removeprefix(f"s3://{BUCKET}/")
            var = CATALOG[file["variable"]]
            try:
                ranges = message_ranges(parse_index(bytes(obstore.get(store, f"{path}.idx").bytes()).decode()), file["size"])
                last = ranges[-1][1]
                check_last_message(bytes(obstore.get_range(store, path, start=last, length=16)), last, file["size"])
            except (StaleIndexError, FileNotFoundError) as e:
                logger.error(f"Skipping {file['url']}: {e}")
                return []
            reset = None if var.step_type == "instant" else WINDOW_RESET_HOURS
            by_window = {(m.element, m.level, m.window): (offset, length) for m, offset, length in ranges}
            rows = []
            for lead in folder_leads(file["folder"]):
                found = by_window.get((var.element, var.level, window(lead, var.step_type, reset)))
                if found is not None:
                    rows.append({REFERENCE_TIME: file[REFERENCE_TIME], MEMBER: file[MEMBER], LEAD_TIME: lead,
                                 "variable": file["variable"], "url": file["url"], "offset": found[0],
                                 "length": found[1], "last_modified": file["last_modified"]})
            return rows

        with ThreadPoolExecutor(IO_THREADS) as pool:
            rows = [row for file_rows in pool.map(refs_of, (row for _, row in files.iterrows())) for row in file_rows]
        refs = pd.DataFrame(rows, columns=[REFERENCE_TIME, MEMBER, LEAD_TIME, "variable", "url", "offset", "length",
                                           "last_modified"])
        return refs.set_index([REFERENCE_TIME, MEMBER, LEAD_TIME])

    def template(self) -> xr.Dataset:
        """One cycle of empty virtual arrays, for creating the repo: one chunk per message, decoded by gribberish."""
        from virtualizarr.manifests import ChunkManifest, ManifestArray
        from virtualizarr.manifests.utils import create_v3_array_metadata

        degrees, (n_lat, n_lon) = GRIDS[self.grid]
        members, leads = self.members(), self.lead_times()
        shape = (1, len(members), len(leads), n_lat, n_lon)
        dims = (REFERENCE_TIME, MEMBER, LEAD_TIME, "latitude", "longitude")
        data_vars = {}
        for name in self.variables:
            var = CATALOG[name]
            metadata = create_v3_array_metadata(
                shape=shape, data_type=np.dtype("float32"), chunk_shape=(1, 1, 1, n_lat, n_lon), fill_value=float("nan"),
                # Longitudes -180 to 180 and latitudes north first, as dynamical.org's GEFS, so weights are shared
                codecs=[{"name": "gribberish", "configuration": {"var": var.element, "adjust_longitude_range": True,
                                                                  "north_up": True}}],
                dimension_names=dims,
            )
            empty = ManifestArray(metadata=metadata, chunkmanifest=ChunkManifest(entries={}, shape=(1, len(members), len(leads), 1, 1)))
            attrs = {"units": var.units, "long_name": var.long_name, "standard_name": var.standard_name,
                     "grib_element": var.element, "grib_level": var.level,
                     CELL_METHODS_ATTR: f"time: {CF_TIME_METHODS[var.step_type]}"}
            if var.step_type != "instant":
                attrs[WINDOW_RESET_HOURS_ATTR] = WINDOW_RESET_HOURS
            data_vars[name] = (dims, empty, attrs)
        coords = {
            REFERENCE_TIME: [self.time_origin()], MEMBER: np.array(members, "int32"), LEAD_TIME: np.array(leads, "int32"),
            "latitude": ("latitude", 90 - degrees * np.arange(n_lat), {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", -180 + degrees * np.arange(n_lon), {"units": "degrees_east", "standard_name": "longitude"}),
        }
        return xr.Dataset(data_vars, coords=coords).rio.write_crs("EPSG:4326")
