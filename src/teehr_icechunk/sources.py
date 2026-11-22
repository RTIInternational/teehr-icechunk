"""The gridded source interface: a source's files, its time grid, and its Icechunk repository."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

import pandas as pd
import xarray as xr
from pydantic import BaseModel

from teehr_icechunk.packed_encoding import PackedEncoding
from teehr_icechunk.repo_const import LEAD_TIME, MEMBER, REFERENCE_TIME, STATUS_COORD

# Columns of a source listing, indexed by each file's timestamp (or forecast step)
LISTING_COLUMNS = ["url", STATUS_COORD, "last_modified"]


class GriddedSource(BaseModel, ABC):
    """A gridded source: its files, its time grid, and its IceChunk repository."""

    source_bucket: ClassVar[str]
    # Source-specific obstore kwargs; deployment obstore_kwargs override them
    store_kwargs: ClassVar[dict] = {}
    # Options for the IceChunk virtual chunk container's object store, e.g. region
    virtual_store_kwargs: ClassVar[dict] = {}
    # Source-specific kwargs for the reference parser, e.g. the HDF5 reader's block size
    parser_kwargs: ClassVar[dict] = {}
    # Ingest's write_materialized when a run doesn't set it
    materialize_by_default: ClassVar[bool] = True
    time_step: ClassVar[pd.Timedelta]

    @abstractmethod
    def list_files(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Files published for the range, one per step: ``LISTING_COLUMNS`` indexed by time.

        Forecast sources index by step, (``reference_time``, [``member``], ``lead_time``), selecting
        cycles by reference time.
        """

    @abstractmethod
    def time_origin(self) -> pd.Timestamp:
        """First step of the source's record; the repo's time grid starts here."""

    @abstractmethod
    def repository_name(self) -> str:
        """IceChunk repository (configuration) name; teehr-valid (letters, digits, underscores)."""

    @abstractmethod
    def ingest_variables(self) -> list[str]:
        """Source variables to materialize."""

    def grid_name(self) -> str:
        """Grid (CRS, pixel size, extent) the source's data is on; sources sharing a grid share a name."""
        return self.repository_name()

    def overview_encoding(self) -> dict[str, PackedEncoding]:
        """Overview packing of each ingested variable, keyed by its own name, in its stored units.

        Variables not listed get float32 overviews.
        """
        return {}

    def lead_times(self) -> list[int] | None:
        """Forecast lead times in hours, fixed when the repo is created; None for analysis sources."""
        return None

    def members(self) -> list[int] | None:
        """Ensemble members, fixed when the repo is created; None for sources without members."""
        return None

    def fixed_axes(self) -> dict[str, list[int]]:
        """The repo's axes after its append axis, in order; empty for analysis sources."""
        axes = {MEMBER: self.members(), LEAD_TIME: self.lead_times()}
        return {name: values for name, values in axes.items() if values is not None}

    def append_dim(self) -> str:
        """The repo's append axis: cycles for forecast sources, valid times otherwise."""
        return REFERENCE_TIME if self.lead_times() is not None else "time"

    def timeseries_type(self) -> str:
        """teehr's timeseries table for the source's values."""
        return "secondary" if self.lead_times() is not None else "primary"

    def credentials(self) -> dict:
        """Source access keys, for obstore and the virtual chunk container; empty means anonymous.

        Both take the same dict, so use keys they share: S3 ``access_key_id``, ``secret_access_key``,
        ``session_token``; GCS ``service_account_key``, ``application_credentials``, ``bearer_token``.
        Keys are read once per run, so they must outlive it.
        """
        return {}

    def preprocess(self, ds: xr.Dataset, url: str) -> xr.Dataset:
        """Adjust one file's virtual dataset before concatenation."""
        return ds


def listing_frame(rows: list[dict], index: tuple[str, ...] = ("time",)) -> pd.DataFrame:
    """A source listing (``LISTING_COLUMNS``) from row dicts, indexed by ``index`` and sorted."""
    df = pd.DataFrame(rows, columns=[*index, *LISTING_COLUMNS])
    if len(index) == 1:
        return df.set_index(pd.DatetimeIndex(df.pop(index[0]), name=index[0])).sort_index()
    return df.set_index(list(index)).sort_index()
