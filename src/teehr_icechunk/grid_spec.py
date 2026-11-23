"""What readers need to know about a gridded Icechunk repo's shape, and its data in a canonical form.

Our repos describe themselves in root attributes; a repo we didn't write is described by an
``override`` (e.g. a registry entry) with the same keys. Readers take the shape from here, never
from dimension names, and address dims by name, never by position.
"""
from dataclasses import dataclass, field

import icechunk as ic
import numpy as np
import xarray as xr

from teehr_icechunk import grid_utils as gu
from teehr_icechunk import time_grid as tg
from teehr_icechunk.repo_const import (
    APPEND_DIM_ATTR,
    DATA_GROUP_ATTR,
    FIXED_DIMS_ATTR,
    FORECAST_COORD_ATTRS,
    FORECAST_LAYOUT_VERSION,
    GRID_NAME_ATTR,
    LAYOUT_VERSION_ATTR,
    LEAD_TIME,
    STEP_TYPE_ATTR,
    VALUE_TIME,
    WINDOW_RESET_HOURS_ATTR,
    WINDOWED_STEP_TYPES,
)

# On-disk layouts these readers understand: 1 is the analysis time grid (also when the attribute is
# missing), 2 adds forecast axes. Any other value is refused rather than guessed at.
KNOWN_LAYOUT_VERSIONS = (1, FORECAST_LAYOUT_VERSION)


@dataclass(frozen=True)
class GridSpec:
    append_dim: str
    # Dims after the append dim, e.g. ("member", "lead_time"); empty for analysis repos
    fixed_dims: tuple[str, ...] = ()
    layout_version: int = 1
    data_group: str = "/data"
    grid_name: str | None = None
    # Source dim name -> canonical name, for repos we didn't write
    renames: dict = field(default_factory=dict)


def read_grid_spec(store: ic.IcechunkStore, override: dict | None = None) -> GridSpec:
    """The repo's grid spec from its root attributes, with ``override`` taking precedence."""
    attrs = {**tg.read_root_attrs(store), **(override or {})}
    version = attrs.get(LAYOUT_VERSION_ATTR, 1)
    if version not in KNOWN_LAYOUT_VERSIONS:
        raise ValueError(f"Unknown layout_version {version!r}; this code reads {list(KNOWN_LAYOUT_VERSIONS)}.")
    return GridSpec(
        append_dim=attrs.get(APPEND_DIM_ATTR, "time"),
        fixed_dims=tuple(attrs.get(FIXED_DIMS_ATTR, ())),
        layout_version=version,
        data_group=attrs.get(DATA_GROUP_ATTR) or gu.read_data_group(store),
        grid_name=attrs.get(GRID_NAME_ATTR),
        renames=dict(attrs.get("renames", {})),
    )


def open_grid_dataset(store: ic.IcechunkStore, spec: GridSpec) -> xr.Dataset:
    """The repo's data group with canonical forecast dims, integer-hour ``lead_time`` and a ``value_time``."""
    ds = xr.open_zarr(store, group=spec.data_group, consolidated=False, chunks=None, decode_coords="all")
    ds = ds.rename({k: v for k, v in spec.renames.items() if k in ds.variables})
    lead = LEAD_TIME
    if lead in ds.coords and np.issubdtype(ds[lead].dtype, np.timedelta64):
        hours = ds[lead].values / np.timedelta64(1, "h")
        if not np.array_equal(hours, np.round(hours)):
            raise ValueError(f"{lead} isn't in whole hours: {ds[lead].values[:5]}")
        ds = ds.assign_coords({lead: (lead, hours.astype("int32"), {**ds[lead].attrs, "units": "hours"})})
    if lead in ds.dims and VALUE_TIME not in ds.coords:
        dim = spec.append_dim
        ds = ds.assign_coords({VALUE_TIME: ((dim, lead), tg.value_times(ds[dim].values, ds[lead].values),
                                               FORECAST_COORD_ATTRS[VALUE_TIME])})
    return ds


def window_start(lead_hours: int, reset_hours: int | None) -> int:
    """The hour after the cycle that a window ending at ``lead_hours`` starts: the last multiple of
    ``reset_hours`` before it, or 0 (the cycle) when windows never reset (``reset_hours`` None or 0)."""
    return (lead_hours - 1) // reset_hours * reset_hours if reset_hours else 0


def window_hours(leads, attrs: dict) -> dict[int, int]:
    """The hours each lead's value covers, from its variable's attributes; leads covering none (lead 0) are left out.

    A windowed variable (``step_type`` acc, ave, max or min, with ``window_reset_hours``) covers the hours since its
    window started: GEFS APCP at 0.25 degrees covers 3 h at leads 3, 9, ... and 6 h at leads 6, 12, ....
    ``window_reset_hours`` 0 means windows never reset, so each value covers the hours since the cycle.
    Anything else covers the hours since the lead before it (the first lead's, since 0).
    """
    windowed = attrs.get(STEP_TYPE_ATTR) in WINDOWED_STEP_TYPES and WINDOW_RESET_HOURS_ATTR in attrs
    reset = int(attrs[WINDOW_RESET_HOURS_ATTR]) if windowed else None
    hours, previous = {}, 0
    for lead in (int(x) for x in leads):
        covered = lead - (window_start(lead, reset) if windowed else previous)
        previous = lead
        if covered > 0:
            hours[lead] = covered
    return hours
