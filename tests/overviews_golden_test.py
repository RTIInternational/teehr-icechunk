"""Overviews built end to end match what they stored when the goldens were recorded (values, encoding, step metadata)."""
import json
import os
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

pytest.importorskip("h5netcdf")
ic = pytest.importorskip("icechunk")

from teehr_icechunk import grid_utils as gu  # noqa: E402
from teehr_icechunk import ingest, overviews  # noqa: E402
from teehr_icechunk import time_grid as tg  # noqa: E402
from teehr_icechunk.data_status import with_status  # noqa: E402
from teehr_icechunk.models import BuildOverviewsDataInput, IngestGriddedDataInput  # noqa: E402
from teehr_icechunk.packed_encoding import PackedEncoding  # noqa: E402
from teehr_icechunk.repo_const import (  # noqa: E402
    APPEND_DIM_ATTR, FIXED_DIMS_ATTR, FORECAST_LAYOUT_VERSION, LAYOUT_VERSION_ATTR, TIME_ORIGIN_ATTR, TIME_STEP_ATTR,
)
from teehr_icechunk.sources import GriddedSource  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
ORIGIN, DAY = pd.Timestamp("2000-01-01"), pd.Timedelta("1D")
DAYS = pd.date_range(ORIGIN, periods=20, freq="D")
NY, NX = 13, 17  # odd sizes, so coarser levels drop partial edge blocks
FILL = -999.0
MODIFIED = pd.Timestamp("2026-01-01")


def _local_repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    prefix = f"file://{src}/"
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(str(src))))
    repo = ic.Repository.create(ic.local_filesystem_storage(str(tmp_path / "repo")), config=config,
                                authorize_virtual_chunk_access={prefix: ic.credentials.LocalFileSystemAccess})
    return repo, src, prefix


def _build_overviews(repo, args, monkeypatch):
    monkeypatch.setattr(overviews.gu, "build_icechunk_s3_storage", lambda **kwargs: None)
    monkeypatch.setattr(overviews.gu, "open_repo_for_reading", lambda storage, credentials=None: repo)
    overviews.build_overviews(args)


def _snapshot(repo) -> dict:
    """The overview groups' attrs, arrays' metadata and stored values (raw, as packed), and the data group's attrs.

    Write times (created_at, updated_at) depend on when the test ran, so only their metadata is kept.
    """
    store = repo.readonly_session("main").store
    root = zarr.open_group(store, mode="r", zarr_format=3)
    data_group = gu.read_data_group(store).strip("/")
    snap = {f"{data_group}/attrs": dict(root[data_group].attrs)}
    for factor, level in sorted(root["overviews"].groups()):
        snap[f"overviews/{factor}/attrs"] = dict(level.attrs)
        for name, arr in sorted(level.arrays()):
            meta = arr.metadata.to_dict()
            meta.pop("attributes", None)
            entry = {"meta": meta, "attrs": dict(arr.attrs)}
            if name not in ("created_at", "updated_at"):
                values = arr[...]
                entry["values"] = np.where(np.isnan(values), None, values).tolist() if values.dtype.kind == "f" else values.tolist()
            snap[f"overviews/{factor}/{name}"] = entry
    return json.loads(json.dumps(snap, default=str))


def _check_golden(snap: dict, name: str) -> None:
    path = FIXTURES / name
    if os.environ.get("RECORD_GOLDEN"):  # only for a deliberate change to what overviews store
        path.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(path.read_text())


# --- Analysis: NWM-like virtual /references, one variable packed and one float32 ---

def _day_file(day: pd.Timestamp, src: Path) -> str:
    """One day's file: RAINRATE with source fills and values beyond its packing range, and T2D."""
    rng = np.random.default_rng(day.dayofyear)
    rain = rng.uniform(0, 0.1, (1, NY, NX)).astype("float32")  # above 0.075 is clipped by the packing
    rain[rng.random(rain.shape) < 0.15] = FILL
    rain[:, :3, :5] = FILL  # a fully missing corner
    t2d = (250 + 40 * rng.random((1, NY, NX))).astype("float32")
    ds = xr.Dataset(
        {"RAINRATE": (("time", "y", "x"), rain, {"units": "mm s^-1"}), "T2D": (("time", "y", "x"), t2d, {"units": "K"})},
        coords={"time": [day], "y": 4000.0 - 1000.0 * np.arange(NY), "x": 1000.0 * np.arange(NX)},
    )
    name = f"f_{day:%Y%m%d}.nc"
    ds.to_netcdf(src / name, engine="h5netcdf", encoding={v: {"_FillValue": FILL} for v in ("RAINRATE", "T2D")})
    return name


def _analysis_source(prefix: str) -> GriddedSource:
    class LocalSource(GriddedSource):
        type: Literal["local"] = "local"
        source_bucket: ClassVar[str] = prefix
        time_step: ClassVar[pd.Timedelta] = DAY

        def list_files(self, start_dt, end_dt):
            raise NotImplementedError

        def time_origin(self):
            return ORIGIN

        def repository_name(self):
            return "golden"

        def ingest_variables(self):
            return ["RAINRATE", "T2D"]

        def overview_encoding(self):
            return {"RAINRATE": PackedEncoding(dtype="uint16", max_value=0.075, units="mm s^-1")}

        def preprocess(self, ds, url):
            return with_status(ds, "stable")

    return LocalSource()


def _write_days(repo, src, prefix, args, days, modified):
    todo = pd.DataFrame(
        {"url": [f"{prefix}{_day_file(d, src)}" for d in days], "status": "stable", "last_modified": modified},
        index=pd.DatetimeIndex(days, name="time"),
    )
    blocks = ingest.virtual_blocks(todo, args.source, {}, args)
    ingest.write_references(repo, blocks, todo, ORIGIN, args.source, args)
    ingest.record_data_group(repo, "/references")


def test_analysis_overviews_match_golden(tmp_path, monkeypatch):
    """Two runs: out-of-order days with one missing, then later days and an upserted one (only stale steps rebuilt)."""
    repo, src, prefix = _local_repo(tmp_path)
    args = IngestGriddedDataInput(
        source=_analysis_source(prefix), dest_bucket="b", base_prefix="p", fallback_source_crs="EPSG:5070",
        num_shard_chunks=4, time_batch_size=8, overview_factors=[2, 4],
    )
    _write_days(repo, src, prefix, args, DAYS[[9, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11]], MODIFIED)
    _build_overviews(repo, args, monkeypatch)
    _write_days(repo, src, prefix, args, DAYS[[3, *range(12, 20)]], MODIFIED + DAY)
    _build_overviews(repo, args, monkeypatch)
    _check_golden(_snapshot(repo), "overviews_analysis_golden.json")


# --- Forecast: an ensemble /data with member and lead axes ---

CYCLES = pd.date_range("2026-10-01", periods=3, freq="6h")
FIXED = {"member": [0, 1], "lead_time": [1, 2, 3, 6]}


def _forecast_block(cycles, members, leads):
    rng = np.random.default_rng(len(cycles) * 100 + sum(leads))
    shape = (len(cycles), len(members), len(leads), NY, NX)
    values = rng.random(shape).astype("float32")
    values[rng.random(shape) < 0.1] = np.nan
    ds = xr.Dataset(
        {"RAINRATE": (("reference_time", "member", "lead_time", "y", "x"), values, {"units": "mm s^-1"})},
        coords={"reference_time": cycles, "member": np.array(members, "int32"), "lead_time": np.array(leads, "int32"),
                "y": 4000.0 - 1000.0 * np.arange(NY), "x": 1000.0 * np.arange(NX)},
        attrs={"spatial:dimensions": ["y", "x"]},
    )
    return ds


def _write_forecast(repo, ds):
    dim, fixed_dims = "reference_time", tuple(FIXED)
    session = repo.writable_session("main")
    stored = tg.read_step_metadata(session.store, "data", dim)
    if stored is None:
        tg.write_grid_attrs(session.store, {
            APPEND_DIM_ATTR: dim, TIME_ORIGIN_ATTR: CYCLES[0].isoformat(), TIME_STEP_ATTR: pd.Timedelta("6h").isoformat(),
            FIXED_DIMS_ATTR: list(fixed_dims), LAYOUT_VERSION_ATTR: FORECAST_LAYOUT_VERSION,
        })
        gu.write_data_group(session.store, "/data")
        tg.create_group(session, "data", ds, CYCLES[0], pd.Timedelta("6h"), dim, ds[dim].to_index()[-1], fixed=FIXED)
        stored = tg.read_step_metadata(session.store, "data", dim)
    else:
        tg.ensure_axis(session, "data", ds[dim].to_index()[-1], pd.Timedelta("6h"), dim)
    n = int(np.prod([ds.sizes[d] for d in (dim, *fixed_dims)]))
    ds = tg.with_step_metadata(ds, dim, np.full(n, 2), np.full(n, MODIFIED.to_datetime64()), tg.utc_now(), stored,
                               fixed_dims=fixed_dims)
    tg.region_write(session, "data", ds, dim, fixed_dims=fixed_dims)
    session.commit("write")


def test_forecast_overviews_match_golden(tmp_path, monkeypatch):
    """Two runs: a partial cycle and a whole one, then the rest of the partial cycle and a new one."""
    repo, _, _ = _local_repo(tmp_path)
    args = BuildOverviewsDataInput(
        configuration_name="golden", dest_bucket="b", base_prefix="p", num_shard_chunks=2, time_batch_size=4,
        overview_factors=[2], overview_encoding={"RAINRATE": PackedEncoding(dtype="uint16", max_value=0.8, units="mm s^-1")},
    )
    _write_forecast(repo, _forecast_block(CYCLES[[1]], [0], [2, 3]))
    _write_forecast(repo, _forecast_block(CYCLES[[0]], [0, 1], FIXED["lead_time"]))
    _build_overviews(repo, args, monkeypatch)
    _write_forecast(repo, _forecast_block(CYCLES[[1]], [1], [1, 2, 3]))
    _write_forecast(repo, _forecast_block(CYCLES[[2]], [0, 1], FIXED["lead_time"]))
    _build_overviews(repo, args, monkeypatch)
    _check_golden(_snapshot(repo), "overviews_forecast_golden.json")
