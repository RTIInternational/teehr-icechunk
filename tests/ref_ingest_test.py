"""Reference ingest on the run model: an index source of raw float32 messages, and a parsed source of netCDF files."""
import os
from datetime import datetime
from typing import ClassVar, Literal

import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr

from teehr_icechunk import grid_spec, ingest
from teehr_icechunk import ref_ingest
from teehr_icechunk import time_grid as tg
from teehr_icechunk.data_status import with_status
from teehr_icechunk.models import IngestGriddedDataInput
from teehr_icechunk.repo_const import LEAD_TIME, NOT_EXPECTED, REFERENCE_TIME, STATUS_COORD, UNWRITTEN
from teehr_icechunk.sources import IndexSource, ParsedSource, listing_frame

ORIGIN = pd.Timestamp("2000-01-01")
SHAPE = (2, 3)
LEADS = [1, 2]
FIELD_BYTES = 4 * SHAPE[0] * SHAPE[1]


class FakeIndexSource(IndexSource):
    """One file per cycle holding a field per lead; lead 2 is published on even days only."""

    type: Literal["fake"] = "fake"
    folder: str

    time_step: ClassVar[pd.Timedelta] = pd.Timedelta("1D")
    manifest_split_steps: ClassVar[int] = 10

    @property
    def source_bucket(self) -> str:
        return f"file://{self.folder}"

    def time_origin(self) -> pd.Timestamp:
        return ORIGIN

    def repository_name(self) -> str:
        return "fake"

    def ingest_variables(self) -> list[str]:
        return ["v"]

    def lead_times(self) -> list[int]:
        return LEADS

    def expected_steps(self, reference_times: pd.DatetimeIndex) -> np.ndarray:
        even = np.asarray(pd.DatetimeIndex(reference_times).day % 2 == 0)
        return np.stack([np.ones(len(even), bool), even], axis=1)

    def list_source_files(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        rows = []
        for cycle in pd.date_range(pd.Timestamp(start_dt).ceil("D"), pd.Timestamp(end_dt).floor("D")):
            path = os.path.join(self.folder, f"{cycle:%Y%m%d}.bin")
            if os.path.exists(path):
                rows.append({REFERENCE_TIME: cycle, "url": f"file://{path}",
                             "last_modified": pd.Timestamp(os.path.getmtime(path), unit="s")})
        return pd.DataFrame(rows, columns=[REFERENCE_TIME, "url", "last_modified"])

    def steps_from_files(self, files: pd.DataFrame) -> pd.DataFrame:
        rows = [{REFERENCE_TIME: f[REFERENCE_TIME], LEAD_TIME: lead, "url": f["url"], STATUS_COORD: "stable",
                 "last_modified": f["last_modified"]}
                for _, f in files.iterrows() for lead in LEADS if lead == 1 or f[REFERENCE_TIME].day % 2 == 0]
        return listing_frame(rows, (REFERENCE_TIME, LEAD_TIME))

    def reference_table(self, files: pd.DataFrame, args, stored) -> pd.DataFrame:
        steps = self.steps_from_files(files)
        refs = pd.DataFrame({"variable": "v", "url": steps["url"], "length": FIELD_BYTES,
                             "offset": (steps.index.get_level_values(LEAD_TIME) - 1) * FIELD_BYTES,
                             "last_modified": steps["last_modified"]}, index=steps.index)
        return refs

    def template(self, files, args) -> xr.Dataset:
        from virtualizarr.manifests import ChunkManifest, ManifestArray
        from virtualizarr.manifests.utils import create_v3_array_metadata

        dims = (REFERENCE_TIME, LEAD_TIME, "y", "x")
        metadata = create_v3_array_metadata(
            shape=(1, len(LEADS), *SHAPE), data_type=np.dtype("float32"), chunk_shape=(1, 1, *SHAPE),
            fill_value=float("nan"), codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
            dimension_names=dims,
        )
        empty = ManifestArray(metadata=metadata, chunkmanifest=ChunkManifest(entries={}, shape=(1, len(LEADS), 1, 1)))
        coords = {REFERENCE_TIME: [ORIGIN], LEAD_TIME: np.array(LEADS, "int32"),
                  "y": np.arange(SHAPE[0], 0, -1.0), "x": np.arange(SHAPE[1], dtype=float)}
        return xr.Dataset({"v": (dims, empty, {"units": "mm"})}, coords=coords).rio.write_crs("EPSG:4326")


class Input(IngestGriddedDataInput):
    source: FakeIndexSource


def _value(cycle: pd.Timestamp, lead: int) -> float:
    return cycle.day * 10 + lead


@pytest.fixture
def setup(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for cycle in pd.date_range("2000-01-02", "2000-01-25"):
        fields = [np.full(SHAPE, _value(cycle, lead), "<f4") for lead in LEADS]
        (src / f"{cycle:%Y%m%d}.bin").write_bytes(b"".join(f.tobytes() for f in fields))
    prefix = f"file://{src}/"
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(str(src))))
    storage = ic.local_filesystem_storage(str(tmp_path / "repo"))
    access = {prefix: ic.credentials.LocalFileSystemAccess}
    repo = ic.Repository.create(storage, config=config, authorize_virtual_chunk_access=access)
    args = Input(source={"type": "fake", "folder": str(src)}, dest_bucket="b", base_prefix="p", write_mode="upsert")
    return repo, args, lambda: ic.Repository.open(storage, authorize_virtual_chunk_access=access)


def _read(open_repo):
    store = open_repo().readonly_session("main").store
    ds = grid_spec.open_grid_dataset(store, grid_spec.read_grid_spec(store))
    return ds, tg.read_step_metadata(store, "/references", REFERENCE_TIME)


def test_references_land_and_metadata_covers_only_the_window(setup):
    repo, args, open_repo = setup
    results = ref_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                             unit_steps=1)
    assert len(results) == 2  # units of 10 cycles (the manifest split): Jan 1-10 and Jan 11-20
    ds, meta = _read(open_repo)
    assert ds["v"].sel({REFERENCE_TIME: "2000-01-04", LEAD_TIME: 2}).values.tolist() == [[42.0] * 3] * 2
    assert ds[REFERENCE_TIME].values[-1] == np.datetime64("2000-01-14")  # the axis ends with the window
    written = meta[meta[STATUS_COORD] >= 0].index.get_level_values(REFERENCE_TIME)
    assert written.min() == pd.Timestamp("2000-01-03") and written.max() == pd.Timestamp("2000-01-14")
    assert meta.loc[(pd.Timestamp("2000-01-03"), 2), STATUS_COORD] == NOT_EXPECTED  # odd day, lead 2
    assert ref_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo) == []


def test_a_failed_unit_leaves_its_steps_for_the_next_run(setup, monkeypatch):
    repo, args, open_repo = setup
    real = FakeIndexSource.reference_table

    def flaky(self, files, args, stored):
        if (files[REFERENCE_TIME] >= pd.Timestamp("2000-01-11")).any():
            raise OSError("connection closed")
        return real(self, files, args, stored)

    monkeypatch.setattr(FakeIndexSource, "reference_table", flaky)
    with pytest.raises(RuntimeError, match="1 of 2 unit"):
        ref_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                       unit_steps=1)
    _, meta = _read(open_repo)
    cycles = meta.index.get_level_values(REFERENCE_TIME)
    assert (meta[(cycles >= "2000-01-03") & (cycles <= "2000-01-10")][STATUS_COORD] != UNWRITTEN).all()
    assert (meta[(cycles >= "2000-01-11") & (cycles <= "2000-01-14")][STATUS_COORD] <= UNWRITTEN).all()

    monkeypatch.setattr(FakeIndexSource, "reference_table", real)
    results = ref_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                             unit_steps=1)
    assert [r.unit.start for r in results] == [pd.Timestamp("2000-01-11")]
    ds, _ = _read(open_repo)
    assert ds["v"].sel({REFERENCE_TIME: "2000-01-12", LEAD_TIME: 2}).values[0, 0] == 122.0


# --- A parsed source: one netCDF file per day, stored in HDF5 tiles ---

GRID = (4, 6)
TILES = (1, 2, 3)  # 2 x 2 tiles per step


class FakeParsedSource(ParsedSource):
    """One file per day; days without a file aren't listed."""

    type: Literal["parsed"] = "parsed"
    folder: str

    time_step: ClassVar[pd.Timedelta] = pd.Timedelta("1D")

    @property
    def source_bucket(self) -> str:
        return f"file://{self.folder}"

    def time_origin(self) -> pd.Timestamp:
        return ORIGIN

    def repository_name(self) -> str:
        return "parsed"

    def ingest_variables(self) -> list[str]:
        return ["v"]

    def list_files(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        rows = []
        for day in pd.date_range(pd.Timestamp(start_dt).ceil("D"), pd.Timestamp(end_dt).floor("D")):
            path = os.path.join(self.folder, f"{day:%Y%m%d}.nc")
            if os.path.exists(path):
                rows.append({"time": day, "url": f"file://{path}", STATUS_COORD: "stable",
                             "last_modified": pd.Timestamp(os.path.getmtime(path), unit="s")})
        return listing_frame(rows)

    def preprocess(self, ds: xr.Dataset, url: str) -> xr.Dataset:
        return with_status(ds, "stable")


class ParsedInput(IngestGriddedDataInput):
    source: FakeParsedSource


def _day_file(folder, day, chunks=TILES):
    values = (day.day * 100 + np.arange(np.prod(GRID)).reshape(1, *GRID)).astype("float32")
    ds = xr.Dataset({"v": (("time", "y", "x"), values, {"units": "mm"})},
                    coords={"time": [day], "y": np.arange(GRID[0], 0, -1.0), "x": np.arange(GRID[1], dtype=float)})
    ds.to_netcdf(folder / f"{day:%Y%m%d}.nc", engine="h5netcdf", encoding={"v": {"chunksizes": chunks, "_FillValue": -9999.0}})


@pytest.fixture
def parsed(tmp_path):
    """Days 2-12 of January, without the 5th and with the 8th unreadable; and a repo builder over them."""
    src = tmp_path / "src"
    src.mkdir()
    for day in pd.date_range("2000-01-02", "2000-01-12"):
        if day.day != 5:
            _day_file(src, day)
    (src / "20000108.nc").write_bytes(b"not a netCDF file")
    prefix = f"file://{src}/"

    def new_repo(name):
        config = ic.RepositoryConfig.default()
        config.set_virtual_chunk_container(ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(str(src))))
        return ic.Repository.create(ic.local_filesystem_storage(str(tmp_path / name)), config=config,
                                    authorize_virtual_chunk_access={prefix: ic.credentials.LocalFileSystemAccess})

    args = ParsedInput(source={"type": "parsed", "folder": str(src)}, dest_bucket="b", base_prefix="p",
                       fallback_source_crs="EPSG:5070", unit_steps=4)
    return src, args, new_repo


def _repo_snapshot(repo) -> dict:
    """Every array's metadata, attrs and values, and the groups' attrs; write times only by their metadata."""
    store = repo.readonly_session("main").store
    root = zarr.open_group(store, mode="r", zarr_format=3)
    # The group's coordinates list is left out: the one-commit path's region writes cut it to the step metadata,
    # where units keep the full list the group was created with (as index sources do)
    group_attrs = {k: v for k, v in root["references"].attrs.items() if k != "coordinates"}
    snap = {"/": dict(root.attrs), "references": group_attrs}
    for name, arr in root["references"].arrays():
        values = arr[...] if name not in ("created_at", "updated_at") else None
        snap[name] = (arr.metadata.to_dict(), None if values is None else np.where(np.isnan(values), -1, values).tolist()
                      if values.dtype.kind == "f" else values.tolist())
    return snap


def test_parsed_references_match_the_one_commit_path(parsed):
    """References written by units (one row per HDF5 tile) store exactly what the one-commit path stored; a missing
    or unreadable file leaves its step unwritten."""
    src, args, new_repo = parsed
    start, end = pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-12")

    old = new_repo("old")
    todo = args.source.list_files(start, end)
    blocks = ingest.virtual_blocks(todo, args.source, {}, args)
    ingest.write_references(old, blocks, todo, ORIGIN, args.source, args)
    ingest.record_data_group(old, "/references")

    new = new_repo("new")
    results = ref_ingest.ingest_references(args, start, end, repo=new, unit_steps=args.unit_steps)
    assert len(results) == 3  # units of 4 days from the origin
    assert _repo_snapshot(new) == _repo_snapshot(old)
    meta = tg.read_step_metadata(new.readonly_session("main").store, "/references", "time")
    assert meta.loc[["2000-01-05", "2000-01-08"], STATUS_COORD].tolist() == [UNWRITTEN, UNWRITTEN]
    v = xr.open_zarr(new.readonly_session("main").store, group="references", consolidated=False)["v"]
    assert v.sel(time="2000-01-09").values[3, 5] == 900 + 23


def test_a_file_laid_out_unlike_the_repo_fails_its_unit(parsed):
    src, args, new_repo = parsed
    repo = new_repo("repo")
    ref_ingest.ingest_references(args, pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-04"), repo=repo)
    _day_file(src, pd.Timestamp("2000-01-13"), chunks=(1, *GRID))  # one tile, where the repo has four
    with pytest.raises(RuntimeError, match="unlike the repo"):
        ref_ingest.ingest_references(args, pd.Timestamp("2000-01-13"), pd.Timestamp("2000-01-13"), repo=repo)
