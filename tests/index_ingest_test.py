"""Index-source ingest on the run model, with a small local source of raw float32 messages."""
import os
from datetime import datetime
from typing import ClassVar, Literal

import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from teehr_icechunk import grid_spec
from teehr_icechunk import index_ingest
from teehr_icechunk import time_grid as tg
from teehr_icechunk.models import IngestGriddedDataInput
from teehr_icechunk.repo_const import LEAD_TIME, NOT_EXPECTED, REFERENCE_TIME, STATUS_COORD, UNWRITTEN
from teehr_icechunk.sources import IndexSource, listing_frame

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

    def reference_table(self, files: pd.DataFrame) -> pd.DataFrame:
        steps = self.steps_from_files(files)
        refs = pd.DataFrame({"variable": "v", "url": steps["url"], "length": FIELD_BYTES,
                             "offset": (steps.index.get_level_values(LEAD_TIME) - 1) * FIELD_BYTES,
                             "last_modified": steps["last_modified"]}, index=steps.index)
        return refs

    def template(self) -> xr.Dataset:
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
    results = index_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                             unit_steps=1)
    assert len(results) == 2  # units of 10 cycles (the manifest split): Jan 1-10 and Jan 11-20
    ds, meta = _read(open_repo)
    assert ds["v"].sel({REFERENCE_TIME: "2000-01-04", LEAD_TIME: 2}).values.tolist() == [[42.0] * 3] * 2
    assert ds[REFERENCE_TIME].values[-1] == np.datetime64("2000-01-14")  # the axis ends with the window
    written = meta[meta[STATUS_COORD] >= 0].index.get_level_values(REFERENCE_TIME)
    assert written.min() == pd.Timestamp("2000-01-03") and written.max() == pd.Timestamp("2000-01-14")
    assert meta.loc[(pd.Timestamp("2000-01-03"), 2), STATUS_COORD] == NOT_EXPECTED  # odd day, lead 2
    assert index_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo) == []


def test_a_failed_unit_leaves_its_steps_for_the_next_run(setup, monkeypatch):
    repo, args, open_repo = setup
    real = FakeIndexSource.reference_table

    def flaky(self, files):
        if (files[REFERENCE_TIME] >= pd.Timestamp("2000-01-11")).any():
            raise OSError("connection closed")
        return real(self, files)

    monkeypatch.setattr(FakeIndexSource, "reference_table", flaky)
    with pytest.raises(RuntimeError, match="1 of 2 unit"):
        index_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                       unit_steps=1)
    _, meta = _read(open_repo)
    cycles = meta.index.get_level_values(REFERENCE_TIME)
    assert (meta[(cycles >= "2000-01-03") & (cycles <= "2000-01-10")][STATUS_COORD] != UNWRITTEN).all()
    assert (meta[(cycles >= "2000-01-11") & (cycles <= "2000-01-14")][STATUS_COORD] <= UNWRITTEN).all()

    monkeypatch.setattr(FakeIndexSource, "reference_table", real)
    results = index_ingest.ingest_references(args, pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-14"), repo=repo,
                                             unit_steps=1)
    assert [r.unit.start for r in results] == [pd.Timestamp("2000-01-11")]
    ds, _ = _read(open_repo)
    assert ds["v"].sel({REFERENCE_TIME: "2000-01-12", LEAD_TIME: 2}).values[0, 0] == 122.0
