"""Virtual references written straight through Icechunk."""
import os

import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teehr_icechunk.refs import write_virtual_refs

TIMES = pd.date_range("2020-01-01", periods=4, freq="D")
SHAPE = (2, 3)


@pytest.fixture
def repo_and_file(tmp_path):
    """A repo with a virtual container over tmp_path/src, and one source file of four raw float32 fields."""
    src = tmp_path / "src"
    src.mkdir()
    fields = [np.full(SHAPE, i + 1, "<f4") for i in range(4)]
    (src / "f.bin").write_bytes(b"".join(f.tobytes() for f in fields))
    prefix = f"file://{src}/"
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(str(src))))
    repo = ic.Repository.create(ic.local_filesystem_storage(str(tmp_path / "repo")), config=config,
                                authorize_virtual_chunk_access={prefix: ic.credentials.LocalFileSystemAccess})
    return repo, f"{prefix}f.bin", fields


def _mtime(url):
    """The source file's last modified time, as a listing gives it (naive UTC)."""
    return pd.Timestamp(os.path.getmtime(url.removeprefix("file://")), unit="s")


def _group(repo, dims, coords, shape):
    """A group whose array ``v`` has one uncompressed chunk per step."""
    session = repo.writable_session("main")
    ds = xr.Dataset({"v": (dims, np.full(shape, np.nan, "<f4"))}, coords=coords)
    chunks = tuple(1 if d in ("time", "reference_time", "lead_time") else s for d, s in zip(dims, shape))
    ds.to_zarr(session.store, group="g", zarr_format=3, consolidated=False,
               encoding={"v": {"chunks": chunks, "compressors": None, "fill_value": np.nan}})
    return session


def _refs(index, url, offsets, last_modified=None):
    return pd.DataFrame({"variable": "v", "url": url, "offset": offsets, "length": 4 * SHAPE[0] * SHAPE[1],
                         "last_modified": last_modified or _mtime(url)}, index=index)


def test_analysis_refs_land_in_their_slots(repo_and_file):
    repo, url, fields = repo_and_file
    session = _group(repo, ("time", "y", "x"), {"time": TIMES}, (4, *SHAPE))
    nbytes = fields[0].nbytes
    # Steps out of order, pointing at messages anywhere in the file
    refs = _refs(pd.DatetimeIndex([TIMES[3], TIMES[1]], name="time"), url, [0, 2 * nbytes])
    assert write_virtual_refs(session, "g", refs) == 2
    session.commit("refs")
    v = xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values
    assert (v[3] == 1).all() and (v[1] == 3).all() and np.isnan(v[[0, 2]]).all()


def test_forecast_refs_land_in_their_slots(repo_and_file):
    repo, url, fields = repo_and_file
    coords = {"reference_time": TIMES[:2], "lead_time": np.array([3, 6], "int32")}
    session = _group(repo, ("reference_time", "lead_time", "y", "x"), coords, (2, 2, *SHAPE))
    index = pd.MultiIndex.from_tuples([(TIMES[1], 6), (TIMES[0], 3)], names=["reference_time", "lead_time"])
    write_virtual_refs(session, "g", _refs(index, url, [fields[0].nbytes, 0]))
    session.commit("refs")
    v = xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values
    assert (v[1, 1] == 2).all() and (v[0, 0] == 1).all() and np.isnan(v[0, 1]).all()


def test_arrays_must_hold_one_chunk_per_step(repo_and_file):
    repo, url, _ = repo_and_file
    session = repo.writable_session("main")
    ds = xr.Dataset({"v": (("time", "y", "x"), np.zeros((4, *SHAPE), "<f4"))}, coords={"time": TIMES})
    ds.to_zarr(session.store, group="g", zarr_format=3, consolidated=False, encoding={"v": {"chunks": (2, *SHAPE)}})
    with pytest.raises(ValueError, match="one chunk per step"):
        write_virtual_refs(session, "g", _refs(pd.DatetimeIndex([TIMES[0]], name="time"), url, [0]))


def test_a_rewritten_source_file_is_refused(repo_and_file):
    """A reference recorded before the file last changed fails to read instead of returning other bytes."""
    repo, url, _ = repo_and_file
    session = _group(repo, ("time", "y", "x"), {"time": TIMES}, (4, *SHAPE))
    older = _mtime(url) - pd.Timedelta("1D")
    write_virtual_refs(session, "g", _refs(pd.DatetimeIndex([TIMES[0]], name="time"), url, [0], older))
    session.commit("refs")
    with pytest.raises(ic.StorageError, match="checksum"):
        xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values
