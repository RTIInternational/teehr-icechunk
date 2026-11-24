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


def _group(repo, dims, coords, shape, chunks=None):
    """A group whose array ``v`` has one uncompressed chunk per step, or ``chunks``."""
    session = repo.writable_session("main")
    ds = xr.Dataset({"v": (dims, np.full(shape, np.nan, "<f4"))}, coords=coords)
    chunks = chunks or tuple(1 if d in ("time", "reference_time", "lead_time") else s for d, s in zip(dims, shape))
    ds.to_zarr(session.store, group="g", zarr_format=3, consolidated=False,
               encoding={"v": {"chunks": chunks, "compressors": None, "fill_value": np.nan}})
    return session


def _refs(index, url, offsets, last_modified=None, length=4 * SHAPE[0] * SHAPE[1]):
    return pd.DataFrame({"variable": "v", "url": url, "offset": offsets, "length": length,
                         "last_modified": _mtime(url) if last_modified is None else last_modified}, index=index)


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
    with pytest.raises(ValueError, match="one step per chunk"):
        write_virtual_refs(session, "g", _refs(pd.DatetimeIndex([TIMES[0]], name="time"), url, [0]))


def test_tiled_refs_land_in_their_chunks(repo_and_file):
    """A step split into tiles (e.g. HDF5 chunks): each row's chunk_<dim> columns place it; a tiled dim needs them."""
    repo, url, fields = repo_and_file
    session = _group(repo, ("time", "y", "x"), {"time": TIMES}, (4, *SHAPE), chunks=(1, 1, SHAPE[1]))
    row = 4 * SHAPE[1]
    # Step 2's tiles come from two fields: row y=0 from field 4, row y=1 from field 2
    refs = _refs(pd.DatetimeIndex([TIMES[2]] * 2, name="time"), url, [3 * fields[0].nbytes, fields[0].nbytes + row],
                 length=row).assign(chunk_y=[0, 1])
    assert write_virtual_refs(session, "g", refs) == 2
    with pytest.raises(ValueError, match="one step per chunk"):
        write_virtual_refs(session, "g", refs.drop(columns="chunk_y"))
    session.commit("refs")
    v = xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values
    assert (v[2, 0] == 4).all() and (v[2, 1] == 2).all() and np.isnan(v[[0, 1, 3]]).all()


def test_a_file_listed_without_last_modified_is_read_unchecked(repo_and_file):
    """A file published after the listing has no last modified time; its reference still reads."""
    repo, url, _ = repo_and_file
    session = _group(repo, ("time", "y", "x"), {"time": TIMES}, (4, *SHAPE))
    write_virtual_refs(session, "g", _refs(pd.DatetimeIndex([TIMES[0]], name="time"), url, [0], pd.NaT))
    session.commit("refs")
    assert (xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values[0] == 1).all()


def test_a_rewritten_source_file_is_refused(repo_and_file):
    """A reference recorded before the file last changed fails to read instead of returning other bytes."""
    repo, url, _ = repo_and_file
    session = _group(repo, ("time", "y", "x"), {"time": TIMES}, (4, *SHAPE))
    older = _mtime(url) - pd.Timedelta("1D")
    write_virtual_refs(session, "g", _refs(pd.DatetimeIndex([TIMES[0]], name="time"), url, [0], older))
    session.commit("refs")
    with pytest.raises(ic.StorageError, match="checksum"):
        xr.open_zarr(repo.readonly_session("main").store, group="g", consolidated=False)["v"].values
