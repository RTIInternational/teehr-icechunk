"""Zonal segments: weights and schema files, segment planning, and staged means equal to the kernel's."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from teehr_icechunk import repo_const as rc
from teehr_icechunk import zonal
from teehr_icechunk.mean_areal_kernel import GridWeights, mean_areal_block
from teehr_icechunk.segments import Segment
from teehr_icechunk.weights import build_weights_matrix

TIMES = pd.date_range("2020-01-01", periods=48, freq="h")
CYCLES = pd.date_range("2026-10-01", periods=4, freq="6h")
MEMBERS, LEADS = [0, 1, 2], [3, 6, 9, 12]
SCHEMA = pa.schema([
    ("reference_time", pa.timestamp("us", tz="UTC")), ("value_time", pa.timestamp("us", tz="UTC")),
    ("value", pa.float32()), ("location_id", pa.string()), ("member", pa.string()),
    ("configuration_name", pa.string()), ("variable_name", pa.string()),
])


def _weights(shape, seed=0):
    """Three zones over random pixels, one sharing pixels with another (nested basins)."""
    rng = np.random.default_rng(seed)
    rows, cols, ids = [], [], []
    for zone, n in (("z-a", 12), ("z-b", 5), ("z-c", 8)):
        rows += list(rng.integers(0, shape[0], n))
        cols += list(rng.integers(0, shape[1], n))
        ids += [zone] * n
    df = pd.DataFrame({"location_id": ids, "row": rows, "col": cols}).drop_duplicates(["location_id", "row", "col"])
    df["weight"] = rng.uniform(0.1, 1.0, len(df))
    matrix, r, c, loc = build_weights_matrix(df)
    return GridWeights(matrix.tocsc(), r, c, loc)


def _analysis():
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 10, (len(TIMES), 16, 12)).astype("float32")
    values[5, 3:9, :] = np.nan
    da = xr.DataArray(values, dims=("time", "y", "x"), coords={"time": TIMES})
    da.encoding = {"chunks": (6, 8, 8)}
    return da


def _forecast():
    rng = np.random.default_rng(2)
    values = rng.uniform(250, 300, (len(CYCLES), len(MEMBERS), len(LEADS), 10, 10)).astype("float32")
    da = xr.DataArray(
        values, dims=(rc.REFERENCE_TIME, rc.MEMBER, rc.LEAD_TIME, "y", "x"),
        coords={rc.REFERENCE_TIME: CYCLES, rc.MEMBER: MEMBERS, rc.LEAD_TIME: LEADS},
    )
    da.encoding = {"chunks": (1, len(MEMBERS), 2, 5, 5)}
    return da


def _stage(tmp_path, weights):
    zonal.save_weights(tmp_path / zonal.WEIGHTS_FILE, weights)
    zonal.save_schema(tmp_path / zonal.SCHEMA_FILE, SCHEMA)


def _read_segments(tmp_path, segments):
    tables = [pq.read_table(p) for s in segments for p in sorted(zonal.segment_dir(tmp_path, s).glob("*.parquet"))]
    return pa.concat_tables(tables).to_pandas()


def _sorted(df, keys):
    return df.sort_values(keys).reset_index(drop=True)


def test_weights_and_schema_round_trip(tmp_path):
    """Weights and the output schema load back exactly as saved."""
    weights = _weights((16, 12))
    zonal.save_weights(tmp_path / "w.npz", weights)
    zonal.save_schema(tmp_path / "s.arrow", SCHEMA)
    loaded = zonal.load_weights(tmp_path / "w.npz")
    assert (loaded.matrix != weights.matrix).nnz == 0
    np.testing.assert_array_equal(loaded.rows, weights.rows)
    np.testing.assert_array_equal(loaded.cols, weights.cols)
    assert list(loaded.location_ids) == list(weights.location_ids)
    assert zonal.load_schema(tmp_path / "s.arrow") == SCHEMA


def test_segments_hold_whole_append_chunks():
    """Segments of about 10 hours on 6-hour chunks are 12 hours from the axis start, skipping unwritten spans."""
    written = TIMES[2:20].append(TIMES[40:44])
    segments = zonal.plan_zonal_segments(_analysis(), written, "time", segment_steps=10)
    assert segments == [Segment(TIMES[0], TIMES[11]), Segment(TIMES[12], TIMES[23]), Segment(TIMES[36], TIMES[47])]


def test_analysis_segments_equal_the_kernel(tmp_path):
    """Means staged by segments (several blocks each, in worker processes) equal one kernel call over every step."""
    da, weights = _analysis(), _weights((16, 12))
    _stage(tmp_path, weights)
    work = zonal.ZonalWork(
        da, TIMES, "time", {}, "y", "x", str(tmp_path), task_memory_gb=2e-6,
        kernel_kwargs={"min_coverage": 0.5, "constants": {"configuration_name": "c", "variable_name": "v"}},
    )
    segments = zonal.plan_zonal_segments(da, TIMES, "time", segment_steps=24)
    for segment in segments:
        work(segment)

    got = _read_segments(tmp_path, segments)
    expected = mean_areal_block(
        da, weights, "y", "x", zonal.step_columns(TIMES, "time"), min_coverage=0.5,
        constants={"configuration_name": "c", "variable_name": "v"}, schema=SCHEMA,
    ).to_pandas()
    keys = ["value_time", "location_id"]
    pd.testing.assert_frame_equal(_sorted(got, keys), _sorted(expected, keys))
    assert len(list(zonal.segment_dir(tmp_path, segments[0]).glob("*.parquet"))) > 1


def test_forecast_segments_equal_the_kernel(tmp_path):
    """A forecast's members and leads, named by the window each lead covers, as one kernel call gives them."""
    da, weights = _forecast(), _weights((10, 10), seed=3)
    _stage(tmp_path, weights)
    fixed = {rc.MEMBER: np.array(MEMBERS), rc.LEAD_TIME: np.array(LEADS)}
    written = pd.MultiIndex.from_product([CYCLES, MEMBERS, LEADS], names=[rc.REFERENCE_TIME, rc.MEMBER, rc.LEAD_TIME])
    lead_names = {3: "t_3hr_max", 6: "t_6hr_max", 9: "t_3hr_max", 12: "t_6hr_max"}
    work = zonal.ZonalWork(
        da, written, rc.REFERENCE_TIME, fixed, "y", "x", str(tmp_path), task_memory_gb=1e-5,
        lead_names=lead_names, kernel_kwargs={"offset": -273.15, "constants": {"configuration_name": "c"}},
    )
    segments = zonal.plan_zonal_segments(da, written, rc.REFERENCE_TIME, segment_steps=2)
    assert len(segments) == 2
    for segment in segments:
        work(segment)

    got = _read_segments(tmp_path, segments)
    expected = mean_areal_block(
        da, weights, "y", "x", zonal.step_columns(written, rc.REFERENCE_TIME, lead_names), offset=-273.15,
        constants={"configuration_name": "c"}, schema=SCHEMA,
    ).to_pandas()
    keys = ["reference_time", "member", "value_time", "location_id"]
    pd.testing.assert_frame_equal(_sorted(got, keys), _sorted(expected, keys))
    assert set(got["variable_name"]) == {"t_3hr_max", "t_6hr_max"}


def test_a_rerun_segment_replaces_its_parts(tmp_path):
    """Parts left by an earlier attempt are removed before a segment runs again."""
    da, weights = _analysis(), _weights((16, 12))
    _stage(tmp_path, weights)
    segment = Segment(TIMES[0], TIMES[11])
    stale = zonal.segment_dir(tmp_path, segment)
    stale.mkdir(parents=True)
    (stale / "part-99999.parquet").write_bytes(b"stale")
    zonal.ZonalWork(da, TIMES, "time", {}, "y", "x", str(tmp_path), task_memory_gb=1.0)(segment)
    assert not (stale / "part-99999.parquet").exists()
    expected = mean_areal_block(da.isel(time=slice(0, 12)), weights, "y", "x",
                                zonal.step_columns(TIMES[:12], "time"), schema=SCHEMA)
    assert len(_read_segments(tmp_path, [segment])) == expected.num_rows


def _grid(ascending: bool):
    """A 1-degree EPSG:4326 grid, 0..10 E and 40..48 N, with latitude stored either way."""
    lats = np.arange(40.5, 48) if ascending else np.arange(47.5, 40, -1)
    da = xr.DataArray(np.ones((len(lats), 10), "float32"), dims=("latitude", "longitude"),
                      coords={"latitude": lats, "longitude": np.arange(0.5, 10)})
    return da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326")


def _polygons():
    from shapely.geometry import Polygon, box
    return gpd.GeoDataFrame({"id": ["sq", "half", "bowtie", "outside"], "geometry": [
        box(2, 44, 4, 46),  # exactly 2 x 2 pixels
        box(6, 42, 6.5, 43),  # half of one pixel
        Polygon([(0, 40), (1, 41), (1, 40), (0, 41)]),  # self-intersecting
        box(9.5, 47.5, 12, 50),  # mostly off the grid
    ]}, crs="EPSG:4326")


@pytest.mark.parametrize("ascending", [False, True])
def test_weights_index_the_stored_grid(ascending):
    """Coverage fractions land on the stored rows and columns, whichever way latitude is stored."""
    da = _grid(ascending)
    got = zonal.weights_for_grid(da, _polygons(), x_dim="longitude", y_dim="latitude", key="id")
    lat = da.latitude.values
    sq = got[got.location_id == "sq"]
    assert len(sq) == 4 and np.allclose(sq.fraction_covered, 1.0)
    assert set(lat[sq.row]) == {44.5, 45.5} and set(sq.col) == {2, 3}
    half = got[got.location_id == "half"]
    assert len(half) == 1 and np.isclose(half.fraction_covered.iloc[0], 0.5)
    assert lat[half.row.iloc[0]] == 42.5 and half.col.iloc[0] == 6
    assert "outside" not in set(got.location_id)
    # The bowtie is repaired into its two triangles over the corner pixel (0..1 E, 40..41 N)
    bowtie = got[got.location_id == "bowtie"]
    assert lat[bowtie.row.iloc[0]] == 40.5 and bowtie.col.iloc[0] == 0
    assert np.isclose(bowtie.fraction_covered.sum(), 0.5)


def test_duplicate_polygon_keys_are_refused():
    """Two polygons with one key would mix their weights."""
    polygons = _polygons()
    polygons.loc[1, "id"] = "sq"
    with pytest.raises(ValueError, match="Duplicate polygon keys"):
        zonal.weights_for_grid(_grid(False), polygons, x_dim="longitude", y_dim="latitude", key="id")
