"""Overview levels and their inputs."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teehr_icechunk import overviews
from teehr_icechunk.models import BuildOverviewsDataInput

DAYS = pd.date_range("1981-10-01", periods=3, freq="D")


@pytest.mark.parametrize("factors", [[2, 4, 8], [3, 9]])
def test_overview_levels_match_xarray_coarsen(factors):
    """Levels built from the previous one match averaging full resolution, skipping NaN; partial edge blocks drop."""
    rng = np.random.default_rng(0)
    values = rng.random((3, 41, 59)).astype("float32")
    values[:, :7, :11] = np.nan  # a fully missing corner
    values[rng.random(values.shape) < 0.2] = np.nan  # and scattered gaps
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), values, {"units": "mm"})},
        coords={"time": DAYS, "y": np.arange(41) * 1000.0, "x": np.arange(59) * 1000.0},
        attrs={"spatial:dimensions": ["y", "x"]},
    )
    levels = overviews._overview_levels(ds, factors)
    for factor in factors:
        expected = ds.coarsen(y=factor, x=factor, boundary="trim").mean()
        level = levels[factor]
        np.testing.assert_allclose(level.v.values, expected.v.values, rtol=1e-6)
        np.testing.assert_allclose(level.x.values, expected.x.values)
        assert level.attrs["spatial:transform"][0] == 1000.0 * factor and level.v.attrs["units"] == "mm"


def test_overview_factors_each_a_multiple_of_the_previous():
    kwargs = {"configuration_name": "r", "dest_bucket": "b", "base_prefix": "p"}
    BuildOverviewsDataInput(overview_factors=[3, 9], **kwargs)
    for bad in ([2, 3], [4, 2], [1, 2], []):
        with pytest.raises(ValueError):
            BuildOverviewsDataInput(overview_factors=bad, **kwargs)
