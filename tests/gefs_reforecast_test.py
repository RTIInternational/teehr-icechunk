"""The GEFS reforecast template, as prepare writes it: CF and GeoZarr metadata on a 0.25 degree grid."""
from teehr_icechunk import grid_utils as gu
from teehr_icechunk.sources.gefs_reforecast import GEFSv12Reforecast


def test_template_metadata():
    ds = gu.standardize_and_inject_geozarr(GEFSv12Reforecast(grid="0p25").template(), x_dim="longitude", y_dim="latitude")
    assert ds.attrs["proj:code"] == "EPSG:4326"
    # Pixel centers from -180 and 90 (north up), so the outer corner is half a pixel beyond
    assert ds.attrs["spatial:transform"] == [0.25, 0.0, -180.125, 0.0, -0.25, 90.125]
    apcp, tmax = ds["APCP_surface"].attrs, ds["TMAX_2m"].attrs
    assert (apcp["standard_name"], apcp["cell_methods"], apcp["window_reset_hours"]) == ("precipitation_amount", "time: sum", 6)
    assert (tmax["standard_name"], tmax["cell_methods"]) == ("air_temperature", "time: maximum")
    assert apcp["proj:code"] == "EPSG:4326" and apcp["grid_mapping"] in ds.coords
