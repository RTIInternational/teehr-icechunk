"""Parsing source files."""
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("h5netcdf")

PARSE = """
import sys
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr.parsers import HDFParser
from teehr_icechunk import grid_utils as gu
folder = sys.argv[1]
registry = ObjectStoreRegistry({f"file://{folder}/": LocalStore(folder)})
print(list(gu.open_virtual_datasets([f"file://{folder}/f.nc"], registry, HDFParser())))
"""


def test_byte_string_variables_parse_without_warnings(tmp_path):
    """NWM files carry a one-character `crs`, whose Zarr v3 type warns on every file parsed; ingests never store it.

    Parsed in a fresh process, as a run does: pytest drops warning filters set while it imports test modules.
    """
    xr.Dataset(
        {"RAINRATE": (("time", "y", "x"), np.zeros((1, 2, 3), "f4")), "crs": ((), np.array(b"", "S1"))},
    ).to_netcdf(tmp_path / "f.nc", engine="h5netcdf")
    done = subprocess.run([sys.executable, "-c", PARSE, str(tmp_path)], capture_output=True, text=True, check=True)
    assert "f.nc" in done.stdout and "NullTerminatedBytes" not in done.stderr
