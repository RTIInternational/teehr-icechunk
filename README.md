# teehr-icechunk

Build, keep up to date and read analysis-ready [Icechunk](https://icechunk.io) datasets of hydrologic and weather data (forecasts and analyses, gridded or per stream reach) straight from the source files.

## What it does

- **Points to the source, or copies it.** Builds a lightweight "virtual" dataset that reads the original files where they sit (NetCDF, GeoTIFF, GRIB2), and optionally a copy rechunked for fast reads, plus map overviews for web maps.
- **Keeps it current.** Every time step is tracked, so daily runs add new data, reprocessed files are picked up, and a failed run simply resumes. The same code does a 20-year backfill or a single day.
- **Speaks one layout.** Analyses on a time axis; forecasts by issue time, ensemble member and lead time. The readers open other people's datasets in the same shape.
- **Summarizes over areas.** Averages a dataset over thousands of watersheds, reading only the parts of the grid they cover.

## Install

```bash
pip install "teehr-icechunk @ git+https://github.com/RTIInternational/teehr-icechunk.git@<sha>"
```

Extras add file formats and features:

| Extra | For |
|---|---|
| `[hdf]` | NetCDF / HDF5 / Zarr sources |
| `[tiff]` | GeoTIFF sources |
| `[grib]` | GRIB2 sources (also needed to read GRIB-backed datasets) |
| `[mean-areal]` | zonal statistics (`teehr_icechunk.zonal`): polygon coverage weights on a grid, and mean areal values staged as Parquet |
| `[prefect]` | running segments of a job as Prefect child flow runs (fan-out across pods) |
| `[all]` | all of the above |

The base install has no workflow orchestrator, teehr or Spark dependency.

## The dataset layout

Every dataset follows the contract in [`spec/gridded-repo-contract.md`](spec/gridded-repo-contract.md), so any tool can read it with plain xarray. The diagrams in [`docs/diagrams/drawio/`](docs/diagrams/drawio/) show the layout and how the package fits with the rest of TEEHR.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

## Status

Under construction. The shared gridded code of TEEHR's deployments (CIROH TEEHR-Hub, USBR FVED) has moved here and is used in production through them. The API may change before 1.0.
