"""Build the teehr-icechunk diagrams as pages of one editable .drawio file.

Usage:
    python build_diagrams.py

Each page has its own function; add new diagrams by writing another function that returns a Page and
adding it to PAGES. Re-running overwrites the output file, so make manual edits in draw.io only once the
generated layout is final (or update the script instead).
"""
from pathlib import Path

from drawio_helpers import (
    CLIENT, CORE, DASHED, EXT, FETCH, ICECHUNK, NOTE, STORE, TEXT, Page, title_body as tb, write_drawio,
)

OUTPUT = Path(__file__).parent / "teehr-icechunk.drawio"


def package_boundaries():
    p = Page("Package boundaries", 1520, 820)
    p.box("sources", tb("Source files", "NetCDF / HDF5, GeoTIFF, GRIB2<br>(NOAA, NASA, universities)"), EXT, 20, 120, 200, 90)

    p.group("grp-ic", "teehr-icechunk (arrays only; no TEEHR data model)", 320, 60, 560, 330)
    p.box("repo", tb("Repo building", "Sources and the NWM configuration catalog, time grid, references, /data, overviews"), ICECHUNK, 350, 100, 160, 130)
    p.box("read", tb("Reading", "Repo contract (spec), read_grid_spec, open_grid_dataset; our repos and external ones"), ICECHUNK, 530, 100, 160, 130)
    p.box("zonal", tb("Zonal statistics", "Weights matrix, chunk-aware read plans, block kernel; Arrow batches out"), ICECHUNK, 710, 100, 150, 130)
    p.box("extras", "<b>Extras:</b> [hdf] [tiff] [grib] [mean-areal] [prefect]. The base install has no Prefect, teehr or Spark",
          TEXT + "align=center;", 350, 250, 510, 40)
    p.box("ic-note", "N-D arrays, gridded or per feature (e.g. NWM streamflow over feature_id)", NOTE, 350, 300, 510, 30)

    p.box("repos", tb("Icechunk repos", "/references → /data → /overviews/&lt;factor&gt;"), STORE, 500, 470, 200, 110)

    p.group("grp-clients", "Client deployments", 980, 60, 520, 210)
    p.box("hub", tb("teehr-hub", "Sources (NWM forcing), input models, Prefect flows and deployments"), CLIENT, 1005, 100, 230, 90)
    p.box("fved", tb("teehr-fved", "Sources (snow), input models, Prefect flows and deployments"), CLIENT, 1250, 100, 230, 90)
    p.box("client-note", "Mean areal ingest: polygons, teehr names and units, warehouse writes (legacy teehr until v1)", NOTE, 1005, 200, 475, 50)

    p.box("fetch", tb("Fetch package (planned, ADR-0015)", "Tabular sources (USGS, RFC); NWM streamflow read from our repos; v1 timeseries"), FETCH + DASHED, 980, 470, 280, 110)
    p.box("warehouse", tb("Warehouse", "Iceberg tables (legacy now, v1 at cutover)"), STORE, 1320, 470, 180, 110)

    p.box("core", tb("teehr-cloud-core", "xpublish-api: map tiles and EDR from the repos; Spark executor image"), CORE, 20, 470, 260, 90)
    p.box("community", tb("Community readers", "xarray, notebooks, other tools"), EXT, 20, 640, 260, 70)

    p.edge("e-src", "sources", "repo", "listed and<br>referenced", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-build", "repo", "repos", "builds and updates", "exitX=0.5;exitY=1;entryX=0.25;entryY=0;")
    p.edge("e-read", "read", "repos", "reads", "exitX=0.5;exitY=1;entryX=0.6;entryY=0;")
    p.edge("e-zonal", "zonal", "read", "uses", "exitX=0.5;exitY=0;entryX=0.5;entryY=0;", points=((785, 80), (610, 80)))
    p.edge("e-clients", "grp-clients", "grp-ic", "depend on<br>(pinned by SHA)", "exitX=0;exitY=0.4;entryX=1;entryY=0.25;")
    p.edge("e-fetch", "fetch", "grp-ic", "depends on", "exitX=0;exitY=0.5;entryX=1;entryY=0.85;" + DASHED, points=((930, 525), (930, 340)))
    p.edge("e-wh-clients", "grp-clients", "warehouse", "means", "exitX=0.88;exitY=1;entryX=0.5;entryY=0;")
    p.edge("e-wh-fetch", "fetch", "warehouse", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;" + DASHED)
    p.edge("e-core", "core", "repos", "serves", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-community", "community", "repos", "read (spec)", "exitX=1;exitY=0.5;entryX=0.3;entryY=1;")
    p.box("arrows", "Arrows point at what is depended on or read. Dashed: planned", NOTE, 980, 610, 280, 40)
    p.legend(1300, 610)
    return p


def repo_layout():
    p = Page("Repo layout", 1300, 700)
    p.box("refs", tb("/references", "Virtual chunks pointing into the source files; always written"), STORE, 40, 60, 260, 110)
    p.box("data", tb("/data", "Optional materialized copy, rechunked and sharded for fast reads"), STORE, 380, 60, 260, 110)
    p.box("ov", tb("/overviews/&lt;factor&gt;", "Block means for zoomed-out map tiles, e.g. 2, 4, 8"), STORE, 720, 60, 260, 110)
    p.edge("e-rd", "refs", "data", "materialize<br>(stale steps)")
    p.edge("e-do", "data", "ov", "overviews of the<br>data group")
    p.box("dg", "Readers use the root <b>data_group</b> attribute: /data when materialized, else /references", NOTE, 40, 180, 940, 30)

    p.box("dims", tb("Dims", "<b>Analysis:</b> (time, y, x) or (time, feature_id)<br>"
                     "<b>Forecast:</b> (reference_time, [member], lead_time, y, x)<br>"
                     "lead_time in int hours; value_time is a 2-D coordinate on (reference_time, lead_time)"),
          ICECHUNK + "align=left;spacingLeft=10;", 40, 250, 460, 120)
    p.box("steps", tb("Per-step metadata (coordinates on the step dims)",
                      "<b>status</b>: -2 not expected, -1 unwritten, 0 early, 1 provisional, 2 stable<br>"
                      "<b>created_at, updated_at, source_last_modified</b>: drive upserts and downstream staleness"),
          ICECHUNK + "align=left;spacingLeft=10;", 520, 250, 460, 120)
    p.box("attrs", tb("Root attributes",
                      "append_dim, time_origin, time_step (the fixed time grid); fixed_dims, layout_version; "
                      "configuration_name, timeseries_type, grid_name, data_group"),
          ICECHUNK + "align=left;spacingLeft=10;", 40, 400, 940, 80)
    p.box("grid", "Every step has a slot on the fixed time grid (time_origin + k × time_step), so steps can be written "
                  "in any order, rewritten (upsert), and resumed after a failed run", NOTE, 40, 500, 940, 40)
    p.box("spec", "Defined in spec/gridded-repo-contract.md; tests/fixtures/*_golden.json pin what each shape stores",
          NOTE, 40, 550, 940, 30)
    return p


PAGES = [package_boundaries, repo_layout]

if __name__ == "__main__":
    write_drawio(OUTPUT, [build() for build in PAGES])
    print(f"Wrote {len(PAGES)} pages to {OUTPUT}")
