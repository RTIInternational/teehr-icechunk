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


def gridded_workflow():
    p = Page("Gridded workflow", 1500, 640)
    p.box("src", tb("Source files", "HDF5 / NetCDF, GeoTIFF, GRIB2 + .idx"), EXT, 20, 60, 180, 80)
    p.box("list", tb("List and select", "Source listing by step; append / upsert against the stored step metadata"), ICECHUNK, 240, 60, 200, 80)
    p.box("prep", tb("Prepare (parent)", "Groups, axes and root attributes; NOT_EXPECTED slots; one commit"), ICECHUNK, 480, 60, 200, 80)
    p.box("plan", tb("Plan units", "Whole shards and manifest splits along the append axis"), ICECHUNK, 720, 60, 200, 80)
    p.box("run", tb("Runner", "In-process (daily runs) or Prefect child runs (backfills)"), ICECHUNK, 960, 60, 200, 80)
    p.box("meta", tb("Record steps (parent)", "Status and audit times of the units that succeeded; one commit"), ICECHUNK, 1200, 60, 260, 80)
    for i, (a, b) in enumerate((("src", "list"), ("list", "prep"), ("prep", "plan"), ("plan", "run"), ("run", "meta"))):
        p.edge(f"e-flow{i}", a, b)
    p.group("grp-unit", "A unit's stages (each writes chunks only and commits with rebase)", 240, 200, 920, 150)
    p.box("refs", tb("/references", "IndexSource: refs from .idx files; parsed sources: VirtualiZarr parsers; written with set_virtual_refs"), STORE, 270, 230, 260, 100)
    p.box("data", tb("/data (optional)", "Materialized copy, rechunked and sharded"), STORE, 580, 230, 240, 100)
    p.box("ov", tb("/overviews/&lt;factor&gt;", "Block means for zoomed-out tiles"), STORE, 870, 230, 260, 100)
    p.edge("e-rd", "refs", "data")
    p.edge("e-do", "data", "ov")
    p.edge("e-run-unit", "run", "grp-unit", "runs", "exitX=0.5;exitY=1;entryX=0.9;entryY=0;")
    p.box("readers", tb("Readers", "open_grid_dataset (ours or external repos), xpublish tiles / EDR, zonal statistics → Arrow batches"), EXT, 240, 420, 920, 70)
    p.edge("e-read", "grp-unit", "readers", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.box("note", "Today index sources (GEFS) run on units; parsed sources (NWM forcing, snow) keep the earlier path until their stages move over", NOTE, 240, 520, 920, 40)
    p.legend(1240, 230, [(ICECHUNK, "teehr-icechunk step"), (STORE, "Repo group"), (EXT, "External")], width=220)
    return p


def run_model():
    p = Page("Run model and fan-out", 1300, 640)
    p.box("parent", tb("Parent flow run", "List, select, prepare (one commit), plan units"), ICECHUNK, 40, 60, 260, 90)
    p.box("prefect", tb("PrefectRunner ([prefect] extra)", "Backfills: one child run per unit via run_deployment, at most max_child_runs at a time"), ICECHUNK, 380, 50, 280, 90)
    p.box("inproc", tb("In-process runner", "Daily runs: units one after another in the parent's pod"), ICECHUNK, 380, 200, 280, 80)
    p.group("grp-children", "Child runs (process-gridded-unit), one pod each", 740, 30, 500, 300)
    for i, y in enumerate((70, 160, 250)):
        p.box(f"child{i}", tb(f"Unit {i + 1}", "Writes its chunks, commits with rebase over the other units"), CLIENT, 770, y, 440, 60)
        p.edge(f"e-pf-c{i}", "prefect", f"child{i}", "", f"exitX=1;exitY={0.3 + 0.2 * i};entryX=0;entryY=0.5;")
    p.box("repo", tb("Icechunk repo", "Units never share a chunk or manifest split; only the parent writes step metadata"), STORE, 380, 430, 330, 110)
    p.box("record", tb("Parent records steps", "For units that succeeded; failed units stay unwritten and the next run retries them"), ICECHUNK, 40, 430, 220, 110)
    p.edge("e-p-pf", "parent", "prefect", "", "exitX=1;exitY=0.4;entryX=0;entryY=0.5;")
    p.edge("e-p-in", "parent", "inproc", "", "exitX=1;exitY=0.8;entryX=0;entryY=0.5;")
    p.edge("e-in-repo", "inproc", "repo", "chunk commits", "exitX=0.5;exitY=1;entryX=0.3;entryY=0;")
    p.edge("e-c-repo", "grp-children", "repo", "chunk commits", "exitX=0.5;exitY=1;entryX=1;entryY=0.5;", points=((990, 485),))
    p.edge("e-p-rec", "parent", "record", "after the units", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.edge("e-rec", "record", "repo", "metadata commit")
    return p

PAGES = [package_boundaries, repo_layout, gridded_workflow, run_model]

if __name__ == "__main__":
    write_drawio(OUTPUT, [build() for build in PAGES])
    print(f"Wrote {len(PAGES)} pages to {OUTPUT}")
