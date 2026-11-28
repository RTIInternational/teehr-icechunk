"""Build the teehr-icechunk diagrams as pages of one editable .drawio file. They show the target design.

Usage:
    python build_diagrams.py

Each page has its own function; add new diagrams by writing another function that returns a Page and
adding it to PAGES. Re-running overwrites the output file, so make manual edits in draw.io only once the
generated layout is final (or update the script instead).
"""
from pathlib import Path

from drawio_helpers import (
    CLIENT, CORE, EXT, FETCH, ICECHUNK, NOTE, STORE, TEXT, Page, title_body as tb, write_drawio,
)

OUTPUT = Path(__file__).parent / "teehr-icechunk.drawio"


def package_boundaries():
    p = Page("Package boundaries", 1520, 820)
    p.box("sources", tb("Source files", "NetCDF / HDF5, GeoTIFF, GRIB2<br>(NOAA, NASA, universities)"), EXT, 20, 120, 200, 90)

    p.group("grp-ic", "teehr-icechunk (arrays only; no TEEHR data model)", 320, 60, 560, 330)
    p.box("repo", tb("Repo building", "Sources and the NWM configuration catalog, time grid, references, /data, overviews"), ICECHUNK, 350, 100, 160, 130)
    p.box("read", tb("Reading", "Repo contract (spec), read_grid_spec, open_grid_dataset; our repos and external ones"), ICECHUNK, 530, 100, 160, 130)
    p.box("zonal", tb("Zonal statistics", "weights_for_grid (exactextract coverage); zonal_means → Arrow batches"), ICECHUNK, 710, 100, 150, 130)
    p.box("extras", "<b>Extras:</b> [hdf] [tiff] [grib] [mean-areal] [prefect]. The base install has no Prefect, teehr or Spark",
          TEXT + "align=center;", 350, 250, 510, 40)
    p.box("ic-note", "N-D arrays, gridded or per feature (e.g. NWM streamflow over feature_id)", NOTE, 350, 300, 510, 30)

    p.box("repos", tb("Icechunk repos", "/references → /data → /overviews/&lt;factor&gt;"), STORE, 500, 470, 200, 110)

    p.group("grp-clients", "Client deployments", 980, 60, 520, 210)
    p.box("hub", tb("teehr-hub", "Sources (NWM forcing), input models, Prefect flows and deployments"), CLIENT, 1005, 100, 230, 90)
    p.box("fved", tb("teehr-fved", "Sources (snow), input models, Prefect flows and deployments"), CLIENT, 1250, 100, 230, 90)
    p.box("client-note", "Mean areal ingest: polygon layers in; Arrow batches mapped to teehr names and segments and written", NOTE, 1005, 200, 475, 50)

    p.box("fetch", tb("Fetch package (ADR-0015)", "Tabular sources (USGS, RFC); NWM streamflow read from our repos; gridded weights and means through teehr-icechunk"), FETCH, 980, 470, 280, 110)
    p.box("warehouse", tb("Warehouse (v1)", "Written through teehr-core's shared writer, teehr_core.warehouse (ADR-0019)"), STORE, 1320, 470, 180, 110)

    p.box("core", tb("teehr-cloud-core", "xpublish-api: map tiles and EDR from the repos; Spark executor image"), CORE, 20, 470, 260, 90)
    p.box("community", tb("Community readers", "xarray, notebooks, other tools"), EXT, 20, 640, 260, 70)

    p.edge("e-src", "sources", "repo", "listed and<br>referenced", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-build", "repo", "repos", "builds and updates", "exitX=0.5;exitY=1;entryX=0.25;entryY=0;")
    p.edge("e-read", "read", "repos", "reads", "exitX=0.5;exitY=1;entryX=0.6;entryY=0;")
    p.edge("e-zonal", "zonal", "read", "uses", "exitX=0.5;exitY=0;entryX=0.5;entryY=0;", points=((785, 80), (610, 80)))
    p.edge("e-clients", "grp-clients", "grp-ic", "depend on<br>(pinned by SHA)", "exitX=0;exitY=0.4;entryX=1;entryY=0.25;")
    p.edge("e-fetch", "fetch", "grp-ic", "depends on", "exitX=0;exitY=0.5;entryX=1;entryY=0.85;", points=((930, 525), (930, 340)))
    p.edge("e-wh-clients", "grp-clients", "warehouse", "v1 timeseries", "exitX=0.88;exitY=1;entryX=0.5;entryY=0;")
    p.edge("e-wh-fetch", "fetch", "warehouse", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-core", "core", "repos", "serves", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-community", "community", "repos", "read (spec)", "exitX=1;exitY=0.5;entryX=0.3;entryY=1;")
    p.box("arrows", "Arrows point at what is depended on, read or written. The target design (see the plan for status)", NOTE, 980, 610, 280, 40)
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
    p.box("plan", tb("Plan segments", "Whole shards and manifest splits along the append axis"), ICECHUNK, 720, 60, 200, 80)
    p.box("run", tb("Runner", "In-process (daily runs) or Prefect child runs (backfills)"), ICECHUNK, 960, 60, 200, 80)
    p.box("meta", tb("Record steps (parent)", "Status and audit times of the segments that succeeded; one commit"), ICECHUNK, 1200, 60, 260, 80)
    for i, (a, b) in enumerate((("src", "list"), ("list", "prep"), ("prep", "plan"), ("plan", "run"), ("run", "meta"))):
        p.edge(f"e-flow{i}", a, b)
    p.group("grp-segment", "A segment's stages (each writes chunks only and commits with rebase)", 240, 200, 920, 150)
    p.box("refs", tb("/references", "IdxRefs (GRIB2 .idx) or ParserRefs (HDF5 / NetCDF, GeoTIFF, Zarr), written with set_virtual_refs"), STORE, 270, 230, 260, 100)
    p.box("data", tb("/data (optional)", "Materialized copy, rechunked and sharded"), STORE, 580, 230, 240, 100)
    p.box("ov", tb("/overviews/&lt;factor&gt;", "Block means for zoomed-out tiles"), STORE, 870, 230, 260, 100)
    p.edge("e-rd", "refs", "data")
    p.edge("e-do", "data", "ov")
    p.edge("e-run-segment", "run", "grp-segment", "runs", "exitX=0.5;exitY=1;entryX=0.9;entryY=0;")
    p.box("readers", tb("Readers", "open_grid_dataset (ours or external repos); xpublish tiles / EDR; zonal_means → Arrow batches"), EXT, 240, 420, 920, 70)
    p.edge("e-read", "grp-segment", "readers", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.box("note", "Every source runs on segments; a new variable is an add-variables run over the steps already written", NOTE, 240, 520, 920, 40)
    p.legend(1240, 230, [(ICECHUNK, "teehr-icechunk step"), (STORE, "Repo group"), (EXT, "External")], width=220)
    return p


def run_model():
    p = Page("Run model and fan-out", 1300, 640)
    p.box("parent", tb("Parent flow run", "List, select, prepare (one commit), plan segments"), ICECHUNK, 40, 60, 260, 90)
    p.box("prefect", tb("PrefectRunner ([prefect] extra)", "Backfills: one child run per segment via run_deployment, at most max_child_runs at a time"), ICECHUNK, 380, 50, 280, 90)
    p.box("inproc", tb("In-process runner", "Daily runs: segments one after another in the parent's pod"), ICECHUNK, 380, 200, 280, 80)
    p.group("grp-children", "Child runs (process-gridded-segment), one pod each", 740, 30, 500, 300)
    for i, y in enumerate((70, 160, 250)):
        p.box(f"child{i}", tb(f"Segment {i + 1}", "Writes its chunks, commits with rebase over the other segments"), CLIENT, 770, y, 440, 60)
        p.edge(f"e-pf-c{i}", "prefect", f"child{i}", "", f"exitX=1;exitY={0.3 + 0.2 * i};entryX=0;entryY=0.5;")
    p.box("repo", tb("Icechunk repo", "Segments never share a chunk or manifest split; only the parent writes step metadata"), STORE, 380, 430, 330, 110)
    p.box("record", tb("Parent records steps", "For segments that succeeded; failed segments stay unwritten and the next run retries them"), ICECHUNK, 40, 430, 220, 110)
    p.edge("e-p-pf", "parent", "prefect", "", "exitX=1;exitY=0.4;entryX=0;entryY=0.5;")
    p.edge("e-p-in", "parent", "inproc", "", "exitX=1;exitY=0.8;entryX=0;entryY=0.5;")
    p.edge("e-in-repo", "inproc", "repo", "chunk commits", "exitX=0.5;exitY=1;entryX=0.3;entryY=0;")
    p.edge("e-c-repo", "grp-children", "repo", "chunk commits", "exitX=0.5;exitY=1;entryX=1;entryY=0.5;", points=((990, 485),))
    p.edge("e-p-rec", "parent", "record", "after the segments", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.edge("e-rec", "record", "repo", "metadata commit")
    return p

def two_paths():
    p = Page("Two paths: inputs and outputs", 1560, 960)
    # 1. Repo building
    p.group("grp-build", "1. Repo building: source files in, a spec repo out", 20, 40, 1520, 290)
    p.box("in-source", tb("Source", "GriddedSource / IndexSource: where the files are, time grid and axes, variables, dataset settings"), CLIENT, 40, 80, 290, 70)
    p.box("in-run", tb("Run request", "Window (start/end or lookback); append / upsert; tiers: references, + data, + overviews; fan-out (max_child_runs)"), CLIENT, 40, 160, 290, 80)
    p.box("in-target", tb("Target storage", "Icechunk storage: bucket and prefix"), CLIENT, 40, 250, 290, 60)
    p.box("ingest", tb("ingest(...)", "List and select steps → prepare (groups, axes, NOT_EXPECTED) → plan segments → run segments "
                       "(in-process, Prefect, or another runner) → record steps"), ICECHUNK, 420, 120, 560, 120)
    p.box("out-repo", tb("Icechunk repo (spec layout)", "/references, optional /data, /overviews/&lt;factor&gt;; step metadata; CF / GeoZarr attributes"), STORE, 1070, 80, 440, 130)
    p.box("out-report", tb("Run report", "Segments written or failed (retried by the next run)"), ICECHUNK, 420, 260, 560, 50)
    for i, src in enumerate(("in-source", "in-run", "in-target")):
        p.edge(f"e-b-in{i}", src, "ingest", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge("e-b-repo", "ingest", "out-repo", "")
    p.edge("e-b-rep", "ingest", "out-report", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")

    # Shared read layer
    p.box("read", tb("Read layer", "Repo contract (spec), read_grid_spec, open_grid_dataset: our repos, or external ones with a spec override"),
          ICECHUNK, 1070, 360, 440, 70)
    p.edge("e-repo-read", "out-repo", "read", "reads", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")

    # 2. Zonal statistics
    p.group("grp-zonal", "2. Zonal statistics: a dataset and polygons in, Arrow batches out", 20, 470, 1520, 320)
    p.box("in-ds", tb("Dataset", "One variable from open_grid_dataset (our repos, dynamical.org, AORC ...)"), CLIENT, 40, 510, 290, 70)
    p.box("in-poly", tb("Polygons", "Key + EPSG:4326 geometry, e.g. a v1 location layer"), CLIENT, 40, 590, 290, 60)
    p.box("in-sel", tb("Selection and settings", "Window, or cycles and members; memory budget; minimum coverage; scale / offset; runner"), CLIENT, 40, 660, 290, 80)
    p.box("weights", tb("weights_for_grid(...)", "Pixel coverage of each polygon (exactextract), on the grid's CRS"), ICECHUNK, 420, 500, 270, 80)
    p.box("w-table", tb("Weights", "Key, row, col, fraction covered; grid_name. Stored by the caller (v1 dataset or cached Parquet; open)"), STORE, 420, 620, 270, 110)
    p.box("zonal", tb("zonal_means(...)", "Written steps → read plans from the chunk layout → block kernel (cos-latitude area weights) "
                      "→ runner: in-process, Prefect, or Spark mapInArrow"), ICECHUNK, 760, 560, 270, 140)
    p.box("out-batches", tb("Arrow batches", "key, value_time, [reference_time], [member], [lead_time], value"), STORE, 1090, 560, 200, 110)
    p.box("out-client", tb("Client (hub, fved, fetch package)", "Maps batches to teehr names and segments (by window for windowed variables) and writes v1 timeseries through teehr_core.warehouse"),
          CLIENT, 1310, 545, 210, 140)
    p.edge("e-z-ds-w", "in-ds", "weights", "grid template", "exitX=1;exitY=0.3;entryX=0;entryY=0.3;")
    p.edge("e-z-poly", "in-poly", "weights", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    p.edge("e-z-w", "weights", "w-table", "")
    p.edge("e-z-wz", "w-table", "zonal", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    p.edge("e-z-ds", "in-ds", "zonal", "", "exitX=1;exitY=0.85;entryX=0;entryY=0.29;", points=((370, 570), (370, 600)))
    p.edge("e-z-sel", "in-sel", "zonal", "", "exitX=1;exitY=0.8;entryX=0.5;entryY=1;", points=((360, 764), (895, 764)))
    p.edge("e-z-out", "zonal", "out-batches", "")
    p.edge("e-z-client", "out-batches", "out-client", "")
    p.edge("e-read-ds", "read", "in-ds", "opens", "exitX=0;exitY=0.5;entryX=0.5;entryY=0;", points=((185, 395),))
    p.legend(40, 820, [(CLIENT, "Given by the caller"), (ICECHUNK, "teehr-icechunk"), (STORE, "Data produced")], width=260)
    p.box("scope", "teehr-icechunk works on arrays only: it never writes the warehouse or knows TEEHR's data model",
          NOTE, 420, 340, 600, 30)
    return p

PAGES = [two_paths, package_boundaries, repo_layout, gridded_workflow, run_model]

if __name__ == "__main__":
    write_drawio(OUTPUT, [build() for build in PAGES])
    print(f"Wrote {len(PAGES)} pages to {OUTPUT}")
