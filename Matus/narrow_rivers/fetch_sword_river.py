"""
Fetch SWORD reaches + nodes for one river straight into narrow_rivers/data/.

Same source as the GEE script that produced the Killik/Sagavanirktok files
(projects/sat-io/open-datasets/SWORD), but pulled locally via the Python API,
so there is no Drive export step.

Columns match the existing CSVs, so extract_river_pixels.py works unchanged
(point NODES_CSV at the new file).

No width filter by default: on the Nenana the median reach is 67 m wide, and a
75-300 m filter would drop most of the river.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\fetch_sword_river.py --river "Nenana River"
"""

import os, csv, argparse, re

import ee

NR_ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(NR_ROOT, "data")
EE_PROJECT = "noaa-river-ice"

REACHES_FC = "projects/sat-io/open-datasets/SWORD/reaches_merged"
NODES_FC   = "projects/sat-io/open-datasets/SWORD/nodes_merged"

# SWORD 'type': 1 = river, 3 = lake on river, 4 = dam/waterfall, 5 = unreliable
# topology, 6 = ghost reach. The earlier GEE script kept type 1 only.
DEFAULT_TYPES = [1]

# (output column, SWORD property) — SWORD's x/y are the centre lon/lat.
REACH_COLS = [("river_name", "river_name"), ("reach_id", "reach_id"),
              ("lat", "y"), ("lon", "x"), ("width_m", "width"),
              ("max_width", "max_width"), ("n_chan_mod", "n_chan_mod"),
              ("n_chan_max", "n_chan_max"), ("length_m", "reach_len"),
              ("facc", "facc"), ("dist_out", "dist_out"), ("lakeflag", "lakeflag")]
NODE_COLS  = [("river_name", "river_name"), ("reach_id", "reach_id"),
              ("node_id", "node_id"), ("lat", "y"), ("lon", "x"),
              ("width_m", "width"), ("max_width", "max_width"),
              ("n_chan_mod", "n_chan_mod"), ("dist_out", "dist_out")]


def _slug(river):
    return re.sub(r"[^a-z0-9]+", "_", river.lower()).strip("_").replace("_river", "")


def _rows(fc, cols):
    """Feature collection -> list of dicts, one per feature."""
    feats = fc.getInfo()["features"]
    return [{out: f["properties"].get(src) for out, src in cols} for f in feats]


def _write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[c for c, _ in cols])
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows):5d} rows -> {path}")


def main(river, types, min_width, max_width, single_channel, write_geojson):
    ee.Initialize(project=EE_PROJECT)

    reaches = (ee.FeatureCollection(REACHES_FC)
                 .filter(ee.Filter.eq("river_name", river)))
    n_all = reaches.size().getInfo()
    if n_all == 0:
        print(f"ERROR: no SWORD reaches named '{river}'. "
              f"Check the exact spelling (e.g. 'Nenana River').")
        raise SystemExit(1)

    reaches = reaches.filter(ee.Filter.inList("type", types))
    if min_width is not None:
        reaches = reaches.filter(ee.Filter.gte("width", min_width))
    if max_width is not None:
        reaches = reaches.filter(ee.Filter.lt("width", max_width))
    if single_channel:
        reaches = reaches.filter(ee.Filter.eq("n_chan_mod", 1))

    reach_ids = reaches.aggregate_array("reach_id")
    nodes = (ee.FeatureCollection(NODES_FC)
               .filter(ee.Filter.eq("river_name", river))
               .filter(ee.Filter.inList("reach_id", reach_ids)))

    print("=" * 60)
    print(f"SWORD fetch — {river}")
    print(f"  reaches with this name: {n_all}")
    print(f"  after filters (type {types}"
          f"{f', width >= {min_width}' if min_width else ''}"
          f"{f', width < {max_width}' if max_width else ''}"
          f"{', single-channel' if single_channel else ''}): "
          f"{reaches.size().getInfo()}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    slug = _slug(river)
    reach_rows = _rows(reaches, REACH_COLS)
    node_rows  = _rows(nodes, NODE_COLS)
    if not node_rows:
        print("ERROR: no nodes for the selected reaches.")
        raise SystemExit(1)

    _write_csv(os.path.join(DATA_DIR, f"sword_reaches_{slug}.csv"), reach_rows, REACH_COLS)
    _write_csv(os.path.join(DATA_DIR, f"sword_nodes_{slug}.csv"), node_rows, NODE_COLS)

    if write_geojson:
        import json
        for name, fc in [(f"sword_reaches_{slug}.geojson", reaches),
                         (f"sword_nodes_{slug}.geojson", nodes)]:
            path = os.path.join(DATA_DIR, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(fc.getInfo(), f)
            print(f"  geometry      -> {path}")

    # Summary — the numbers worth knowing before siting a patch
    lats = [r["lat"] for r in node_rows]
    lons = [r["lon"] for r in node_rows]
    widths = sorted(r["width_m"] for r in reach_rows)
    nwidths = sorted(r["width_m"] for r in node_rows)
    zone = int((sum(lons) / len(lons) + 180) // 6) + 1
    print(f"\n  nodes: {len(node_rows)}  (200 m spacing along the centerline)")
    print(f"  lat {min(lats):.3f}–{max(lats):.3f}   lon {min(lons):.3f}–{max(lons):.3f}")
    print(f"  reach width  min/median/max: {widths[0]:.0f} / {widths[len(widths)//2]:.0f} / {widths[-1]:.0f} m")
    print(f"  node  width  min/median/max: {nwidths[0]:.0f} / {nwidths[len(nwidths)//2]:.0f} / {nwidths[-1]:.0f} m")
    print(f"  nodes narrower than 375 m (one VIIRS pixel): "
          f"{sum(1 for w in nwidths if w < 375)}/{len(nwidths)}")
    print(f"  UTM zone {zone}N  (EPSG:326{zone:02d}) — check this matches "
          f"UTM_CRS in check_geolocation.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch SWORD reaches/nodes for one river.")
    p.add_argument("--river", default="Nenana River",
                   help="SWORD river_name, exactly as spelled in SWORD")
    p.add_argument("--types", type=int, nargs="+", default=DEFAULT_TYPES,
                   help="SWORD reach types to keep (1 = river)")
    p.add_argument("--min-width", type=float, default=None,
                   help="Minimum reach width in m (default: no filter)")
    p.add_argument("--max-width", type=float, default=None,
                   help="Maximum reach width in m (default: no filter)")
    p.add_argument("--single-channel", action="store_true",
                   help="Keep only n_chan_mod == 1 reaches (drops braided)")
    p.add_argument("--geojson", action="store_true",
                   help="Also write GeoJSON with the geometries")
    a = p.parse_args()
    main(a.river, a.types, a.min_width, a.max_width, a.single_channel, a.geojson)
