"""
Narrow-river VIIRS pixel extractor (SWORD corridor).

Same approach as Matus/extract_training_pixels.py, but candidates are limited
to one river's SWORD centerline corridor instead of the whole Landsat bbox:

1. Build the corridor: SWORD nodes of one river, buffered by --buffer-km.
2. Load the VIIRS pair via viirs_training_loader and resample it to 375 m on
   the Alaska shared grid, clipped to the corridor bbox (no Alaska TIF write).
3. Read the JRC occurrence + seasonality masks for the same window.
4. Candidates = JRC narrow-river filter AND within the corridor, spaced
   evenly along the river (by SWORD dist_out).
5. Sample Landsat at 30 m via Earth Engine, auto-label, save CSV + PNG +
   GEE inspector script.
   -> User visually classifies each pixel in GEE.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\pixel_extraction\\extract_river_pixels.py --scene 2024-05-27-j01
"""

import os, sys, math, csv
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt

NR_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # narrow_rivers/
PIPELINE_ROOT = os.path.dirname(NR_ROOT)                                      # Matus/
sys.path.insert(0, PIPELINE_ROOT)

from viirs_training_loader import load_viirs_training_pair

# ── SCENE REGISTRY ────────────────────────────────────────────────────────────
# Paths are relative to narrow_rivers/.

SCENES = {
    # key = date + satellite, so several overpasses of one date can coexist.
    "2024-06-12-j02": {
        "date":          "2024-06-12",
        "gitco":         "data/viirs/GITCO_j02_d20240612_t2247556_e2249185_b08238_c20240612230426997000_oebc_ops.h5",
        "gimgo":         "data/viirs/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_j02_d20240612_t2247556_e2249185_b08238_stitched.h5",
        "landsat_scene": "LC08_L2SP_073011_20240612_20240628_02_T1",   # LANDSAT_PRODUCT_ID
        "ee_collection": "LANDSAT/LC08/C02/T1_L2",
        "output_dir":    "pixel_extraction/output/20240612_j02",
    },
    # Shortlisted by scene_pairing/shortlist_overpasses.py: VZA ~7°, 2 min
    # before the Landsat acquisition.
    "2024-06-12-npp": {
        "date":          "2024-06-12",
        "gitco":         "data/viirs/GITCO_npp_d20240612_t2132308_e2133550_b65426_c20240612231406696000_oebc_ops.h5",
        "gimgo":         "data/viirs/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_npp_d20240612_t2132308_e2133550_b65426_stitched.h5",
        "landsat_scene": "LC08_L2SP_073011_20240612_20240628_02_T1",
        "ee_collection": "LANDSAT/LC08/C02/T1_L2",
        "output_dir":    "pixel_extraction/output/20240612_npp",
    },
    # Breakup-window scene; QA_PIXEL flags river ice as cloud, so the ~30 %
    # corridor-cloud figure overstates the real cloud.
    # J01, not the closer NPP pass: NPP t2132184 is geolocated ~3 km off
    # (check_geolocation.py); J01 checks out at 0 m.
    "2024-05-27-j01": {
        "date":          "2024-05-27",
        "gitco":         "data/viirs/GITCO_j01_d20240527_t2157530_e2159175_b33803_c20240527222120313000_oeac_ops.h5",
        "gimgo":         "data/viirs/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_j01_d20240527_t2157530_e2159175_b33803_stitched.h5",
        "landsat_scene": "LC08_L2SP_073011_20240527_20240611_02_T1",
        "ee_collection": "LANDSAT/LC08/C02/T1_L2",
        "output_dir":    "pixel_extraction/output/20240527_j01",
    },
}

NODES_CSV     = os.path.join(NR_ROOT, "data", "sword_nodes_75_300m_csv.csv")
OCC_MASK_PATH = os.path.join(PIPELINE_ROOT, "alaska_occ_375m.tif")
SEA_MASK_PATH = os.path.join(PIPELINE_ROOT, "alaska_sea_375m.tif")
EE_PROJECT    = "noaa-river-ice"

DEFAULT_RIVER     = "Sagavanirktok River"
DEFAULT_BUFFER_KM = 2.0
N_CANDIDATES      = 25

# ── Full Alaska domain (fixed, never scene-specific) ──────────────────────────
# Same grid as the JRC masks and the main pipeline's shared grid.

ALASKA_LAT_MAX = 72.0
ALASKA_LON_MIN = -171.0
ALASKA_W = 12432
ALASKA_H = 5328

RESOLUTION_M = 375.0
SHARED_RES   = RESOLUTION_M / 111_000.0   # degrees per pixel ≈ 0.003378°

# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr, pct_lo=2, pct_hi=98):
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(fin, pct_lo), np.nanpercentile(fin, pct_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def _load_nodes(river):
    """SWORD nodes of one river as a list of dicts (numeric fields as float)."""
    nodes = []
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["river_name"] != river:
                continue
            nodes.append({
                "node_id":   row["node_id"],
                "reach_id":  row["reach_id"],
                "lat":       float(row["lat"]),
                "lon":       float(row["lon"]),
                "width_m":   float(row["width_m"]),
                "max_width": float(row["max_width"]),
                "dist_out":  float(row["dist_out"]),
            })
    return nodes


def _to_km(lat, lon, lon0):
    """Local equirectangular km coordinates — fine over a ~2° corridor."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x = (lon - lon0) * 111.320 * np.cos(np.radians(lat))
    y = lat * 110.574
    return np.column_stack([x, y])


def _resample_band_to_grid(lat_swath, lon_swath, data, grid):
    """
    Resample one VIIRS swath band to the corridor grid (EPSG:4326)
    using pyresample kd_tree — same settings as the main extractor.
    """
    from pyresample import geometry, kd_tree

    valid = np.isfinite(lat_swath) & np.isfinite(lon_swath) & np.isfinite(data)
    lats_m = np.ma.masked_array(lat_swath, mask=~valid)
    lons_m = np.ma.masked_array(lon_swath, mask=~valid)
    data_m = np.ma.masked_array(data.astype(np.float32), mask=~valid)

    swath_def = geometry.SwathDefinition(lons=lons_m, lats=lats_m)
    area_def  = geometry.AreaDefinition(
        "corridor_4326", "Corridor EPSG:4326", "corridor_4326",
        {"proj": "longlat", "datum": "WGS84"},
        grid["W"], grid["H"],
        (grid["lon_min"], grid["lat_min"], grid["lon_max"], grid["lat_max"]),
    )
    result = kd_tree.resample_nearest(
        swath_def, data_m, area_def,
        radius_of_influence=7500,
        epsilon=0.5,
        fill_value=np.nan,
    )
    arr = np.ma.filled(result, np.nan) if np.ma.is_masked(result) else np.asarray(result)
    return np.asarray(arr, dtype=np.float32)


def _read_mask_window(path, grid):
    """Read a 375 m Alaska-grid mask for the corridor window (no warp needed)."""
    with rasterio.open(path) as src:
        win = Window(grid["c0"], grid["r0"], grid["W"], grid["H"])
        arr = src.read(1, window=win, boundless=True,
                       fill_value=np.nan).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _sample_landsat_ee(product_id, ee_collection, candidates_latlon):
    """
    Sample Landsat Collection 2 L2 at 30m at each candidate point
    using Earth Engine Python API.
    Returns dict: row_idx -> {SR_B2, SR_B3, SR_B4, SR_B5, SR_B6,
                               ST_B10, NDSI, NDWI, landsat_valid}
    candidates_latlon: list of (row_idx, lat, lon)
    """
    try:
        import ee
        ee.Initialize(project=EE_PROJECT)
    except Exception as ex:
        print(f"  WARNING: EE not available ({ex}) — "
              f"Landsat values will be null")
        return {}

    def _scale(image):
        opt = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thm = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(opt, None, True).addBands(thm, None, True)

    img = _scale(
        ee.ImageCollection(ee_collection)
          .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', product_id))
          .first()
    )
    ndsi = img.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI')
    ndwi = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    img  = img.addBands(ndsi).addBands(ndwi)

    features = [
        ee.Feature(
            ee.Geometry.Point([lon, lat]),
            {'row_idx': row_idx}
        )
        for row_idx, lat, lon in candidates_latlon
    ]
    fc = ee.FeatureCollection(features)

    sampled = img.select(
        ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','ST_B10','NDSI','NDWI']
    ).sampleRegions(
        collection=fc, scale=30, geometries=False
    ).getInfo()

    results = {}
    for feat in sampled['features']:
        props = feat['properties']
        idx   = props['row_idx']
        st    = props.get('ST_B10')
        ndsi_v = props.get('NDSI')
        results[idx] = {
            'SR_B2':  props.get('SR_B2'),
            'SR_B3':  props.get('SR_B3'),
            'SR_B4':  props.get('SR_B4'),
            'SR_B5':  props.get('SR_B5'),
            'SR_B6':  props.get('SR_B6'),
            'ST_B10': st,
            'NDSI':   ndsi_v,
            'NDWI':   props.get('NDWI'),
            'landsat_valid': (st is not None and ndsi_v is not None),
        }
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main(scene_key: str, river: str, buffer_km: float):
    from pykdtree.kdtree import KDTree

    scene         = SCENES[scene_key]
    date          = scene.get("date", scene_key)
    LANDSAT_SCENE = scene["landsat_scene"]
    OUTPUT_DIR    = os.path.join(NR_ROOT, scene["output_dir"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Corridor from SWORD nodes
    nodes = _load_nodes(river)
    if not nodes:
        print(f"ERROR: No SWORD nodes for river '{river}' in {NODES_CSV}")
        sys.exit(1)
    node_lat = np.array([n["lat"] for n in nodes])
    node_lon = np.array([n["lon"] for n in nodes])

    lat_mid = float(node_lat.mean())
    buf_lat = buffer_km / 110.574
    buf_lon = buffer_km / (111.320 * math.cos(math.radians(lat_mid)))

    # Snap the corridor bbox to the Alaska shared grid so rows/cols are
    # directly the shared-grid indices and the JRC masks need no warp.
    r0 = int(math.floor((ALASKA_LAT_MAX - (node_lat.max() + buf_lat)) / SHARED_RES))
    r1 = int(math.ceil((ALASKA_LAT_MAX - (node_lat.min() - buf_lat)) / SHARED_RES))
    c0 = int(math.floor(((node_lon.min() - buf_lon) - ALASKA_LON_MIN) / SHARED_RES))
    c1 = int(math.ceil(((node_lon.max() + buf_lon) - ALASKA_LON_MIN) / SHARED_RES))
    grid = {
        "r0": r0, "c0": c0, "H": r1 - r0, "W": c1 - c0,
        "lat_max": ALASKA_LAT_MAX - r0 * SHARED_RES,
        "lat_min": ALASKA_LAT_MAX - r1 * SHARED_RES,
        "lon_min": ALASKA_LON_MIN + c0 * SHARED_RES,
        "lon_max": ALASKA_LON_MIN + c1 * SHARED_RES,
    }
    H, W = grid["H"], grid["W"]
    pix_lats = grid["lat_max"] - (np.arange(H) + 0.5) * SHARED_RES
    pix_lons = grid["lon_min"] + (np.arange(W) + 0.5) * SHARED_RES

    print("=" * 60)
    print("Narrow-River VIIRS Pixel Extractor")
    print(f"River: {river}  ({len(nodes)} SWORD nodes)  buffer: ±{buffer_km} km")
    print(f"Landsat scene: {LANDSAT_SCENE}  scene: {scene_key}")
    print(f"Corridor grid: {W} × {H} px  "
          f"lat {grid['lat_min']:.4f}–{grid['lat_max']:.4f}  "
          f"lon {grid['lon_min']:.4f}–{grid['lon_max']:.4f}")
    print("=" * 60)

    # Distance from every grid cell centre to its nearest SWORD node
    lon0 = float(node_lon.mean())
    tree = KDTree(_to_km(node_lat, node_lon, lon0))
    LAT2D, LON2D = np.meshgrid(pix_lats, pix_lons, indexing="ij")
    dist_km, nearest = tree.query(_to_km(LAT2D.ravel(), LON2D.ravel(), lon0), k=1)
    dist_km = dist_km.reshape(H, W)
    nearest = nearest.reshape(H, W)
    corridor = dist_km <= buffer_km
    print(f"\nCorridor cells (<= {buffer_km} km from a node): {corridor.sum():,}")

    # 2. Load VIIRS pair and resample to the corridor grid
    GITCO_PATH = os.path.join(NR_ROOT, scene["gitco"])
    GIMGO_PATH = os.path.join(NR_ROOT, scene["gimgo"])
    buf = 0.1
    clip_bbox = (
        grid["lat_min"] - buf, grid["lat_max"] + buf,
        grid["lon_min"] - buf, grid["lon_max"] + buf,
    )
    lat, lon, bands, angles, meta = load_viirs_training_pair(
        GITCO_PATH, GIMGO_PATH, clip_bbox=clip_bbox)

    print(f"\nResampling to corridor grid ({W}×{H} px)...")
    grids = {}
    for name, data in {**bands, **angles}.items():
        grids[name] = _resample_band_to_grid(lat, lon, data, grid)
        fin = np.sum(np.isfinite(grids[name][corridor]))
        print(f"  {name}: {fin:,} valid corridor pixels")

    # 3. JRC masks for the same window
    print("\nLoading JRC occurrence + seasonality masks (corridor window)...")
    wm  = _read_mask_window(OCC_MASK_PATH, grid)
    sea = _read_mask_window(SEA_MASK_PATH, grid)

    # 4. Same dual-mask filter as the main extractor, restricted to the corridor:
    #    occurrence: 0.05 < occ < 0.90  (mixed land-water, not open water)
    #    seasonality: sea < 1.0         (not permanent water year-round)
    #    VIIRS I1 must be valid
    jrc_narrow = (np.isfinite(wm) & (wm > 0.05) & (wm < 0.90) &
                  np.isfinite(sea) & (sea < 1.0))
    viirs_ok   = np.isfinite(grids["I1"])
    narrow     = jrc_narrow & corridor & viirs_ok
    print(f"  JRC narrow-river cells in window:  {jrc_narrow.sum():,}")
    print(f"  ... of those inside the corridor:  {(jrc_narrow & corridor).sum():,}")
    print(f"  ... with valid VIIRS I1:           {narrow.sum():,}")

    rows_n, cols_n = np.where(narrow)
    if len(rows_n) == 0:
        print("WARNING: No narrow river pixels found in the corridor.")
        return

    # Even spacing along the river: order by nearest node's dist_out
    along = np.array([nodes[nearest[r, c]]["dist_out"] for r, c in zip(rows_n, cols_n)])
    order = np.argsort(along)
    n_pick = min(N_CANDIDATES, len(order))
    picks = order[np.linspace(0, len(order) - 1, n_pick).round().astype(int)]
    candidates = [(int(rows_n[i]), int(cols_n[i])) for i in picks]
    print(f"Selecting {len(candidates)} candidate pixels, spaced along the river")

    # 4b. Sample Landsat at 30m at each candidate location via EE
    EE_COLLECTION = scene.get("ee_collection")
    ls_data = {}
    if EE_COLLECTION:
        print(f"\nSampling Landsat at 30m via Earth Engine "
              f"({len(candidates)} points)...")
        candidates_latlon = [(idx, float(pix_lats[r]), float(pix_lons[c]))
                             for idx, (r, c) in enumerate(candidates)]
        ls_data = _sample_landsat_ee(
            LANDSAT_SCENE, EE_COLLECTION, candidates_latlon)
        n_valid = sum(1 for v in ls_data.values() if v.get('landsat_valid'))
        print(f"  {n_valid}/{len(candidates)} candidates inside "
              f"Landsat swath")
    else:
        print("\nNo EE collection — Landsat values will be null")

    # 5. Extract values for each candidate
    def _val(name, r, c, nd):
        v = grids[name][r, c]
        return round(float(v), nd) if np.isfinite(v) else ""

    rows_out = []
    for idx, (r, c) in enumerate(candidates):
        node = nodes[nearest[r, c]]
        ls = ls_data.get(idx, {})

        def _safe(key):
            v = ls.get(key)
            return round(float(v), 4) if v is not None else ""

        st     = ls.get('ST_B10')
        ndsi_v = ls.get('NDSI')

        # Auto-label ground truth class from Landsat 30m thermal + NDSI
        if st is not None and ndsi_v is not None:
            is_ice  = float(st)     < 273.0
            is_snow = float(ndsi_v) > 0.4
            if   not is_ice and not is_snow:
                auto_class = "ice_free_river_snow_free_land"
            elif not is_ice and is_snow:
                auto_class = "ice_free_river_snow_covered_land"
            elif is_ice and not is_snow:
                auto_class = "ice_covered_river_snow_free_land"
            else:
                auto_class = "ice_covered_river_snow_covered_land"
            auto_note = (f"Landsat confirmed at 30m. "
                         f"ST_B10={round(float(st),2)}K "
                         f"NDSI={round(float(ndsi_v),4)}.")
        else:
            auto_class = ""
            auto_note  = "Landsat null - outside swath. Classify manually."

        rows_out.append({
            "viirs_date":      meta["date"][:4]+"-"+meta["date"][4:6]+"-"+meta["date"][6:],
            "landsat_scene":   LANDSAT_SCENE,
            "river":           river,
            "reach_id":        node["reach_id"],
            "nearest_node_id": node["node_id"],
            "dist_to_node_m":  round(float(dist_km[r, c]) * 1000.0, 1),
            "sword_width_m":   node["width_m"],
            "sword_max_width_m": node["max_width"],
            "row_shared_grid": grid["r0"] + r,
            "col_shared_grid": grid["c0"] + c,
            "lat":             round(float(pix_lats[r]), 5),
            "lon":             round(float(pix_lons[c]), 5),
            "water_fraction":  round(float(wm[r, c]), 4),
            "I1":  _val("I1", r, c, 4),
            "I2":  _val("I2", r, c, 4),
            "I3":  _val("I3", r, c, 4),
            "I4":  _val("I4", r, c, 4),
            "I5":  _val("I5", r, c, 4),
            "SZA": _val("SZA", r, c, 3),
            "SAA": _val("SAA", r, c, 3),
            "VZA": _val("VZA", r, c, 3),
            "VAA": _val("VAA", r, c, 3),
            "LS_SR_B2":  _safe('SR_B2'),
            "LS_SR_B3":  _safe('SR_B3'),
            "LS_SR_B4":  _safe('SR_B4'),
            "LS_SR_B5":  _safe('SR_B5'),
            "LS_SR_B6":  _safe('SR_B6'),
            "LS_ST_B10": _safe('ST_B10'),
            "LS_NDWI":   _safe('NDWI'),
            "LS_NDSI":   _safe('NDSI'),
            "ground_truth_class": auto_class,
            "notes": auto_note,
        })

    # 6. Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"training_candidates_{date}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved CSV: {csv_path}")
    print("\nCandidate rows:")
    for row in rows_out:
        print(f"  lat={row['lat']}, lon={row['lon']}  reach={row['reach_id']}  "
              f"wf={row['water_fraction']}  VZA={row['VZA']}  "
              f"{row['ground_truth_class'] or 'no Landsat'}")

    n_confirmed = sum(1 for row in rows_out if row["ground_truth_class"])
    n_null      = len(rows_out) - n_confirmed
    from collections import Counter
    class_counts = Counter(
        row["ground_truth_class"]
        for row in rows_out if row["ground_truth_class"])
    print(f"\nSummary: {n_confirmed} Landsat-confirmed / "
          f"{n_null} Landsat-null")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")

    # 7. Side-by-side PNG: VIIRS I2-I2-I1 false color vs water mask,
    #    with the SWORD centerline and corridor outline
    i2 = grids["I2"]
    i1 = grids["I1"]
    extent = [grid["lon_min"], grid["lon_max"], grid["lat_min"], grid["lat_max"]]
    aspect = 1.0 / math.cos(math.radians(lat_mid))   # true ground shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 12))
    fig.suptitle(
        f"{river} — VIIRS 2-2-1 false color vs JRC water fraction\n"
        f"Landsat {LANDSAT_SCENE}  |  VIIRS {meta['date'][:4]}-"
        f"{meta['date'][4:6]}-{meta['date'][6:]}  |  corridor ±{buffer_km} km",
        fontsize=10)

    rgb      = np.stack([_normalize(i2), _normalize(i2), _normalize(i1)], axis=-1)
    nan_mask = ~(np.isfinite(i1) & np.isfinite(i2))
    alpha    = np.where(nan_mask, 0.0, 1.0)
    axes[0].imshow(np.dstack([rgb, alpha]), interpolation="nearest",
                   extent=extent, aspect=aspect, origin="upper")
    axes[0].set_title("VIIRS 2-2-1 (I2→R, I2→G, I1→B)\n"
                      "Ice/snow=bright | Water=dark | NaN=transparent")

    im2 = axes[1].imshow(wm, cmap="Blues", vmin=0, vmax=1,
                         interpolation="nearest",
                         extent=extent, aspect=aspect, origin="upper")
    plt.colorbar(im2, ax=axes[1], fraction=0.03, label="Water fraction")
    axes[1].set_title("JRC water fraction (occurrence)\n"
                      "red × = Landsat-confirmed | gray ○ = outside swath")

    for ax in axes:
        ax.scatter(node_lon, node_lat, s=0.5, color="yellow", label="SWORD nodes")
        ax.contour(pix_lons, pix_lats, corridor.astype(float), levels=[0.5],
                   colors="magenta", linewidths=0.8)
        for idx, (r, c) in enumerate(candidates):
            if ls_data.get(idx, {}).get('landsat_valid', False):
                ax.plot(pix_lons[c], pix_lats[r], "rx", markersize=8, markeredgewidth=2)
            else:
                ax.plot(pix_lons[c], pix_lats[r], "o", color="gray", markersize=6,
                        markeredgewidth=1.5, markerfacecolor="none")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"viirs_vs_watermask_{date}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved PNG: {png_path}")
    plt.close()

    # 8. Generate GEE inspector script for visual verification
    confirmed_rows = [r for r in rows_out if r["ground_truth_class"]]
    if confirmed_rows:
        from datetime import datetime as _dt, timedelta as _td
        _d = _dt.strptime(date, "%Y-%m-%d")
        s2_start = (_d - _td(days=7)).strftime("%Y-%m-%d")
        s2_end   = (_d + _td(days=7)).strftime("%Y-%m-%d")

        js_candidates = []
        for i, r in enumerate(confirmed_rows, 1):
            js_candidates.append(
                f"  {{id: {i}, lat: {r['lat']}, lon: {r['lon']}, "
                f"auto_class: '{r['ground_truth_class']}', "
                f"st: {r['LS_ST_B10']}, ndsi: {r['LS_NDSI']}, "
                f"wf: {r['water_fraction']}}}"
            )
        candidates_js = ",\n".join(js_candidates)
        center_lon = float(np.mean([float(r['lon']) for r in confirmed_rows]))
        center_lat = float(np.mean([float(r['lat']) for r in confirmed_rows]))

        gee_script = f"""/*
 * GEE Visual Inspector - {date} - {river}
 * Landsat scene: {LANDSAT_SCENE}
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('{EE_COLLECTION}')
  .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', '{LANDSAT_SCENE}'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var corridorBbox = ee.Geometry.Rectangle(
  [{grid['lon_min']}, {grid['lat_min']}, {grid['lon_max']}, {grid['lat_max']}]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(corridorBbox)
  .filterDate('{s2_start}', '{s2_end}')
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

// ── Map layers ──────────────────────────────────────────────────────────────
Map.addLayer(full, {{bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}}, 'RGB (True Color)', true);
Map.addLayer(full, {{bands: ['SR_B5', 'SR_B4', 'SR_B3'], min: 0, max: 0.4}}, 'False Color NIR', false);
Map.addLayer(full, {{bands: ['SR_B6', 'SR_B5', 'SR_B4'], min: 0, max: 0.5}}, 'SWIR false colour (snow/ice = cyan, cloud = white)', false);
Map.addLayer(full, {{bands: ['ST_B10'], min: 250, max: 290, palette: ['purple','blue','cyan','yellow','red']}}, 'Thermal (ST_B10)', false);
Map.addLayer(ndsi, {{min: -0.5, max: 1.0, palette: ['brown','white','cyan']}}, 'NDSI', false);
Map.addLayer(s2, {{bands: ['B4','B3','B2'], min: 0, max: 3000}}, 'Sentinel-2 10m (nearest pass)', false);

var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
Map.addLayer(cloudMask.selfMask(), {{palette: ['red']}}, 'Cloud Mask', false);

// ── Candidate pixels (with 375 m VIIRS cell outline) ────────────────────────
var candidates = [
{candidates_js}
];

var half = {SHARED_RES / 2};
var iceSnow = [], iceNoSnow = [], freeNoSnow = [], freeSnow = [], unknown = [], cells = [];
candidates.forEach(function(c) {{
  var feat = ee.Feature(ee.Geometry.Point([c.lon, c.lat]),
    {{'Pixel': c.id, 'Auto_Class': c.auto_class, 'ST_B10_K': c.st, 'NDSI': c.ndsi, 'JRC_wf': c.wf}});
  cells.push(ee.Feature(ee.Geometry.Rectangle([c.lon - half, c.lat - half, c.lon + half, c.lat + half])));
  if (c.auto_class === 'ice_covered_river_snow_covered_land') iceSnow.push(feat);
  else if (c.auto_class === 'ice_covered_river_snow_free_land') iceNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_free_land') freeNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_covered_land') freeSnow.push(feat);
  else unknown.push(feat);
}});

Map.addLayer(ee.FeatureCollection(cells).style({{color: 'FFFF00', fillColor: '00000000', width: 1}}), {{}}, 'VIIRS 375 m cells', true);
if (iceSnow.length > 0)   Map.addLayer(ee.FeatureCollection(iceSnow),   {{color: 'FF0000'}}, 'ice_covered+snow_covered (' + iceSnow.length + ')', true);
if (iceNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(iceNoSnow), {{color: 'FF8800'}}, 'ice_covered+snow_free (' + iceNoSnow.length + ')', true);
if (freeNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(freeNoSnow), {{color: '00FF00'}}, 'ice_free+snow_free (' + freeNoSnow.length + ')', true);
if (freeSnow.length > 0)  Map.addLayer(ee.FeatureCollection(freeSnow),  {{color: '0088FF'}}, 'ice_free+snow_covered (' + freeSnow.length + ')', true);
if (unknown.length > 0)   Map.addLayer(ee.FeatureCollection(unknown),   {{color: 'AAAAAA'}}, 'unlabeled / legacy class (' + unknown.length + ')', true);

Map.setCenter({center_lon:.2f}, {center_lat:.2f}, 8);

print('{date} - {river} - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {{
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
}});
"""
        gee_path = os.path.join(OUTPUT_DIR, f"gee_inspector_{date.replace('-', '')}.js")
        with open(gee_path, "w", encoding="utf-8") as f:
            f.write(gee_script)
        print(f"\nSaved GEE inspector script: {gee_path}")
        print("  -> Paste into https://code.earthengine.google.com/ to verify pixels")
    else:
        print("\nNo Landsat-confirmed pixels — skipping GEE script generation")

    print("\nNext steps:")
    print("1. Check the PNG: candidates should sit on the SWORD centerline")
    print("2. Paste the GEE inspector script into code.earthengine.google.com")
    print("3. Click Inspector tab, then click each marker to verify class")
    print("4. Update ground_truth_class in the CSV if needed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract VIIRS pixels along a SWORD river corridor.")
    parser.add_argument(
        "--scene", required=False, default="2024-06-12-npp",
        choices=list(SCENES.keys()),
        help="Scene to process (key of SCENES: date + satellite)")
    parser.add_argument(
        "--river", required=False, default=DEFAULT_RIVER,
        help="SWORD river_name to build the corridor from")
    parser.add_argument(
        "--buffer-km", type=float, required=False, default=DEFAULT_BUFFER_KM,
        help="Corridor half-width around the SWORD nodes, in km")
    args = parser.parse_args()
    main(args.scene, args.river, args.buffer_km)
