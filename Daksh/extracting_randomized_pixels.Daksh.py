"""
Standalone pixel extraction — runs AFTER main.py has produced the
reprojected VIIRS and Landsat TIFs in output/<YYYYMMDD>/.

Scans output/ for date folders containing viirs_alaska_*.tif and
landsat_*.tif, then generates:
    training_candidates_YYYY-MM-DD.csv
    viirs_vs_watermask_YYYY-MM-DD.png
    training_candidates_YYYY-MM-DD_report.html      (GEE report, built in)
    training_candidates_YYYY-MM-DD_gee_results.csv  (GEE report, built in)

Usage:
    python extract_pixels.py                  # all dates, random 25 pixels
    python extract_pixels.py 20240102         # single date
    python extract_pixels.py 20240102 20240206  # multiple dates
    python extract_pixels.py --overwrite      # regenerate existing
    python extract_pixels.py --seed 42        # reproducible random selection
    python extract_pixels.py --uniform        # evenly spaced (main.py behavior)
    python extract_pixels.py --pure-random    # fully random (no spatial spread)
    python extract_pixels.py --no-modis       # skip MODIS NDVI / land cover columns
    python extract_pixels.py --no-report      # skip the GEE HTML report per CSV
    python extract_pixels.py --view           # open report(s) + start live map server
                                              # (reports already there -> server only)
    python extract_pixels.py --no-view        # never ask to open reports
    (default asks at the end whether to open the report(s) + start the map server)
    (default is stratified random — spread out but different each run)
"""

import os, re, math, csv, sys, logging, socket, webbrowser
import base64, functools, html, http.server, json, socketserver
import urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject as rio_reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


# ---------------------------------------------------------------------------
# Configuration — same constants as main.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
WATER_MASK_OCC = os.path.join(DATA_DIR, "alaska_occ_375m.tif")
WATER_MASK_SEA = os.path.join(DATA_DIR, "alaska_sea_375m.tif")
LANDSAT_DIR  = os.path.join(DATA_DIR, "landsat")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
PIXEL_DIR    = os.path.join(PROJECT_ROOT, "output_pixels")

TARGET_CRS = "EPSG:4326"

ALASKA_LAT_MIN = 54.0
ALASKA_LAT_MAX = 72.0
ALASKA_LON_MIN = -171.0
ALASKA_LON_MAX = -129.0

VIIRS_RES_M = 375.0
VIIRS_RES   = VIIRS_RES_M / 111_000.0
VIIRS_W     = int(round((ALASKA_LON_MAX - ALASKA_LON_MIN) / VIIRS_RES))
VIIRS_H     = int(round((ALASKA_LAT_MAX - ALASKA_LAT_MIN) / VIIRS_RES))

NODATA       = -9999.0
N_CANDIDATES = 25

# MODIS enrichment (Earth Engine). Set False (or pass --no-modis) to skip.
ENABLE_MODIS = True
EE_PROJECT   = "noaa-river-ice"

# GEE report run on every CSV. Set False (or pass --no-report) to skip.
ENABLE_REPORT = True
MAP_SERVER_PORT = 8765   # live-map server port (same as output/report_main.py)

VIIRS_BAND_ORDER = ["I1", "I2", "I3", "I4", "I5", "SZA", "SAA", "VZA", "VAA"]

log = logging.getLogger("extract_pixels")
log.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%H:%M:%S"))
log.addHandler(_ch)

# ---------------------------------------------------------------------------
# MODIS enrichment (Earth Engine)
#   modis_ndvi      MOD13Q1 NDVI (250 m, 16-day composite, scaled to -1..1)
#   modis_lc_type1  MCD12Q1 LC_Type1 IGBP class (500 m, annual)
#   modis_lc_name   readable IGBP class name
# ---------------------------------------------------------------------------

IGBP_LOOKUP = {
    0:   'Water Bodies',
    1:   'Evergreen Needleleaf Forests',
    2:   'Evergreen Broadleaf Forests',
    3:   'Deciduous Needleleaf Forests',
    4:   'Deciduous Broadleaf Forests',
    5:   'Mixed Forests',
    6:   'Closed Shrublands',
    7:   'Open Shrublands',
    8:   'Woody Savannas',
    9:   'Savannas',
    10:  'Grasslands',
    11:  'Permanent Wetlands',
    12:  'Croplands',
    13:  'Urban and Built-up Lands',
    14:  'Cropland/Natural Vegetation Mosaics',
    15:  'Permanent Snow and Ice',
    16:  'Barren',
    17:  'Unclassified',
    255: 'Fill/NoData',
}


def init_ee(project=EE_PROJECT):
    """Initialise Earth Engine; return the ee module, or None if unavailable."""
    try:
        import ee
        ee.Initialize(project=project)
        log.info(f"Earth Engine initialised (project={project})")
        return ee
    except Exception as exc:
        log.warning(f"Earth Engine init failed: {exc}")
        log.warning("MODIS columns will be empty — "
                    "check 'earthengine authenticate' and project access.")
        return None


def sample_modis(ee, points, scene_date):
    """Sample MOD13Q1 NDVI and MCD12Q1 LC_Type1 at the given points.

    points     : list of (key, lat, lon); key is any int/str identifier
    scene_date : 'YYYY-MM-DD'

    Returns {key: {'modis_ndvi': float|None, 'modis_lc_type1': int|None}}.
    NDVI: 16-day lookback window (MOD13Q1 is a 16-day composite, 250 m).
    LC:   scene year, else up to 2 prior years (MCD12Q1 is annual, 500 m).
    """
    if not points:
        return {}

    dt = datetime.strptime(scene_date, "%Y-%m-%d")

    # EE feature properties round-trip keys as strings; map back afterwards.
    keys = {str(k): k for k, _, _ in points}
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"key": str(k)})
        for k, lat, lon in points
    ])

    results = {k: {"modis_ndvi": None, "modis_lc_type1": None}
               for k, _, _ in points}

    # NDVI — MOD13Q1 (250 m, 16-day composite)
    try:
        ndvi_img = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
              .filterDate(
                  ee.Date(scene_date).advance(-16, "day"),
                  ee.Date(scene_date).advance(1,   "day"),
              )
              .sort("system:time_start", False)
              .first()
              .select("NDVI")
        )
        ndvi_sampled = ndvi_img.sampleRegions(
            collection=fc, scale=250,
            projection="EPSG:4326", geometries=True,
        ).getInfo()
        for feat in ndvi_sampled.get("features", []):
            props = feat["properties"]
            raw   = props.get("NDVI")
            if raw is not None:
                results[keys[props["key"]]]["modis_ndvi"] = round(float(raw) * 0.0001, 6)
    except Exception as exc:
        log.warning(f"  MODIS NDVI sampling failed for {scene_date}: {exc}")

    # Land Cover — MCD12Q1 (500 m, annual). It lags real time by ~1 year, so
    # fall back to up to 2 prior years when the scene year isn't published.
    lc_img = None
    for year_offset in range(0, 3):
        lc_year = dt.year - year_offset
        col = (ee.ImageCollection("MODIS/061/MCD12Q1")
                 .filterDate(f"{lc_year}-01-01", f"{lc_year + 1}-01-01"))
        try:
            if col.size().getInfo() > 0:
                lc_img = col.first().select("LC_Type1")
                if year_offset > 0:
                    log.info(f"  MCD12Q1 {dt.year} not available — "
                             f"using {lc_year} land cover instead")
                break
        except Exception as exc:
            log.warning(f"  MCD12Q1 {lc_year} lookup failed: {exc}")
    if lc_img is None:
        log.warning(f"  No MCD12Q1 data within 3 years of {scene_date}")
        return results

    try:
        lc_sampled = lc_img.sampleRegions(
            collection=fc, scale=500,
            projection="EPSG:4326", geometries=True,
        ).getInfo()
        for feat in lc_sampled.get("features", []):
            props = feat["properties"]
            raw   = props.get("LC_Type1")
            if raw is not None:
                results[keys[props["key"]]]["modis_lc_type1"] = int(raw)
    except Exception as exc:
        log.warning(f"  MODIS land cover sampling failed for {scene_date}: {exc}")

    return results


def enrich_rows(ee, rows_out, scene_date):
    """Add modis_ndvi / modis_lc_type1 / modis_lc_name to every row (in place).

    Columns are always added so the CSV schema is stable; only rows that are
    Landsat-confirmed (notes without "Landsat null") are sampled.
    """
    for r in rows_out:
        r["modis_ndvi"]     = ""
        r["modis_lc_type1"] = ""
        r["modis_lc_name"]  = ""

    if ee is None:
        return

    confirmed = [
        (i, r["lat"], r["lon"])
        for i, r in enumerate(rows_out)
        if "Landsat null" not in str(r.get("notes", ""))
    ]
    if not confirmed:
        log.info("  No Landsat-confirmed pixels — skipping MODIS sampling")
        return

    log.info(f"  Sampling MODIS for {len(confirmed)} "
             f"Landsat-confirmed pixel(s)...")
    modis = sample_modis(ee, confirmed, scene_date)
    n_ndvi = n_lc = 0
    for idx, vals in modis.items():
        ndvi = vals.get("modis_ndvi")
        lc   = vals.get("modis_lc_type1")
        if ndvi is not None:
            rows_out[idx]["modis_ndvi"] = ndvi
            n_ndvi += 1
        if lc is not None:
            rows_out[idx]["modis_lc_type1"] = lc
            rows_out[idx]["modis_lc_name"]  = IGBP_LOOKUP.get(lc, f"Unknown({lc})")
            n_lc += 1
    log.info(f"  MODIS returned: NDVI {n_ndvi}/{len(confirmed)}, "
             f"LC {n_lc}/{len(confirmed)}")


# ---------------------------------------------------------------------------
# Helpers (same as main.py)
# ---------------------------------------------------------------------------

def parse_mtl(mtl_path):
    """Parse MTL.txt (same as main_reproject.daksh.py): thermal/reflectance
    constants + LANDSAT_PRODUCT_ID / DATE_ACQUIRED / SCENE_CENTER_TIME."""
    numeric_keys = {
        "K1_CONSTANT_BAND_10": "K1",
        "K2_CONSTANT_BAND_10": "K2",
        "RADIANCE_MULT_BAND_10": "RADIANCE_MULT",
        "RADIANCE_ADD_BAND_10": "RADIANCE_ADD",
        "REFLECTANCE_MULT_BAND_3": "REFL_MULT_B3",
        "REFLECTANCE_ADD_BAND_3": "REFL_ADD_B3",
        "REFLECTANCE_MULT_BAND_6": "REFL_MULT_B6",
        "REFLECTANCE_ADD_BAND_6": "REFL_ADD_B6",
    }
    string_keys = {
        "DATE_ACQUIRED": "DATE_ACQUIRED",
        "SCENE_CENTER_TIME": "SCENE_CENTER_TIME",
        "LANDSAT_PRODUCT_ID": "LANDSAT_PRODUCT_ID",
    }
    vals = {}
    try:
        with open(mtl_path, "r") as f:
            for line in f:
                key, eq, raw = line.strip().partition("=")
                if not eq:
                    continue
                key = key.strip().upper()
                raw = raw.strip().strip('"').strip()
                if key in numeric_keys:
                    try:
                        vals[numeric_keys[key]] = float(raw)
                    except ValueError:
                        pass
                elif key in string_keys:
                    vals[string_keys[key]] = raw
    except Exception:
        return None
    if all(k in vals for k in ("K1", "K2", "RADIANCE_MULT", "RADIANCE_ADD")):
        return vals
    return None


def dn_to_kelvin(dn, mtl_vals):
    rad = mtl_vals["RADIANCE_MULT"] * dn + mtl_vals["RADIANCE_ADD"]
    if rad <= 0:
        return None
    return mtl_vals["K2"] / math.log(mtl_vals["K1"] / rad + 1)


def dn_to_toa_reflectance(dn, band, mtl_vals):
    """Convert Landsat DN to TOA reflectance using MTL REFLECTANCE_MULT/ADD.
    Sun-elevation correction is skipped — it divides every band by the same
    sin(elev), so it cancels in band ratios like NDSI. Returns dn unchanged
    if dn is None or the MTL coefficients for this band aren't available."""
    if dn is None or not mtl_vals:
        return dn
    mult = mtl_vals.get(f"REFL_MULT_B{band}")
    add  = mtl_vals.get(f"REFL_ADD_B{band}")
    if mult is None or add is None:
        return dn
    return mult * dn + add


def _normalize(arr, pct_lo=2, pct_hi=98):
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(fin, pct_lo), np.nanpercentile(fin, pct_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

# ---------------------------------------------------------------------------
# Discover B10 + MTL from raw Landsat data
# ---------------------------------------------------------------------------

def find_b10_and_mtl(date_str, landsat_path=None):
    """Return (b10_path, mtl_vals) for the raw scene behind landsat_path.

    main_reproject.daksh.py writes landsat_<date>_<N>.tif where N is the index
    of the scene in data/landsat/<date>/ grouped by scene ID, sorted, keeping
    scenes with >= 5 spectral bands (no suffix when there is only one scene).
    Mirror that here so thermal + LANDSAT_PRODUCT_ID come from the right scene.
    """
    ls_folder = os.path.join(LANDSAT_DIR, date_str)
    if not os.path.isdir(ls_folder):
        return None, None
    groups = {}
    for f in os.listdir(ls_folder):
        m = re.match(r'^(.+)_(B\d+|SAA|SZA|VAA|VZA)\.TIF$', f, re.I)
        if m:
            sid = m.group(1)
        elif f.upper().endswith("_MTL.TXT"):
            sid = f[:-8]
        else:
            continue
        groups.setdefault(sid, []).append(f)
    scenes = [sid for sid in sorted(groups)
              if len([f for f in groups[sid] if re.search(r'_B[1-6]\.TIF$', f, re.I)]) >= 5]
    if not scenes:
        return None, None

    idx = 0
    if landsat_path:
        m = re.search(rf'landsat_{date_str}_(\d+)\.tif$', os.path.basename(landsat_path))
        if m:
            idx = int(m.group(1))
        elif len(scenes) > 1:
            log.warning(f"  {os.path.basename(landsat_path)} has no scene index but "
                        f"{len(scenes)} scenes exist — using the first")
    if idx >= len(scenes):
        log.warning(f"  No raw scene #{idx} in {ls_folder} — thermal/scene ID unavailable")
        return None, None

    sfiles = groups[scenes[idx]]
    b10 = sorted(f for f in sfiles if re.search(r'_B10\.TIF$', f, re.I))
    mtl = [f for f in sfiles if f.upper().endswith("MTL.TXT")]
    if not b10 or not mtl:
        return None, None
    mtl_vals = parse_mtl(os.path.join(ls_folder, mtl[0]))
    if mtl_vals is None:
        return None, None
    return os.path.join(ls_folder, b10[0]), mtl_vals


# ---------------------------------------------------------------------------
# Discover processed TIFs in output/
# ---------------------------------------------------------------------------

def discover_output_dates(requested_dates=None):
    """Find date folders in output/ that have both viirs and landsat TIFs."""
    pairs = []
    for folder in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, folder)
        if not os.path.isdir(fp):
            continue
        if not re.match(r'^\d{8}$', folder):
            continue
        if requested_dates and folder not in requested_dates:
            continue

        files = os.listdir(fp)
        viirs_tifs   = sorted([f for f in files
                               if f.startswith("viirs_alaska_") and f.endswith(".tif")])
        landsat_tifs = sorted([f for f in files
                               if f.startswith("landsat_") and f.endswith(".tif")])

        if not viirs_tifs or not landsat_tifs:
            log.info(f"Skipping {folder}: missing VIIRS or Landsat TIF")
            continue

        for vt in viirs_tifs:
            for lt in landsat_tifs:
                pairs.append((
                    folder,
                    os.path.join(fp, vt),
                    os.path.join(fp, lt),
                    fp,
                ))
    return pairs

# ---------------------------------------------------------------------------
# Pixel extraction + CSV + PNG  (same logic as main.py generate_csv)
# ---------------------------------------------------------------------------

def extract_and_save(viirs_path, landsat_path, date_str, out_dir,
                     b10_path=None, mtl_vals=None, overwrite=False,
                     seed=None, mode="stratified", region=None, side=None,
                     ee=None):

    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    pixel_date_dir = os.path.join(PIXEL_DIR, date_str)
    os.makedirs(pixel_date_dir, exist_ok=True)

    # Derive suffixes from filenames so multi-scene/granule dates don't collide
    # e.g. landsat_20240102.tif → "" , landsat_20240102_1.tif → "_ls1"
    #       viirs_alaska_20240102.tif → "" , viirs_alaska_20240102_0.tif → "_v0"
    ls_base = os.path.basename(landsat_path).replace(".tif", "")
    ls_suffix = ls_base.replace(f"landsat_{date_str}", "")
    v_base = os.path.basename(viirs_path).replace(".tif", "")
    v_suffix = v_base.replace(f"viirs_alaska_{date_str}", "")

    tag = ""
    if v_suffix:
        tag += f"_v{v_suffix.lstrip('_')}"
    if ls_suffix:
        tag += f"_ls{ls_suffix.lstrip('_')}"

    csv_path = os.path.join(pixel_date_dir, f"training_candidates_{date_fmt}{tag}.csv")
    png_path = os.path.join(pixel_date_dir, f"viirs_vs_watermask_{date_fmt}{tag}.png")
    if os.path.exists(csv_path) and os.path.exists(png_path) and not overwrite:
        log.info(f"  CSV+PNG exist, skip: {date_fmt}{tag}")
        return csv_path

    # --- Load Landsat metadata ---
    with rasterio.open(landsat_path) as ls_src:
        ls_bounds = ls_src.bounds
        ls_tf     = ls_src.transform
        ls_w, ls_h = ls_src.width, ls_src.height
        ls_band_names = []
        for bi in range(1, ls_src.count + 1):
            tags = ls_src.tags(bi)
            ls_band_names.append(tags.get("name", f"band{bi}"))

    # Real USGS scene ID from MTL when available (needed by output/report_main.py);
    # fall back to the output filename.
    if mtl_vals and mtl_vals.get("LANDSAT_PRODUCT_ID"):
        ls_scene_id = mtl_vals["LANDSAT_PRODUCT_ID"]
    else:
        ls_scene_id = os.path.basename(landsat_path).replace(".tif", "")

    scene_lon_min, scene_lat_min = ls_bounds.left, ls_bounds.bottom
    scene_lon_max, scene_lat_max = ls_bounds.right, ls_bounds.top

    scene_w = int(math.ceil((scene_lon_max - scene_lon_min) / VIIRS_RES))
    scene_h = int(math.ceil((scene_lat_max - scene_lat_min) / VIIRS_RES))
    scene_tf = from_bounds(scene_lon_min, scene_lat_min,
                           scene_lon_max, scene_lat_max,
                           scene_w, scene_h)

    log.info(f"  Scene grid: {scene_w} x {scene_h} px  res: {VIIRS_RES:.6f}\u00b0")

    # Read VIIRS bands windowed to scene extent
    log.info(f"  Loading VIIRS from: {viirs_path}")
    viirs_grids = {}
    with rasterio.open(viirs_path) as v_src:
        v_nodata = v_src.nodata if v_src.nodata is not None else NODATA
        win = window_from_bounds(scene_lon_min, scene_lat_min,
                                 scene_lon_max, scene_lat_max,
                                 transform=v_src.transform)
        v_win_tf = v_src.window_transform(win)
        for bi, name in enumerate(VIIRS_BAND_ORDER, 1):
            raw = v_src.read(bi, window=win).astype(np.float32)
            raw[raw == v_nodata] = np.nan
            out = np.full((scene_h, scene_w), np.nan, np.float32)
            rio_reproject(
                source=raw, destination=out,
                src_transform=v_win_tf, src_crs=TARGET_CRS,
                src_nodata=np.nan,
                dst_transform=scene_tf, dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            viirs_grids[name] = out

    log.info(f"  VIIRS bands loaded")
    for name, arr in viirs_grids.items():
        fin = int(np.sum(np.isfinite(arr)))
        log.info(f"    {name}: {fin:,} valid pixels")

    # Read both water masks (occurrence + seasonality)
    def _load_wm(path):
        with rasterio.open(path) as src:
            win = window_from_bounds(scene_lon_min, scene_lat_min,
                                     scene_lon_max, scene_lat_max,
                                     transform=src.transform)
            raw = src.read(1, window=win).astype(np.float32)
            win_tf = src.window_transform(win)
            if src.nodata is not None:
                raw[raw == src.nodata] = np.nan
        out = np.full((scene_h, scene_w), np.nan, np.float32)
        rio_reproject(source=raw, destination=out,
                      src_transform=win_tf, src_crs=TARGET_CRS,
                      src_nodata=np.nan,
                      dst_transform=scene_tf, dst_crs=TARGET_CRS,
                      dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        return out

    log.info(f"  Loading occurrence mask ...")
    wm_occ = _load_wm(WATER_MASK_OCC)
    log.info(f"    occ shape: {wm_occ.shape}  valid: {int(np.sum(np.isfinite(wm_occ))):,}")
    log.info(f"  Loading seasonality mask ...")
    wm_sea = _load_wm(WATER_MASK_SEA)
    log.info(f"    sea shape: {wm_sea.shape}  valid: {int(np.sum(np.isfinite(wm_sea))):,}")
    wm = wm_occ  # used by the "JRC water" panel

    # Landsat valid mask
    log.info(f"  Building Landsat valid mask ...")
    ls_valid = np.zeros((scene_h, scene_w), dtype=bool)
    with rasterio.open(landsat_path) as ls_src:
        for bi in range(1, ls_src.count + 1):
            band = ls_src.read(bi, out_shape=(scene_h, scene_w),
                               resampling=Resampling.nearest).astype(np.float32)
            ls_valid |= (band != NODATA) & np.isfinite(band) & (band != 0)
    log.info(f"    Landsat valid: {int(ls_valid.sum()):,} px")

    # Mixed-pixel candidates: 0.05 < occ < 0.90 AND sea < 1.0
    narrow = (np.isfinite(wm_occ) & (wm_occ > 0.05) & (wm_occ < 0.90) &
              np.isfinite(wm_sea) & (wm_sea < 1.0) &
              np.isfinite(viirs_grids["I1"]) & ls_valid)
    rows_n, cols_n = np.where(narrow)
    log.info(f"  Mixed-pixel candidates: {len(rows_n):,}")

    if len(rows_n) == 0:
        log.warning(f"  No mixed-pixel candidates — skipping {date_str}")
        return

    all_pixels = list(zip(rows_n, cols_n))

    # Optional region (row-based: top/middle/bottom) and side (column-based: left/right).
    if region is not None:
        if region == "top":
            all_pixels = [(r, c) for r, c in all_pixels if r < scene_h / 2]
        elif region == "bottom":
            all_pixels = [(r, c) for r, c in all_pixels if r >= scene_h / 2]
        elif region == "middle":
            all_pixels = [(r, c) for r, c in all_pixels
                          if scene_h / 4 <= r < 3 * scene_h / 4]
        else:
            log.warning(f"  Unknown region '{region}' — ignored")
        log.info(f"  Region '{region}': pool now {len(all_pixels):,} pixels")

    if side is not None:
        if side == "left":
            all_pixels = [(r, c) for r, c in all_pixels if c < scene_w / 2]
        elif side == "right":
            all_pixels = [(r, c) for r, c in all_pixels if c >= scene_w / 2]
        else:
            log.warning(f"  Unknown side '{side}' — ignored")
        log.info(f"  Side '{side}': pool now {len(all_pixels):,} pixels")

    if not all_pixels:
        log.warning(f"  No candidates after region/side filter — skipping {date_str}")
        return

    rng = np.random.default_rng(seed)
    seed_msg = f", seed={seed}" if seed is not None else ""

    if mode == "uniform":
        # Same as main.py — evenly spaced, always identical
        step = max(1, len(all_pixels) // N_CANDIDATES)
        candidates = all_pixels[::step][:N_CANDIDATES]
        log.info(f"  Selecting {len(candidates)} candidates (uniform spacing)")
    elif mode == "pure-random":
        # Fully random — no spatial guarantees
        n_pick = min(N_CANDIDATES, len(all_pixels))
        indices = rng.choice(len(all_pixels), size=n_pick, replace=False)
        candidates = [all_pixels[i] for i in sorted(indices)]
        log.info(f"  Selecting {len(candidates)} candidates "
                 f"(pure random{seed_msg})")
    else:
        # Stratified random (default): split pool into N bins, pick one
        # random from each — spread out like uniform but different each run
        n_bins = min(N_CANDIDATES, len(all_pixels))
        bin_edges = np.linspace(0, len(all_pixels), n_bins + 1, dtype=int)
        candidates = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if lo < hi:
                idx = int(rng.integers(lo, hi))
                candidates.append(all_pixels[idx])
        log.info(f"  Selecting {len(candidates)} candidates "
                 f"(stratified random{seed_msg})")

    # --- Sample pixels ---
    rows_out = []
    n_skipped = 0
    n_ls_valid = 0
    with rasterio.open(landsat_path) as ls_src:
        for idx, (r, c) in enumerate(candidates):
            lat_px = scene_lat_max - (r + 0.5) * VIIRS_RES
            lon_px = scene_lon_min + (c + 0.5) * VIIRS_RES

            r_shared = int((ALASKA_LAT_MAX - lat_px) / VIIRS_RES)
            c_shared = int((lon_px - ALASKA_LON_MIN) / VIIRS_RES)
            if not (0 <= r_shared < VIIRS_H and 0 <= c_shared < VIIRS_W):
                n_skipped += 1
                continue

            ls_col, ls_row = ~ls_tf * (lon_px, lat_px)
            ls_col, ls_row = int(ls_col), int(ls_row)

            ls_vals = {}
            ls_inside = False
            if 0 <= ls_row < ls_h and 0 <= ls_col < ls_w:
                for bi, bname in enumerate(ls_band_names, 1):
                    val = float(ls_src.read(bi, window=rasterio.windows.Window(
                        ls_col, ls_row, 1, 1))[0, 0])
                    ls_vals[bname] = val if val != NODATA else None
                ls_inside = any(v is not None for v in ls_vals.values())
            else:
                for bname in ls_band_names:
                    ls_vals[bname] = None

            if ls_inside:
                n_ls_valid += 1

            def _safe_viirs(name):
                v = viirs_grids[name][r, c]
                return round(float(v), 4) if np.isfinite(v) else ""

            def _safe_ls(name):
                v = ls_vals.get(name)
                return round(float(v), 4) if v is not None else ""

            row = {
                "id":              idx + 1,
                "viirs_date":      date_fmt,
                "landsat_scene":   ls_scene_id,
                "row_shared_grid": r_shared,
                "col_shared_grid": c_shared,
                "lat":                  round(lat_px, 5),
                "lon":                  round(lon_px, 5),
                "water_fraction_occ":   round(float(wm_occ[r, c]), 4),
                "water_fraction_sea":   round(float(wm_sea[r, c]), 4),
            }
            for name in VIIRS_BAND_ORDER:
                row[name] = _safe_viirs(name)
            for bname in ls_band_names:
                row[f"LS_{bname}"] = _safe_ls(bname)

            # NDSI needs reflectance: DN has a -0.1 offset that biases it low
            b3 = dn_to_toa_reflectance(ls_vals.get("B3"), 3, mtl_vals)
            b6 = dn_to_toa_reflectance(ls_vals.get("B6"), 6, mtl_vals)
            if b3 is not None and b6 is not None and (b3 + b6) != 0:
                ndsi = (b3 - b6) / (b3 + b6)
                row["LS_NDSI"] = round(ndsi, 4)
            else:
                ndsi = None
                row["LS_NDSI"] = ""

            st_b10 = None
            if b10_path and mtl_vals and ls_inside:
                try:
                    with rasterio.open(b10_path) as b10_src:
                        b10_x, b10_y = Transformer.from_crs(
                            "EPSG:4326", b10_src.crs, always_xy=True
                        ).transform(lon_px, lat_px)
                        b10_col, b10_row = ~b10_src.transform * (b10_x, b10_y)
                        b10_col, b10_row = int(b10_col), int(b10_row)
                        if (0 <= b10_row < b10_src.height and
                                0 <= b10_col < b10_src.width):
                            dn = float(b10_src.read(
                                1, window=rasterio.windows.Window(
                                    b10_col, b10_row, 1, 1))[0, 0])
                            if dn > 0:
                                st_b10 = dn_to_kelvin(dn, mtl_vals)
                except Exception:
                    pass
            row["LS_ST_B10"] = round(st_b10, 2) if st_b10 is not None else ""

            if ndsi is not None and st_b10 is not None:
                is_ice  = st_b10 < 273.0
                is_snow = ndsi > 0.4
                if   not is_ice and not is_snow:
                    auto_class = "ice_free_river_snow_free_land"
                elif not is_ice and is_snow:
                    auto_class = "ice_free_river_snow_land"
                elif is_ice and not is_snow:
                    auto_class = "ice_covered_river_snow_free_land"
                else:
                    auto_class = "ice_covered_river_snow_covered_land"
                row["ground_truth_class"] = auto_class
                row["manual_verified_class"] = ""
                row["notes"] = (f"Landsat confirmed at 30m. "
                                f"ST_B10={round(st_b10, 2)}K "
                                f"NDSI={round(ndsi, 4)}.")
            elif ndsi is not None:
                is_snow = ndsi > 0.4
                row["ground_truth_class"] = ("snow_covered_land" if is_snow
                                             else "snow_free_land")
                row["manual_verified_class"] = ""
                row["notes"] = (f"NDSI={round(ndsi, 4)}. "
                                f"No thermal — ice status unknown.")
            else:
                row["ground_truth_class"] = ""
                row["manual_verified_class"] = ""
                row["notes"] = ("Landsat null — outside swath. Classify manually."
                                if not ls_inside else
                                "NDSI unavailable — classify manually.")

            rows_out.append(row)

    log.info(f"  {n_ls_valid}/{len(candidates)} candidates inside Landsat swath")
    log.info(f"  {n_skipped} skipped (outside Alaska domain)")

    if not rows_out:
        log.warning(f"  No valid candidates — no CSV for {date_fmt}")
        return

    # --- MODIS enrichment (NDVI + IGBP land cover) ---
    enrich_rows(ee, rows_out, date_fmt)

    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    log.info(f"  Saved CSV: {csv_path} ({len(rows_out)} rows)")

    for row in rows_out:
        log.info(f"    lat={row['lat']}, lon={row['lon']}  "
                 f"row={row['row_shared_grid']}, col={row['col_shared_grid']}")

    # --- PNG ---
    i2 = viirs_grids["I2"]
    i1 = viirs_grids["I1"]

    # Load Landsat bands (downsampled) for visualization
    log.info(f"  Loading Landsat bands for visualization ...")
    LS_VIZ_W = min(2048, ls_w)
    LS_VIZ_H = max(1, int(round(LS_VIZ_W * ls_h / ls_w)))
    ls_viz_bands = {}
    with rasterio.open(landsat_path) as ls_src:
        for bname in ("B4", "B5"):
            if bname in ls_band_names:
                bi = ls_band_names.index(bname) + 1
                arr = ls_src.read(bi, out_shape=(LS_VIZ_H, LS_VIZ_W),
                                  resampling=Resampling.bilinear).astype(np.float32)
                arr[arr == NODATA] = np.nan
                ls_viz_bands[bname] = arr

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f"VIIRS 2-2-1 false color | JRC water fraction | Landsat 5-5-4 false color\n"
        f"VIIRS {date_fmt}  |  "
        f"lon {scene_lon_min:.1f}\u2013{scene_lon_max:.1f}  "
        f"lat {scene_lat_min:.1f}\u2013{scene_lat_max:.1f}",
        fontsize=10)

    rgb      = np.stack([_normalize(i2), _normalize(i2), _normalize(i1)], axis=-1)
    nan_mask = ~(np.isfinite(i1) & np.isfinite(i2))
    alpha    = np.where(nan_mask, 0.0, 1.0)
    rgba_viirs = np.dstack([rgb, alpha])

    axes[0].imshow(rgba_viirs, interpolation="nearest",
                   extent=[scene_lon_min, scene_lon_max,
                           scene_lat_min, scene_lat_max],
                   aspect="auto", origin="upper")
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        axes[0].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = axes[0].text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                         str(i + 1), color="white", fontsize=8, fontweight="bold",
                         ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    axes[0].set_title("VIIRS 2-2-1 (I2\u2192R, I2\u2192G, I1\u2192B)\n"
                      "Ice/snow=bright | Water=dark | NaN=transparent")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    im2 = axes[1].imshow(wm, cmap="Blues", vmin=0, vmax=1,
                          interpolation="nearest",
                          extent=[scene_lon_min, scene_lon_max,
                                  scene_lat_min, scene_lat_max],
                          aspect="auto", origin="upper")
    plt.colorbar(im2, ax=axes[1], fraction=0.03, label="Water fraction")
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        axes[1].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = axes[1].text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                         str(i + 1), color="white", fontsize=8, fontweight="bold",
                         ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    axes[1].set_title("JRC water fraction (occurrence)\n"
                      "\u00d7 = candidate training pixels")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

    b5 = ls_viz_bands.get("B5")
    b4 = ls_viz_bands.get("B4")
    if b5 is not None and b4 is not None:
        ls_rgb = np.stack([_normalize(b5), _normalize(b5), _normalize(b4)], axis=-1)
        ls_nan = ~(np.isfinite(b5) & np.isfinite(b4))
        ls_alpha = np.where(ls_nan, 0.0, 1.0)
        ls_rgba = np.dstack([ls_rgb, ls_alpha])
        axes[2].imshow(ls_rgba, interpolation="nearest",
                       extent=[ls_bounds.left, ls_bounds.right,
                               ls_bounds.bottom, ls_bounds.top],
                       aspect="auto", origin="upper")
        ls_title = ("Landsat 5-5-4 (B5→R, B5→G, B4→B)\n"
                    "Ice/snow=bright | Water=dark | × = candidate pixels")
    else:
        axes[2].set_facecolor("black")
        axes[2].set_xlim(ls_bounds.left, ls_bounds.right)
        axes[2].set_ylim(ls_bounds.bottom, ls_bounds.top)
        ls_title = "Landsat (B4/B5 unavailable)\n× = candidate pixels"
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        axes[2].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = axes[2].text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                         str(i + 1), color="white", fontsize=8, fontweight="bold",
                         ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    axes[2].set_title(ls_title)
    axes[2].set_xlabel("Longitude"); axes[2].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved PNG: {png_path}")

    # --- Individual panels (separate PNGs) ---
    panels_dir = os.path.join(pixel_date_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    fig_v, ax_v = plt.subplots(figsize=(8, 6))
    ax_v.imshow(rgba_viirs, interpolation="nearest",
                extent=[scene_lon_min, scene_lon_max,
                        scene_lat_min, scene_lat_max],
                aspect="auto", origin="upper")
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        ax_v.plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = ax_v.text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                      str(i + 1), color="white", fontsize=8, fontweight="bold",
                      ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    ax_v.set_title(f"VIIRS 2-2-1 — {date_fmt}\n× = candidate pixels")
    ax_v.set_xlabel("Longitude"); ax_v.set_ylabel("Latitude")
    plt.tight_layout()
    viirs_panel = os.path.join(panels_dir, f"viirs_{date_fmt}{tag}.png")
    plt.savefig(viirs_panel, dpi=150, bbox_inches="tight")
    plt.close()

    fig_w, ax_w = plt.subplots(figsize=(8, 6))
    im_w = ax_w.imshow(wm, cmap="Blues", vmin=0, vmax=1,
                        interpolation="nearest",
                        extent=[scene_lon_min, scene_lon_max,
                                scene_lat_min, scene_lat_max],
                        aspect="auto", origin="upper")
    plt.colorbar(im_w, ax=ax_w, fraction=0.03, label="Water fraction")
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        ax_w.plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = ax_w.text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                      str(i + 1), color="white", fontsize=8, fontweight="bold",
                      ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    ax_w.set_title(f"JRC water fraction — {date_fmt}\n× = candidate pixels")
    ax_w.set_xlabel("Longitude"); ax_w.set_ylabel("Latitude")
    plt.tight_layout()
    wm_panel = os.path.join(panels_dir, f"watermask_{date_fmt}{tag}.png")
    plt.savefig(wm_panel, dpi=150, bbox_inches="tight")
    plt.close()

    fig_l, ax_l = plt.subplots(figsize=(8, 6))
    if b5 is not None and b4 is not None:
        ax_l.imshow(ls_rgba, interpolation="nearest",
                    extent=[ls_bounds.left, ls_bounds.right,
                            ls_bounds.bottom, ls_bounds.top],
                    aspect="auto", origin="upper")
        l_title = f"Landsat 5-5-4 — {date_fmt}\n× = candidate pixels"
    else:
        ax_l.set_facecolor("black")
        ax_l.set_xlim(ls_bounds.left, ls_bounds.right)
        ax_l.set_ylim(ls_bounds.bottom, ls_bounds.top)
        l_title = f"Landsat (B4/B5 unavailable) — {date_fmt}\n× = candidate pixels"
    for i, (r, c) in enumerate(candidates):
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        ax_l.plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        t = ax_l.text(lon_pt + VIIRS_RES * 0.5, lat_pt + VIIRS_RES * 0.5,
                      str(i + 1), color="white", fontsize=8, fontweight="bold",
                      ha="left", va="bottom")
        t.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="black"),
                            path_effects.Normal()])
    ax_l.set_title(l_title)
    ax_l.set_xlabel("Longitude"); ax_l.set_ylabel("Latitude")
    plt.tight_layout()
    ls_panel = os.path.join(panels_dir, f"landsat_{date_fmt}{tag}.png")
    plt.savefig(ls_panel, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved 3 panel PNGs to: {panels_dir}")
    return csv_path


# ---------------------------------------------------------------------------
# GEE report + live map server  (same as output/report_main.py and
# output/report_map_server.py, embedded so this script runs on its own)
# ---------------------------------------------------------------------------

try:
    import ee
except ImportError:
    ee = None

# ============================================================
# SETTINGS
# ============================================================
BUFFER_M = 90            # radius (m) for cloud % around each point
CHIP_M = 1500            # half-width (m) of each image chip
CHIP_PX = 256            # chip size in pixels
THERMAL_K = 273.0        # river frozen if thermal < this (K)
NDSI_SNOW = 0.4          # land snow-covered if NDSI > this

# Which pixels count as CLOUD (-> CLASS 0):
#   "qa_cloud"       QA cloud bit only                         (default)
#   "qa_any"         cloud OR dilated cloud OR cirrus OR shadow (strictest)
#   "qa_conf_medium" QA cloud confidence medium or high
#   "simple"         B10 < 260 K and B2 > 0.2 (the GEE-script rule;
#                    flags lots of clear winter ground as cloud)
CLOUD_METHOD = "qa_cloud"

# Thermal band for the frozen test:
#   "ST_B10" Level-2 surface temp (falls back to TOA B10 where missing)
#   "B10"    TOA brightness temperature
THERMAL_SOURCE = "ST_B10"

# ------------------------------------------------------------
# YOUR OWN INDICES — any formula of TOA bands B1..B11.
# They become bands you can sample and use in VIEWS.
# ------------------------------------------------------------
INDICES = {
    "NDSI":  "(B3 - B6) / (B3 + B6)",   # snow
    "NDWI":  "(B3 - B5) / (B3 + B5)",   # water
    "NDVI":  "(B5 - B4) / (B5 + B4)",   # vegetation
    "MNDWI": "(B3 - B6) / (B3 + B6)",   # same as NDSI on Landsat 8/9
    "NDSII": "(B4 - B6) / (B4 + B6)",   # snow/ice alternative (red-SWIR)
}

# ------------------------------------------------------------
# LIVE MAP — the report embeds a real pannable/zoomable Leaflet map fed by
# live Earth Engine tiles, with a band-combo picker. It needs the map server (--view)
# running locally (it talks to Earth Engine on demand). Bands offered in
# the "Custom RGB" picker, and the default
# stretch (min, max) used if you don't override it there.
# ------------------------------------------------------------
BAND_CHOICES = (
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "ST_B10"]
    + list(INDICES.keys())
    + ["cloud", "class"]
    + [f"qa_{n}" for n in ["cloud", "dilated", "cirrus", "shadow", "snow", "clear", "water"]]
    + [f"qa_{n}_conf" for n in ["cloud", "shadow", "snow", "cirrus"]]
)
DEFAULT_VIS = {
    **{b: (0, 0.4) for b in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]},
    "B10": (220, 320), "B11": (220, 320), "ST_B10": (220, 320),
    "NDSI": (-0.5, 0.8), "NDWI": (-0.5, 0.5), "NDVI": (-0.2, 0.8),
    "MNDWI": (-0.5, 0.8), "NDSII": (-0.5, 0.8),
    "cloud": (0, 1), "class": (0, 4),
    **{f"qa_{n}": (0, 1) for n in ["cloud", "dilated", "cirrus", "shadow", "snow", "clear", "water"]},
    **{f"qa_{n}_conf": (0, 3) for n in ["cloud", "shadow", "snow", "cirrus"]},
}

# ------------------------------------------------------------
# VIEWS — one image chip per point per view. "on": False to skip.
#
# Bands you can use:
#   TOA reflectance  B1 B2 B3 B4 B5 B6 B7 B8 B9
#   thermal (K)      B10 B11 (TOA)   ST_B10 (Level-2 surface temp)
#   indices          any name in INDICES above
#   QA flags (0/1)   qa_cloud qa_dilated qa_cirrus qa_shadow qa_snow
#                    qa_water qa_clear
#   QA confidence    qa_cloud_conf qa_shadow_conf qa_snow_conf
#                    qa_cirrus_conf   (0 none, 1 low, 2 medium, 3 high)
#   classification   cloud (mask used for CLASS 0), class (0-4)
#
# 3 bands = RGB composite; 1 band = needs a palette.
# "max": "auto" = 1.5 for low winter sun (<15°), 0.4 otherwise.
# "qa": True paints the QA cloud layers on top of the image.
# ------------------------------------------------------------
VIEWS = {
    "true_color":  {"on": True,  "bands": ["B4", "B3", "B2"], "min": 0.05, "max": "auto"},
    "false_nir":   {"on": True,  "bands": ["B5", "B4", "B3"], "min": 0.05, "max": "auto"},
    "swir":        {"on": True,  "bands": ["B7", "B5", "B3"], "min": 0.05, "max": "auto"},
    "swir_654":    {"on": False, "bands": ["B6", "B5", "B4"], "min": 0.05, "max": "auto"},
    "rgb_554":     {"on": False, "bands": ["B5", "B5", "B4"], "min": 0.05, "max": "auto"},
    "snow_221":    {"on": True,  "bands": ["B2", "B2", "B1"], "min": 0.03, "max": "auto",
                    "gamma": [1.4, 1.4, 1.2]},
    "ndsi":        {"on": True,  "bands": ["NDSI"], "min": -0.2, "max": 0.8,
                    "palette": ["000000", "ffffff", "00ffff"]},
    "ndwi":        {"on": True,  "bands": ["NDWI"], "min": -0.5, "max": 0.5,
                    "palette": ["8b4513", "ffffff", "0000ff"]},
    "ndvi":        {"on": False, "bands": ["NDVI"], "min": -0.2, "max": 0.8,
                    "palette": ["8b4513", "ffffff", "006400"]},
    "thermal_st":  {"on": True,  "bands": ["ST_B10"], "min": 240, "max": 280,
                    "palette": ["0000ff", "ffffff", "ff0000"]},
    "thermal_toa": {"on": False, "bands": ["B10"], "min": 220, "max": 280,
                    "palette": ["0000ff", "ffffff", "ff0000"]},
    "cloud_qa":    {"on": True,  "bands": ["B4", "B3", "B2"], "min": 0.05, "max": "auto",
                    "qa": True},
    "cloud_conf":  {"on": False, "bands": ["qa_cloud_conf"], "min": 0, "max": 3,
                    "palette": ["000000", "ffff00", "ff8800", "ff0000"]},
    "class_map":   {"on": True,  "bands": ["class"], "min": 0, "max": 4,
                    "palette": ["808080", "2e7d32", "ffffff", "7b1fa2", "00bcd4"]},
}

# QA overlay colours (drawn bottom -> top), used by views with "qa": True
QA_OVERLAY = [
    ("qa_snow",    "00e5ff", "snow/ice"),
    ("qa_shadow",  "1f3a93", "cloud shadow"),
    ("qa_cirrus",  "ffeb3b", "cirrus"),
    ("qa_dilated", "ff9800", "dilated cloud (edge)"),
    ("qa_cloud",   "ff00ff", "cloud"),
]
QA_OPACITY = 0.6

CLASS_NAMES = {
    0: "cloud",
    1: "ice_free_river_snow_free_land",
    2: "ice_covered_river_snow_covered_land",
    3: "ice_covered_river_snow_free_land",
    4: "ice_free_river_snow_land",
}
CLASS_FROM_STATES = {
    ("ice_free", "snow_free"): 1,
    ("ice_covered", "snow_covered"): 2,
    ("ice_covered", "snow_free"): 3,
    ("ice_free", "snow_covered"): 4,
}
CLASS_COLORS = {0: "808080", 1: "2e7d32", 2: "ffffff", 3: "7b1fa2", 4: "00bcd4"}

QA_BITS = {  # bit -> band name (Landsat Collection 2 QA_PIXEL)
    1: "qa_dilated", 2: "qa_cirrus", 3: "qa_cloud", 4: "qa_shadow",
    5: "qa_snow", 6: "qa_clear", 7: "qa_water",
}
QA_CONF = {  # first bit of each 2-bit confidence field
    8: "qa_cloud_conf", 10: "qa_shadow_conf", 12: "qa_snow_conf", 14: "qa_cirrus_conf",
}
CONF_WORDS = ["none", "low", "medium", "high"]
PCT_FLAGS = ["qa_cloud", "qa_dilated", "qa_cirrus", "qa_shadow", "qa_snow", "cloud"]


# ============================================================
# Plain-Python helpers (no Earth Engine needed)
# ============================================================
def read_candidates(path):
    """Read the CSV; handles Excel's Windows encoding (e.g. the — dash)."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    missing = {"lat", "lon", "landsat_scene"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}")
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    df["pid"] = df["id"].astype(str)
    return df


def scene_ids(scene):
    """LC09_L1TP_073018_20241127_20241128_02_T1 -> (TOA id, L2 id, date)."""
    p = str(scene).strip().split("_")
    if len(p) < 7:
        raise ValueError(f"Not a Landsat Collection 2 file name: {scene}")
    sensor, pathrow, date, tier = p[0], p[2], p[3], p[6]
    base = f"{sensor}_{pathrow}_{date}"
    return (
        f"LANDSAT/{sensor}/C02/{tier}_TOA/{base}",
        f"LANDSAT/{sensor}/C02/{tier}_L2/{base}",
        f"{date[:4]}-{date[4:6]}-{date[6:8]}",
    )


def decode_qa(q):
    """Decode a QA_PIXEL integer into readable flags + confidences."""
    if q is None or (isinstance(q, float) and pd.isna(q)):
        return {"qa_summary": "no QA"}
    q = int(q)
    flags = {name: (q >> bit) & 1 for bit, name in QA_BITS.items()}
    confs = {name: CONF_WORDS[(q >> bit) & 3] for bit, name in QA_CONF.items()}
    on = [n.replace("qa_", "") for n, v in flags.items() if v]
    summary = (", ".join(on) if on else "no flags") + (
        f" | conf: cloud {confs['qa_cloud_conf']}, shadow {confs['qa_shadow_conf']}, "
        f"snow {confs['qa_snow_conf']}, cirrus {confs['qa_cirrus_conf']}")
    if q & 1:
        summary = "FILL (no data) | " + summary
    return {"QA_PIXEL": q, **{f"{k}_word": v for k, v in confs.items()},
            "qa_summary": summary}


def parse_gt(label):
    """Split a ground-truth label into (river state, land state)."""
    s = str(label) if isinstance(label, str) else ""
    river = ("ice_covered" if "ice_covered_river" in s
             else "ice_free" if "ice_free_river" in s else None)
    land = ("snow_covered" if ("snow_covered_land" in s or "snow_land" in s)
            else "snow_free" if "snow_free_land" in s else None)
    return river, land


def classify(v):
    """Classify one point from its sampled values (dict)."""
    st, b10 = v.get("ST_B10"), v.get("B10")
    if THERMAL_SOURCE == "ST_B10" and st is not None:
        thermal, source = st, "ST_B10"
    elif b10 is not None:
        thermal, source = b10, "B10"
    else:
        thermal, source = None, None

    ndsi = v.get("NDSI")
    cloud = v.get("cloud")

    river = None if thermal is None else ("ice_covered" if thermal < THERMAL_K else "ice_free")
    land = None if ndsi is None else ("snow_covered" if ndsi > NDSI_SNOW else "snow_free")

    if cloud == 1:
        cls = 0
    elif river and land:
        cls = CLASS_FROM_STATES[(river, land)]
    else:
        cls = None

    if cls is not None:
        label = CLASS_NAMES[cls]
    elif land:
        label = f"{land}_land (river unknown — no thermal)"
    elif river:
        label = f"{river}_river (land unknown — no NDSI)"
    else:
        label = "NO DATA"
    return {"gee_class": cls, "gee_label": label, "gee_river": river,
            "gee_land": land, "thermal_used_K": thermal, "thermal_source": source}


def compare(gt_label, c):
    """MATCH / MISMATCH / CLOUD / NO DATA, checked separately for river and land."""
    gt_r, gt_l = parse_gt(gt_label)
    river_match = (gt_r == c["gee_river"]) if (gt_r and c["gee_river"]) else None
    land_match = (gt_l == c["gee_land"]) if (gt_l and c["gee_land"]) else None
    if c["gee_class"] == 0:
        status = "CLOUD"
    elif river_match is None and land_match is None:
        status = "NO DATA"
    elif False in (river_match, land_match):
        status = "MISMATCH"
    else:
        status = "MATCH"
    return {"river_match": river_match, "land_match": land_match, "status": status}


def fmt(v, d):
    try:
        if v is None or pd.isna(v):
            return "NA"
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "NA"


# ============================================================
# Earth Engine part
# ============================================================
def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def build_full_image(toa_id, l2_id):
    """TOA bands + indices + ST_B10 + decoded QA + cloud + class, as one image."""
    toa = ee.Image(toa_id)
    refl = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
    band_map = {b: toa.select(b) for b in refl}

    parts = [toa.select(refl + ["B10", "B11", "QA_PIXEL"])]

    # indices
    for name, expr in INDICES.items():
        parts.append(toa.expression(expr, band_map).rename(name))

    # Level-2 surface temperature
    if l2_id:
        st = (ee.Image(l2_id).select("ST_B10")
              .multiply(0.00341802).add(149.0).rename("ST_B10"))
        parts.append(st)
    else:
        st = None

    # QA_PIXEL decoded
    qa = toa.select("QA_PIXEL")
    for bit, name in QA_BITS.items():
        parts.append(qa.rightShift(bit).bitwiseAnd(1).rename(name))
    for bit, name in QA_CONF.items():
        parts.append(qa.rightShift(bit).bitwiseAnd(3).rename(name))
    q = lambda bit: qa.rightShift(bit).bitwiseAnd(1)

    # cloud mask used for CLASS 0
    if CLOUD_METHOD == "qa_any":
        cloud = q(3).Or(q(1)).Or(q(2)).Or(q(4))
    elif CLOUD_METHOD == "qa_conf_medium":
        cloud = qa.rightShift(8).bitwiseAnd(3).gte(2)
    elif CLOUD_METHOD == "simple":
        cloud = toa.select("B10").lt(260).And(toa.select("B2").gt(0.2))
    else:  # qa_cloud
        cloud = q(3)
    cloud = cloud.rename("cloud")
    parts.append(cloud)

    full = ee.Image.cat(parts)

    # % of each flag within BUFFER_M of every pixel
    pct = (full.select(PCT_FLAGS)
           .focalMean(BUFFER_M, "circle", "meters").multiply(100)
           .rename([f + "_pct" for f in PCT_FLAGS]))

    # classification
    thermal = toa.select("B10")
    if st is not None and THERMAL_SOURCE == "ST_B10":
        thermal = st.unmask(toa.select("B10"))
    ndsi = full.select("NDSI")
    frozen = thermal.lt(THERMAL_K)
    snow = ndsi.gt(NDSI_SNOW)
    class_img = (thermal.multiply(0).add(1).toInt()
                 .where(frozen.And(snow), 2)
                 .where(frozen.And(snow.Not()), 3)
                 .where(frozen.Not().And(snow), 4)
                 .where(cloud, 0)
                 .updateMask(ndsi.mask())
                 .rename("class"))

    return toa, full.addBands(pct).addBands(class_img)


def make_view(full, spec, rgb_max):
    """Turn a VIEWS entry into a visualized (RGB) ee.Image."""
    vis = {"bands": spec["bands"], "min": spec.get("min", 0)}
    vis["max"] = rgb_max if spec.get("max") == "auto" else spec.get("max", 1)
    if "palette" in spec:
        vis["palette"] = spec["palette"]
    if "gamma" in spec:
        vis["gamma"] = spec["gamma"]
    img = full.visualize(**vis)
    if spec.get("qa"):
        for band, color, _ in QA_OVERLAY:
            img = img.blend(full.select(band).selfMask()
                            .visualize(palette=[color], opacity=QA_OPACITY))
    return img


def qa_overlay_image(full):
    """QA cloud/shadow/snow flags only, transparent everywhere else — a standalone
    overlay you can drop on top of any live-map view, independent of its band combo."""
    layers, any_flag = None, None
    for band, color, _ in QA_OVERLAY:
        flag = full.select(band)
        vis = flag.selfMask().visualize(palette=[color])
        layers = vis if layers is None else layers.blend(vis)
        any_flag = flag if any_flag is None else any_flag.Or(flag)
    return layers.updateMask(any_flag)


def sample_points(full, rows):
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"pid": r.pid})
        for r in rows.itertuples()
    ])
    out = full.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                             scale=30).getInfo()
    return {f["properties"]["pid"]: f["properties"] for f in out["features"]}


def chip_data_uri(vis_img, lat, lon):
    """PNG chip around the point with a red ring, as an embeddable data URI."""
    try:
        pt = ee.Geometry.Point([float(lon), float(lat)])
        ring = (ee.Image().byte()
                .paint(ee.FeatureCollection([ee.Feature(pt.buffer(BUFFER_M))]), 1, 2)
                .visualize(palette=["ff0000"]))
        url = vis_img.blend(ring).getThumbURL({
            "region": pt.buffer(CHIP_M).bounds(),
            "dimensions": CHIP_PX,
            "format": "png",
        })
        with urllib.request.urlopen(url, timeout=180) as r:
            return "data:image/png;base64," + base64.b64encode(r.read()).decode()
    except Exception as e:
        log.warning(f"   chip failed at {lat}, {lon}: {e}")
        return None


# ============================================================
# Report
# ============================================================
def write_report(res, chips, view_names, scenes_meta, path, title):
    counts = res["status"].value_counts().to_dict()
    badge = {"MATCH": "#2e7d32", "MISMATCH": "#c62828",
             "CLOUD": "#616161", "NO DATA": "#ef6c00"}

    # live-map data: one entry per scene, with every point as a marker
    map_scenes = []
    for meta in scenes_meta.values():
        toa_id = meta.get("toa_id")
        if not toa_id:
            continue
        sub = res[res["gee_scene"] == toa_id]
        if sub.empty:
            continue
        sun = meta.get("SUN_ELEVATION")
        pts = [{"id": str(row["id"]), "lat": float(row["lat"]), "lon": float(row["lon"]),
                "status": row["status"], "label": row["gee_label"],
                "gt": str(row.get("ground_truth_class", "") or ""),
                "notes": "" if pd.isna(row.get("notes")) else str(row.get("notes", ""))}
               for _, row in sub.iterrows()]
        map_scenes.append({"toa": toa_id, "l2": meta.get("l2_id", ""),
                            "rgb_max": 1.5 if (sun is not None and sun < 15) else 0.4,
                            "points": pts})
    preset_names = list(VIEWS.keys())
    if "false_nir" in preset_names:
        preset_names.remove("false_nir")
        preset_names.insert(0, "false_nir")
    map_data = {"port": MAP_SERVER_PORT, "bands": BAND_CHOICES, "bufferM": BUFFER_M,
                "presets": preset_names,
                "presetBands": {k: v["bands"] for k, v in VIEWS.items()},
                "statusColor": {"MATCH": "#2e7d32", "MISMATCH": "#c62828",
                                 "CLOUD": "#616161", "NO DATA": "#ef6c00"},
                "scenes": map_scenes}

    rows_html = []
    for _, r in res.iterrows():
        c = chips.get(r["pid"], {})
        chip_html = "".join(
            f'<figure class="view v-{v}"><img src="{c[v]}" alt="{v}"><figcaption>{v}</figcaption></figure>'
            if c.get(v) else
            f'<figure class="view v-{v}"><div class="nochip">no chip</div><figcaption>{v}</figcaption></figure>'
            for v in view_names)
        pcts = " · ".join(f"{f.replace('qa_', '')} {fmt(r.get(f + '_pct'), 0)}%"
                          for f in PCT_FLAGS if f != "cloud")
        notes = r.get("notes", "")
        notes = "" if (notes is None or (isinstance(notes, float) and pd.isna(notes))) else notes
        gee_scene = str(r.get("gee_scene", ""))
        lc_name = r.get("modis_lc_name")
        lc_name = "—" if (lc_name is None or (isinstance(lc_name, float) and pd.isna(lc_name))) else lc_name
        modis_html = (
            f"<br>MODIS NDVI {fmt(r.get('modis_ndvi'), 3)} · LC {html.escape(str(lc_name))}"
            if "modis_ndvi" in r.index else "")
        gee_snippet = (
            f"var img = ee.Image('{gee_scene}');\n"
            f"Map.centerObject(ee.Geometry.Point([{r['lon']}, {r['lat']}]), 13);\n"
            "Map.addLayer(img, {bands: ['B4','B3','B2'], min: 0.05, max: 0.4}, 'true_color');"
        )
        rows_html.append(f"""
<tr class="st-{r['status'].replace(' ', '_')}">
  <td><b>{html.escape(str(r['id']))}</b><br><span class="small">[{r['lat']}, {r['lon']}]</span><br>
      <button class="gee-copy small" data-snippet="{html.escape(gee_snippet)}">copy GEE snippet &#128203;</button>
      <a class="small" href="https://code.earthengine.google.com/" target="_blank" rel="noopener">open editor &#8599;</a></td>
  <td>{html.escape(str(r.get('ground_truth_class', '')))}</td>
  <td>{html.escape(str(r['gee_label']))}<br>
      <span class="badge" style="background:{badge.get(r['status'], '#555')}">{r['status']}</span></td>
  <td class="small">
    thermal {fmt(r['thermal_used_K'], 1)} K ({r['thermal_source'] or '—'})<br>
    ST_B10 {fmt(r.get('ST_B10'), 1)} K · B10 {fmt(r.get('B10'), 1)} K<br>
    NDSI {fmt(r.get('NDSI'), 3)} · NDWI {fmt(r.get('NDWI'), 3)} · NDVI {fmt(r.get('NDVI'), 3)}{modis_html}
  </td>
  <td class="small">
    <b>pixel QA:</b> {html.escape(str(r.get('qa_summary', '')))}<br>
    <b>within {BUFFER_M} m:</b> {pcts}<br>
    <b>cloud used ({CLOUD_METHOD}):</b> {fmt(r.get('cloud_pct'), 0)}%
  </td>
  <td class="small">{html.escape(str(notes))}</td>
  <td class="chips">{chip_html}</td>
</tr>""")

    view_boxes = "".join(
        f'<label><input type="checkbox" class="vt" value="{v}" checked> {v}</label> '
        for v in view_names)
    class_legend = "".join(
        f'<span class="sw" style="background:#{CLASS_COLORS[k]}"></span>{k} {v} &nbsp; '
        for k, v in CLASS_NAMES.items())
    qa_legend = "".join(
        f'<span class="sw" style="background:#{col}"></span>{lbl} &nbsp; '
        for _, col, lbl in QA_OVERLAY)
    scene_lines = "<br>".join(
        f"{html.escape(s)} — acquired {m.get('date')}, sun {fmt(m.get('SUN_ELEVATION'), 1)}°, "
        f"scene cloud {fmt(m.get('CLOUD_COVER'), 1)}% (land {fmt(m.get('CLOUD_COVER_LAND'), 1)}%)"
        for s, m in scenes_meta.items())
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    map_data_json = json.dumps(map_data)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; color: #222; }}
 h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
 .meta {{ color: #444; margin-bottom: 10px; line-height: 1.6; }}
 .controls {{ background: #fff; border: 1px solid #ddd; padding: 8px 10px; margin-bottom: 10px;
              position: sticky; top: 0; z-index: 2; }}
 .controls label {{ margin-right: 10px; white-space: nowrap; }}
 table {{ border-collapse: collapse; width: 100%; background: #fff; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }}
 th {{ background: #eee; }}
 .small {{ font-size: 0.85em; color: #333; }}
 .gee-copy {{ font-size: 0.85em; padding: 2px 6px; border: 1px solid #ccc; border-radius: 4px;
              background: #fff; cursor: pointer; display: block; margin-bottom: 3px; }}
 .gee-copy:hover {{ background: #f0f0f0; }}
 .badge {{ color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8em;
           display: inline-block; margin-top: 4px; }}
 .chips {{ min-width: 360px; }}
 figure.view {{ display: inline-block; margin: 0 6px 6px 0; }}
 figure.view img, .nochip {{ width: 160px; height: 160px; border: 1px solid #ccc; display: block; }}
 .nochip {{ display: flex; align-items: center; justify-content: center; color: #999; }}
 figcaption {{ font-size: 0.75em; color: #555; text-align: center; }}
 .sw {{ display: inline-block; width: 12px; height: 12px; border: 1px solid #999;
        margin-right: 4px; vertical-align: middle; }}
 .wrap {{ overflow-x: auto; }}
 figure.view img {{ cursor: zoom-in; }}
 .lightbox-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.88);
                       z-index: 50; align-items: center; justify-content: center; flex-direction: column; }}
 .lightbox-overlay.open {{ display: flex; }}
 .lightbox-overlay img {{ max-width: 90vw; max-height: 78vh; width: auto; height: auto;
                           border: 2px solid #fff; image-rendering: pixelated; }}
 .lightbox-caption {{ color: #fff; margin-top: 12px; font-size: 1em; text-align: center; }}
 .lightbox-nav {{ position: fixed; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,.15);
                   color: #fff; border: none; font-size: 2em; padding: 10px 18px; cursor: pointer; border-radius: 4px; }}
 .lightbox-nav:hover {{ background: rgba(255,255,255,.3); }}
 .lightbox-nav.prev {{ left: 16px; }}
 .lightbox-nav.next {{ right: 16px; }}
 .lightbox-close {{ position: fixed; top: 16px; right: 20px; color: #fff; font-size: 2.2em;
                     cursor: pointer; background: none; border: none; line-height: 1; }}
 .livemap-block {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px;
                    margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
 .livemap-status {{ font-size: 0.85em; margin-bottom: 10px; padding: 7px 10px; border-radius: 6px;
                     font-weight: 500; }}
 .livemap-status.ok {{ background: #e8f5e9; color: #2e7d32; }}
 .livemap-status.bad {{ background: #fff3e0; color: #ef6c00; }}
 .livemap-status code {{ background: #00000012; padding: 1px 6px; border-radius: 4px; font-size: 0.95em; }}
 .livemap-controls {{ margin-bottom: 10px; font-size: 0.9em; display: flex; flex-wrap: wrap;
                       align-items: center; gap: 10px; }}
 .livemap-controls select, .livemap-controls button {{
   margin: 0 2px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 6px;
   background: #fff; font: inherit; }}
 .livemap-controls button {{ background: #1565c0; color: #fff; border-color: #1565c0; cursor: pointer; }}
 .livemap-controls button:hover {{ background: #0d47a1; }}
 .livemap-controls label {{ white-space: nowrap; }}
 .lm-bandinfo {{ font-size: 0.85em; color: #555; background: #f0f4f8; padding: 4px 10px;
                  border-radius: 6px; border: 1px solid #dde5ee; }}
 .lm-bandinfo b {{ color: #1565c0; }}
 .livemap-help {{ font-size: 0.85em; color: #444; margin-bottom: 8px; }}
 .livemap-help summary {{ cursor: pointer; color: #1565c0; }}
 .livemap-help ul {{ margin: 6px 0 0 0; padding-left: 20px; }}
 .livemap-help li {{ margin-bottom: 3px; }}
 .livemap {{ height: 480px; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #ddd; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">{summary}<br>{scene_lines}<br>
Red ring = {BUFFER_M} m around the point · chip width {2 * CHIP_M / 1000:.1f} km<br>
Class colours: {class_legend}<br>
QA overlay: {qa_legend}<br>
Rules: frozen if {THERMAL_SOURCE} &lt; {THERMAL_K:g} K · snow if NDSI &gt; {NDSI_SNOW} · cloud = {CLOUD_METHOD}</div>
<div id="livemap-root"></div>
<div class="controls"><b>Views:</b> {view_boxes}<br>
<label><input type="checkbox" id="onlybad"> show only MISMATCH / CLOUD / NO DATA</label>
&nbsp; <span class="small">Click any chip to zoom · use &larr;/&rarr; or the arrows to page through every point</span></div>
<div class="wrap"><table>
<tr><th>ID</th><th>Your ground truth</th><th>GEE result</th><th>Values</th><th>Cloud (QA)</th><th>Notes</th><th>Chips</th></tr>
{''.join(rows_html)}
</table></div>
<div class="lightbox-overlay" id="lightbox">
  <button class="lightbox-close" id="lb-close" aria-label="close">&times;</button>
  <button class="lightbox-nav prev" id="lb-prev" aria-label="previous">&#10094;</button>
  <img id="lb-img" src="" alt="">
  <button class="lightbox-nav next" id="lb-next" aria-label="next">&#10095;</button>
  <div class="lightbox-caption" id="lb-caption"></div>
</div>
<script>
document.querySelectorAll('input.vt').forEach(function(cb) {{
  cb.addEventListener('change', function() {{
    document.querySelectorAll('.v-' + cb.value).forEach(function(el) {{
      el.style.display = cb.checked ? '' : 'none';
    }});
  }});
}});
document.getElementById('onlybad').addEventListener('change', function(e) {{
  document.querySelectorAll('tr.st-MATCH').forEach(function(tr) {{
    tr.style.display = e.target.checked ? 'none' : '';
  }});
}});
(function() {{
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lb-img');
  var lbCap = document.getElementById('lb-caption');
  var idx = 0, imgs = [];
  function visible(im) {{ return im.offsetParent !== null; }}
  function collect() {{
    imgs = Array.prototype.filter.call(document.querySelectorAll('.chips img'), visible);
  }}
  function show(i) {{
    if (!imgs.length) return;
    idx = (i + imgs.length) % imgs.length;
    var im = imgs[idx];
    var row = im.closest('tr');
    var id = row ? row.querySelector('td b').textContent : '';
    lbImg.src = im.src;
    lbImg.alt = im.alt;
    lbCap.textContent = 'ID ' + id + ' — ' + im.alt + '  (' + (idx + 1) + ' / ' + imgs.length + ')';
    lb.classList.add('open');
  }}
  document.querySelector('.wrap table').addEventListener('click', function(e) {{
    if (e.target.classList.contains('gee-copy')) {{
      var btn = e.target;
      navigator.clipboard.writeText(btn.dataset.snippet).then(function() {{
        var old = btn.textContent;
        btn.textContent = 'copied ✓';
        setTimeout(function() {{ btn.textContent = old; }}, 1500);
      }}).catch(function() {{ alert(btn.dataset.snippet); }});
      return;
    }}
    if (e.target.tagName !== 'IMG') return;
    collect();
    show(imgs.indexOf(e.target));
  }});
  document.getElementById('lb-close').addEventListener('click', function() {{ lb.classList.remove('open'); }});
  document.getElementById('lb-prev').addEventListener('click', function() {{ show(idx - 1); }});
  document.getElementById('lb-next').addEventListener('click', function() {{ show(idx + 1); }});
  lb.addEventListener('click', function(e) {{ if (e.target === lb) lb.classList.remove('open'); }});
  document.addEventListener('keydown', function(e) {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') lb.classList.remove('open');
    if (e.key === 'ArrowLeft') show(idx - 1);
    if (e.key === 'ArrowRight') show(idx + 1);
  }});
}})();
</script>
<script>
var MAP_DATA = {map_data_json};
(function() {{
  var root = document.getElementById('livemap-root');
  var base = 'http://127.0.0.1:' + MAP_DATA.port;

  function bandOptions(selected) {{
    return MAP_DATA.bands.map(function(b) {{
      return '<option value="' + b + '"' + (b === selected ? ' selected' : '') + '>' + b + '</option>';
    }}).join('');
  }}

  MAP_DATA.scenes.forEach(function(scene, i) {{
    var block = document.createElement('div');
    block.className = 'livemap-block';
    block.innerHTML =
      '<div class="livemap-status bad" id="lmstatus-' + i + '">checking for the map server&hellip;</div>' +
      '<div class="livemap-controls">' +
        '<b>Live map</b> &nbsp; View: ' +
        '<select class="lm-preset" data-i="' + i + '">' +
          MAP_DATA.presets.map(function(p) {{ return '<option value="' + p + '">' + p + '</option>'; }}).join('') +
          '<option value="__custom__">Custom RGB…</option>' +
        '</select>' +
        '<span class="lm-custom" data-i="' + i + '" style="display:none">' +
          ' R ' + '<select class="lm-r">' + bandOptions('B4') + '</select>' +
          ' G ' + '<select class="lm-g">' + bandOptions('B3') + '</select>' +
          ' B ' + '<select class="lm-b">' + bandOptions('B2') + '</select>' +
          ' <button class="lm-apply">Apply</button>' +
        '</span>' +
        '<span class="lm-bandinfo" id="lmbandinfo-' + i + '"></span>' +
        '<label><input type="checkbox" class="lm-dots" checked> point markers</label>' +
        '<label><input type="checkbox" class="lm-ring"> ' + MAP_DATA.bufferM + ' m buffer ring</label>' +
        '<label><input type="checkbox" class="lm-qa"> cloud/QA overlay</label>' +
      '</div>' +
      '<details class="livemap-help"><summary>what do these do?</summary><ul>' +
        '<li><b>View</b> — a ready-made band combo (true_color, false_nir, ndsi, cloud_qa…), rendered live from Earth Engine.</li>' +
        '<li><b>Custom RGB…</b> — pick any band for R/G/B yourself (raw bands, indices, QA flags/confidences, or the classification) and hit Apply.</li>' +
        '<li><b>' + MAP_DATA.bufferM + ' m buffer ring</b> — the same radius used to compute the cloud/QA percentages in the table; a hollow ring (unlike the solid dot) so you can see the pixels underneath.</li>' +
        '<li><b>point markers</b> — small solid dot per point, colored by MATCH/MISMATCH/CLOUD/NO DATA — hover for a quick id/ground-truth/GEE comparison, click for full detail.</li>' +
        '<li><b>cloud/QA overlay</b> — paints the QA cloud/shadow/cirrus/snow/dilated-cloud flags on top of whatever View you’re looking at (everything else stays transparent), independent of the band combo.</li>' +
        '<li><b>click anywhere on the map</b> (not on a marker) — reads the live pixel at that spot and shows its class, thermal, NDSI/NDWI/NDVI and QA summary.</li>' +
      '</ul></details>' +
      '<div class="livemap" id="lmmap-' + i + '"></div>';
    root.appendChild(block);

    var map = L.map('lmmap-' + i);
    // Esri imagery: works from file:// (OSM tiles return 403 without a Referer)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19, attribution: 'Tiles &copy; Esri'
    }}).addTo(map);
    var eeLayer = null;
    var lats = scene.points.map(function(p) {{ return p.lat; }});
    var lons = scene.points.map(function(p) {{ return p.lon; }});
    if (lats.length) {{
      map.fitBounds([[Math.min.apply(null, lats), Math.min.apply(null, lons)],
                      [Math.max.apply(null, lats), Math.max.apply(null, lons)]], {{padding: [30, 30]}});
    }} else {{
      map.setView([0, 0], 2);
    }}
    var dotGroup = L.layerGroup().addTo(map);
    var ringGroup = L.layerGroup();
    function tooltipHtml(p) {{
      var match = p.gt && p.gt === p.label ? ' ✅' : (p.gt ? ' ⚠️' : '');
      return '<b>id ' + p.id + '</b>' + match + '<br>' +
        '<b>ground truth:</b> ' + (p.gt || '—') + '<br>' +
        '<b>GEE:</b> ' + p.label + '<br>' +
        '<span style="color:#666">' + p.status + '</span>';
    }}
    function popupHtml(p) {{
      return tooltipHtml(p) + '<br>' + p.lat.toFixed(5) + ', ' + p.lon.toFixed(5) +
        (p.notes ? '<br><i>' + p.notes + '</i>' : '');
    }}
    scene.points.forEach(function(p) {{
      var color = MAP_DATA.statusColor[p.status] || '#555';
      dotGroup.addLayer(L.circleMarker([p.lat, p.lon], {{radius: 7, color: '#fff', weight: 1.5, fillColor: color, fillOpacity: 0.9}})
        .bindTooltip(tooltipHtml(p), {{sticky: true, direction: 'top', opacity: 0.95}})
        .bindPopup(popupHtml(p))
        .on('click', function(ev) {{ L.DomEvent.stopPropagation(ev); }}));
      ringGroup.addLayer(L.circle([p.lat, p.lon], {{radius: MAP_DATA.bufferM, color: color, weight: 2, fillOpacity: 0}})
        .bindTooltip(tooltipHtml(p), {{sticky: true, direction: 'top', opacity: 0.95}})
        .bindPopup(popupHtml(p))
        .on('click', function(ev) {{ L.DomEvent.stopPropagation(ev); }}));
    }});
    block.querySelector('.lm-dots').addEventListener('change', function(e) {{
      if (e.target.checked) map.addLayer(dotGroup); else map.removeLayer(dotGroup);
    }});

    function fmtNum(x, d) {{
      return (x === undefined || x === null) ? 'NA' : (typeof x === 'number' ? x.toFixed(d) : x);
    }}
    map.on('click', function(e) {{
      var popup = L.popup().setLatLng(e.latlng).setContent('reading pixel…').openOn(map);
      var qs = new URLSearchParams({{toa: scene.toa, l2: scene.l2, lat: e.latlng.lat, lon: e.latlng.lng}});
      fetch(base + '/api/inspect?' + qs.toString()).then(function(r) {{ return r.json(); }})
        .then(function(j) {{
          if (j.error) {{ popup.setContent('error: ' + j.error); return; }}
          var v = j.values, c = j.classify;
          popup.setContent(
            '<b>pixel @ ' + j.lat.toFixed(5) + ', ' + j.lon.toFixed(5) + '</b><br>' +
            '<b>class:</b> ' + c.gee_label + '<br>' +
            'thermal ' + fmtNum(c.thermal_used_K, 1) + ' K (' + (c.thermal_source || '—') + ')<br>' +
            'NDSI ' + fmtNum(v.NDSI, 3) + ' · NDWI ' + fmtNum(v.NDWI, 3) + ' · NDVI ' + fmtNum(v.NDVI, 3) + '<br>' +
            '<span class="small">' + (j.qa.qa_summary || '') + '</span>'
          );
        }})
        .catch(function() {{ popup.setContent('map server unreachable'); }});
    }});
    block.querySelector('.lm-ring').addEventListener('change', function(e) {{
      if (e.target.checked) map.addLayer(ringGroup); else map.removeLayer(ringGroup);
    }});

    var qaLayer = null, qaUrlPromise = null;
    block.querySelector('.lm-qa').addEventListener('change', function(e) {{
      if (!e.target.checked) {{
        if (qaLayer) map.removeLayer(qaLayer);
        return;
      }}
      if (!qaUrlPromise) {{
        var qs = new URLSearchParams({{toa: scene.toa, l2: scene.l2}});
        qaUrlPromise = fetch(base + '/api/qa_overlay?' + qs.toString()).then(function(r) {{ return r.json(); }});
      }}
      qaUrlPromise.then(function(j) {{
        if (j.error) {{ alert('map server error: ' + j.error); return; }}
        qaLayer = L.tileLayer(j.url, {{maxZoom: 20}}).addTo(map);
      }}).catch(function() {{ alert('map server unreachable'); }});
    }});

    var bandInfoEl = document.getElementById('lmbandinfo-' + i);
    function setBandInfo(bands) {{
      var labels = ['R', 'G', 'B'];
      bandInfoEl.innerHTML = bands.map(function(b, idx) {{
        return bands.length === 3 ? '<b>' + labels[idx] + '</b>=' + b : '<b>band</b>=' + b;
      }}).join(' &nbsp; ');
    }}

    function setLayer(url) {{
      if (eeLayer) map.removeLayer(eeLayer);
      eeLayer = L.tileLayer(url, {{maxZoom: 20, opacity: 0.95}}).addTo(map);
    }}
    function requestTile(params) {{
      var qs = new URLSearchParams(Object.assign({{toa: scene.toa, l2: scene.l2, rgb_max: scene.rgb_max}}, params));
      fetch(base + '/api/tile?' + qs.toString()).then(function(r) {{ return r.json(); }})
        .then(function(j) {{
          if (j.error) {{ alert('map server error: ' + j.error); return; }}
          setLayer(j.url);
        }})
        .catch(function(e) {{ alert('map server unreachable: ' + e); }});
    }}

    var presetSel = block.querySelector('.lm-preset');
    var customSpan = block.querySelector('.lm-custom');
    presetSel.addEventListener('change', function() {{
      if (presetSel.value === '__custom__') {{
        customSpan.style.display = '';
        setBandInfo([block.querySelector('.lm-r').value, block.querySelector('.lm-g').value, block.querySelector('.lm-b').value]);
        requestTile({{
          view: '__custom__',
          r: block.querySelector('.lm-r').value,
          g: block.querySelector('.lm-g').value,
          b: block.querySelector('.lm-b').value
        }});
      }} else {{
        customSpan.style.display = 'none';
        setBandInfo(MAP_DATA.presetBands[presetSel.value] || []);
        requestTile({{view: presetSel.value}});
      }}
    }});
    block.querySelector('.lm-apply').addEventListener('click', function() {{
      var bands = [block.querySelector('.lm-r').value, block.querySelector('.lm-g').value, block.querySelector('.lm-b').value];
      setBandInfo(bands);
      requestTile({{view: '__custom__', r: bands[0], g: bands[1], b: bands[2]}});
    }});

    var statusEl = document.getElementById('lmstatus-' + i);
    fetch(base + '/api/ping').then(function(r) {{ return r.json(); }}).then(function() {{
      statusEl.className = 'livemap-status ok';
      statusEl.textContent = 'map server connected — live Earth Engine tiles';
      setBandInfo(MAP_DATA.presetBands[presetSel.value] || []);
      requestTile({{view: presetSel.value}});
    }}).catch(function() {{
      statusEl.className = 'livemap-status bad';
      statusEl.innerHTML = 'map server not running — start it with <code>python extracting_randomized_pixels.Daksh.py DATE --view</code> ' +
        'then reload this page.';
    }});
  }});
}})();
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def generate_report(csv_path, use_modis):
    """GEE check of one candidates CSV (same as output/report_main.py):
    writes <name>_gee_results.csv + <name>_report.html. Returns the HTML path."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as e:
        log.warning(f"  Earth Engine init failed: {e} — no report")
        return None

    wanted = [k for k, s in VIEWS.items() if s.get("on", True)]
    df = read_candidates(csv_path)
    log.info(f"  Loaded {len(df)} points from {os.path.basename(csv_path)}")
    log.info(f"  Views: {', '.join(wanted) if wanted else '(none)'}")

    results, chips, scenes_meta, used_views = [], {}, {}, []

    for scene, rows in df.groupby("landsat_scene", sort=False):
        log.info(f"  Scene {scene}  ({len(rows)} points)")
        try:
            toa_id, l2_id, date = scene_ids(scene)
        except ValueError as e:
            log.info(f"   skipped: {e}")
            continue
        if not asset_exists(toa_id):
            log.info(f"   skipped: {toa_id} not found in Earth Engine")
            continue
        if not asset_exists(l2_id):
            log.info("   Level-2 not available — ST_B10 missing, using TOA B10 for thermal")
            l2_id = None

        toa, full = build_full_image(toa_id, l2_id)
        meta = toa.toDictionary(["SUN_ELEVATION", "CLOUD_COVER", "CLOUD_COVER_LAND"]).getInfo()
        meta["date"] = date
        meta["toa_id"] = toa_id
        meta["l2_id"] = l2_id or ""
        scenes_meta[scene] = meta
        sun = meta.get("SUN_ELEVATION")
        rgb_max = 1.5 if (sun is not None and sun < 15) else 0.4
        log.info(f"   acquired {date}, sun {fmt(sun, 1)}°, scene cloud {fmt(meta.get('CLOUD_COVER'), 1)}%")

        vals = sample_points(full, rows)

        # MODIS (same rule as main_reproject: Landsat-confirmed points only).
        # Reuse values already in the CSV, sample the rest.
        modis = {}
        if use_modis:
            need = [(r.pid, r.lat, r.lon) for r in rows.itertuples()
                    if "Landsat null" not in str(getattr(r, "notes", ""))
                    and (pd.isna(getattr(r, "modis_ndvi", None))
                         or pd.isna(getattr(r, "modis_lc_type1", None)))]
            if need:
                log.info(f"   sampling MODIS NDVI + land cover for {len(need)} points ...")
                modis = sample_modis(ee, need, date)

        for r in rows.itertuples():
            v = vals.get(r.pid, {})
            c = classify(v)
            m = {}
            if r.pid in modis:
                ndvi = modis[r.pid]["modis_ndvi"]
                lc = modis[r.pid]["modis_lc_type1"]
                m = {"modis_ndvi": ndvi,
                     "modis_lc_type1": lc,
                     "modis_lc_name": (IGBP_LOOKUP.get(lc, f"Unknown({lc})")
                                       if lc is not None else None)}
            results.append({
                **r._asdict(), **m, **c,
                **compare(getattr(r, "ground_truth_class", None), c),
                **decode_qa(v.get("QA_PIXEL")),
                **{k: val for k, val in v.items() if k not in ("pid", "QA_PIXEL")},
                "gee_scene": toa_id, "sun_elevation": sun,
            })

        if wanted:
            band_names = set(full.bandNames().getInfo())
            views_here = []
            for name in wanted:
                missing = [b for b in VIEWS[name]["bands"] if b not in band_names]
                if missing:
                    log.info(f"   view '{name}' skipped (missing band {', '.join(missing)})")
                else:
                    views_here.append(name)
            for name in views_here:
                if name not in used_views:
                    used_views.append(name)

            vis = {name: make_view(full, VIEWS[name], rgb_max) for name in views_here}
            log.info(f"   downloading {len(views_here) * len(rows)} chips ...")
            jobs = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for r in rows.itertuples():
                    for name in views_here:
                        jobs[(r.pid, name)] = pool.submit(chip_data_uri, vis[name], r.lat, r.lon)
            for (pid, name), fut in jobs.items():
                chips.setdefault(pid, {})[name] = fut.result()

    if not results:
        log.warning("  No points processed — no report")
        return None

    res = pd.DataFrame(results).drop(columns=["Index"], errors="ignore")

    base = os.path.splitext(os.path.abspath(csv_path))[0]
    out_csv = base + "_gee_results.csv"
    out_html = base + "_report.html"

    res.drop(columns=["pid"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_report(res, chips, used_views, scenes_meta, out_html,
                 f"GEE check — {os.path.basename(csv_path)}")

    log.info("  === SUMMARY ===")
    for status, n in res["status"].value_counts().items():
        log.info(f"   {status:9s} {n}")
    for _, r in res[res["status"] != "MATCH"].iterrows():
        log.info(f"   id {r['id']}: {r['status']:9s} truth={r.get('ground_truth_class')}  "
              f"GEE={r['gee_label']}  QA: {r.get('qa_summary')}")
    log.info(f"  Results CSV : {out_csv}")
    log.info(f"Report      : {out_html}")

    return out_html


_image_cache = {}


def get_full_image(toa_id, l2_id):
    key = (toa_id, l2_id or None)
    if key not in _image_cache:
        _, full = build_full_image(toa_id, l2_id or None)
        _image_cache[key] = full
    return _image_cache[key]


class MapServerHandler(http.server.SimpleHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # lets the report work even when opened as a file:// page instead of
        # through this server (some browsers still allow it with these set)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/ping":
                return self._json(200, {"ok": True})
            if parsed.path == "/api/tile":
                return self._handle_tile(q)
            if parsed.path == "/api/inspect":
                return self._handle_inspect(q)
            if parsed.path == "/api/qa_overlay":
                return self._handle_qa_overlay(q)
            if parsed.path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            return super().do_GET()
        except Exception as e:
            return self._json(400, {"error": str(e)})

    def _handle_tile(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        rgb_max = float(q.get("rgb_max", ["0.4"])[0])
        view_name = q["view"][0]
        full = get_full_image(toa_id, l2_id)

        if view_name == "__custom__":
            bands = [q["r"][0], q["g"][0], q["b"][0]]
            mins, maxs = [], []
            for b in bands:
                dmn, dmx = DEFAULT_VIS.get(b, (0, 1))
                mins.append(float(q.get(f"{b}_min", [dmn])[0]))
                maxs.append(float(q.get(f"{b}_max", [dmx])[0]))
            vis_img = full.select(bands).visualize(min=mins, max=maxs)
        else:
            spec = VIEWS.get(view_name)
            if not spec:
                return self._json(400, {"error": f"unknown view {view_name}"})
            vis_img = make_view(full, spec, rgb_max)

        mapid = vis_img.getMapId()
        self._json(200, {"url": mapid["tile_fetcher"].url_format})

    def _handle_qa_overlay(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        full = get_full_image(toa_id, l2_id)
        vis_img = qa_overlay_image(full)
        mapid = vis_img.getMapId()
        self._json(200, {"url": mapid["tile_fetcher"].url_format})

    def _handle_inspect(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        lat = float(q["lat"][0])
        lon = float(q["lon"][0])
        full = get_full_image(toa_id, l2_id)
        pt = ee.Geometry.Point([lon, lat])
        values = full.reduceRegion(ee.Reducer.first(), pt, scale=30).getInfo()
        cls = classify(values)
        qa = decode_qa(values.get("QA_PIXEL"))
        self._json(200, {"lat": lat, "lon": lon, "values": values, "classify": cls, "qa": qa})

    def log_message(self, fmt, *args):
        print("  " + (fmt % args))


def run_report(csv_path, use_modis, overwrite=False):
    """Make the GEE report for one candidates CSV."""
    html_path = os.path.splitext(csv_path)[0] + "_report.html"
    if os.path.exists(html_path) and not overwrite:
        log.info(f"  Report exists, skip: {html_path}")
        return
    log.info(f"  Generating GEE report for {os.path.basename(csv_path)} ...")
    try:
        html_path = generate_report(csv_path, use_modis)
    except Exception as e:
        log.warning(f"  Report generation failed for {csv_path}: {e}")
        return None
    if html_path:
        log.info(f"  Saved report: {html_path}")
    return html_path


def _port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def existing_reports(requested_dates=None):
    """*_report.html files already in output_pixels/<date>/ (all dates if none given)."""
    if not os.path.isdir(PIXEL_DIR):
        return []
    dates = requested_dates or sorted(os.listdir(PIXEL_DIR))
    out = []
    for d in dates:
        dd = os.path.join(PIXEL_DIR, d)
        if os.path.isdir(dd):
            out += [os.path.join(dd, f) for f in sorted(os.listdir(dd))
                    if f.endswith("_report.html")]
    return out


def _open_reports(report_paths):
    for rp in report_paths:
        rel = os.path.relpath(rp, PIXEL_DIR).replace(os.sep, "/")
        url = f"http://127.0.0.1:{MAP_SERVER_PORT}/{rel}"
        log.info(f"  Opening {url}")
        webbrowser.open(url)


def view_reports(report_paths):
    """Start the live Earth Engine map server (in this process) and open the
    reports through it. Blocks until Ctrl+C."""
    if _port_open(MAP_SERVER_PORT):
        log.info(f"Map server already running on port {MAP_SERVER_PORT} — reusing it")
        _open_reports(report_paths)
        return
    log.info("Starting map server (Earth Engine init takes a few seconds) ...")
    try:
        ee.Initialize(project=EE_PROJECT)
        handler = functools.partial(MapServerHandler, directory=PIXEL_DIR)
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", MAP_SERVER_PORT), handler)
    except Exception as e:
        log.error(f"Map server failed to start ({e}) — opening report files directly "
                  "(live map panel will be unavailable)")
        for rp in report_paths:
            webbrowser.open("file:///" + os.path.abspath(rp).replace(os.sep, "/"))
        return
    log.info(f"Map server running on http://127.0.0.1:{MAP_SERVER_PORT}  (serving {PIXEL_DIR})")
    _open_reports(report_paths)
    log.info("Press Ctrl+C to stop it.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        log.info("Map server stopped.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    overwrite    = "--overwrite" in sys.argv
    uniform      = "--uniform" in sys.argv
    pure_random  = "--pure-random" in sys.argv
    use_modis    = ENABLE_MODIS and "--no-modis" not in sys.argv
    use_report   = ENABLE_REPORT and "--no-report" not in sys.argv
    view_flag    = ("yes" if "--view" in sys.argv
                    else "no" if "--no-view" in sys.argv else "ask")
    seed      = None
    region    = None
    args = sys.argv[1:]
    if "--seed" in args:
        si = args.index("--seed")
        seed = int(args[si + 1])
        args = args[:si] + args[si + 2:]
    if "--region" in args:
        ri = args.index("--region")
        region = args[ri + 1]
        if region not in ("top", "bottom", "middle"):
            log.error(f"--region must be one of: top, bottom, middle (got '{region}')")
            return
        args = args[:ri] + args[ri + 2:]
    side = None
    if "--side" in args:
        si = args.index("--side")
        side = args[si + 1]
        if side not in ("left", "right"):
            log.error(f"--side must be one of: left, right (got '{side}')")
            return
        args = args[:si] + args[si + 2:]
    requested = [a for a in args if not a.startswith("--")]
    mode = "uniform" if uniform else ("pure-random" if pure_random else "stratified")

    # --view and the reports already exist -> just start the map server
    if view_flag == "yes" and not overwrite:
        ready = existing_reports(requested if requested else None)
        if ready:
            log.info(f"Found {len(ready)} existing report(s) — starting the map server "
                     "only (pass --overwrite to regenerate)")
            view_reports(ready)
            return

    log.info("=" * 60)
    log.info("Pixel Extraction (standalone)")
    log.info("=" * 60)

    pairs = discover_output_dates(requested if requested else None)
    log.info(f"Found {len(pairs)} VIIRS+Landsat pair(s) to process")

    if not pairs:
        log.error("No processed TIF pairs found in output/")
        return

    # Initialise Earth Engine once for MODIS enrichment.  Degrade gracefully
    # if EE is unavailable: MODIS columns are still written, just empty.
    ee = init_ee() if use_modis else None

    reports = []
    for date_str, viirs_path, landsat_path, out_dir in pairs:
        log.info(f"\n{'='*60}")
        log.info(f"Date {date_str}")
        log.info(f"  VIIRS:   {os.path.basename(viirs_path)}")
        log.info(f"  Landsat: {os.path.basename(landsat_path)}")

        b10_path, mtl_vals = find_b10_and_mtl(date_str, landsat_path)
        if b10_path:
            log.info(f"  B10 + MTL found — thermal enabled "
                     f"({mtl_vals.get('LANDSAT_PRODUCT_ID', '?')})")

        csv_path = extract_and_save(viirs_path, landsat_path, date_str, out_dir,
                                    b10_path, mtl_vals, overwrite, seed, mode,
                                    region, side, ee)
        if use_report and csv_path and os.path.exists(csv_path):
            html_path = run_report(csv_path, use_modis, overwrite)
            if html_path is None:
                existing = os.path.splitext(csv_path)[0] + "_report.html"
                html_path = existing if os.path.exists(existing) else None
            if html_path:
                reports.append(html_path)

    log.info(f"\n{'='*60}")
    log.info("Pixel extraction complete.")

    if reports:
        log.info("Reports:")
        for rp in reports:
            log.info(f"  {rp}")
        want = view_flag == "yes"
        if view_flag == "ask" and sys.stdin.isatty():
            ans = input("\nOpen the report(s) in the browser with the live map "
                        "server? [y/N]: ").strip().lower()
            want = ans in ("y", "yes")
        if want:
            view_reports(reports)
        else:
            log.info("To view later with the live map:  python extracting_randomized_pixels.Daksh.py "
                     "<date> --view")


if __name__ == "__main__":
    main()
