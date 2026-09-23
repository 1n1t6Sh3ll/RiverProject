"""
Standalone pixel extraction — runs AFTER main.py has produced the
reprojected VIIRS and Landsat TIFs in output/<YYYYMMDD>/.

Scans output/ for date folders containing viirs_alaska_*.tif and
landsat_*.tif, then generates:
    training_candidates_YYYY-MM-DD.csv
    viirs_vs_watermask_YYYY-MM-DD.png
    training_candidates_YYYY-MM-DD_report.html      (via output/report_main.py)
    training_candidates_YYYY-MM-DD_gee_results.csv  (via output/report_main.py)

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
    python extract_pixels.py --no-view        # never ask to open reports
    (default asks at the end whether to open the report(s) + start the map server)
    (default is stratified random — spread out but different each run)
"""

import os, re, math, csv, sys, logging, subprocess, socket, time, webbrowser
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

import modis_enrich

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

# GEE report (output/report_main.py) run on every CSV. Set False (or pass --no-report) to skip.
ENABLE_REPORT = True
REPORT_SCRIPT = os.path.join(OUTPUT_DIR, "report_main.py")
MAP_SERVER_SCRIPT = os.path.join(OUTPUT_DIR, "report_map_server.py")
MAP_SERVER_PORT = 8765   # must match MAP_SERVER_PORT in output/report_main.py

VIIRS_BAND_ORDER = ["I1", "I2", "I3", "I4", "I5", "SZA", "SAA", "VZA", "VAA"]

log = logging.getLogger("extract_pixels")
log.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%H:%M:%S"))
log.addHandler(_ch)
_modis_log = logging.getLogger("modis_enrich")
_modis_log.setLevel(logging.INFO)
_modis_log.addHandler(_ch)

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
    modis_enrich.enrich_rows(ee, rows_out, date_fmt)

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


def run_report(csv_path, use_modis, overwrite=False):
    """Run the GEE report generator (output/report_main.py) on one candidates CSV."""
    html_path = os.path.splitext(csv_path)[0] + "_report.html"
    if os.path.exists(html_path) and not overwrite:
        log.info(f"  Report exists, skip: {html_path}")
        return
    cmd = [sys.executable, REPORT_SCRIPT, csv_path,
           "--project", modis_enrich.EE_PROJECT, "--no-open"]
    if not use_modis:
        cmd.append("--no-modis")
    log.info(f"  Generating GEE report for {os.path.basename(csv_path)} ...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        log.info(f"  Saved report: {html_path}")
        return html_path
    log.warning(f"  Report generation failed (exit {result.returncode}) "
                f"for {csv_path}")
    return None


def _port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def view_reports(report_paths):
    """Start output/report_map_server.py (live Earth Engine map) and open the reports
    through it. Blocks until Ctrl+C, then stops the server it started."""
    server = None
    if _port_open(MAP_SERVER_PORT):
        log.info(f"Map server already running on port {MAP_SERVER_PORT} — reusing it")
    else:
        log.info("Starting map server (Earth Engine init takes a few seconds) ...")
        server = subprocess.Popen(
            [sys.executable, MAP_SERVER_SCRIPT, "--project", modis_enrich.EE_PROJECT,
             "--port", str(MAP_SERVER_PORT), "--root", PIXEL_DIR])
        deadline = time.time() + 60
        while not _port_open(MAP_SERVER_PORT):
            if server.poll() is not None or time.time() > deadline:
                log.error("Map server failed to start — opening report files directly "
                          "(live map panel will be unavailable)")
                for rp in report_paths:
                    webbrowser.open("file:///" + os.path.abspath(rp).replace(os.sep, "/"))
                if server.poll() is None:
                    server.terminate()
                return
            time.sleep(0.5)

    for rp in report_paths:
        rel = os.path.relpath(rp, PIXEL_DIR).replace(os.sep, "/")
        url = f"http://127.0.0.1:{MAP_SERVER_PORT}/{rel}"
        log.info(f"  Opening {url}")
        webbrowser.open(url)

    if server is None:
        return
    log.info("Map server running. Press Ctrl+C to stop it.")
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
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
    ee = modis_enrich.init_ee() if use_modis else None

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
