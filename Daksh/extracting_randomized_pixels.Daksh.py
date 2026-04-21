"""
Standalone pixel extraction — runs AFTER main.py has produced the
reprojected VIIRS and Landsat TIFs in output/<YYYYMMDD>/.

Scans output/ for date folders containing viirs_alaska_*.tif and
landsat_*.tif, then generates:
    training_candidates_YYYY-MM-DD.csv
    viirs_vs_watermask_YYYY-MM-DD.png

Usage:
    python extract_pixels.py                  # all dates, random 25 pixels
    python extract_pixels.py 20240102         # single date
    python extract_pixels.py 20240102 20240206  # multiple dates
    python extract_pixels.py --overwrite      # regenerate existing
    python extract_pixels.py --seed 42        # reproducible random selection
    python extract_pixels.py --uniform        # evenly spaced (main.py behavior)
    python extract_pixels.py --pure-random    # fully random (no spatial spread)
    (default is stratified random — spread out but different each run)
"""

import os, re, math, csv, sys, logging
import numpy as np
import rasterio
from rasterio.warp import reproject as rio_reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration — same constants as main.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
WATER_MASK   = os.path.join(DATA_DIR, "alaska_occ_375m.tif")
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

VIIRS_BAND_ORDER = ["I1", "I2", "I3", "I4", "I5", "SZA", "SAA", "VZA", "VAA"]

log = logging.getLogger("extract_pixels")
log.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%H:%M:%S"))
log.addHandler(_ch)

# ---------------------------------------------------------------------------
# Helpers (same as main.py)
# ---------------------------------------------------------------------------

def parse_mtl(mtl_path):
    keys = {
        "K1_CONSTANT_BAND_10": "K1",
        "K2_CONSTANT_BAND_10": "K2",
        "RADIANCE_MULT_BAND_10": "RADIANCE_MULT",
        "RADIANCE_ADD_BAND_10": "RADIANCE_ADD",
    }
    vals = {}
    try:
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.strip()
                for mtl_key, short in keys.items():
                    if line.upper().startswith(mtl_key):
                        vals[short] = float(line.split("=")[1].strip())
    except Exception:
        return None
    return vals if len(vals) == 4 else None


def dn_to_kelvin(dn, mtl_vals):
    rad = mtl_vals["RADIANCE_MULT"] * dn + mtl_vals["RADIANCE_ADD"]
    if rad <= 0:
        return None
    return mtl_vals["K2"] / math.log(mtl_vals["K1"] / rad + 1)


def _normalize(arr, pct_lo=2, pct_hi=98):
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(fin, pct_lo), np.nanpercentile(fin, pct_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

# ---------------------------------------------------------------------------
# Discover B10 + MTL from raw Landsat data
# ---------------------------------------------------------------------------

def find_b10_and_mtl(date_str):
    """Look in data/landsat/<date_str>/ for B10 and MTL files."""
    ls_folder = os.path.join(LANDSAT_DIR, date_str)
    if not os.path.isdir(ls_folder):
        return None, None
    files = os.listdir(ls_folder)
    b10 = [f for f in files if re.search(r'_B10\.TIF$', f, re.I)]
    mtl = [f for f in files if f.upper().endswith("MTL.TXT")]
    if not b10 or not mtl:
        return None, None
    b10_path = os.path.join(ls_folder, b10[0])
    mtl_vals = parse_mtl(os.path.join(ls_folder, mtl[0]))
    if mtl_vals is None:
        return None, None
    return b10_path, mtl_vals

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
                     seed=None, mode="stratified"):

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
        return

    # --- Load Landsat metadata ---
    with rasterio.open(landsat_path) as ls_src:
        ls_bounds = ls_src.bounds
        ls_tf     = ls_src.transform
        ls_w, ls_h = ls_src.width, ls_src.height
        ls_band_names = []
        for bi in range(1, ls_src.count + 1):
            tags = ls_src.tags(bi)
            ls_band_names.append(tags.get("name", f"band{bi}"))

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

    # Read water mask
    log.info(f"  Loading water mask ...")
    with rasterio.open(WATER_MASK) as wm_src:
        wm_win = window_from_bounds(scene_lon_min, scene_lat_min,
                                     scene_lon_max, scene_lat_max,
                                     transform=wm_src.transform)
        wm_raw = wm_src.read(1, window=wm_win).astype(np.float32)
        wm_win_tf = wm_src.window_transform(wm_win)
        if wm_src.nodata is not None:
            wm_raw[wm_raw == wm_src.nodata] = np.nan

    wm = np.full((scene_h, scene_w), np.nan, np.float32)
    rio_reproject(
        source=wm_raw, destination=wm,
        src_transform=wm_win_tf, src_crs=TARGET_CRS,
        src_nodata=np.nan,
        dst_transform=scene_tf, dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    log.info(f"    Shape: {wm.shape}  valid: {int(np.sum(np.isfinite(wm))):,}")

    # Landsat valid mask
    log.info(f"  Building Landsat valid mask ...")
    ls_valid = np.zeros((scene_h, scene_w), dtype=bool)
    with rasterio.open(landsat_path) as ls_src:
        for bi in range(1, ls_src.count + 1):
            band = ls_src.read(bi, out_shape=(scene_h, scene_w),
                               resampling=Resampling.nearest).astype(np.float32)
            ls_valid |= (band != NODATA) & np.isfinite(band) & (band != 0)
    log.info(f"    Landsat valid: {int(ls_valid.sum()):,} px")

    # Narrow river pixels
    narrow = (np.isfinite(wm) & (wm > 0.05) & (wm < 0.95) &
              np.isfinite(viirs_grids["I1"]) & ls_valid)
    rows_n, cols_n = np.where(narrow)
    log.info(f"  Narrow river pixels: {len(rows_n):,}")

    if len(rows_n) == 0:
        log.warning(f"  No narrow river pixels — skipping {date_str}")
        return

    all_pixels = list(zip(rows_n, cols_n))
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
                "viirs_date":      date_fmt,
                "landsat_scene":   ls_scene_id,
                "row_shared_grid": r_shared,
                "col_shared_grid": c_shared,
                "lat":             round(lat_px, 5),
                "lon":             round(lon_px, 5),
                "water_fraction":  round(float(wm[r, c]), 4),
            }
            for name in VIIRS_BAND_ORDER:
                row[name] = _safe_viirs(name)
            for bname in ls_band_names:
                row[f"LS_{bname}"] = _safe_ls(bname)

            b3 = ls_vals.get("B3")
            b6 = ls_vals.get("B6")
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
                        b10_col, b10_row = ~b10_src.transform * (lon_px, lat_px)
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
                row["notes"] = (f"Landsat confirmed at 30m. "
                                f"ST_B10={round(st_b10, 2)}K "
                                f"NDSI={round(ndsi, 4)}.")
            elif ndsi is not None:
                is_snow = ndsi > 0.4
                row["ground_truth_class"] = ("snow_covered_land" if is_snow
                                             else "snow_free_land")
                row["notes"] = (f"NDSI={round(ndsi, 4)}. "
                                f"No thermal — ice status unknown.")
            else:
                row["ground_truth_class"] = ""
                row["notes"] = ("Landsat null — outside swath. Classify manually."
                                if not ls_inside else
                                "NDSI unavailable — classify manually.")

            rows_out.append(row)

    log.info(f"  {n_ls_valid}/{len(candidates)} candidates inside Landsat swath")
    log.info(f"  {n_skipped} skipped (outside Alaska domain)")

    if not rows_out:
        log.warning(f"  No valid candidates — no CSV for {date_fmt}")
        return

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"VIIRS 2-2-1 false color vs JRC water fraction\n"
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
    axes[0].set_title("VIIRS 2-2-1 (I2\u2192R, I2\u2192G, I1\u2192B)\n"
                      "Ice/snow=bright | Water=dark | NaN=transparent")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    im2 = axes[1].imshow(wm, cmap="Blues", vmin=0, vmax=1,
                          interpolation="nearest",
                          extent=[scene_lon_min, scene_lon_max,
                                  scene_lat_min, scene_lat_max],
                          aspect="auto", origin="upper")
    plt.colorbar(im2, ax=axes[1], fraction=0.03, label="Water fraction")
    for r, c in candidates:
        lon_pt = scene_lon_min + (c + 0.5) * VIIRS_RES
        lat_pt = scene_lat_max - (r + 0.5) * VIIRS_RES
        axes[1].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
    axes[1].set_title("JRC water fraction (occurrence)\n"
                      "\u00d7 = candidate training pixels")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved PNG: {png_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    overwrite    = "--overwrite" in sys.argv
    uniform      = "--uniform" in sys.argv
    pure_random  = "--pure-random" in sys.argv
    seed      = None
    args = sys.argv[1:]
    if "--seed" in args:
        si = args.index("--seed")
        seed = int(args[si + 1])
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

    for date_str, viirs_path, landsat_path, out_dir in pairs:
        log.info(f"\n{'='*60}")
        log.info(f"Date {date_str}")
        log.info(f"  VIIRS:   {os.path.basename(viirs_path)}")
        log.info(f"  Landsat: {os.path.basename(landsat_path)}")

        b10_path, mtl_vals = find_b10_and_mtl(date_str)
        if b10_path:
            log.info(f"  B10 + MTL found — thermal enabled")

        extract_and_save(viirs_path, landsat_path, date_str, out_dir,
                         b10_path, mtl_vals, overwrite, seed, mode)

    log.info(f"\n{'='*60}")
    log.info("Pixel extraction complete.")


if __name__ == "__main__":
    main()
