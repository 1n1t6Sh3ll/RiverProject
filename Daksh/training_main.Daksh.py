"""
Landsat + VIIRS reprojection pipeline — Alaska AOI.

Exact same logic as valen/extract_training_pixels.py:
  - VIIRS:   full Alaska EPSG:4326 grid, 12432 x 5328, res = 375/111000°
             9-band float32 GeoTIFF (I1-I5, SZA, SAA, VZA, VAA)
             pyresample resample_nearest, radius=7500, epsilon=0.5
  - Landsat: USGS TIFFs reprojected to EPSG:4326 at 30/111000°,
             scene-extent only (pixel grid aligned to Alaska domain origin),
             Lanczos for spectral, nearest for angles
  - No GEE dependency.

Output:
    output/<YYYYMMDD>/
        viirs_<granule>.tif       full Alaska, 9-band
        landsat_<scene>.tif       scene extent, N-band
        samples/
            *_sample.tif
"""

import os, re, math, csv, logging
from datetime import datetime, timedelta
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject as rio_reproject, Resampling
from rasterio.transform import from_bounds, Affine
from rasterio.windows import from_bounds as window_from_bounds
from pyresample import geometry as pr_geometry
from pyresample.kd_tree import resample_nearest
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# ---------------------------------------------------------------------------
# Configuration — matches valen/extract_training_pixels.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
WATER_MASK_OCC = os.path.join(DATA_DIR, "alaska_occ_375m.tif")
WATER_MASK_SEA = os.path.join(DATA_DIR, "alaska_sea_375m.tif")
VIIRS_DIR    = os.path.join(DATA_DIR, "viirs")
LANDSAT_DIR  = os.path.join(DATA_DIR, "landsat")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")

TARGET_CRS = "EPSG:4326"

# Alaska domain (identical to valen/)
ALASKA_LAT_MIN = 54.0
ALASKA_LAT_MAX = 72.0
ALASKA_LON_MIN = -171.0
ALASKA_LON_MAX = -129.0

# VIIRS grid (identical to valen/)
VIIRS_RES_M  = 375.0
VIIRS_RES    = VIIRS_RES_M / 111_000.0           # ≈ 0.003378°
VIIRS_W      = int(round((ALASKA_LON_MAX - ALASKA_LON_MIN) / VIIRS_RES))  # 12432
VIIRS_H      = int(round((ALASKA_LAT_MAX - ALASKA_LAT_MIN) / VIIRS_RES))  # 5328
VIIRS_TF     = from_bounds(ALASKA_LON_MIN, ALASKA_LAT_MIN,
                            ALASKA_LON_MAX, ALASKA_LAT_MAX,
                            VIIRS_W, VIIRS_H)

# Landsat pixel size in degrees (same conversion as valen/)
LANDSAT_RES_M = 30.0
LANDSAT_RES   = LANDSAT_RES_M / 111_000.0        # ≈ 0.000270°

# Resampling (identical to valen/)
RADIUS_OF_INFLUENCE = 7500
EPSILON             = 0.5

# Output format (identical to valen/)
NODATA          = -9999.0
OVERWRITE       = False
SAMPLE_TILE_PX  = 128

VIIRS_BAND_ORDER = ["I1", "I2", "I3", "I4", "I5", "SZA", "SAA", "VZA", "VAA"]

log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%H:%M:%S"))
log.addHandler(_ch)

# File handler — one log per run in output/
os.makedirs(OUTPUT_DIR, exist_ok=True)
_fh = logging.FileHandler(
    os.path.join(OUTPUT_DIR, "pipeline_run.log"), mode="a", encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_fh)

# ---------------------------------------------------------------------------
# VIIRS H5 loaders  (replicated from valen/viirs_training_loader.py)
# ---------------------------------------------------------------------------

def _h5_datasets(h5, prefix=""):
    paths = []
    for key in h5.keys():
        p = f"{prefix}/{key}" if prefix else key
        if isinstance(h5[key], h5py.Dataset):
            paths.append(p)
        else:
            paths.extend(_h5_datasets(h5[key], p))
    return paths


def _h5_find(h5, *keywords):
    for path in _h5_datasets(h5):
        if all(kw.lower() in path.lower() for kw in keywords):
            return path
    return None


def load_viirs_geo(gitco_path):
    """Load lat/lon from GITCO — same as valen/viirs_training_loader."""
    with h5py.File(gitco_path, "r") as h5:
        lat_p = _h5_find(h5, "Latitude")
        lon_p = _h5_find(h5, "Longitude")
        if not lat_p or not lon_p:
            raise KeyError(f"Lat/Lon not found in {gitco_path}")
        lat = h5[lat_p][...].astype(np.float32)
        lon = h5[lon_p][...].astype(np.float32)
    lat[(lat < -90) | (lat > 90)]   = np.nan
    lon[(lon < -180) | (lon > 180)] = np.nan
    return lat, lon


def load_viirs_bands(gimgo_path):
    """Load I1-I5 with scale/offset — same as valen/viirs_training_loader."""
    bands = {}
    with h5py.File(gimgo_path, "r") as h5:
        all_paths = _h5_datasets(h5)
        for i in range(1, 6):
            token   = f"VIIRS-I{i}-SDR_All"
            targets = ["Reflectance"] if i <= 3 else ["BrightnessTemperature"]
            for target in targets:
                cands = [p for p in all_paths
                         if token in p and p.endswith(target)]
                if cands:
                    raw = h5[cands[0]][...].astype(np.float32)
                    fp = cands[0] + "Factors"
                    if fp in h5:
                        fac = h5[fp][...].ravel()
                        scale  = float(fac[0]) if len(fac) >= 1 else 1.0
                        offset = float(fac[1]) if len(fac) >= 2 else 0.0
                    else:
                        scale, offset = 1.0, 0.0
                    arr = raw * scale + offset
                    arr[raw > 65529] = np.nan
                    bands[f"I{i}"] = arr
                    break
            if f"I{i}" not in bands:
                raise KeyError(f"Band I{i} not found in {gimgo_path}")
    return bands


def load_viirs_angles(gitco_path):
    """Load SZA/SAA/VZA/VAA — same as valen/viirs_training_loader."""
    angle_map = {
        "SZA": ["SolarZenithAngle", "SZA"],
        "SAA": ["SolarAzimuthAngle", "SAA"],
        "VZA": ["SatelliteZenithAngle", "SatZenithAngle", "VZA"],
        "VAA": ["SatelliteAzimuthAngle", "SatAzimuthAngle", "VAA"],
    }
    angles = {}
    with h5py.File(gitco_path, "r") as h5:
        all_paths = _h5_datasets(h5)
        for key, candidates in angle_map.items():
            for cand in candidates:
                matches = [p for p in all_paths if cand.lower() in p.lower()]
                for p in matches:
                    arr = h5[p][...].astype(np.float32)
                    if arr.ndim == 2:
                        arr[arr > 1e6] = np.nan
                        angles[key] = arr
                        break
                if key in angles:
                    break
            if key not in angles:
                log.warning(f"  VIIRS angle {key} not found — will be NaN")
    return angles

# ---------------------------------------------------------------------------
# VIIRS reprojection — resample_nearest, same as valen/
# ---------------------------------------------------------------------------

def _reproject_viirs_band(lat, lon, data, area_def):
    """Mirrors valen/extract_training_pixels._resample_band_to_alaska() exactly."""
    valid  = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(data)
    lats_m = np.ma.masked_array(lat,  mask=~valid)
    lons_m = np.ma.masked_array(lon,  mask=~valid)
    data_m = np.ma.masked_array(data.astype(np.float32), mask=~valid)

    swath_def = pr_geometry.SwathDefinition(lons=lons_m, lats=lats_m)
    result = resample_nearest(
        swath_def, data_m, area_def,
        radius_of_influence=RADIUS_OF_INFLUENCE,
        epsilon=EPSILON,
        fill_value=np.nan,
    )
    arr = np.ma.filled(result, np.nan) if np.ma.is_masked(result) else np.asarray(result)
    return np.asarray(arr, dtype=np.float32)

# ---------------------------------------------------------------------------
# Landsat scene-extent grid (aligned to Alaska domain origin)
# ---------------------------------------------------------------------------

def compute_landsat_scene_grid(folder_path, band_files):
    """Read Landsat scene footprint, compute an EPSG:4326 grid at LANDSAT_RES
    whose pixels are aligned with the Alaska domain origin (-171, 72)."""
    # Get scene bounds in its native CRS, then transform to 4326
    sample_tif = os.path.join(folder_path, band_files[0])
    with rasterio.open(sample_tif) as src:
        src_crs = src.crs
        b = src.bounds

    # Dense edge sampling for accurate reprojection of scene footprint
    t = Transformer.from_crs(src_crs, TARGET_CRS, always_xy=True)
    n = 100
    xs_src, ys_src = [], []
    for i in range(n + 1):
        frac = i / n
        xs_src += [b.left + frac * (b.right - b.left), b.right,
                   b.right - frac * (b.right - b.left), b.left]
        ys_src += [b.top, b.top - frac * (b.top - b.bottom),
                   b.bottom, b.bottom + frac * (b.top - b.bottom)]
    lons, lats = t.transform(xs_src, ys_src)

    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    # Snap to Alaska domain pixel grid: pixel edges at
    #   lon = ALASKA_LON_MIN + k * LANDSAT_RES
    #   lat = ALASKA_LAT_MAX - k * LANDSAT_RES
    col_min = math.floor((lon_min - ALASKA_LON_MIN) / LANDSAT_RES)
    col_max = math.ceil((lon_max - ALASKA_LON_MIN) / LANDSAT_RES)
    row_min = math.floor((ALASKA_LAT_MAX - lat_max) / LANDSAT_RES)
    row_max = math.ceil((ALASKA_LAT_MAX - lat_min) / LANDSAT_RES)

    scene_lon_min = ALASKA_LON_MIN + col_min * LANDSAT_RES
    scene_lon_max = ALASKA_LON_MIN + col_max * LANDSAT_RES
    scene_lat_max = ALASKA_LAT_MAX - row_min * LANDSAT_RES
    scene_lat_min = ALASKA_LAT_MAX - row_max * LANDSAT_RES

    scene_w = col_max - col_min
    scene_h = row_max - row_min
    scene_tf = from_bounds(scene_lon_min, scene_lat_min,
                           scene_lon_max, scene_lat_max,
                           scene_w, scene_h)

    return scene_tf, scene_w, scene_h

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

VIIRS_GRANULE_MINUTES = 6  # Suomi NPP IMG granule duration


def _viirs_match_dates(folder_name):
    """Return all UTC dates a VIIRS granule overlaps.

    VIIRS folders are named ``YYYYMMDD_tHHMMSSS``. Granules are ~6 minutes
    long, so a granule beginning late in a UTC day can end on the next UTC
    day. We return both dates when that happens, so the granule can match a
    Landsat scene dated either day.
    """
    parts = folder_name.split("_")
    date_str = parts[0]
    if len(parts) < 2 or not parts[1].startswith("t") or len(parts[1]) < 7:
        return [date_str]
    try:
        start = datetime.strptime(date_str + parts[1][1:7], "%Y%m%d%H%M%S")
    except Exception:
        return [date_str]
    end = start + timedelta(minutes=VIIRS_GRANULE_MINUTES)
    dates = [start.strftime("%Y%m%d")]
    if start.date() != end.date():
        dates.append(end.strftime("%Y%m%d"))
        log.info(f"  VIIRS {folder_name}: spans {start:%Y-%m-%d %H:%M} → "
                 f"{end:%Y-%m-%d %H:%M} UTC — matching both days")
    return dates


def discover_viirs():
    granules = {}
    for folder in sorted(os.listdir(VIIRS_DIR)):
        fp = os.path.join(VIIRS_DIR, folder)
        if not os.path.isdir(fp):
            continue
        files = os.listdir(fp)
        gitco = [f for f in files if f.upper().startswith("GITCO")]
        gimgo = [f for f in files if f.upper().startswith("GIMGO")]
        if not gitco or not gimgo:
            log.warning(f"Skipping VIIRS {folder}: missing GITCO or GIMGO")
            continue
        entry = (fp, os.path.join(fp, gitco[0]),
                 os.path.join(fp, gimgo[0]), folder)
        for date_str in _viirs_match_dates(folder):
            granules.setdefault(date_str, []).append(entry)
    return granules


def parse_mtl(mtl_path):
    """Parse MTL.txt for thermal calibration constants.
    Returns dict with K1, K2, RADIANCE_MULT, RADIANCE_ADD for B10, or None."""
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
    if len(vals) == 4:
        return vals
    return None


def dn_to_kelvin(dn, mtl_vals):
    """Convert Landsat B10 DN to brightness temperature in Kelvin.
    radiance = RADIANCE_MULT * DN + RADIANCE_ADD
    T = K2 / ln(K1 / radiance + 1)"""
    rad = mtl_vals["RADIANCE_MULT"] * dn + mtl_vals["RADIANCE_ADD"]
    if rad <= 0:
        return None
    return mtl_vals["K2"] / math.log(mtl_vals["K1"] / rad + 1)


def discover_landsat():
    scenes = {}
    for folder in sorted(os.listdir(LANDSAT_DIR)):
        fp = os.path.join(LANDSAT_DIR, folder)
        if not os.path.isdir(fp):
            continue
        date_str = folder
        files = os.listdir(fp)

        # Group files by scene ID = filename prefix before the suffix.
        # Suffixes we recognize: _B<n>.TIF, _SAA.TIF, _SZA.TIF, _VAA.TIF, _VZA.TIF, _MTL.txt
        groups = {}
        for f in files:
            sid = None
            m = re.match(r'^(.+)_(B\d+|SAA|SZA|VAA|VZA)\.TIF$', f, re.I)
            if m:
                sid = m.group(1)
            elif f.upper().endswith("_MTL.TXT"):
                sid = f[:-8]
            if sid is None:
                continue
            groups.setdefault(sid, []).append(f)

        for sid in sorted(groups):
            sfiles = groups[sid]
            bands  = sorted([f for f in sfiles if re.search(r'_B[1-6]\.TIF$', f, re.I)])
            angles = sorted([f for f in sfiles if re.search(r'_(SAA|SZA|VAA|VZA)\.TIF$', f, re.I)])
            b10    = sorted([f for f in sfiles if re.search(r'_B10\.TIF$', f, re.I)])
            mtl    = [f for f in sfiles if f.upper().endswith("MTL.TXT")]
            if len(bands) < 5:
                log.warning(f"Skipping Landsat {folder}/{sid}: only {len(bands)} spectral bands")
                continue
            mtl_vals = None
            if b10 and mtl:
                mtl_vals = parse_mtl(os.path.join(fp, mtl[0]))
                if mtl_vals:
                    log.info(f"  {folder}/{sid}: B10 + MTL.txt found — thermal enabled")
            scenes.setdefault(date_str, []).append((
                fp, bands, angles, sid,
                os.path.join(fp, b10[0]) if b10 else None,
                mtl_vals))
    return scenes


def match_dates(viirs, landsat):
    matched = []
    for d in sorted(landsat):
        if d in viirs:
            matched.append((d, viirs[d], landsat[d]))
        else:
            log.info(f"No VIIRS match for Landsat {d} — skipping")
    return matched

# ---------------------------------------------------------------------------
# Step 1 — VIIRS → full Alaska GeoTIFF (same as valen/)
# ---------------------------------------------------------------------------

def process_viirs(gitco, gimgo, area_def, out_path):
    if os.path.exists(out_path) and not OVERWRITE:
        log.info(f"  VIIRS exists, skip: {os.path.basename(out_path)}")
        return True

    log.info("  Loading VIIRS geo + bands + angles ...")
    lat, lon = load_viirs_geo(gitco)
    bands    = load_viirs_bands(gimgo)
    angles   = load_viirs_angles(gitco)
    all_data = {**bands, **angles}

    log.info(f"  Swath shape {lat.shape}  "
             f"lat [{np.nanmin(lat):.2f}, {np.nanmax(lat):.2f}]  "
             f"lon [{np.nanmin(lon):.2f}, {np.nanmax(lon):.2f}]")

    n_bands = len(VIIRS_BAND_ORDER)
    log.info(f"  Reprojecting {n_bands} bands → Alaska 4326 "
             f"({VIIRS_W}x{VIIRS_H}) ...")

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=VIIRS_H, width=VIIRS_W, count=n_bands,
        dtype=np.float32, crs=TARGET_CRS, transform=VIIRS_TF,
        nodata=NODATA, compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
        BIGTIFF="IF_SAFER",
    ) as dst:
        for i, name in enumerate(VIIRS_BAND_ORDER, 1):
            if name in all_data:
                arr = _reproject_viirs_band(lat, lon, all_data[name], area_def)
                arr[~np.isfinite(arr)] = NODATA
            else:
                arr = np.full((VIIRS_H, VIIRS_W), NODATA, np.float32)
            dst.write(arr, i)
            dst.update_tags(i, name=name)
            valid = int(np.sum(arr != NODATA))
            log.info(f"    {name}: {valid:,} valid px")
            del arr

    mb = os.path.getsize(out_path) / 1048576
    log.info(f"  Wrote {out_path} ({mb:.1f} MB)")
    return True

# ---------------------------------------------------------------------------
# Step 2 — Landsat → scene-extent GeoTIFF in EPSG:4326 at 30m
# ---------------------------------------------------------------------------

def process_landsat(folder, band_files, angle_files, out_path):
    if os.path.exists(out_path) and not OVERWRITE:
        log.info(f"  Landsat exists, skip: {os.path.basename(out_path)}")
        return True

    scene_tf, scene_w, scene_h = compute_landsat_scene_grid(folder, band_files)
    log.info(f"  Landsat scene grid: {scene_w}x{scene_h} px  "
             f"res={LANDSAT_RES:.7f}°")

    entries = []
    for f in band_files:
        m = re.search(r'_B(\d+)\.TIF$', f, re.I)
        entries.append((os.path.join(folder, f), f"B{m.group(1)}",
                        Resampling.lanczos))
    for f in angle_files:
        m = re.search(r'_(SAA|SZA|VAA|VZA)\.TIF$', f, re.I)
        entries.append((os.path.join(folder, f), m.group(1),
                        Resampling.nearest))

    n_bands = len(entries)
    log.info(f"  Reprojecting {n_bands} bands → 4326 scene extent ...")

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=scene_h, width=scene_w, count=n_bands,
        dtype=np.float32, crs=TARGET_CRS, transform=scene_tf,
        nodata=NODATA, compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
        BIGTIFF="YES",
    ) as dst_ds:
        for i, (path, name, resamp) in enumerate(entries, 1):
            with rasterio.open(path) as src:
                src_nodata = src.nodata
                out_arr = np.full((scene_h, scene_w), np.nan, np.float32)
                rio_reproject(
                    source=rasterio.band(src, 1),
                    destination=out_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=scene_tf,
                    dst_crs=TARGET_CRS,
                    dst_nodata=np.nan,
                    resampling=resamp,
                )
                out_arr[~np.isfinite(out_arr)] = NODATA
                dst_ds.write(out_arr, i)
                dst_ds.update_tags(i, name=name)
                valid = int(np.sum(out_arr != NODATA))
                log.info(f"    {name} ({resamp.name}): {valid:,} valid px")
                del out_arr

    mb = os.path.getsize(out_path) / 1048576
    log.info(f"  Wrote {out_path} ({mb:.1f} MB)")
    return True

# ---------------------------------------------------------------------------
# Step 3 — Validate alignment
# ---------------------------------------------------------------------------

def validate_alignment(viirs_path, landsat_path, date_str):
    with rasterio.open(viirs_path) as v, rasterio.open(landsat_path) as l:
        vc, lc = str(v.crs), str(l.crs)
        vb, lb = v.bounds, l.bounds
        v_res  = v.res
        l_res  = l.res

    ok = True
    if vc != lc:
        log.warning(f"[{date_str}] CRS mismatch: VIIRS={vc} Landsat={lc}")
        ok = False

    # Landsat is scene-extent (subset of VIIRS Alaska grid), so we check that
    # Landsat falls within VIIRS bounds and pixel grids are aligned
    tol = LANDSAT_RES * 0.01
    if lb.left < vb.left - tol or lb.right > vb.right + tol:
        log.warning(f"[{date_str}] Landsat lon outside VIIRS: "
                    f"LS=[{lb.left:.4f},{lb.right:.4f}] "
                    f"VIIRS=[{vb.left:.4f},{vb.right:.4f}]")
        ok = False
    if lb.bottom < vb.bottom - tol or lb.top > vb.top + tol:
        log.warning(f"[{date_str}] Landsat lat outside VIIRS: "
                    f"LS=[{lb.bottom:.4f},{lb.top:.4f}] "
                    f"VIIRS=[{vb.bottom:.4f},{vb.top:.4f}]")
        ok = False

    # Check pixel grid alignment: Landsat pixel edges should fall on
    # integer multiples of LANDSAT_RES from the Alaska domain origin
    lon_offset = (lb.left - ALASKA_LON_MIN) / LANDSAT_RES
    lat_offset = (ALASKA_LAT_MAX - lb.top) / LANDSAT_RES
    if abs(lon_offset - round(lon_offset)) > 0.001:
        log.warning(f"[{date_str}] Landsat lon grid not aligned to Alaska origin "
                    f"(offset={lon_offset:.4f} px)")
        ok = False
    if abs(lat_offset - round(lat_offset)) > 0.001:
        log.warning(f"[{date_str}] Landsat lat grid not aligned to Alaska origin "
                    f"(offset={lat_offset:.4f} px)")
        ok = False

    if ok:
        log.info(f"[{date_str}] Alignment OK — CRS={vc}, "
                 f"Landsat inside VIIRS, grids aligned")
    return ok

# ---------------------------------------------------------------------------
# Step 4 — Sample extraction
# ---------------------------------------------------------------------------

def extract_sample(src_path, out_path, tile_px):
    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        r0 = max(0, (h - tile_px) // 2)
        c0 = max(0, (w - tile_px) // 2)
        rh = min(tile_px, h - r0)
        cw = min(tile_px, w - c0)
        win = rasterio.windows.Window(c0, r0, cw, rh)
        data = src.read(window=win)
        profile = src.profile.copy()
        profile.update(height=rh, width=cw,
                       transform=src.window_transform(win))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    log.info(f"  Sample {rh}x{cw} px: {os.path.basename(out_path)}")

# ---------------------------------------------------------------------------
# Step 5 — CSV: candidate pixels + values  (same as valen/ lines 440-567)
# ---------------------------------------------------------------------------

N_CANDIDATES = 25   # same as valen/


def _normalize(arr, pct_lo=2, pct_hi=98):
    """Percentile stretch for display — same as valen/."""
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(fin, pct_lo), np.nanpercentile(fin, pct_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

def generate_csv(viirs_path, landsat_path, date_str, out_dir,
                  b10_path=None, mtl_vals=None):
    """Sample narrow-river candidate pixels from reprojected VIIRS + Landsat.
    Same logic as valen/extract_training_pixels.main(), but reads Landsat from
    local TIF instead of GEE.  If b10_path + mtl_vals are provided, computes
    brightness temperature for ice classification (same as valen/)."""

    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    # Derive suffixes so multi-scene/granule dates don't overwrite each other
    ls_base = os.path.basename(landsat_path).replace(".tif", "")
    ls_suffix = ls_base.replace(f"landsat_{date_str}", "")
    v_base = os.path.basename(viirs_path).replace(".tif", "")
    v_suffix = v_base.replace(f"viirs_alaska_{date_str}", "")

    tag = ""
    if v_suffix:
        tag += f"_v{v_suffix.lstrip('_')}"
    if ls_suffix:
        tag += f"_ls{ls_suffix.lstrip('_')}"

    csv_path = os.path.join(out_dir, f"training_candidates_{date_fmt}{tag}.csv")
    png_path = os.path.join(out_dir, f"viirs_vs_watermask_{date_fmt}{tag}.png")
    if (os.path.exists(csv_path) and os.path.exists(png_path)
            and not OVERWRITE):
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

    # Extract Landsat scene ID from output filename or band tag
    ls_scene_id = os.path.basename(landsat_path).replace(".tif", "")

    scene_lon_min, scene_lat_min = ls_bounds.left, ls_bounds.bottom
    scene_lon_max, scene_lat_max = ls_bounds.right, ls_bounds.top

    # Build a scene-level VIIRS grid matching valen/ (SHARED_RES = VIIRS_RES)
    scene_w = int(math.ceil((scene_lon_max - scene_lon_min) / VIIRS_RES))
    scene_h = int(math.ceil((scene_lat_max - scene_lat_min) / VIIRS_RES))
    scene_tf = from_bounds(scene_lon_min, scene_lat_min,
                           scene_lon_max, scene_lat_max,
                           scene_w, scene_h)

    log.info(f"  Scene grid: {scene_w} x {scene_h} px  res: {VIIRS_RES:.6f}\u00b0")

    # Read VIIRS bands windowed to scene extent, warp to scene grid
    # (same as valen/_load_viirs_from_tif)
    log.info(f"  Loading VIIRS from pre-computed TIF: {viirs_path}")
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

    log.info(f"  VIIRS bands on scene grid (loaded from TIF)")
    for name, arr in viirs_grids.items():
        fin = int(np.sum(np.isfinite(arr)))
        log.info(f"    {name}: {fin:,} valid pixels")

    # Read both water masks (occurrence + seasonality) windowed to scene extent
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

    log.info(f"  Loading occurrence mask (scene window)...")
    wm_occ = _load_wm(WATER_MASK_OCC)
    log.info(f"    occ shape: {wm_occ.shape}  valid: {int(np.sum(np.isfinite(wm_occ))):,}")
    log.info(f"  Loading seasonality mask (scene window)...")
    wm_sea = _load_wm(WATER_MASK_SEA)
    log.info(f"    sea shape: {wm_sea.shape}  valid: {int(np.sum(np.isfinite(wm_sea))):,}")
    wm = wm_occ  # used by the "JRC water" panel

    # Build Landsat valid mask on scene grid — any band with real data
    # (excludes the dead-zone triangles outside the tilted swath)
    log.info(f"  Loading Landsat valid mask (scene grid)...")
    ls_valid = np.zeros((scene_h, scene_w), dtype=bool)
    with rasterio.open(landsat_path) as ls_src:
        for bi in range(1, ls_src.count + 1):
            band = ls_src.read(bi, out_shape=(scene_h, scene_w),
                               resampling=Resampling.nearest).astype(np.float32)
            ls_valid |= (band != NODATA) & np.isfinite(band) & (band != 0)
    log.info(f"    Landsat valid on scene grid: {int(ls_valid.sum()):,} px")

    # Find mixed-pixel candidates: 0.05 < occ < 0.90 AND sea < 1.0
    narrow = (np.isfinite(wm_occ) & (wm_occ > 0.05) & (wm_occ < 0.90) &
              np.isfinite(wm_sea) & (wm_sea < 1.0) &
              np.isfinite(viirs_grids["I1"]) & ls_valid)
    rows_n, cols_n = np.where(narrow)
    log.info(f"  Mixed-pixel candidates in scene: {len(rows_n):,}")

    if len(rows_n) == 0:
        log.warning(f"  No narrow river pixels — falling back to any valid covered pixel")
        fallback = (np.isfinite(viirs_grids["I1"]) & ls_valid)
        rows_n, cols_n = np.where(fallback)
        if len(rows_n) == 0:
            log.warning(f"  No valid covered pixels either — skipping CSV for {date_str}")
            return

    step = max(1, len(rows_n) // N_CANDIDATES)
    candidates = list(zip(rows_n[::step], cols_n[::step]))[:N_CANDIDATES]
    log.info(f"  Selecting {len(candidates)} candidate pixels for training table")

    # --- Sample Landsat at 30m from local TIF (replaces GEE) ---
    log.info(f"  Sampling Landsat at 30m from local TIF "
             f"({len(candidates)} points)...")
    rows_out = []
    n_skipped = 0
    n_ls_valid = 0
    with rasterio.open(landsat_path) as ls_src:
        for idx, (r, c) in enumerate(candidates):
            lat_px = scene_lat_max - (r + 0.5) * VIIRS_RES
            lon_px = scene_lon_min + (c + 0.5) * VIIRS_RES

            # Alaska shared-grid indices (same as valen/)
            r_shared = int((ALASKA_LAT_MAX - lat_px) / VIIRS_RES)
            c_shared = int((lon_px - ALASKA_LON_MIN) / VIIRS_RES)
            if not (0 <= r_shared < VIIRS_H and 0 <= c_shared < VIIRS_W):
                n_skipped += 1
                continue

            # Sample Landsat: convert lat/lon to Landsat pixel coords
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
            # VIIRS bands (I1-I5 to 4dp, angles to 3dp — same as valen/)
            for name in VIIRS_BAND_ORDER:
                row[name] = _safe_viirs(name)
            # Landsat bands
            for bname in ls_band_names:
                row[f"LS_{bname}"] = _safe_ls(bname)
            # NDSI = (B3 - B6) / (B3 + B6)  — same index as valen/
            b3 = ls_vals.get("B3")
            b6 = ls_vals.get("B6")
            if b3 is not None and b6 is not None and (b3 + b6) != 0:
                ndsi = (b3 - b6) / (b3 + b6)
                row["LS_NDSI"] = round(ndsi, 4)
            else:
                ndsi = None
                row["LS_NDSI"] = ""

            # B10 thermal → brightness temperature (Kelvin)
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

            # Ground truth — same 4-class logic as valen/ (lines 500-516)
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
    log.info(f"  {n_skipped} candidate(s) skipped — outside Alaska domain bounds")

    if not rows_out:
        log.warning(f"  All candidates outside domain — no CSV for {date_fmt}")
        return

    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    log.info(f"  Saved CSV: {csv_path} ({len(rows_out)} rows)")

    # --- Summary (same as valen/ lines 569-578) ---
    for row in rows_out:
        log.info(f"    lat={row['lat']}, lon={row['lon']}  "
                 f"row={row['row_shared_grid']}, col={row['col_shared_grid']}")

    # --- PNG: side-by-side VIIRS false color vs water mask (valen/ lines 580-623) ---
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
    panels_dir = os.path.join(out_dir, "panels")
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
    plt.savefig(os.path.join(panels_dir, f"viirs_{date_fmt}{tag}.png"),
                dpi=150, bbox_inches="tight")
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
    plt.savefig(os.path.join(panels_dir, f"watermask_{date_fmt}{tag}.png"),
                dpi=150, bbox_inches="tight")
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
    plt.savefig(os.path.join(panels_dir, f"landsat_{date_fmt}{tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved 3 panel PNGs to: {panels_dir}")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Landsat + VIIRS Reprojection Pipeline — Alaska AOI")
    log.info("Same specs as valen/ — no GEE")
    log.info("=" * 60)

    log.info(f"CRS:        {TARGET_CRS}")
    log.info(f"Domain:     lat [{ALASKA_LAT_MIN}, {ALASKA_LAT_MAX}]  "
             f"lon [{ALASKA_LON_MIN}, {ALASKA_LON_MAX}]")
    log.info(f"VIIRS:      {VIIRS_W} x {VIIRS_H}  res={VIIRS_RES:.6f}° ({VIIRS_RES_M}m)")
    log.info(f"Landsat:    scene-extent  res={LANDSAT_RES:.7f}° ({LANDSAT_RES_M}m)")
    log.info(f"Resampling: nearest (r={RADIUS_OF_INFLUENCE}, eps={EPSILON})")
    log.info(f"Nodata:     {NODATA}")

    viirs_area = pr_geometry.AreaDefinition(
        "alaska_4326", "Alaska EPSG:4326", "alaska_4326",
        {"proj": "longlat", "datum": "WGS84"},
        VIIRS_W, VIIRS_H,
        (ALASKA_LON_MIN, ALASKA_LAT_MIN, ALASKA_LON_MAX, ALASKA_LAT_MAX),
    )

    viirs_map   = discover_viirs()
    landsat_map = discover_landsat()
    matched     = match_dates(viirs_map, landsat_map)

    log.info(f"VIIRS dates: {len(viirs_map)}, Landsat dates: {len(landsat_map)}, "
             f"matched: {len(matched)}")
    if not matched:
        log.error("No matched date pairs — nothing to do.")
        return

    for date_str, v_list, l_list in matched:
        log.info(f"\n{'='*60}")
        log.info(f"Date {date_str}  |  "
                 f"{len(v_list)} VIIRS, {len(l_list)} Landsat")

        date_out   = os.path.join(OUTPUT_DIR, date_str)
        sample_out = os.path.join(date_out, "samples")
        os.makedirs(sample_out, exist_ok=True)

        # Step 1 — VIIRS (naming matches valen/: viirs_alaska_YYYYMMDD.tif)
        viirs_outs = []
        for gi, (_, gitco, gimgo, fname) in enumerate(v_list):
            suffix = "" if len(v_list) == 1 else f"_{gi}"
            op = os.path.join(date_out, f"viirs_alaska_{date_str}{suffix}.tif")
            if process_viirs(gitco, gimgo, viirs_area, op):
                viirs_outs.append(op)

        # Step 2 — Landsat (only if VIIRS has coverage over scene footprint)
        landsat_outs = []
        for li, (fp, bfiles, afiles, fname, b10_path, mtl_vals) in enumerate(l_list):
            suffix = "" if len(l_list) == 1 else f"_{li}"
            op = os.path.join(date_out, f"landsat_{date_str}{suffix}.tif")

            # Check VIIRS coverage over this Landsat scene before reprojecting
            skip = False
            if viirs_outs and not (os.path.exists(op) and not OVERWRITE):
                scene_tf_chk, sw_chk, sh_chk = compute_landsat_scene_grid(fp, bfiles)
                scene_bounds = rasterio.transform.array_bounds(sh_chk, sw_chk, scene_tf_chk)
                for vp in viirs_outs:
                    with rasterio.open(vp) as v_src:
                        v_nodata = v_src.nodata if v_src.nodata is not None else NODATA
                        try:
                            win = window_from_bounds(
                                scene_bounds[0], scene_bounds[1],
                                scene_bounds[2], scene_bounds[3],
                                transform=v_src.transform)
                            patch = v_src.read(1, window=win)
                            n_valid = int(np.sum(patch != v_nodata))
                        except Exception:
                            n_valid = 0
                    pct = n_valid / max(1, sw_chk * sh_chk) * 100
                    log.info(f"  VIIRS coverage over Landsat {fname}: "
                             f"{n_valid:,} valid px ({pct:.1f}%)")
                    if n_valid == 0:
                        log.warning(f"  NO VIIRS data over Landsat {fname} — "
                                    f"skipping this scene")
                        skip = True

            if skip:
                continue
            if process_landsat(fp, bfiles, afiles, op):
                landsat_outs.append((op, b10_path, mtl_vals))

        # Step 3 — Validate
        for vp in viirs_outs:
            for lp, _, _ in landsat_outs:
                validate_alignment(vp, lp, date_str)

        # Step 4 — Samples
        for vp in viirs_outs:
            sn = os.path.basename(vp).replace(".tif", "_sample.tif")
            extract_sample(vp, os.path.join(sample_out, sn), SAMPLE_TILE_PX)
        for lp, _, _ in landsat_outs:
            sn = os.path.basename(lp).replace(".tif", "_sample.tif")
            extract_sample(lp, os.path.join(sample_out, sn), SAMPLE_TILE_PX)

        # Step 5 — CSV (candidate pixels, same as valen/)
        for vp in viirs_outs:
            for lp, lp_b10, lp_mtl in landsat_outs:
                generate_csv(vp, lp, date_str, date_out, lp_b10, lp_mtl)

    log.info(f"\n{'='*60}")
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
