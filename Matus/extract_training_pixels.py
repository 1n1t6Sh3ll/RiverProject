"""
Step 2 of training data workflow:

1. Load VIIRS pair via viirs_training_loader.
2. Resample VIIRS directly to a scene-level EPSG:4326 grid (Landsat bbox).
3. Load water mask clipped to the same scene window.
4. Find narrow-river pixels (0 < wf < 1) inside the scene.
5. For each candidate pixel: extract I1-I5 band values + SZA/SAA/VZA/VAA.
6. Save side-by-side PNG: VIIRS I2-I2-I1 false color vs water mask.
7. Save CSV with candidate pixel coordinates and VIIRS values.
   → User visually classifies each pixel in GEE (ground truth for table).

Run from project root:
    python extract_training_pixels.py --date 2024-03-14
"""

import os, sys, math, csv
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject as _warp, Resampling
from rasterio.windows import from_bounds as window_from_bounds
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from viirs_training_loader import load_viirs_training_pair

# ── SCENE REGISTRY ────────────────────────────────────────────────────────────

SCENES = {
    "2022-10-23": {
        "gitco":         "viirs_data/GITCO_j01_d20221023_t2053479_e2055106_b25545_c20221023210616118927_oeac_ops.h5",
        "gimgo":         "viirs_data/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_j01_d20221023_t2053479_e2055106_b25545_stitched.h5",
        "landsat_scene": "LC90710162022296LGN01",
        "landsat_bbox":  (61.733758606360816, 63.95661083361867, -154.46161174807895, -149.68068983997256),
        "output_dir":    "20221023",
        "ee_collection": "LANDSAT/LC09/C02/T1_L2",
        "viirs_tif":     "20221023/viirs_alaska_20221023.tif",
    },
    "2024-04-27": {
        "gitco":         "viirs_data/GITCO_j02_d20240427_t2028217_e2029464_b07584_c20240427204149474000_oebc_ops.h5",
        "gimgo":         "viirs_data/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_j02_d20240427_t2028217_e2029464_b07584_stitched.h5",
        "landsat_scene": "LC80710142024118LGN00",
        "landsat_bbox":  (64.45474013450857, 66.70519256447828, -152.42881658598267, -147.0838276763541),
        "output_dir":    "20240427",
        "ee_collection": "LANDSAT/LC08/C02/T1_L2",
        "viirs_tif":     "20240427/viirs_alaska_20240427.tif",
    },
    "2025-04-20": {
        "gitco":         "viirs_data/GITCO_npp_d20250420_t2221517_e2223159_b69853_c20250421000632694000_oeac_ops.h5",
        "gimgo":         "viirs_data/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_npp_d20250420_t2221517_e2223159_b69853_stitched.h5",
        "landsat_scene": "LC90730162025110LGN00",
        "landsat_bbox":  (61.73524864911403, 63.95595546251622, -157.61019607961842, -152.82892400172676),
        "output_dir":    "20250420",
        "ee_collection": "LANDSAT/LC09/C02/T1_L2",
        "viirs_tif":     "20250420/viirs_alaska_20250420.tif",
    },
}

OCC_MASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alaska_occ_375m.tif")
SEA_MASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alaska_sea_375m.tif")
# ── Full Alaska domain (fixed, never scene-specific) ──────────────────────────

ALASKA_LAT_MIN = 54.0
ALASKA_LAT_MAX = 72.0   # do not redefine
ALASKA_LON_MIN = -171.0  # do not redefine
ALASKA_LON_MAX = -129.0
ALASKA_W = 12432
ALASKA_H = 5328
ALASKA_TF = from_bounds(ALASKA_LON_MIN, ALASKA_LAT_MIN,
                         ALASKA_LON_MAX, ALASKA_LAT_MAX,
                         ALASKA_W, ALASKA_H)

RESOLUTION_M = 375.0
SHARED_RES   = RESOLUTION_M / 111_000.0   # degrees per pixel ≈ 0.003378°

# Scene-level grid vars — set as globals inside main() for the requested date
# so that _resample_band_to_scene() can reference them without modification.
SCENE_LAT_MIN = None
SCENE_LAT_MAX = None
SCENE_LON_MIN = None
SCENE_LON_MAX = None
SCENE_W = None
SCENE_H = None
SCENE_TF = None

# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr, pct_lo=2, pct_hi=98):
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(fin, pct_lo), np.nanpercentile(fin, pct_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def _resample_band_to_scene(lat_swath, lon_swath, data):
    """
    Directly resample one VIIRS swath band to the scene EPSG:4326 grid
    using pyresample kd_tree (memory-efficient, no UTM intermediate step).
    """
    from pyresample import geometry, kd_tree

    valid = np.isfinite(lat_swath) & np.isfinite(lon_swath) & np.isfinite(data)
    lats_m = np.ma.masked_array(lat_swath, mask=~valid)
    lons_m = np.ma.masked_array(lon_swath, mask=~valid)
    data_m = np.ma.masked_array(data.astype(np.float32), mask=~valid)

    swath_def = geometry.SwathDefinition(lons=lons_m, lats=lats_m)
    area_def  = geometry.AreaDefinition(
        "scene_4326", "Scene EPSG:4326", "scene_4326",
        {"proj": "longlat", "datum": "WGS84"},
        SCENE_W, SCENE_H,
        (SCENE_LON_MIN, SCENE_LAT_MIN, SCENE_LON_MAX, SCENE_LAT_MAX),
    )
    result = kd_tree.resample_nearest(
        swath_def, data_m, area_def,
        radius_of_influence=7500,
        epsilon=0.5,
        fill_value=np.nan,
    )
    arr = np.ma.filled(result, np.nan) if np.ma.is_masked(result) else np.asarray(result)
    return np.asarray(arr, dtype=np.float32)


def _resample_band_to_alaska(lat_swath, lon_swath, data):
    """
    Resample one VIIRS swath band to the full Alaska EPSG:4326 grid.
    Mirrors _resample_band_to_scene() exactly — only AreaDefinition changes.
    """
    from pyresample import geometry, kd_tree

    valid = np.isfinite(lat_swath) & np.isfinite(lon_swath) & np.isfinite(data)
    lats_m = np.ma.masked_array(lat_swath, mask=~valid)
    lons_m = np.ma.masked_array(lon_swath, mask=~valid)
    data_m = np.ma.masked_array(data.astype(np.float32), mask=~valid)

    swath_def = geometry.SwathDefinition(lons=lons_m, lats=lats_m)
    area_def = geometry.AreaDefinition(
        "alaska_4326", "Alaska EPSG:4326", "alaska_4326",
        {"proj": "longlat", "datum": "WGS84"},
        ALASKA_W, ALASKA_H,
        (ALASKA_LON_MIN, ALASKA_LAT_MIN, ALASKA_LON_MAX, ALASKA_LAT_MAX),
    )
    result = kd_tree.resample_nearest(
        swath_def, data_m, area_def,
        radius_of_influence=7500,
        epsilon=0.5,
        fill_value=np.nan,
    )
    arr = np.ma.filled(result, np.nan) if np.ma.is_masked(result) else np.asarray(result)
    return np.asarray(arr, dtype=np.float32)


def _sample_landsat_ee(scene_id, ee_collection, candidates_latlon):
    """
    Sample Landsat Collection 2 L2 at 30m at each candidate point
    using Earth Engine Python API.
    Returns dict: row_idx -> {SR_B2, SR_B3, SR_B4, SR_B5, SR_B6,
                               ST_B10, NDSI, NDWI, landsat_valid}
    candidates_latlon: list of (row_idx, lat, lon)
    """
    try:
        import ee
        ee.Initialize()
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
          .filter(ee.Filter.eq('LANDSAT_SCENE_ID', scene_id))
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


def _load_viirs_from_tif(tif_path, scene_lon_min, scene_lat_min,
                          scene_lon_max, scene_lat_max,
                          scene_w, scene_h, scene_tf):
    """
    Read VIIRS bands from a pre-computed Alaska domain GeoTIFF and
    clip/reproject to the scene grid.
    Band order in TIF: I1, I2, I3, I4, I5, SZA, SAA, VZA, VAA (1-9).
    Returns: lat (None), lon (None), bands dict, angles dict, meta dict
    The lat/lon return values are None because swath coords are not
    needed when reading from a TIF.
    """
    BAND_NAMES = ['I1', 'I2', 'I3', 'I4', 'I5', 'SZA', 'SAA', 'VZA', 'VAA']
    bands  = {}
    angles = {}

    with rasterio.open(tif_path) as src:
        win = window_from_bounds(
            scene_lon_min, scene_lat_min,
            scene_lon_max, scene_lat_max,
            transform=src.transform)
        src_tf = src.window_transform(win)
        nodata = src.nodata if src.nodata is not None else -9999.0

        for band_idx, name in enumerate(BAND_NAMES, start=1):
            raw = src.read(band_idx, window=win).astype(np.float32)
            raw[raw == nodata] = np.nan

            out = np.full((scene_h, scene_w), np.nan, dtype=np.float32)
            _warp(
                source=raw, destination=out,
                src_transform=src_tf, src_crs="EPSG:4326",
                src_nodata=np.nan,
                dst_transform=scene_tf, dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            if name in ('I1', 'I2', 'I3', 'I4', 'I5'):
                bands[name] = out
            else:
                angles[name] = out

    # Build minimal meta dict matching load_viirs_training_pair output
    date_str = os.path.basename(tif_path)[13:21]  # viirs_alaska_YYYYMMDD
    meta = {'date': date_str, 'source': 'tif'}
    fin = np.sum(np.isfinite(bands.get('I1', np.array([]))))
    print(f"  Loaded from TIF: {fin:,} valid I1 pixels on scene grid")
    return None, None, bands, angles, meta


# ── main ──────────────────────────────────────────────────────────────────────

def main(date: str, region: str | None = None):
    global SCENE_LAT_MIN, SCENE_LAT_MAX, SCENE_LON_MIN, SCENE_LON_MAX
    global SCENE_W, SCENE_H, SCENE_TF

    try:
        from pyresample import geometry, kd_tree
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Ensure pyresample is installed.")
        sys.exit(1)

    scene         = SCENES[date]
    LANDSAT_SCENE = scene["landsat_scene"]
    LANDSAT_BBOX  = scene["landsat_bbox"]
    OUTPUT_DIR    = os.path.join(PROJECT_ROOT, scene["output_dir"])
    if region is not None:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, region)
    LANDSAT_DATE  = date
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SCENE_LAT_MIN = LANDSAT_BBOX[0]
    SCENE_LAT_MAX = LANDSAT_BBOX[1]
    SCENE_LON_MIN = LANDSAT_BBOX[2]
    SCENE_LON_MAX = LANDSAT_BBOX[3]
    SCENE_W = int(math.ceil((SCENE_LON_MAX - SCENE_LON_MIN) / SHARED_RES))
    SCENE_H = int(math.ceil((SCENE_LAT_MAX - SCENE_LAT_MIN) / SHARED_RES))
    SCENE_TF = from_bounds(SCENE_LON_MIN, SCENE_LAT_MIN,
                            SCENE_LON_MAX, SCENE_LAT_MAX,
                            SCENE_W, SCENE_H)

    sub_bbox = None
    if region is not None:
        lat_span = (SCENE_LAT_MAX - SCENE_LAT_MIN) / 3.0
        lon_span = (SCENE_LON_MAX - SCENE_LON_MIN) / 3.0
        if len(region) == 1:
            # Full-row mode: T/M/B spans the entire longitude range.
            row_idx = {"T": 0, "M": 1, "B": 2}[region]
            sub_lat_max = SCENE_LAT_MAX - row_idx * lat_span
            sub_lat_min = sub_lat_max - lat_span
            sub_lon_min = SCENE_LON_MIN
            sub_lon_max = SCENE_LON_MAX
        else:
            row_idx = {"T": 0, "M": 1, "B": 2}[region[0]]
            col_idx = {"L": 0, "M": 1, "R": 2}[region[1]]
            sub_lat_max = SCENE_LAT_MAX - row_idx * lat_span
            sub_lat_min = sub_lat_max - lat_span
            sub_lon_min = SCENE_LON_MIN + col_idx * lon_span
            sub_lon_max = sub_lon_min + lon_span
        sub_bbox = (sub_lat_min, sub_lat_max, sub_lon_min, sub_lon_max)
        print(f"[region] {region}: lat {sub_lat_min:.4f}–{sub_lat_max:.4f}  "
              f"lon {sub_lon_min:.4f}–{sub_lon_max:.4f}")

    print("=" * 60)
    print("VIIRS Training Pixel Extractor")
    print(f"Landsat scene: {LANDSAT_SCENE}  date: {LANDSAT_DATE}")
    print(f"Scene grid: {SCENE_W} × {SCENE_H} px  res: {SHARED_RES:.6f}°")
    print("=" * 60)

    # 1. Load VIIRS data — from pre-computed TIF if available,
    #    otherwise from H5 files
    VIIRS_TIF = scene.get("viirs_tif")
    tif_abs   = os.path.join(PROJECT_ROOT, VIIRS_TIF) if VIIRS_TIF else None

    if tif_abs and os.path.exists(tif_abs):
        print(f"\nLoading VIIRS from pre-computed TIF: {tif_abs}")
        lat, lon, bands, angles, meta = _load_viirs_from_tif(
            tif_abs,
            SCENE_LON_MIN, SCENE_LAT_MIN,
            SCENE_LON_MAX, SCENE_LAT_MAX,
            SCENE_W, SCENE_H, SCENE_TF)
    else:
        # Fall back to H5 loading
        gitco = scene.get("gitco")
        gimgo = scene.get("gimgo")
        if not gitco or not gimgo:
            print("ERROR: VIIRS H5 files not yet available for this "
                  "scene and no pre-computed TIF found. Exiting.")
            sys.exit(1)
        GITCO_PATH = os.path.join(PROJECT_ROOT, gitco)
        GIMGO_PATH = os.path.join(PROJECT_ROOT, gimgo)
        buf = 0.5
        clip_bbox = (
            LANDSAT_BBOX[0] - buf, LANDSAT_BBOX[1] + buf,
            LANDSAT_BBOX[2] - buf, LANDSAT_BBOX[3] + buf,
        )
        lat, lon, bands, angles, meta = load_viirs_training_pair(
            GITCO_PATH, GIMGO_PATH, clip_bbox=clip_bbox)

    # 2. Place VIIRS bands onto scene grid
    if lat is None:
        # Data already on scene grid — loaded from TIF
        print(f"\nVIIRS bands already on scene grid (loaded from TIF)")
        grids = {**bands, **angles}
        for name, arr in grids.items():
            fin = np.sum(np.isfinite(arr))
            print(f"  {name}: {fin:,} valid pixels")
    else:
        # Resample from swath (H5 path)
        print(f"\nResampling to scene grid ({SCENE_W}×{SCENE_H} px)...")
        grids = {}
        for name, data in {**bands, **angles}.items():
            grids[name] = _resample_band_to_scene(lat, lon, data)
            fin = np.sum(np.isfinite(grids[name]))
            print(f"  {name}: {fin:,} valid pixels")

    # 2b. Write full Alaska domain GeoTIFF (9-band, float32, deflate)
    #     Only when loading from H5 — skip if TIF already exists
    if lat is not None:
        BAND_ORDER = ["I1", "I2", "I3", "I4", "I5", "SZA", "SAA", "VZA", "VAA"]
        NODATA_VAL = -9999.0
        tif_path = os.path.join(OUTPUT_DIR, f"viirs_alaska_{LANDSAT_DATE.replace('-', '')}.tif")

        print(f"\nWriting Alaska VIIRS GeoTIFF: {tif_path}")
        print(f"  Grid: {ALASKA_W} × {ALASKA_H} | 9 bands | float32 | deflate")

        with rasterio.open(
            tif_path, "w",
            driver="GTiff",
            height=ALASKA_H, width=ALASKA_W,
            count=9,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=ALASKA_TF,
            nodata=NODATA_VAL,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            src_data = {**bands, **angles}
            for band_idx, name in enumerate(BAND_ORDER, start=1):
                print(f"  Resampling band {band_idx}/9: {name} ...", flush=True)
                alaska_band = _resample_band_to_alaska(lat, lon, src_data[name])
                alaska_band[~np.isfinite(alaska_band)] = NODATA_VAL
                dst.write(alaska_band, band_idx)
                dst.update_tags(band_idx, name=name)
                del alaska_band  # free ~250 MB before next band
                print(f"  band {band_idx} written")

        file_mb = os.path.getsize(tif_path) / (1024 * 1024)
        print(f"  Done. File saved: {tif_path}  ({file_mb:.1f} MB)")
        print(f"  -> H5 source files can now be deleted to reclaim disk space")
    else:
        print(f"\nSkipping Alaska TIF write — already exists: {tif_abs}")

    # 3. Load water masks — occurrence AND seasonality, clipped to scene bbox
    print("\nLoading JRC occurrence mask (scene window)...")
    with rasterio.open(OCC_MASK_PATH) as src:
        win = window_from_bounds(
            SCENE_LON_MIN, SCENE_LAT_MIN,
            SCENE_LON_MAX, SCENE_LAT_MAX,
            transform=src.transform)
        occ_raw = src.read(1, window=win).astype(np.float32)
        occ_src_tf = src.window_transform(win)
        if src.nodata is not None:
            occ_raw[occ_raw == src.nodata] = np.nan

    # Warp occurrence mask to exact scene grid
    wm = np.full((SCENE_H, SCENE_W), np.nan, dtype=np.float32)
    _warp(
        source=occ_raw, destination=wm,
        src_transform=occ_src_tf, src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=SCENE_TF, dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    print(f"  Occurrence shape: {wm.shape}  valid: {np.sum(np.isfinite(wm)):,}")

    print("Loading JRC seasonality mask (scene window)...")
    with rasterio.open(SEA_MASK_PATH) as src:
        win_s = window_from_bounds(
            SCENE_LON_MIN, SCENE_LAT_MIN,
            SCENE_LON_MAX, SCENE_LAT_MAX,
            transform=src.transform)
        sea_raw = src.read(1, window=win_s).astype(np.float32)
        sea_src_tf = src.window_transform(win_s)
        if src.nodata is not None:
            sea_raw[sea_raw == src.nodata] = np.nan

    # Warp seasonality mask to exact scene grid
    sea = np.full((SCENE_H, SCENE_W), np.nan, dtype=np.float32)
    _warp(
        source=sea_raw, destination=sea,
        src_transform=sea_src_tf, src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=SCENE_TF, dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    print(f"  Seasonality shape: {sea.shape}  valid: {np.sum(np.isfinite(sea)):,}")

    # 4. Find narrow river pixels using dual-mask filter:
    #    occurrence: 0.05 < occ < 0.90  (mixed land-water, not open water)
    #    seasonality: sea < 1.0         (not permanent water year-round)
    #    VIIRS I1 must be valid
    narrow = (np.isfinite(wm) & (wm > 0.05) & (wm < 0.90) &
              np.isfinite(sea) & (sea < 1.0) &
              np.isfinite(grids["I1"]))

    if sub_bbox is not None:
        sub_lat_min, sub_lat_max, sub_lon_min, sub_lon_max = sub_bbox
        cols_idx = np.arange(SCENE_W)
        rows_idx = np.arange(SCENE_H)
        pixel_lons = SCENE_LON_MIN + (cols_idx + 0.5) * SHARED_RES
        pixel_lats = SCENE_LAT_MAX - (rows_idx + 0.5) * SHARED_RES
        lon_in = (pixel_lons >= sub_lon_min) & (pixel_lons < sub_lon_max)
        lat_in = (pixel_lats >= sub_lat_min) & (pixel_lats < sub_lat_max)
        region_mask = np.outer(lat_in, lon_in)
        narrow &= region_mask

    rows_n, cols_n = np.where(narrow)
    print(f"\nNarrow river pixels (occ>0.05 & occ<0.90 & sea<1.0): {len(rows_n):,}")

    if len(rows_n) == 0:
        print("WARNING: No narrow river pixels found. Check VIIRS coverage.")
        return

    N_CANDIDATES = 25
    step = max(1, len(rows_n) // N_CANDIDATES)
    candidates = list(zip(rows_n[::step], cols_n[::step]))[:N_CANDIDATES]
    print(f"Selecting {len(candidates)} candidate pixels for training table")

    # 4b. Sample Landsat at 30m at each candidate location via EE
    EE_COLLECTION    = scene.get("ee_collection")
    LANDSAT_SCENE_ID = scene["landsat_scene"]
    ls_data = {}

    if EE_COLLECTION:
        print(f"\nSampling Landsat at 30m via Earth Engine "
              f"({len(candidates)} points)...")
        candidates_latlon = []
        for idx, (r, c) in enumerate(candidates):
            lat_px = SCENE_LAT_MAX - (r + 0.5) * SHARED_RES
            lon_px = SCENE_LON_MIN + (c + 0.5) * SHARED_RES
            candidates_latlon.append((idx, lat_px, lon_px))
        ls_data = _sample_landsat_ee(
            LANDSAT_SCENE_ID, EE_COLLECTION, candidates_latlon)
        n_valid = sum(1 for v in ls_data.values() if v.get('landsat_valid'))
        print(f"  {n_valid}/{len(candidates)} candidates inside "
              f"Landsat swath")
    else:
        print("\nNo EE collection — Landsat values will be null")

    # 5. Extract values for each candidate
    rows_out = []
    n_skipped = 0
    for idx, (r, c) in enumerate(candidates):
        lat_px = SCENE_LAT_MAX - (r + 0.5) * SHARED_RES
        lon_px = SCENE_LON_MIN + (c + 0.5) * SHARED_RES
        # Alaska shared-grid indices (canonical: top-left origin 72°N, -171°E)
        r_shared = int((ALASKA_LAT_MAX - lat_px) / SHARED_RES)
        c_shared = int((lon_px - ALASKA_LON_MIN) / SHARED_RES)
        if not (0 <= r_shared < ALASKA_H and 0 <= c_shared < ALASKA_W):
            n_skipped += 1
            continue

        # Look up EE-sampled Landsat values for this candidate
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
                auto_class = "ice_free_river_snow_land"
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

        row = {
            "viirs_date":      meta["date"][:4]+"-"+meta["date"][4:6]+"-"+meta["date"][6:],
            "landsat_scene":   LANDSAT_SCENE,
            "row_shared_grid": r_shared,
            "col_shared_grid": c_shared,
            "lat":             round(lat_px, 5),
            "lon":             round(lon_px, 5),
            "water_fraction":  round(float(wm[r, c]), 4),
            "I1":  round(float(grids["I1"][r,c]),4) if np.isfinite(grids["I1"][r,c]) else "",
            "I2":  round(float(grids["I2"][r,c]),4) if np.isfinite(grids["I2"][r,c]) else "",
            "I3":  round(float(grids["I3"][r,c]),4) if np.isfinite(grids["I3"][r,c]) else "",
            "I4":  round(float(grids["I4"][r,c]),4) if np.isfinite(grids["I4"][r,c]) else "",
            "I5":  round(float(grids["I5"][r,c]),4) if np.isfinite(grids["I5"][r,c]) else "",
            "SZA": round(float(grids["SZA"][r,c]),3) if np.isfinite(grids["SZA"][r,c]) else "",
            "SAA": round(float(grids["SAA"][r,c]),3) if np.isfinite(grids["SAA"][r,c]) else "",
            "VZA": round(float(grids["VZA"][r,c]),3) if np.isfinite(grids["VZA"][r,c]) else "",
            "VAA": round(float(grids["VAA"][r,c]),3) if np.isfinite(grids["VAA"][r,c]) else "",
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
        }
        rows_out.append(row)
    print(f"  {n_skipped} candidate(s) skipped — outside Alaska domain bounds")

    if not rows_out:
        print("WARNING: All candidates were outside the Alaska domain bounds.")
        return

    # 6. Save CSV
    csv_path = os.path.join(OUTPUT_DIR,
        f"training_candidates_{LANDSAT_DATE}.csv")
    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved CSV: {csv_path}")
    print("\nCandidate rows (spot-check row_shared_grid, col_shared_grid):")
    for row in rows_out:
        print(
            f"  lat={row['lat']}, lon={row['lon']}  "
            f"row={row['row_shared_grid']}, col={row['col_shared_grid']}"
        )

    n_confirmed = sum(1 for row in rows_out if row["ground_truth_class"])
    n_null      = sum(1 for row in rows_out if not row["ground_truth_class"])
    from collections import Counter
    class_counts = Counter(
        row["ground_truth_class"]
        for row in rows_out if row["ground_truth_class"])
    print(f"\nSummary: {n_confirmed} Landsat-confirmed / "
          f"{n_null} Landsat-null")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")

    # 7. Side-by-side PNG: VIIRS I2-I2-I1 false color vs water mask
    i2 = grids["I2"]
    i1 = grids["I1"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"VIIRS 2-2-1 false color vs JRC water fraction\n"
        f"Landsat scene {LANDSAT_SCENE}  |  VIIRS {meta['date'][:4]}-"
        f"{meta['date'][4:6]}-{meta['date'][6:]}  |  "
        f"lon {SCENE_LON_MIN:.1f}–{SCENE_LON_MAX:.1f}  "
        f"lat {SCENE_LAT_MIN:.1f}–{SCENE_LAT_MAX:.1f}",
        fontsize=10)

    rgb      = np.stack([_normalize(i2), _normalize(i2), _normalize(i1)], axis=-1)
    nan_mask = ~(np.isfinite(i1) & np.isfinite(i2))
    alpha    = np.where(nan_mask, 0.0, 1.0)
    rgba_viirs = np.dstack([rgb, alpha])

    axes[0].imshow(rgba_viirs, interpolation="nearest",
                   extent=[SCENE_LON_MIN, SCENE_LON_MAX, SCENE_LAT_MIN, SCENE_LAT_MAX],
                   aspect="auto", origin="upper")
    axes[0].set_title("VIIRS 2-2-1 (I2→R, I2→G, I1→B)\n"
                      "Ice/snow=bright | Water=dark | NaN=transparent")
    for idx, (r, c) in enumerate(candidates):
        lon_pt = SCENE_LON_MIN + (c + 0.5) * SHARED_RES
        lat_pt = SCENE_LAT_MAX - (r + 0.5) * SHARED_RES
        confirmed = ls_data.get(idx, {}).get('landsat_valid', False)
        if confirmed:
            axes[0].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        else:
            axes[0].plot(lon_pt, lat_pt, "o", color="gray", markersize=6,
                         markeredgewidth=1.5, markerfacecolor="none")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    im2 = axes[1].imshow(wm, cmap="Blues", vmin=0, vmax=1,
                          interpolation="nearest",
                          extent=[SCENE_LON_MIN, SCENE_LON_MAX, SCENE_LAT_MIN, SCENE_LAT_MAX],
                          aspect="auto", origin="upper")
    plt.colorbar(im2, ax=axes[1], fraction=0.03, label="Water fraction")
    for idx, (r, c) in enumerate(candidates):
        lon_pt = SCENE_LON_MIN + (c + 0.5) * SHARED_RES
        lat_pt = SCENE_LAT_MAX - (r + 0.5) * SHARED_RES
        confirmed = ls_data.get(idx, {}).get('landsat_valid', False)
        if confirmed:
            axes[1].plot(lon_pt, lat_pt, "rx", markersize=8, markeredgewidth=2)
        else:
            axes[1].plot(lon_pt, lat_pt, "o", color="gray", markersize=6,
                         markeredgewidth=1.5, markerfacecolor="none")
    axes[1].set_title("JRC water fraction (occurrence)\n"
                      "red × = Landsat-confirmed | gray ○ = outside swath")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR,
        f"viirs_vs_watermask_{LANDSAT_DATE}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved PNG: {png_path}")
    plt.close()

    # 8. Generate GEE inspector script for visual verification
    confirmed_rows = [r for r in rows_out if r["ground_truth_class"]]
    if confirmed_rows:
        # Determine Landsat collection number for the JS script
        ls_id = LANDSAT_SCENE
        if ls_id.startswith("LC9"):
            ls_collection_js = "LANDSAT/LC09/C02/T1_L2"
        else:
            ls_collection_js = "LANDSAT/LC08/C02/T1_L2"

        # Sentinel-2 search window: ±7 days around the scene date
        from datetime import datetime as _dt, timedelta as _td
        _d = _dt.strptime(LANDSAT_DATE, "%Y-%m-%d")
        s2_start = (_d - _td(days=7)).strftime("%Y-%m-%d")
        s2_end   = (_d + _td(days=7)).strftime("%Y-%m-%d")

        # Build candidate JS array entries
        js_candidates = []
        for i, r in enumerate(confirmed_rows, 1):
            js_candidates.append(
                f"  {{id: {i}, lat: {r['lat']}, lon: {r['lon']}, "
                f"auto_class: '{r['ground_truth_class']}', "
                f"st: {r['LS_ST_B10']}, ndsi: {r['LS_NDSI']}}}"
            )
        candidates_js = ",\n".join(js_candidates)

        gee_script = f"""/*
 * GEE Visual Inspector - {LANDSAT_DATE}
 * Landsat scene: {LANDSAT_SCENE}
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('{ls_collection_js}')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', '{LANDSAT_SCENE}'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [{SCENE_LON_MIN}, {SCENE_LAT_MIN}, {SCENE_LON_MAX}, {SCENE_LAT_MAX}]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('{s2_start}', '{s2_end}')
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

// ── Map layers ──────────────────────────────────────────────────────────────
Map.addLayer(full, {{bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}}, 'RGB (True Color)', true);
Map.addLayer(full, {{bands: ['SR_B5', 'SR_B4', 'SR_B3'], min: 0, max: 0.4}}, 'False Color NIR', false);
Map.addLayer(full, {{bands: ['ST_B10'], min: 250, max: 290, palette: ['purple','blue','cyan','yellow','red']}}, 'Thermal (ST_B10)', false);
Map.addLayer(ndsi, {{min: -0.5, max: 1.0, palette: ['brown','white','cyan']}}, 'NDSI', false);
Map.addLayer(s2, {{bands: ['B4','B3','B2'], min: 0, max: 3000}}, 'Sentinel-2 10m (nearest pass)', false);

var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
Map.addLayer(cloudMask.selfMask(), {{palette: ['red']}}, 'Cloud Mask', false);

// ── Candidate pixels ────────────────────────────────────────────────────────
var candidates = [
{candidates_js}
];

var iceSnow = [], iceNoSnow = [], freeNoSnow = [], freeSnow = [];
candidates.forEach(function(c) {{
  var feat = ee.Feature(ee.Geometry.Point([c.lon, c.lat]),
    {{'Pixel': c.id, 'Auto_Class': c.auto_class, 'ST_B10_K': c.st, 'NDSI': c.ndsi}});
  if (c.auto_class === 'ice_covered_river_snow_covered_land') iceSnow.push(feat);
  else if (c.auto_class === 'ice_covered_river_snow_free_land') iceNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_free_land') freeNoSnow.push(feat);
  else freeSnow.push(feat);
}});

if (iceSnow.length > 0)   Map.addLayer(ee.FeatureCollection(iceSnow),   {{color: 'FF0000'}}, 'ice_covered+snow_covered (' + iceSnow.length + ')', true);
if (iceNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(iceNoSnow), {{color: 'FF8800'}}, 'ice_covered+snow_free (' + iceNoSnow.length + ')', true);
if (freeNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(freeNoSnow), {{color: '00FF00'}}, 'ice_free+snow_free (' + freeNoSnow.length + ')', true);
if (freeSnow.length > 0)  Map.addLayer(ee.FeatureCollection(freeSnow),  {{color: '0088FF'}}, 'ice_free+snow (' + freeSnow.length + ')', true);

Map.setCenter({(float(confirmed_rows[0]['lon']) + float(confirmed_rows[-1]['lon'])) / 2:.1f}, {(float(confirmed_rows[0]['lat']) + float(confirmed_rows[-1]['lat'])) / 2:.1f}, 8);

print('{LANDSAT_DATE} - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {{
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
}});
"""
        gee_path = os.path.join(OUTPUT_DIR,
            f"gee_inspector_{LANDSAT_DATE.replace('-', '')}.js")
        with open(gee_path, "w", encoding="utf-8") as f:
            f.write(gee_script)
        print(f"\nSaved GEE inspector script: {gee_path}")
        print("  -> Paste into https://code.earthengine.google.com/ to verify pixels")
    else:
        print("\nNo Landsat-confirmed pixels — skipping GEE script generation")

    print("\nNext steps:")
    print("1. Paste the GEE inspector script into code.earthengine.google.com")
    print("2. Click Inspector tab, then click each marker to verify class")
    print("3. Update ground_truth_class in the CSV if needed")
    print("4. Copy confirmed rows into the shared training table")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract VIIRS training pixels for a given scene date.")
    parser.add_argument(
        "--date", required=False, default="2022-10-23",
        choices=list(SCENES.keys()),
        help="Scene date to process (YYYY-MM-DD)")
    parser.add_argument(
        "--region", required=False, default=None,
        choices=["T", "M", "B",
                 "TL", "TM", "TR", "ML", "MM", "MR", "BL", "BM", "BR"],
        help="Restrict candidates to a sub-region of the scene bbox. "
             "Single letter T/M/B = full row (all 3 columns combined). "
             "Two letters: T/M/B (north/middle/south) + L/M/R (west/middle/east). "
             "Omit to use the full scene.")
    args = parser.parse_args()
    main(args.date, region=args.region)
