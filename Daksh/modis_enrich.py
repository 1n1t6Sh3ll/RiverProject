"""
MODIS enrichment (Earth Engine) shared by the pixel-extraction scripts and
the GEE report generator (output/main.py).

For each point it samples:
    modis_ndvi      MOD13Q1 NDVI (250 m, 16-day composite, scaled to -1..1)
    modis_lc_type1  MCD12Q1 LC_Type1 IGBP class (500 m, annual)
    modis_lc_name   readable IGBP class name
"""
import logging
from datetime import datetime

log = logging.getLogger("modis_enrich")

EE_PROJECT = "noaa-river-ice"

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
