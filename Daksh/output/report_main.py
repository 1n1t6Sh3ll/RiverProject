#!/usr/bin/env python3
"""
auto_classify_points.py  (v2 — full QA cloud detail + custom band views)
=========================================================================
Automatic Landsat check of training-candidate points in Google Earth Engine.

Give it your candidates CSV (needs: lat, lon, landsat_scene;
optional: id, ground_truth_class, notes). For every point it:

  1. loads the exact Landsat scene named in `landsat_scene`
  2. samples every band, your indices, ST_B10, and the FULL Landsat QA
     cloud information (cloud, dilated cloud, cirrus, shadow, snow, water,
     clear + confidence levels) and % of each within BUFFER_M of the point
  3. classifies the point (CLASS 0-4, same rules as the GEE script)
  4. compares the result with your ground_truth_class
  5. downloads an image chip for every view in VIEWS (band combos,
     indices, thermal, QA cloud overlay, class map)

Output, next to your CSV:
  <name>_gee_results.csv   one row per point, all values
  <name>_report.html       table + chips; tick views on/off at the top

Setup (once):
    pip install earthengine-api pandas
    earthengine authenticate

Run:
    python auto_classify_points.py training_candidates_2024-11-27_ls2.csv --project YOUR_GCP_PROJECT

Options:
    --views true_color,cloud_qa   download only these views (names from VIEWS)
    --no-chips                    table only, no images (fast)
    --no-open                     don't open the report automatically
    --no-modis                    skip MODIS NDVI / IGBP land cover sampling
"""

import argparse
import base64
import html
import json
import os
import sys
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# Shared MODIS sampler lives one level up (D:\Reprojection\modis_enrich.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import modis_enrich
except ImportError:
    modis_enrich = None

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
# live Earth Engine tiles, with a band-combo picker. It needs report_map_server.py
# running locally (it talks to Earth Engine on demand); see that file's
# docstring. Bands offered in the "Custom RGB" picker, and the default
# stretch (min, max) used if you don't override it there.
# ------------------------------------------------------------
MAP_SERVER_PORT = 8765
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
        sys.exit(f"CSV is missing column(s): {', '.join(sorted(missing))}")
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
        print(f"   chip failed at {lat}, {lon}: {e}")
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
      '<div class="livemap-status bad" id="lmstatus-' + i + '">checking for report_map_server.py&hellip;</div>' +
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
      statusEl.innerHTML = 'map server not running — start it once with <code>python report_map_server.py --project YOUR_PROJECT</code> ' +
        'from the output folder, then reload this page.';
    }});
  }});
}})();
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="training candidates CSV")
    ap.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--views", help="comma-separated view names to download (overrides 'on')")
    ap.add_argument("--no-chips", action="store_true", help="skip image chips")
    ap.add_argument("--no-open", action="store_true", help="don't open the report")
    ap.add_argument("--no-modis", action="store_true",
                    help="skip MODIS NDVI / IGBP land cover sampling")
    args = ap.parse_args()

    if args.views:
        wanted = [v.strip() for v in args.views.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in VIEWS]
        if unknown:
            sys.exit(f"Unknown view(s): {', '.join(unknown)}. Available: {', '.join(VIEWS)}")
    else:
        wanted = [k for k, s in VIEWS.items() if s.get("on", True)]
    if args.no_chips:
        wanted = []

    if ee is None:
        sys.exit("earthengine-api is not installed:  pip install earthengine-api")
    try:
        ee.Initialize(project=args.project) if args.project else ee.Initialize()
    except Exception as e:
        sys.exit(f"Earth Engine init failed: {e}\n"
                 "Run  earthengine authenticate  once, and pass --project YOUR_PROJECT")

    use_modis = not args.no_modis
    if use_modis and modis_enrich is None:
        print("modis_enrich.py not found next to the project root — MODIS disabled")
        use_modis = False

    df = read_candidates(args.csv)
    print(f"Loaded {len(df)} points from {args.csv}")
    print(f"Views: {', '.join(wanted) if wanted else '(none)'}")

    results, chips, scenes_meta, used_views = [], {}, {}, []

    for scene, rows in df.groupby("landsat_scene", sort=False):
        print(f"\nScene {scene}  ({len(rows)} points)")
        try:
            toa_id, l2_id, date = scene_ids(scene)
        except ValueError as e:
            print(f"   skipped: {e}")
            continue
        if not asset_exists(toa_id):
            print(f"   skipped: {toa_id} not found in Earth Engine")
            continue
        if not asset_exists(l2_id):
            print("   Level-2 not available — ST_B10 missing, using TOA B10 for thermal")
            l2_id = None

        toa, full = build_full_image(toa_id, l2_id)
        meta = toa.toDictionary(["SUN_ELEVATION", "CLOUD_COVER", "CLOUD_COVER_LAND"]).getInfo()
        meta["date"] = date
        meta["toa_id"] = toa_id
        meta["l2_id"] = l2_id or ""
        scenes_meta[scene] = meta
        sun = meta.get("SUN_ELEVATION")
        rgb_max = 1.5 if (sun is not None and sun < 15) else 0.4
        print(f"   acquired {date}, sun {fmt(sun, 1)}°, scene cloud {fmt(meta.get('CLOUD_COVER'), 1)}%")

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
                print(f"   sampling MODIS NDVI + land cover for {len(need)} points ...")
                modis = modis_enrich.sample_modis(ee, need, date)

        for r in rows.itertuples():
            v = vals.get(r.pid, {})
            c = classify(v)
            m = {}
            if r.pid in modis:
                ndvi = modis[r.pid]["modis_ndvi"]
                lc = modis[r.pid]["modis_lc_type1"]
                m = {"modis_ndvi": ndvi,
                     "modis_lc_type1": lc,
                     "modis_lc_name": (modis_enrich.IGBP_LOOKUP.get(lc, f"Unknown({lc})")
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
                    print(f"   view '{name}' skipped (missing band {', '.join(missing)})")
                else:
                    views_here.append(name)
            for name in views_here:
                if name not in used_views:
                    used_views.append(name)

            vis = {name: make_view(full, VIEWS[name], rgb_max) for name in views_here}
            print(f"   downloading {len(views_here) * len(rows)} chips ...")
            jobs = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for r in rows.itertuples():
                    for name in views_here:
                        jobs[(r.pid, name)] = pool.submit(chip_data_uri, vis[name], r.lat, r.lon)
            for (pid, name), fut in jobs.items():
                chips.setdefault(pid, {})[name] = fut.result()

    if not results:
        sys.exit("No points processed.")

    res = pd.DataFrame(results).drop(columns=["Index"], errors="ignore")

    base = os.path.splitext(os.path.abspath(args.csv))[0]
    out_csv = base + "_gee_results.csv"
    out_html = base + "_report.html"

    res.drop(columns=["pid"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_report(res, chips, used_views, scenes_meta, out_html,
                 f"GEE check — {os.path.basename(args.csv)}")

    print("\n=== SUMMARY ===")
    for status, n in res["status"].value_counts().items():
        print(f"   {status:9s} {n}")
    for _, r in res[res["status"] != "MATCH"].iterrows():
        print(f"   id {r['id']}: {r['status']:9s} truth={r.get('ground_truth_class')}  "
              f"GEE={r['gee_label']}  QA: {r.get('qa_summary')}")
    print(f"\nResults CSV : {out_csv}")
    print(f"Report      : {out_html}")

    if not args.no_open:
        webbrowser.open("file://" + out_html)


if __name__ == "__main__":
    main()