"""
Add Sentinel-2 (10 m) to a patch folder, as a digitising aid.

Landsat at 30 m makes a 70 m channel 2-3 pixels wide; Sentinel-2 at 10 m makes
it 7, so banks and ice edges can be traced instead of guessed.

Sentinel-2 rarely flies on the same day as Landsat, and the labels must describe
the LANDSAT moment (that is what pairs with VIIRS), so the time gap is reported
in the console, the file names and the README. A gap of hours is as good as
simultaneous; days are fine for banks but not for ice edges.

Grid: one third of the Landsat cell, so 3 x 3 Sentinel cells nest exactly in a
Landsat cell and 37.5 per VIIRS cell (patch_config).

Outputs (in patches/<patch>/):
  <date>_s2_truecolour_10m.tif  B4/B3/B2, 8-bit — banks, gravel, general view
  <date>_s2_swir_10m.tif        B11/B8/B4, 8-bit — snow/ice cyan, water black
  <date>_s2_ndwi_10m.tif        (B3-B8)/(B3+B8), float — sharpest water edge
  *.qml styles

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\add_sentinel.py --dates 2024-06-22
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\add_sentinel.py --dates 2024-06-22 --max-days 3
"""

import os, sys, csv, math, argparse, datetime
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C

NR_ROOT    = os.path.dirname(HERE)
EE_PROJECT = "noaa-river-ice"
S2_SUB     = 3                       # Sentinel cells per Landsat cell, per side
S2_RES     = C.LS_RES / S2_SUB       # degrees
TILE_ROWS  = 800        # keeps each computePixels request under the 50 MB limit
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def _grid():
    lg = C.landsat_grid()
    return {"x0": lg["x0"], "y1": lg["y1"], "W": lg["W"] * S2_SUB,
            "H": lg["H"] * S2_SUB, "res": S2_RES, "crs": C.CRS,
            "col0": lg["col0"] * S2_SUB, "row0": lg["row0"] * S2_SUB}


def _pick_scene(ee, date, max_days):
    """Least-cloudy Sentinel-2 acquisition within +-max_days of the Landsat date."""
    box = ee.Geometry.Rectangle(list(C.BOX))
    d = ee.Date(date)
    col = (ee.ImageCollection(COLLECTION).filterBounds(box)
           .filterDate(d.advance(-max_days, "day"), d.advance(max_days + 1, "day")))
    n = col.size().getInfo()
    if n == 0:
        raise SystemExit(f"ERROR: no Sentinel-2 within +-{max_days} days of {date}")

    # cloud+shadow over the study box, per acquisition day, from the scene
    # classification layer (SCL): 3 shadow, 8/9 cloud medium/high, 10 cirrus
    days = sorted({t[:10] for t in col.aggregate_array("DATATAKE_IDENTIFIER").getInfo()
                   and [datetime.datetime.fromtimestamp(ms / 1000, datetime.UTC).strftime("%Y-%m-%d")
                        for ms in col.aggregate_array("system:time_start").getInfo()]})
    best = None
    print(f"  Sentinel-2 acquisitions within +-{max_days} days:")
    for day in days:
        sub = col.filterDate(day, (datetime.date.fromisoformat(day) +
                                   datetime.timedelta(days=1)).isoformat())
        mosaic = sub.mosaic()
        frac = (mosaic.select("SCL").remap([3, 8, 9, 10], [1, 1, 1, 1], 0)
                .reduceRegion(ee.Reducer.mean(), box, 60, maxPixels=1e9)
                .getInfo().get("remapped"))
        if frac is None:
            continue
        t_ms = sub.aggregate_min("system:time_start").getInfo()
        t = datetime.datetime.fromtimestamp(t_ms / 1000, datetime.UTC)
        gap_h = (t - datetime.datetime.fromisoformat(date + " 21:15:00+00:00")).total_seconds() / 3600
        print(f"    {day}  {t:%H:%M} UTC  cloud+shadow over box {100*frac:5.1f} %  "
              f"({gap_h:+.0f} h vs Landsat)")
        if best is None or frac < best["frac"]:
            best = {"day": day, "frac": frac, "time": t, "gap_h": gap_h, "col": sub}
    return best


def _fetch(ee, img, bands, g, scale_divisor=None):
    out = {b: [] for b in bands}
    for r0 in range(0, g["H"], TILE_ROWS):
        h = min(TILE_ROWS, g["H"] - r0)
        arr = ee.data.computePixels({
            "expression": img.select(bands),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": g["W"], "height": h},
                     "affineTransform": {
                         "scaleX": g["res"], "shearX": 0, "translateX": g["x0"],
                         "shearY": 0, "scaleY": -g["res"],
                         "translateY": g["y1"] - r0 * g["res"]},
                     "crsCode": g["crs"]}})
        for b in bands:
            out[b].append(arr[b].astype(np.float32))
        print(f"      rows {r0}-{r0 + h} of {g['H']}")
    res = {b: np.concatenate(v, axis=0) for b, v in out.items()}
    if scale_divisor:
        res = {b: v / scale_divisor for b, v in res.items()}
    return res


def _write(path, data, g, dtype, nodata, descriptions=None):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, "w", driver="GTiff", height=g["H"], width=g["W"],
                       count=data.shape[0], dtype=dtype, crs=g["crs"],
                       transform=from_origin(g["x0"], g["y1"], g["res"], g["res"]),
                       nodata=nodata, compress="deflate", predictor=2, tiled=True) as dst:
        dst.write(data.astype(dtype))
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    print(f"    {os.path.basename(path):42} {os.path.getsize(path)/1e6:6.1f} MB")


def _scale8(band, vmax):
    return np.clip(np.nan_to_num(band, nan=0.0) / vmax * 255.0, 0, 255).astype(np.uint8)


def _rgb_qml():
    def enh(b):
        return (f'<{b}ContrastEnhancement><minValue>0</minValue><maxValue>255</maxValue>'
                f'<algorithm>StretchToMinimumMaximum</algorithm></{b}ContrastEnhancement>')
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="multibandcolor" redBand="1" greenBand="2" blueBand="3" opacity="1">\n'
            f'   {enh("red")}{enh("green")}{enh("blue")}\n'
            '  </rasterrenderer></pipe></qgis>\n')


def _ndwi_qml():
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">\n'
            '   <rastershader><colorrampshader colorRampType="INTERPOLATED" clip="0">\n'
            '     <item value="-0.6" color="#8c510a" label="-0.6 dry land"/>\n'
            '     <item value="-0.1" color="#f6e8c3" label="-0.1"/>\n'
            '     <item value="0"    color="#ffffff" label="0 (usual water edge)"/>\n'
            '     <item value="0.2"  color="#80cdc1" label="0.2"/>\n'
            '     <item value="0.6"  color="#01665e" label="0.6 open water"/>\n'
            '   </colorrampshader></rastershader>\n'
            '  </rasterrenderer></pipe></qgis>\n')


def main(dates, max_days):
    import ee
    ee.Initialize(project=EE_PROJECT)
    g = _grid()

    for date in dates:
        patch_dir = os.path.join(NR_ROOT, "patches", f"nenana_{date.replace('-', '')}")
        if not os.path.isdir(patch_dir):
            raise SystemExit(f"ERROR: no patch folder for {date}")
        print(f"\n{date}  ->  {os.path.basename(patch_dir)}")
        print("  " + C.describe(g, "Sentinel grid: "))

        best = _pick_scene(ee, date, max_days)
        print(f"  chosen: {best['day']} {best['time']:%H:%M} UTC, "
              f"{best['gap_h']:+.1f} h from Landsat, "
              f"{100*best['frac']:.1f} % cloud+shadow over the box")
        if abs(best["gap_h"]) > 48:
            print("  NOTE: more than 2 days from the Landsat scene — fine for banks, "
                  "but do not trust it for ice or snow edges")

        img = best["col"].mosaic()
        print("    reflectance (B2,B3,B4,B8,B11):")
        b = _fetch(ee, img, ["B2", "B3", "B4", "B8", "B11"], g, scale_divisor=10000.0)

        base = os.path.join(patch_dir, f"{date}_s2")
        _write(base + "_truecolour_10m.tif",
               np.stack([_scale8(b["B4"], 0.3), _scale8(b["B3"], 0.3), _scale8(b["B2"], 0.3)]),
               g, "uint8", None, ["red B4", "green B3", "blue B2"])
        _write(base + "_swir_10m.tif",
               np.stack([_scale8(b["B11"], 0.5), _scale8(b["B8"], 0.5), _scale8(b["B4"], 0.5)]),
               g, "uint8", None, ["swir B11", "nir B8", "red B4"])
        ndwi = (b["B3"] - b["B8"]) / (b["B3"] + b["B8"] + 1e-9)
        ndwi[~np.isfinite(ndwi)] = np.nan
        _write(base + "_ndwi_10m.tif", ndwi, g, "float32", np.nan, ["NDWI (B3-B8)/(B3+B8)"])

        open(base + "_truecolour_10m.qml", "w", encoding="utf-8").write(_rgb_qml())
        open(base + "_swir_10m.qml", "w", encoding="utf-8").write(_rgb_qml())
        open(base + "_ndwi_10m.qml", "w", encoding="utf-8").write(_ndwi_qml())

        # README
        readme = os.path.join(patch_dir, "README.txt")
        text = open(readme, encoding="utf-8").read().split("\nSENTINEL-2")[0].rstrip("\n")
        text += (f"\n\nSENTINEL-2 (10 m, digitising aid)\n"
                 f"  Acquired  : {best['day']} {best['time']:%H:%M} UTC, "
                 f"{best['gap_h']:+.1f} h from the Landsat scene\n"
                 f"  Cloud+shadow over the box: {100*best['frac']:.1f} %\n"
                 f"  {date}_s2_truecolour_10m.tif  B4/B3/B2 — banks and gravel\n"
                 f"  {date}_s2_swir_10m.tif        B11/B8/B4 — snow/ice cyan, water black\n"
                 f"  {date}_s2_ndwi_10m.tif        (B3-B8)/(B3+B8); water is high,\n"
                 f"                                the edge sits near 0\n"
                 f"  Grid: 3 x 3 Sentinel cells per Landsat cell, so everything stays aligned.\n"
                 f"  Trace borders on these, but remember the labels describe the LANDSAT scene.\n")
        open(readme, "w", encoding="utf-8").write(text)
        print(f"    README.txt updated")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add Sentinel-2 10 m layers to a patch.")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--max-days", type=int, default=2,
                   help="how far from the Landsat date to look (default 2)")
    a = p.parse_args()
    main(a.dates, a.max_days)
