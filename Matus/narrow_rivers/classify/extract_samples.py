"""
Turn the hand-digitised polygons of one date into a table of training samples.

One row per labelled Landsat pixel: the 6 surface-reflectance bands, the thermal
band, three indices, the class, and -- importantly -- which polygon the pixel
came from. Pixels inside one hand-drawn polygon are near-duplicates, so any
honest validation has to hold out whole polygons, and that needs the polygon id.

The label shapefiles are opened READ-ONLY. Nothing here writes a shapefile.

All 10 patches cover the identical box, so there is nothing to mosaic: the
"combine everything" step is this table, with a `date` column.

Outputs (in narrow_rivers/classify/):
  samples_<date>.csv

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\extract_samples.py --dates 2024-06-22
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\extract_samples.py --dates 2024-06-22 2023-10-04
"""

import os, sys, csv, argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
import patch_config as C
import make_fractions as MF          # reuse the .shp/.dbf readers, read-only

# QA_PIXEL bits worth distrusting: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud,
# 4 cloud shadow. Kept as a column so training can decide, not dropped here.
QA_BAD_BITS = (0, 1, 2, 3, 4)

FEATURES = ["B2", "B3", "B4", "B5", "B6", "B7", "thermal", "ndwi", "ndsi", "ndvi"]
COLUMNS = ["date", "row", "col", "poly_id", "class_id", "qa_bad"] + FEATURES


def features(sr, thermal):
    """Stack of the 10 per-pixel features, in FEATURES order."""
    b2, b3, b4, b5, b6, b7 = sr

    def norm(a, b):
        out = (a - b) / (a + b)
        out[~np.isfinite(out)] = np.nan
        return out

    ndwi = norm(b3, b5)          # green - NIR   : open water is high
    ndsi = norm(b3, b6)          # green - SWIR1 : snow and ice are high
    ndvi = norm(b5, b4)          # NIR - red     : vegetation is high
    return np.stack([b2, b3, b4, b5, b6, b7, thermal, ndwi, ndsi, ndvi])


def qa_bad_mask(path):
    """True where QA_PIXEL flags fill, cloud, dilated cloud, cirrus or shadow."""
    with rasterio.open(path) as src:
        qa = src.read(1)
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in QA_BAD_BITS:
        bad |= (qa & (1 << bit)) > 0
    return bad


def label_rasters(shp, shape, transform, class_field="class_id"):
    """(class raster, polygon-id raster, polygon table). Read-only."""
    shapes = MF._read_shp(shp)
    attrs = MF._read_dbf(shp[:-4] + ".dbf")
    if len(shapes) != len(attrs):
        raise SystemExit("ERROR: .shp and .dbf record counts differ")

    valid = set(C.CLASSES) | {C.NODATA_CLASS}
    polys = []                                  # (geom, class, polygon id)
    for g, a in zip(shapes, attrs):
        if g is None or a.get("_deleted"):
            continue
        cls = a.get(class_field)
        if cls not in valid:
            continue
        polys.append((g, int(cls), len(polys) + 1))

    if not polys:
        raise SystemExit(f"ERROR: no polygons with a valid {class_field} in {shp}")

    y = rasterize([(g, c) for g, c, _ in polys], out_shape=shape,
                  transform=transform, fill=0, dtype="uint8", all_touched=False)
    # uint16: a date could exceed 255 polygons later
    gid = rasterize([(g, i) for g, _, i in polys], out_shape=shape,
                    transform=transform, fill=0, dtype="uint16", all_touched=False)
    return y, gid, polys


def process(date):
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    sr_path = os.path.join(patch, f"{date}_sr.tif")
    if not os.path.exists(sr_path):
        raise SystemExit(f"ERROR: {sr_path} missing — run build_patches.py first")
    shp = os.path.join(patch, "labels", f"labels_{date}.shp")
    if not os.path.exists(shp):
        raise SystemExit(f"ERROR: {shp} missing")

    with rasterio.open(sr_path) as src:
        sr = src.read().astype(np.float32)
        transform, shape = src.transform, (src.height, src.width)
    with rasterio.open(os.path.join(patch, f"{date}_thermal.tif")) as src:
        thermal = src.read(1).astype(np.float32)

    print(f"\n{date}  ->  {os.path.basename(patch)}")
    print(f"  grid {shape[1]} x {shape[0]} Landsat cells")

    y, gid, polys = label_rasters(shp, shape, transform)
    bad = qa_bad_mask(os.path.join(patch, f"{date}_qa_pixel.tif"))
    feat = features(sr, thermal)

    labelled = y > 0
    usable = labelled & np.isfinite(feat).all(axis=0)
    dropped = int(labelled.sum() - usable.sum())
    if dropped:
        print(f"  {dropped} labelled pixel(s) dropped: a feature is not finite")

    rows, cols = np.nonzero(usable)
    table = feat[:, usable]

    out = os.path.join(HERE, f"samples_{date}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for k in range(rows.size):
            w.writerow([date, rows[k], cols[k], gid[rows[k], cols[k]],
                        y[rows[k], cols[k]], int(bad[rows[k], cols[k]])]
                       + [f"{v:.6g}" for v in table[:, k]])

    names = dict(C.CLASSES); names[C.NODATA_CLASS] = "no data"
    print(f"  {os.path.basename(out)}: {rows.size:,} rows, "
          f"{len(polys)} polygons, {os.path.getsize(out)/1e6:.1f} MB")
    for c in sorted(set(cls for _, cls, _ in polys)):
        m = y[usable] == c
        n_poly = sum(1 for _, k, _ in polys if k == c)
        n_bad = int(bad[usable][m].sum())
        print(f"    {names.get(c, c):8} {int(m.sum()):7,} px  {n_poly:3} polygons"
              + (f"   ({n_bad} QA-flagged)" if n_bad else ""))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hand labels -> training table.")
    p.add_argument("--dates", nargs="+", required=True)
    for d in p.parse_args().dates:
        process(d)
