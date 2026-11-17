"""
Fix the date's base mask, and create the empty layers you draw on in QGIS.

Codes follow the label convention: 1 river/water, 2 ice, 3 snow, 4 land, 9 unsure.

Two modes:
  summer (--variant)   base = a union variant from union_mask.py
                       (union 1/2/3 -> 1 water, 0 -> 4 land, 9 -> 9)
  frozen (--channel)   base = the date's RF class map + the channel rule:
                       ice or snow INSIDE the channel mask -> 2 ice,
                       ice or snow OUTSIDE it -> 3 snow ("ice on river, snow on land").
                       River and land stay as the RF says. The channel mask is a
                       final summer export (value 1 = water), a proxy for where the
                       river is.

Hand labels win in both modes: wherever the date's label shapefile has class 1-4,
the base takes that class (same rule as train_rf.py). The shapefile is only read.

The override and done layers are created ONCE. If they already exist they are
never touched -- they hold hand work. Scripts only ever read them.

Outputs (in reference_masks/<date>/):
  <date>_base_water.tif + .qml        base mask (land transparent)
  base_variant.txt                    where the base came from
  overrides/overrides_<date>.shp      empty polygon layer, field class_id
  done/done_<date>.shp                empty polygon layer: draw around finished cells

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\make_base.py --date 2024-06-22 --variant s2_ndwi010_k1
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\make_base.py --date 2023-10-04 --channel narrow_rivers\\reference_masks\\2024-06-22\\final\\2024-06-22_final_2026-11-03.tif --rf-dir rf_frozengate
"""

import os, sys, argparse, datetime
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(os.path.dirname(HERE), "patches")
sys.path.insert(0, PATCHES)
import build_patches as B          # _empty_shapefile, _labels_qml
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "classify"))
import extract_samples as EX       # label_rasters (read-only)

WATER, ICE, SNOW, LAND, UNSURE = 1, 2, 3, 4, 9
CLASSES = (WATER, ICE, SNOW, LAND)

BASE_QML = """<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
        <paletteEntry value="1" color="#0066ff" alpha="255" label="1 water"/>
        <paletteEntry value="2" color="#ff00ff" alpha="255" label="2 ice"/>
        <paletteEntry value="3" color="#00ffff" alpha="255" label="3 snow"/>
        <paletteEntry value="4" color="#8b4513" alpha="0" label="4 land"/>
        <paletteEntry value="9" color="#9e9e9e" alpha="140" label="9 RF unsure"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def empty_layer(path, what):
    if os.path.exists(path):
        print(f"  {os.path.relpath(path, HERE)} exists — left untouched (hand work)")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    B._empty_shapefile(path)
    open(path[:-4] + ".qml", "w", encoding="utf-8").write(B._labels_qml())
    print(f"  created empty {os.path.relpath(path, HERE)} ({what})")


def main(date, variant, channel, rf_dir="rf"):
    out_dir = os.path.join(HERE, date)
    os.makedirs(out_dir, exist_ok=True)
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")

    if variant:
        src = os.path.join(out_dir, f"{date}_union_{variant}.tif")
        if not os.path.exists(src):
            raise SystemExit(f"ERROR: {src} missing — run union_mask.py first")
        with rasterio.open(src) as s:
            u, profile = s.read(1), s.profile
        base = np.full(u.shape, LAND, np.uint8)
        base[np.isin(u, (1, 2, 3))] = WATER
        base[u == 9] = UNSURE
        origin = f"{variant}\nfrom {os.path.basename(src)}"
    else:
        src = os.path.join(patch, rf_dir, f"{date}_rf_class.tif")
        with rasterio.open(src) as s:
            base, profile = s.read(1).astype(np.uint8), s.profile
        with rasterio.open(channel) as s:
            assert s.transform == profile["transform"] and s.shape == base.shape, \
                "channel mask is not on this date's grid"
            chan = s.read(1) == WATER
        frozen = np.isin(base, (ICE, SNOW))
        n_flip = int((frozen & chan & (base == SNOW)).sum()
                     + (frozen & ~chan & (base == ICE)).sum())
        base[frozen & chan] = ICE
        base[frozen & ~chan] = SNOW
        print(f"  channel rule: {int(frozen.sum()):,} frozen px, {n_flip:,} switched "
              f"ice<->snow by location")
        origin = (f"rf + channel rule\nfrom {os.path.relpath(src, HERE)}, channel "
                  f"{os.path.basename(channel)}")

    # hand labels win
    y, _, _ = EX.label_rasters(os.path.join(patch, "labels", f"labels_{date}.shp"),
                               base.shape, profile["transform"])
    changed = 0
    for cls in CLASSES:
        changed += int(((y == cls) & (base != cls)).sum())
        base[y == cls] = cls
    print(f"  hand labels win: {int(np.isin(y, CLASSES).sum()):,} labelled px, "
          f"{changed:,} changed")

    path = os.path.join(out_dir, f"{date}_base_water.tif")
    with rasterio.open(path, "w", **dict(profile, dtype="uint8", nodata=None)) as d:
        d.write(base, 1)
    open(path[:-4] + ".qml", "w", encoding="utf-8").write(BASE_QML)
    with open(os.path.join(out_dir, "base_variant.txt"), "w", encoding="utf-8") as f:
        f.write(f"{origin}, written {datetime.datetime.now():%Y-%m-%d %H:%M}, "
                f"hand labels win ({changed} px changed)\n")
    n = base.size
    print(f"  {os.path.basename(path)}: " + ", ".join(
        f"{nm} {100 * (base == c).sum() / n:.2f} %" for c, nm in
        ((WATER, "water"), (ICE, "ice"), (SNOW, "snow"), (LAND, "land"), (UNSURE, "unsure"))))

    empty_layer(os.path.join(out_dir, "overrides", f"overrides_{date}.shp"),
                "class_id: 1 water, 2 ice, 3 snow, 4 land")
    empty_layer(os.path.join(out_dir, "done", f"done_{date}.shp"),
                "draw around cells you have finished; class_id can be anything")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fix the base mask and create the hand-drawn layers.")
    p.add_argument("--date", required=True)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--variant", help="summer: union variant, e.g. s2_ndwi010_k1")
    m.add_argument("--channel", help="frozen dates: path to a final summer water mask")
    p.add_argument("--rf-dir", default="rf",
                   help="frozen dates: RF output folder inside the patch (default rf)")
    a = p.parse_args()
    main(a.date, a.variant, a.channel, a.rf_dir)
