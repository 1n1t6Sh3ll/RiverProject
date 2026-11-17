"""
Download NLCD 2016 Alaska land cover from Earth Engine onto a patch's 30 m grid.

NLCD 2016 is the newest Alaska release in Earth Engine (2019 and 2021 are
CONUS only). It is a reference layer for QGIS -- it does NOT go into the union.

Outputs (in reference_masks/<date>/):
  <date>_nlcd2016_landcover.tif   NLCD class codes (11 = open water, 0 = no data)
  <date>_nlcd2016_landcover.qml   standard NLCD colours
  <date>_nlcd2016_water.tif       1 = open water (class 11), 0 = anything else

Grid: copied from the patch's water_occurrence.tif (EPSG:4326, nearest neighbour).

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\fetch_nlcd.py --date 2024-06-22
"""

import os, argparse
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(os.path.dirname(HERE), "patches")
EE_PROJECT = "noaa-river-ice"
NLCD_ASSET = "USGS/NLCD_RELEASES/2016_REL/2016_AK"
TILE_ROWS = 700

# NLCD legend (Alaska adds 51 dwarf scrub, 72 sedge, 73 lichens, 74 moss)
NLCD = {
    11: ("Open water", "#466b9f"), 12: ("Perennial ice/snow", "#d1def8"),
    21: ("Developed, open", "#dec5c5"), 22: ("Developed, low", "#d99282"),
    23: ("Developed, medium", "#eb0000"), 24: ("Developed, high", "#ab0000"),
    31: ("Barren", "#b3ac9f"), 41: ("Deciduous forest", "#68ab5f"),
    42: ("Evergreen forest", "#1c5f2c"), 43: ("Mixed forest", "#b5c58f"),
    51: ("Dwarf scrub", "#af963c"), 52: ("Shrub/scrub", "#ccb879"),
    71: ("Grassland", "#dfdfc2"), 72: ("Sedge", "#d1d182"),
    73: ("Lichens", "#a3cc51"), 74: ("Moss", "#82ba9e"),
    81: ("Pasture", "#dcd939"), 82: ("Crops", "#ab6c28"),
    90: ("Woody wetlands", "#b8d9eb"), 95: ("Herbaceous wetlands", "#6c9fb8"),
}


def qml():
    items = "\n".join(
        f'        <paletteEntry value="{v}" color="{c}" alpha="255" label="{v} {n}"/>'
        for v, (n, c) in NLCD.items())
    return f"""<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{items}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def main(date):
    import ee
    ee.Initialize(project=EE_PROJECT)
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    with rasterio.open(os.path.join(patch, f"{date}_water_occurrence.tif")) as g:
        profile, tf, H, W = g.profile, g.transform, g.height, g.width
    out_dir = os.path.join(HERE, date)
    os.makedirs(out_dir, exist_ok=True)

    img = ee.Image(NLCD_ASSET).select("landcover").unmask(0).toUint8()
    # same tiled computePixels call as patches/build_patches.py (exact grid)
    print(f"  downloading {NLCD_ASSET} on the {W} x {H} patch grid")
    tiles = []
    for r0 in range(0, H, TILE_ROWS):
        h = min(TILE_ROWS, H - r0)
        arr = ee.data.computePixels({
            "expression": img,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": W, "height": h},
                     "affineTransform": {
                         "scaleX": tf.a, "shearX": 0, "translateX": tf.c,
                         "shearY": 0, "scaleY": tf.e,
                         "translateY": tf.f + r0 * tf.e},
                     "crsCode": "EPSG:4326"}})
        tiles.append(arr["landcover"])
    lc = np.concatenate(tiles, axis=0).astype(np.uint8)
    assert lc.shape == (H, W), f"grid mismatch {lc.shape} vs {(H, W)}"

    prof = dict(profile, dtype="uint8", count=1, nodata=0, compress="deflate")
    base = os.path.join(out_dir, f"{date}_nlcd2016")
    with rasterio.open(base + "_landcover.tif", "w", **prof) as d:
        d.write(lc, 1)
    open(base + "_landcover.qml", "w", encoding="utf-8").write(qml())
    prof_w = dict(prof, nodata=None)
    with rasterio.open(base + "_water.tif", "w", **prof_w) as d:
        d.write((lc == 11).astype(np.uint8), 1)

    vals, n = np.unique(lc, return_counts=True)
    print("  classes in the box:")
    for v, k in sorted(zip(vals, n), key=lambda t: -t[1]):
        print(f"    {v:3d} {NLCD.get(int(v), ('no data', ''))[0]:22} {k:9,} px "
              f"{100*k/lc.size:5.1f} %")
    print(f"  wrote {os.path.basename(base)}_landcover.tif / _water.tif")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--date", required=True)
    main(p.parse_args().date)
