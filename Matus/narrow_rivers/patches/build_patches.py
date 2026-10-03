"""
Build ready-to-share QGIS folders for the Nenana patches, one per Landsat date.

For each date in patch_config.SCENES:
  <patch_dir>/
    <date>_sr.tif                SR_B2..B7 reflectance (float32)
    <date>_thermal.tif           ST_B10 in kelvin
    <date>_qa_pixel.tif          QA_PIXEL (uint16, bit flags intact)
    <date>_water_occurrence.tif  JRC occurrence 0-100 from Source/EarthEngineExports
    <date>_truecolour.tif        8-bit RGB, fixed 0-0.4 scaling
    <date>_swir.tif              8-bit RGB, fixed 0-0.5 scaling
    *.qml                        styles, so nobody sets colours by hand
    labels/labels_<date>.shp     EMPTY polygon layer, EPSG:4326, field class_id
    README.txt                   scene, grid, classes, how to start

Everything is EPSG:4326 on the Alaska grid (see patch_config).

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\build_patches.py
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\build_patches.py --dates 2021-07-01
"""

import os, sys, glob, struct, argparse, datetime
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as rio_merge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C

NR_ROOT   = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(NR_ROOT))
JRC_TILES = os.path.join(REPO_ROOT, "Source", "EarthEngineExports", "*.tif")
EE_PROJECT = "noaa-river-ice"
TILE_ROWS  = 700


# ── Earth Engine ────────────────────────────────────────────────────────────

def _fetch(img, bands, g):
    import ee
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
            out[b].append(arr[b])
        print(f"      rows {r0}-{r0 + h} of {g['H']}")
    return {b: np.concatenate(v, axis=0) for b, v in out.items()}


# ── raster helpers ──────────────────────────────────────────────────────────

def _tf(g):
    return from_origin(g["x0"], g["y1"], g["res"], g["res"])


def _write(path, data, g, dtype, nodata, descriptions=None):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, "w", driver="GTiff", height=g["H"], width=g["W"],
                       count=data.shape[0], dtype=dtype, crs=g["crs"],
                       transform=_tf(g), nodata=nodata, compress="deflate",
                       predictor=2, tiled=True) as dst:
        dst.write(data.astype(dtype))
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    print(f"    {os.path.basename(path):42} {os.path.getsize(path)/1e6:6.1f} MB")


def _scale8(band, vmax):
    """Fixed scaling 0..vmax -> 0..255 (same for every band, so colours mean
    the same thing on every date)."""
    out = np.clip(band / vmax * 255.0, 0, 255)
    return np.nan_to_num(out, nan=0.0).astype(np.uint8)


def _water_mask(g):
    lon0, lat1 = g["x0"], g["y1"]
    lon1, lat0 = lon0 + g["W"] * g["res"], lat1 - g["H"] * g["res"]
    box = (lon0 - 0.02, lat0 - 0.02, lon1 + 0.02, lat1 + 0.02)
    hits = []
    for f in glob.glob(JRC_TILES):
        with rasterio.open(f) as s:
            b = s.bounds
            if b.left < box[2] and b.right > box[0] and b.bottom < box[3] and b.top > box[1]:
                hits.append(f)
    if not hits:
        print("    WARNING: no JRC tiles overlap the box — water mask skipped")
        return None
    srcs = [rasterio.open(f) for f in hits]
    mosaic, mtf = rio_merge(srcs, bounds=box)
    nod = srcs[0].nodata
    for s in srcs:
        s.close()
    dst = np.full((g["H"], g["W"]), 255, dtype=np.float32)
    reproject(source=mosaic[0].astype(np.float32), destination=dst,
              src_transform=mtf, src_crs="EPSG:4326", src_nodata=nod,
              dst_transform=_tf(g), dst_crs=g["crs"], dst_nodata=255,
              resampling=Resampling.bilinear)
    print(f"    water mask from {len(hits)} JRC tile(s)")
    return np.clip(np.nan_to_num(dst, nan=255), 0, 255)


# ── empty shapefile (no fiona/GDAL in this venv) ─────────────────────────────

WGS84_WKT = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
             '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
             'UNIT["Degree",0.0174532925199433]]')


def _empty_shapefile(path):
    """Polygon shapefile with zero records and one integer field, class_id."""
    hdr = bytearray(100)
    struct.pack_into(">i", hdr, 0, 9994)
    struct.pack_into(">i", hdr, 24, 50)          # file length in 16-bit words
    struct.pack_into("<i", hdr, 28, 1000)        # version
    struct.pack_into("<i", hdr, 32, 5)           # shape type: polygon
    for f, ext in ((path, ".shp"), (path, ".shx")):
        with open(f[:-4] + ext, "wb") as fh:
            fh.write(bytes(hdr))

    today = datetime.date.today()
    dbf = bytearray()
    dbf += struct.pack("<BBBB", 3, today.year - 1900, today.month, today.day)
    dbf += struct.pack("<IHH", 0, 65, 11)        # records, header len, record len
    dbf += bytes(20)
    name = b"class_id".ljust(11, b"\0")
    dbf += name + b"N" + bytes(4) + struct.pack("<BB", 10, 0) + bytes(14)
    dbf += b"\x0D" + b"\x1A"
    with open(path[:-4] + ".dbf", "wb") as fh:
        fh.write(bytes(dbf))
    with open(path[:-4] + ".prj", "w", encoding="utf-8") as fh:
        fh.write(WGS84_WKT)


# ── QGIS styles ─────────────────────────────────────────────────────────────

def _rgb_qml(red, green, blue, vmax):
    def enh(b):
        return (f'<{b}ContrastEnhancement><minValue>0</minValue>'
                f'<maxValue>{vmax}</maxValue>'
                f'<algorithm>StretchToMinimumMaximum</algorithm></{b}ContrastEnhancement>')
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology">\n'
            f'  <pipe><rasterrenderer type="multibandcolor" redBand="{red}" '
            f'greenBand="{green}" blueBand="{blue}" opacity="1">\n'
            f'    {enh("red")}{enh("green")}{enh("blue")}\n'
            '  </rasterrenderer></pipe>\n</qgis>\n')


def _water_qml():
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology">\n'
            '  <pipe><rasterrenderer type="singlebandpseudocolor" band="1" opacity="0.75">\n'
            '    <rastershader><colorrampshader colorRampType="INTERPOLATED" '
            'classificationMode="1" clip="1">\n'
            '      <item value="0" color="#ffffff" alpha="0" label="0"/>\n'
            '      <item value="10" color="#c6e2ff" alpha="180" label="10"/>\n'
            '      <item value="50" color="#3f8fd2" alpha="220" label="50"/>\n'
            '      <item value="100" color="#08306b" alpha="255" label="100"/>\n'
            '    </colorrampshader></rastershader>\n'
            '  </rasterrenderer></pipe>\n</qgis>\n')


def _labels_qml():
    cats, syms = [], []
    for i, (cid, name) in enumerate(list(C.CLASSES.items()) + [(C.NODATA_CLASS, "no data")]):
        cats.append(f'      <category value="{cid}" symbol="{i}" '
                    f'label="{cid} {name}" render="true"/>')
        rgb = tuple(int(C.CLASS_COLOURS[cid][k:k+2], 16) for k in (1, 3, 5))
        syms.append(
            f'      <symbol type="fill" name="{i}" alpha="0.5" clip_to_extent="1">\n'
            f'        <layer class="SimpleFill" enabled="1">\n'
            f'          <prop k="color" v="{rgb[0]},{rgb[1]},{rgb[2]},255"/>\n'
            f'          <prop k="outline_color" v="{rgb[0]},{rgb[1]},{rgb[2]},255"/>\n'
            f'          <prop k="outline_width" v="0.4"/><prop k="style" v="solid"/>\n'
            f'        </layer>\n      </symbol>')
    i += 1
    cats.append(f'      <category value="" symbol="{i}" label="unclassified" render="true"/>')
    syms.append(f'      <symbol type="fill" name="{i}" alpha="0.5" clip_to_extent="1">\n'
                f'        <layer class="SimpleFill" enabled="1">\n'
                f'          <prop k="color" v="0,255,0,255"/>\n'
                f'          <prop k="outline_color" v="0,255,0,255"/>\n'
                f'          <prop k="style" v="solid"/>\n        </layer>\n      </symbol>')
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology">\n'
            '  <renderer-v2 type="categorizedSymbol" attr="class_id" symbollevels="0">\n'
            '    <categories>\n' + "\n".join(cats) + '\n    </categories>\n'
            '    <symbols>\n' + "\n".join(syms) + '\n    </symbols>\n'
            '  </renderer-v2>\n</qgis>\n')


README = """Nenana River patch — {date} ({season})
==========================================================

Landsat scene : {scene_id}
                {product_id}
                acquired {acq} UTC, cloud {cloud} % (whole scene), sun {sun:.1f} deg
Box           : lon {lon0:.5f} .. {lon1:.5f}, lat {lat0:.5f} .. {lat1:.5f}
Grid          : EPSG:4326 on the pipeline's Alaska grid
                Landsat {lw} x {lh} cells of {lres:.8f} deg
                VIIRS   {vw} x {vh} cells of {vres:.8f} deg
                Pixels are rectangular on the ground at this latitude.

HOW TO START (QGIS)
  1. Drag every .tif from this folder into QGIS. Styles load automatically.
  2. Drag labels/labels_{date}.shp in as well — it is empty and ready to draw on.
  3. Digitise polygons and set class_id on each:
        1 = river (open water)     2 = ice
        3 = snow                   4 = land
        9 = no data
     Do not let polygons of different classes overlap
     (Project > Snapping Options > Avoid overlap on active layer).
  4. A VIIRS cell only becomes usable when it is labelled edge to edge.

LAYERS
  {date}_sr.tif                SR_B2..B7 surface reflectance (float)
                               default style = true colour (B4/B3/B2, 0..0.4)
                               alternative   = {date}_sr_swir.qml
                               (Layer Properties > Style > Load Style)
  {date}_thermal.tif           ST_B10, kelvin
  {date}_qa_pixel.tif          QA_PIXEL bit flags (CFMask; it often marks river
                               ice as cloud, so treat it as a hint only)
  {date}_water_occurrence.tif  JRC water occurrence 0-100 %, a guide for where
                               the channel usually is
  {date}_truecolour.tif        8-bit quicklooks, fixed scaling, for a fast look
  {date}_swir.tif              snow/ice = cyan, water = black, cloud = white

VIIRS
{viirs}

CLOUD CHECK (wide-area VIIRS)
  {date}_viirs_wide_falsecolour.tif  I3/I2/I1 over ~110 x 270 km around the box
                                         cloud = white (bright in SWIR as well)
                                         snow/ice = cyan (dark in SWIR)
                                         water = black, land = green/brown
  {date}_viirs_wide_thermal.tif      I5 brightness temperature (K);
                                         cloud tops are colder than the surface
  {date}_viirs_wide_375m.tif         I1..I5 + SZA/SAA/VZA/VAA over the wide area
  {date}_study_box.geojson           the study box, red outline, no fill
"""


def build(date, scene, ee_module):
    ee = ee_module
    lg, vg = C.landsat_grid(), C.viirs_grid()
    patch_dir = os.path.join(NR_ROOT, "patches", f"nenana_{date.replace('-', '')}")
    os.makedirs(os.path.join(patch_dir, "labels"), exist_ok=True)
    print(f"\n{date}  ({scene['season']})  -> {patch_dir}")

    img = (ee.ImageCollection(scene["collection"])
             .filter(ee.Filter.eq("LANDSAT_SCENE_ID", scene["scene_id"])).first())
    props = img.getInfo()["properties"]

    print("    surface reflectance")
    raw = _fetch(img, C.SR_BANDS, lg)
    sr = {}
    for b in C.SR_BANDS:
        v = raw[b].astype(np.float32)
        r = v * 0.0000275 - 0.2
        r[v == 0] = np.nan
        sr[b] = r
    _write(os.path.join(patch_dir, f"{date}_sr.tif"),
           np.stack([sr[b] for b in C.SR_BANDS]), lg, "float32", np.nan, C.SR_BANDS)

    print("    thermal + QA")
    aux = _fetch(img, ["ST_B10", "QA_PIXEL"], lg)
    st = aux["ST_B10"].astype(np.float32)
    k = st * 0.00341802 + 149.0
    k[st == 0] = np.nan
    _write(os.path.join(patch_dir, f"{date}_thermal.tif"), k, lg, "float32", np.nan,
           ["ST_B10_kelvin"])
    _write(os.path.join(patch_dir, f"{date}_qa_pixel.tif"), aux["QA_PIXEL"], lg,
           "uint16", 1, ["QA_PIXEL"])

    print("    water mask")
    w = _water_mask(lg)
    if w is not None:
        _write(os.path.join(patch_dir, f"{date}_water_occurrence.tif"), w, lg,
               "uint8", 255, ["water_occurrence_pct"])

    print("    quicklooks")
    _write(os.path.join(patch_dir, f"{date}_truecolour.tif"),
           np.stack([_scale8(sr["SR_B4"], 0.4), _scale8(sr["SR_B3"], 0.4),
                     _scale8(sr["SR_B2"], 0.4)]), lg, "uint8", None,
           ["red", "green", "blue"])
    _write(os.path.join(patch_dir, f"{date}_swir.tif"),
           np.stack([_scale8(sr["SR_B6"], 0.5), _scale8(sr["SR_B5"], 0.5),
                     _scale8(sr["SR_B4"], 0.5)]), lg, "uint8", None,
           ["swir1", "nir", "red"])

    # styles
    styles = {
        f"{date}_sr.qml":            _rgb_qml(3, 2, 1, 0.4),    # true colour default
        f"{date}_sr_swir.qml":       _rgb_qml(5, 4, 3, 0.5),
        f"{date}_water_occurrence.qml": _water_qml(),
        "labels/labels_%s.qml" % date: _labels_qml(),
    }
    for name, text in styles.items():
        with open(os.path.join(patch_dir, name), "w", encoding="utf-8") as f:
            f.write(text)

    shp = os.path.join(patch_dir, "labels", f"labels_{date}.shp")
    if os.path.exists(shp):
        print(f"    labels/{os.path.basename(shp)} exists — left alone")
    else:
        _empty_shapefile(shp)
        print(f"    labels/{os.path.basename(shp)} (empty, EPSG:4326, field class_id)")

    lat_mid = lg["y1"] - lg["H"] * lg["res"] / 2
    import math
    readme = README.format(
        date=date, season=scene["season"], scene_id=scene["scene_id"],
        product_id=props["LANDSAT_PRODUCT_ID"], acq=f"{props['DATE_ACQUIRED']} "
        f"{props['SCENE_CENTER_TIME'][:8]}", cloud=props["CLOUD_COVER"],
        sun=props["SUN_ELEVATION"],
        lon0=C.BOX[0], lat0=C.BOX[1], lon1=C.BOX[2], lat1=C.BOX[3],
        lw=lg["W"], lh=lg["H"], lres=lg["res"], vw=vg["W"], vh=vg["H"], vres=vg["res"],
        ew=lg["res"] * 111.320 * math.cos(math.radians(lat_mid)) * 1000,
        ns=lg["res"] * 110.574 * 1000,
        viirs="  (pass not selected yet)")
    with open(os.path.join(patch_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)
    return {"date": date, "season": scene["season"], "scene_id": scene["scene_id"],
            "product_id": props["LANDSAT_PRODUCT_ID"],
            "acquired_utc": f"{props['DATE_ACQUIRED']} {props['SCENE_CENTER_TIME'][:8]}",
            "cloud_scene_pct": props["CLOUD_COVER"], "sun_elevation": round(props["SUN_ELEVATION"], 1),
            "folder": os.path.basename(patch_dir)}


def main(dates):
    import ee, csv
    ee.Initialize(project=EE_PROJECT)
    lg, vg = C.landsat_grid(), C.viirs_grid()
    print("=" * 70)
    print(f"Nenana patches — {C.RIVER}")
    print(f"  box  lon {C.BOX[0]:.5f}..{C.BOX[2]:.5f}  lat {C.BOX[1]:.5f}..{C.BOX[3]:.5f}")
    print("  " + C.describe(lg, "Landsat: "))
    print("  " + C.describe(vg, "VIIRS  : "))
    print("=" * 70)

    rows = [build(d, C.SCENES[d], ee) for d in dates]

    track = os.path.join(NR_ROOT, "patches", "patch_tracking.csv")
    cols = ["date", "season", "scene_id", "product_id", "acquired_utc",
            "cloud_scene_pct", "sun_elevation", "folder", "viirs_granule",
            "viirs_gap_min", "viirs_vza", "geoloc_offset_m", "labelled_by", "status"]
    existing = {}
    if os.path.exists(track):
        with open(track, newline="", encoding="utf-8") as f:
            existing = {r["date"]: r for r in csv.DictReader(f)}
    for r in rows:
        prev = existing.get(r["date"], {})
        existing[r["date"]] = {**{c: prev.get(c, "") for c in cols}, **r,
                               "status": prev.get("status") or "downloaded"}
    with open(track, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in sorted(existing):
            w.writerow({c: existing[d].get(c, "") for c in cols})
    print(f"\ntracking table: {track}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build shareable Nenana patch folders.")
    p.add_argument("--dates", nargs="+", default=sorted(C.SCENES),
                   choices=sorted(C.SCENES))
    main(p.parse_args().dates)
