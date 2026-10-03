"""
Wide-area VIIRS image per date, for judging cloud by eye.

The patch-sized VIIRS grid is too small to recognise cloud by its pattern, so
this writes the same granule over a much larger area (the box grown by
--pad degrees), on the same 375 m Alaska grid:

  <date>_viirs_wide_375m.tif        I1..I5 + SZA/SAA/VZA/VAA (float)
  <date>_viirs_wide_falsecolour.tif 8-bit RGB = I3/I2/I1
                                    cloud  = white (bright in SWIR too)
                                    snow/ice = cyan (dark in SWIR)
                                    water  = black, land = green/brown
  <date>_viirs_wide_thermal.tif     I5 brightness temperature (K); cloud tops cold
  *.qml                             styles, plus an outline of the study box
  <date>_study_box.geojson          the box, to show where the patch sits

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\make_viirs_wide.py
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\make_viirs_wide.py --dates 2024-06-22 --pad 1.5
"""

import os, sys, json, math, argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C
from add_viirs import PASSES, _granule, _resample

NR_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(NR_ROOT))
from viirs_training_loader import load_viirs_training_pair

BOX_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0">
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1">
        <layer class="SimpleFill" enabled="1">
          <prop k="style" v="no"/>
          <prop k="color" v="255,0,0,0"/>
          <prop k="outline_color" v="255,0,0,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.6"/>
          <prop k="outline_width_unit" v="MM"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""

BANDS  = ["I1", "I2", "I3", "I4", "I5"]
ANGLES = ["SZA", "SAA", "VZA", "VAA"]


def _scale8(band, vmax):
    out = np.clip(np.nan_to_num(band, nan=0.0) / vmax * 255.0, 0, 255)
    return out.astype(np.uint8)


def _rgb_qml():
    def enh(b, mx):
        return (f'<{b}ContrastEnhancement><minValue>0</minValue><maxValue>{mx}</maxValue>'
                f'<algorithm>StretchToMinimumMaximum</algorithm></{b}ContrastEnhancement>')
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="multibandcolor" redBand="1" greenBand="2" blueBand="3" opacity="1">\n'
            f'   {enh("red",255)}{enh("green",255)}{enh("blue",255)}\n'
            '  </rasterrenderer></pipe></qgis>\n')


def _thermal_qml():
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">\n'
            '   <rastershader><colorrampshader colorRampType="INTERPOLATED" clip="0">\n'
            '     <item value="230" color="#6a00a8" label="230 K (high cloud)"/>\n'
            '     <item value="250" color="#2171b5" label="250 K"/>\n'
            '     <item value="265" color="#41b6c4" label="265 K"/>\n'
            '     <item value="273" color="#ffffbf" label="273 K (freezing)"/>\n'
            '     <item value="290" color="#f46d43" label="290 K"/>\n'
            '     <item value="305" color="#a50026" label="305 K"/>\n'
            '   </colorrampshader></rastershader>\n'
            '  </rasterrenderer></pipe></qgis>\n')


def main(dates, pad):
    for date in dates:
        sat, t_code, gap, vza = PASSES[date]
        patch_dir = os.path.join(NR_ROOT, "patches", f"nenana_{date.replace('-', '')}")
        box = (C.BOX[0] - pad, C.BOX[1] - pad, C.BOX[2] + pad, C.BOX[3] + pad)
        g = C.viirs_grid(box)
        print(f"\n{date}  {sat} {t_code}")
        print("  " + C.describe(g, "wide grid: "))

        gitco, gimgo = _granule(sat, t_code)
        clip = (box[1] - 0.1, box[3] + 0.1, box[0] - 0.1, box[2] + 0.1)
        lat, lon, bands, angles, meta = load_viirs_training_pair(gitco, gimgo, clip_bbox=clip)

        stack, names = [], []
        for name in BANDS + ANGLES:
            stack.append(_resample(lat, lon, bands.get(name, angles.get(name)), g))
            names.append(name)
        stack = np.stack(stack)
        filled = np.isfinite(stack[0]).sum()
        print(f"  {filled}/{g['W']*g['H']} cells have data "
              f"({100*filled/(g['W']*g['H']):.0f} % — the granule is a strip, so edges can be empty)")

        tf = from_origin(g["x0"], g["y1"], g["res"], g["res"])
        base = os.path.join(patch_dir, f"{date}_viirs_wide")

        with rasterio.open(base + "_375m.tif", "w", driver="GTiff", height=g["H"],
                           width=g["W"], count=len(names), dtype="float32", crs=C.CRS,
                           transform=tf, nodata=np.nan, compress="deflate") as dst:
            dst.write(stack)
            for i, n in enumerate(names, start=1):
                dst.set_band_description(i, n)

        # I3/I2/I1 false colour: cloud white, snow/ice cyan, water black
        rgb = np.stack([_scale8(stack[2], 0.5), _scale8(stack[1], 0.5), _scale8(stack[0], 0.5)])
        with rasterio.open(base + "_falsecolour.tif", "w", driver="GTiff", height=g["H"],
                           width=g["W"], count=3, dtype="uint8", crs=C.CRS,
                           transform=tf, compress="deflate") as dst:
            dst.write(rgb)
            for i, n in enumerate(["I3 (SWIR)", "I2 (NIR)", "I1 (red)"], start=1):
                dst.set_band_description(i, n)
        open(base + "_falsecolour.qml", "w", encoding="utf-8").write(_rgb_qml())

        with rasterio.open(base + "_thermal.tif", "w", driver="GTiff", height=g["H"],
                           width=g["W"], count=1, dtype="float32", crs=C.CRS,
                           transform=tf, nodata=np.nan, compress="deflate") as dst:
            dst.write(stack[4][None])
            dst.set_band_description(1, "I5 brightness temperature (K)")
        open(base + "_thermal.qml", "w", encoding="utf-8").write(_thermal_qml())

        # the study box, so it is obvious where the patch sits in the wide view
        # (red outline, no fill — it sits on top of the imagery)
        open(os.path.join(patch_dir, f"{date}_study_box.qml"), "w",
             encoding="utf-8").write(BOX_QML)
        x0, y0, x1, y1 = C.BOX
        with open(os.path.join(patch_dir, f"{date}_study_box.geojson"), "w",
                  encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection",
                       "crs": {"type": "name",
                               "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
                       "features": [{"type": "Feature", "properties": {"name": "study box"},
                                     "geometry": {"type": "Polygon", "coordinates": [[
                                         [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}]}, f)

        for suffix in ("_375m.tif", "_falsecolour.tif", "_thermal.tif"):
            p = base + suffix
            print(f"    {os.path.basename(p):46} {os.path.getsize(p)/1e6:5.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Wide-area VIIRS image for cloud checking.")
    p.add_argument("--dates", nargs="+", default=sorted(PASSES), choices=sorted(PASSES))
    p.add_argument("--pad", type=float, default=1.0,
                   help="degrees to grow the study box by (default 1.0 ~ 110 km)")
    a = p.parse_args()
    main(a.dates, a.pad)
