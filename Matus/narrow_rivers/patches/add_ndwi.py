"""
Write the Landsat NDWI of one patch as a viewable layer.

NDWI = (B3 - B5) / (B3 + B5)   green - NIR; open water is high, the edge near 0.
Exactly the formula extract_samples.py feeds the model, so what you see here is
what the classifier sees -- not a separate definition.

30 m, on the patch grid. For a finer look at the same quantity the Sentinel-2
<date>_s2_ndwi_10m.tif is 3x sharper, where that date has one.

Reads <date>_sr.tif, writes beside it:
  <date>_ndwi.tif    float32, NaN where the scene has no data
  <date>_ndwi.qml    brown -> white at 0 -> blue, so the water edge is the
                     colour break rather than something you have to judge

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\add_ndwi.py --date 2024-06-22
"""

import os, sys, argparse
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C          # noqa: E402  (same folder)

# Diverging about 0, which is where the water edge sits. The blue end matches
# the river colour in C.CLASS_COLOURS so the layers read together.
RAMP = [(-1.0, "#8b4513", "-1.0  dry land"),
        (-0.3, "#d8b365", "-0.3"),
        (0.0,  "#ffffff", " 0.00  water edge"),
        (0.3,  "#67a9cf", " 0.3"),
        (1.0,  "#0066ff", " 1.0  open water")]


def ndwi_qml():
    items = "\n".join(
        f'      <item value="{v}" color="{c}" alpha="255" label="{lab}"/>'
        for v, c, lab in RAMP)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.0" styleCategories="Symbology">\n'
            ' <pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1"\n'
            '                  opacity="1" classificationMin="-1"\n'
            '                  classificationMax="1" nodataColor="">\n'
            '   <rastershader>\n'
            '    <colorrampshader colorRampType="INTERPOLATED" clip="0"\n'
            '                     classificationMode="1" labelPrecision="2"\n'
            '                     minimumValue="-1" maximumValue="1">\n'
            f'{items}\n'
            '    </colorrampshader>\n'
            '   </rastershader>\n'
            '  </rasterrenderer>\n'
            '  <brightnesscontrast brightness="0" contrast="0"/>\n'
            ' </pipe>\n'
            ' <blendMode>0</blendMode>\n'
            '</qgis>\n')


def main(date):
    patch = os.path.join(HERE, f"nenana_{date.replace('-', '')}")
    sr = os.path.join(patch, f"{date}_sr.tif")
    if not os.path.exists(sr):
        raise SystemExit(f"ERROR: missing {sr}")

    with rasterio.open(sr) as d:
        if d.count != len(C.SR_BANDS):
            raise SystemExit(f"ERROR: {sr} has {d.count} bands, expected "
                             f"{len(C.SR_BANDS)} ({', '.join(C.SR_BANDS)})")
        b3 = d.read(C.SR_BANDS.index("SR_B3") + 1)
        b5 = d.read(C.SR_BANDS.index("SR_B5") + 1)
        prof = d.profile

    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (b3 - b5) / (b3 + b5)
    ndwi = ndwi.astype(np.float32)
    ndwi[~np.isfinite(ndwi)] = np.nan

    prof.update(count=1, dtype="float32", nodata=float("nan"),
                compress="deflate")
    out = os.path.join(patch, f"{date}_ndwi.tif")
    with rasterio.open(out, "w", **prof) as d:
        d.write(ndwi, 1)
        d.set_band_description(1, "ndwi")
    open(os.path.join(patch, f"{date}_ndwi.qml"), "w",
         encoding="utf-8").write(ndwi_qml())

    ok = np.isfinite(ndwi)
    print(f"  {date}  NDWI = (B3 - B5) / (B3 + B5)")
    print(f"  {int(ok.sum()):,} valid cells of {ndwi.size:,}")
    print(f"  min {np.nanmin(ndwi):+.3f}   median {np.nanmedian(ndwi):+.3f}   "
          f"max {np.nanmax(ndwi):+.3f}")
    print(f"  above 0: {100.0 * np.nansum(ndwi > 0) / max(ok.sum(), 1):.2f}% "
          f"of valid cells")
    print(f"  wrote {os.path.basename(out)} + .qml")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Landsat NDWI layer for one patch.")
    p.add_argument("--date", required=True, help="patch date, e.g. 2024-06-22")
    main(p.parse_args().date)
