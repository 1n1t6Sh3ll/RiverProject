"""
Landsat band composites and MNDWI for one patch, from the local sr.tif.

No download: everything here comes out of <date>_sr.tif, so it runs offline and
can be re-run freely.

Band numbering differs between the sensors. These are the same wavelengths:

    composite        Sentinel-2      Landsat 8/9     what it is for
    true colour      B4 B3 B2        B4 B3 B2  (432) banks, gravel, dry channels
    colour infrared  B8 B4 B3        B5 B4 B3  (543) water black, vegetation red

So the professor's "432 and 843" is 432 and 543 in Landsat numbering.

MNDWI = (B3 - B6) / (B3 + B6), green vs SWIR1 -- the Landsat counterpart of
Sentinel's (B3 - B11). Water absorbs SWIR far more strongly than NIR, so a
narrow channel holds a larger contrast here than in NDWI. That is the whole
reason for computing it: NDWI can leave a sub-pixel channel below its
threshold, and MNDWI is the usual answer.

Outputs (beside sr.tif):
  <date>_432.tif    red B4, green B3, blue B2   float32 reflectance
  <date>_543.tif    nir B5, red  B4, green B3   float32 reflectance
  <date>_mndwi.tif  float32
  + .qml for each

Reflectance, not 8-bit: the existing <date>_truecolour.tif is a display-stretched
quicklook, which is fine to look at but useless for a threshold. These keep the
physical values, and the styles carry the stretch instead.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\make_composites.py --date 2024-06-22
"""

import os, sys, argparse
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C          # noqa: E402

# (name, bands as (red, green, blue), per-band display max, what it shows)
COMPOSITES = [
    ("432", ("SR_B4", "SR_B3", "SR_B2"), 0.30,
     "true colour -- banks, gravel bars, dry channels"),
    ("543", ("SR_B5", "SR_B4", "SR_B3"), 0.45,
     "colour infrared -- water black, vegetation red"),
]

MNDWI_RAMP = [(-1.0, "#8b4513", "-1.0  dry land"),
              (-0.3, "#d8b365", "-0.3"),
              (0.0,  "#ffffff", " 0.00  usual water edge"),
              (0.3,  "#67a9cf", " 0.3"),
              (1.0,  "#0066ff", " 1.0  open water")]


def rgb_qml(vmax):
    def enh(b):
        return (f'   <{b}ContrastEnhancement><minValue>0</minValue>'
                f'<maxValue>{vmax}</maxValue>'
                f'<algorithm>StretchToMinimumMaximum</algorithm>'
                f'</{b}ContrastEnhancement>\n')
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology">\n <pipe>\n'
            '  <rasterrenderer type="multibandcolor" redBand="1" greenBand="2"'
            ' blueBand="3" opacity="1">\n'
            f'{enh("red")}{enh("green")}{enh("blue")}'
            '  </rasterrenderer>\n </pipe>\n</qgis>\n')


def ramp_qml(ramp):
    items = "\n".join(
        f'      <item value="{v}" color="{c}" alpha="255" label="{lab}"/>'
        for v, c, lab in ramp)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology">\n <pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1"'
            ' opacity="1" classificationMin="-1" classificationMax="1">\n'
            '   <rastershader>\n'
            '    <colorrampshader colorRampType="INTERPOLATED" clip="0"'
            ' classificationMode="1" labelPrecision="2"'
            ' minimumValue="-1" maximumValue="1">\n'
            f'{items}\n'
            '    </colorrampshader>\n   </rastershader>\n'
            '  </rasterrenderer>\n </pipe>\n</qgis>\n')


def main(date):
    patch = os.path.join(HERE, f"nenana_{date.replace('-', '')}")
    sr = os.path.join(patch, f"{date}_sr.tif")
    if not os.path.exists(sr):
        raise SystemExit(f"ERROR: missing {sr}")

    with rasterio.open(sr) as d:
        if d.count != len(C.SR_BANDS):
            raise SystemExit(f"ERROR: {sr} has {d.count} bands, expected "
                             f"{len(C.SR_BANDS)}")
        band = {b: d.read(i + 1) for i, b in enumerate(C.SR_BANDS)}
        prof = d.profile

    print(f"  {date}  from {os.path.basename(sr)}")

    for name, bands, vmax, what in COMPOSITES:
        stack = np.stack([band[b] for b in bands]).astype(np.float32)
        out = os.path.join(patch, f"{date}_{name}.tif")
        p = dict(prof, count=3, dtype="float32", nodata=float("nan"),
                 compress="deflate", predictor=2, tiled=True)
        with rasterio.open(out, "w", **p) as d:
            d.write(stack)
            for i, b in enumerate(bands, start=1):
                d.set_band_description(i, b)
        open(os.path.join(patch, f"{date}_{name}.qml"), "w",
             encoding="utf-8").write(rgb_qml(vmax))
        print(f"    {date}_{name}.tif   {'/'.join(bands)}   "
              f"stretch 0-{vmax}   {what}")

    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (band["SR_B3"] - band["SR_B6"]) / (band["SR_B3"] + band["SR_B6"])
    mndwi = mndwi.astype(np.float32)
    mndwi[~np.isfinite(mndwi)] = np.nan
    out = os.path.join(patch, f"{date}_mndwi.tif")
    p = dict(prof, count=1, dtype="float32", nodata=float("nan"),
             compress="deflate")
    with rasterio.open(out, "w", **p) as d:
        d.write(mndwi, 1)
        d.set_band_description(1, "mndwi")
    open(os.path.join(patch, f"{date}_mndwi.qml"), "w",
         encoding="utf-8").write(ramp_qml(MNDWI_RAMP))

    ok = np.isfinite(mndwi)
    print(f"    {date}_mndwi.tif   (B3-B6)/(B3+B6)   "
          f"median {np.nanmedian(mndwi):+.3f}, "
          f"above 0: {100.0 * np.nansum(mndwi > 0) / max(ok.sum(), 1):.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Landsat 432/543 composites + MNDWI.")
    p.add_argument("--date", required=True, help="patch date, e.g. 2024-06-22")
    main(p.parse_args().date)
