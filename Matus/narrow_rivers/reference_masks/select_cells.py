"""
Pick the VIIRS 375 m cells worth perfecting by hand, for one date.

Every cell of the study box that contains any water in the base mask is a
candidate. A candidate is USABLE when all of these hold (reason code if not):

  outside_box     the cell lies fully inside the study box
  no_landsat      every Landsat cell under it has data
  no_viirs        VIIRS I1..I5 are all valid
  unsure          RF-unsure (class 9) covers < --max-unsure of the cell
  geoloc          the channel stays in the cell: after the geolocation correction,
                  shifting by the remaining uncertainty in any of 8 directions
                  keeps at least --keep-frac of the cell's water fraction.
                  Relative, so it works the same for a 3 % cell as a 60 % one.

Geolocation correction: <date>_geoloc_offset.json (from geoloc_offset.py) gives
where a VIIRS cell really looks (east_m, north_m). Fractions are computed on the
base mask moved by that amount; the remaining uncertainty is the peak half-width
from the same file. Without the file (or with --no-correct) the uncorrected
fractions are used and the uncertainty is the tracked offset length.

Cells with only a trace of water are kept on purpose -- they are the narrow
channels this work is about. Cloud is NOT checked here: that stays visual.

Grid arithmetic is the exact overlap from classify/rf_fractions.py (12.5 Landsat
cells per VIIRS cell, 50 x 50 fine cells per VIIRS cell).

Outputs (in reference_masks/<date>/):
  <date>_viirs_cells.geojson           every candidate cell, its numbers, usable, reason
  <date>_viirs_cells.qml               usable = green outline, rejected = red
  <date>_usable_outline.geojson        usable cells dissolved into one clean outline
  <date>_usable_outline.qml            thick yellow line, no fill
  <date>_waterfrac_375m_raw.tif        water fraction, NO geolocation correction
  <date>_waterfrac_375m_shifted.tif    water fraction after the shift (only if one is applied)
  <date>_cell_status_375m.tif          1 usable, 2 rejected geoloc, 3 rejected other
  runs/<timestamp>/config.json + summary.csv

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\select_cells.py --date 2024-06-22
"""

import os, sys, csv, json, math, argparse, datetime
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box as sbox, mapping
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
sys.path.insert(0, os.path.join(NR_ROOT, "classify"))
import patch_config as C
from rf_fractions import _to_fine, SUB

WATER, LAND, UNSURE, OUTSIDE = 1, 4, 9, 0
CHANNEL = (1, 2)     # a cell is a candidate if it holds river/water or river ice
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def geoloc_offset_m(date):
    with open(os.path.join(PATCHES, "patch_tracking.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["date"] == date:
                return float(r["geoloc_offset_m"])
    raise SystemExit(f"ERROR: {date} not in patch_tracking.csv")


def shifted(a, sy, sx, fill):
    """a moved by (sy, sx) fine cells; vacated cells get `fill`."""
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(0, H - sy), slice(sy, H)) if sy >= 0 else (slice(-sy, H), slice(0, H + sy))
    xs, xd = (slice(0, W - sx), slice(sx, W)) if sx >= 0 else (slice(-sx, W), slice(0, W + sx))
    out[yd, xd] = a[ys, xs]
    return out


def per_cell(a, ny, nx):
    return a.reshape(ny, SUB, nx, SUB).sum(axis=(1, 3))


FRAC_QML = """<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1">
      <rastershader><colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
        <item value="0" color="#ffffff" alpha="0" label="0"/>
        <item value="0.001" color="#c6dbef" alpha="200" label="trace"/>
        <item value="0.1" color="#6baed6" alpha="230" label="0.1"/>
        <item value="0.3" color="#2171b5" alpha="255" label="0.3"/>
        <item value="1" color="#08306b" alpha="255" label="1"/>
      </colorrampshader></rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""

STATUS_QML = """<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
        <paletteEntry value="0" color="#000000" alpha="0" label="no water"/>
        <paletteEntry value="1" color="#2ca02c" alpha="170" label="1 usable"/>
        <paletteEntry value="2" color="#d62728" alpha="170" label="2 rejected: channel leaves cell"/>
        <paletteEntry value="3" color="#7f7f7f" alpha="170" label="3 rejected: other"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def qml_cells():
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="usable" symbollevels="0">
    <categories>
      <category value="1" symbol="0" label="usable" render="true"/>
      <category value="0" symbol="1" label="rejected (see reason)" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="1"><layer class="SimpleFill" enabled="1">
        <prop k="style" v="no"/><prop k="outline_color" v="0,200,0,255"/>
        <prop k="outline_width" v="0.3"/></layer></symbol>
      <symbol type="fill" name="1" alpha="1"><layer class="SimpleFill" enabled="1">
        <prop k="style" v="no"/><prop k="outline_color" v="220,30,30,255"/>
        <prop k="outline_width" v="0.2"/></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


def qml_outline():
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0">
    <symbols>
      <symbol type="fill" name="0" alpha="1"><layer class="SimpleFill" enabled="1">
        <prop k="style" v="no"/><prop k="outline_color" v="255,215,0,255"/>
        <prop k="outline_width" v="0.8"/></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


def main(date, keep_frac, max_unsure, correct, unc_override=(None, None)):
    out_dir = os.path.join(HERE, date)
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    base_path = os.path.join(out_dir, f"{date}_base_water.tif")
    if not os.path.exists(base_path):
        raise SystemExit(f"ERROR: {base_path} missing — run make_base.py first")

    vg = C.viirs_grid()
    ny, nx, res = vg["H"], vg["W"], vg["res"]
    with rasterio.open(base_path) as s:
        base = s.read(1)
        fine = _to_fine(base, s.transform.c, s.transform.f, vg, np.uint8(OUTSIDE))
    with rasterio.open(os.path.join(patch, f"{date}_viirs_375m.tif")) as s:
        assert abs(s.transform.c - vg["x0"]) < 1e-9 and abs(s.transform.f - vg["y1"]) < 1e-9 \
            and (s.height, s.width) == (ny, nx), "VIIRS file is not on the box's VIIRS grid"
        v = s.read()
        desc = list(s.descriptions)
    viirs_ok = np.isfinite(v[[desc.index(b) for b in ("I1", "I2", "I3", "I4", "I5")]]).all(0)
    vza = v[desc.index("VZA")]

    n = SUB * SUB
    lat_c = vg["y1"] - ny * res / 2
    fine_ns = res / SUB * 110_574.0
    fine_ew = res / SUB * 111_320.0 * math.cos(math.radians(lat_c))

    def move(a, east_m, north_m, fill):
        """content moved so each cell sees the ground (east, north) metres away"""
        return shifted(a, int(round(north_m / fine_ns)), -int(round(east_m / fine_ew)), fill)

    off_path = os.path.join(out_dir, f"{date}_geoloc_offset.json")
    if correct and os.path.exists(off_path):
        o = json.load(open(off_path, encoding="utf-8"))
        east, north = o["east_m"], o["north_m"]
        unc_e, unc_n = o["peak_halfwidth_e_m"], o["peak_halfwidth_n_m"]
    else:
        east = north = 0.0
        unc_e = unc_n = geoloc_offset_m(date)
    if unc_override[0] is not None:
        unc_e = unc_override[0]
    if unc_override[1] is not None:
        unc_n = unc_override[1]
    print(f"Cell selection — {date}: correction {east:+.0f} m E, {north:+.0f} m N; "
          f"remaining uncertainty ±{unc_e:.0f} m E-W, ±{unc_n:.0f} m N-S")

    wet0 = np.isin(fine, CHANNEL).astype(np.uint8)     # river/water + river ice
    valid0 = (fine != OUTSIDE).astype(np.uint8)
    raw_f = per_cell(wet0, ny, nx) / n
    wet = move(wet0, east, north, 0)
    valid = move(valid0, east, north, 0)
    fine_c = move(fine, east, north, np.uint8(OUTSIDE))
    water_f = per_cell(wet, ny, nx) / n
    unsure_f = per_cell(fine_c == UNSURE, ny, nx) / n
    outside_f = per_cell(valid == 0, ny, nx) / n

    # does the channel stay in the cell within the remaining uncertainty?
    keep = np.ones((ny, nx))
    for dy, dx in DIRECTIONS:
        w_s = per_cell(move(wet, dx * unc_e, dy * unc_n, 0), ny, nx)
        v_s = per_cell(move(valid, dx * unc_e, dy * unc_n, 0), ny, nx)
        f_s = w_s / np.maximum(v_s, 1)
        ratio = np.where(water_f > 0, f_s / np.maximum(water_f, 1e-9), 1.0)
        keep = np.minimum(keep, np.where(v_s >= 0.9 * n, ratio, 1.0))

    # independent check: VIIRS NIR (I2) is dark over water, so the better-aligned
    # fraction should anti-correlate more strongly with it
    i2 = v[desc.index("I2")]
    m = np.isfinite(i2) & ((raw_f > 0) | (water_f > 0))
    r_raw = float(np.corrcoef(raw_f[m], i2[m])[0, 1])
    r_cor = float(np.corrcoef(water_f[m], i2[m])[0, 1])
    print(f"  check vs VIIRS I2 (NIR) over {int(m.sum())} water cells: r = {r_raw:+.3f} "
          f"uncorrected, {r_cor:+.3f} corrected (more negative = better aligned)")

    # inside the study box?
    lon0, lat0, lon1, lat1 = C.BOX
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    cx0 = vg["x0"] + cols * res
    cy1 = vg["y1"] - rows * res
    inside = (cx0 >= lon0) & (cx0 + res <= lon1) & (cy1 <= lat1) & (cy1 - res >= lat0)

    candidate = water_f > 0
    reason = np.full((ny, nx), "", dtype=object)
    for mask, code in ((~inside, "outside_box"), (outside_f > 0, "no_landsat"),
                       (~viirs_ok, "no_viirs"), (unsure_f >= max_unsure, "unsure"),
                       (keep < keep_frac, "geoloc")):
        reason[(reason == "") & mask] = code
    usable = candidate & (reason == "")

    feats, boxes = [], []
    for r, c in zip(*np.nonzero(candidate)):
        g = sbox(cx0[r, c], cy1[r, c] - res, cx0[r, c] + res, cy1[r, c])
        if usable[r, c]:
            boxes.append(g)
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": {
            "cell_id": f"{vg['row0'] + r}_{vg['col0'] + c}",
            "alaska_row": int(vg["row0"] + r), "alaska_col": int(vg["col0"] + c),
            "water_frac": round(float(water_f[r, c]), 4),
            "water_frac_raw": round(float(raw_f[r, c]), 4),
            "unsure_frac": round(float(unsure_f[r, c]), 4),
            "geoloc_keep": round(float(keep[r, c]), 3),
            "vza": round(float(vza[r, c]), 2) if np.isfinite(vza[r, c]) else None,
            "usable": int(usable[r, c]), "reason": reason[r, c] or None}})
    base_out = os.path.join(out_dir, f"{date}_viirs_cells")
    json.dump({"type": "FeatureCollection", "features": feats},
              open(base_out + ".geojson", "w", encoding="utf-8"))
    open(base_out + ".qml", "w", encoding="utf-8").write(qml_cells())
    outline = unary_union(boxes)
    ol = os.path.join(out_dir, f"{date}_usable_outline")
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature",
               "geometry": mapping(outline), "properties": {"date": date,
               "n_cells": int(usable.sum())}}]}, open(ol + ".geojson", "w", encoding="utf-8"))
    open(ol + ".qml", "w", encoding="utf-8").write(qml_outline())

    # report
    print(f"  {nx*ny:,} cells in the grid, {int(candidate.sum()):,} contain water")
    for code in ("outside_box", "no_landsat", "no_viirs", "unsure", "geoloc"):
        print(f"    rejected {code:12} {int((candidate & (reason == code)).sum()):5,}")
    print(f"  USABLE {int(usable.sum()):,} cells, {len(getattr(outline, 'geoms', [outline]))} "
          f"separate piece(s)")
    pre_geoloc = candidate & np.isin(reason, ["", "geoloc"])
    rows_out = []
    for t in (0.25, 0.5, 0.75):
        k = int((pre_geoloc & (keep >= t)).sum())
        rows_out.append({"keep_frac": t, "usable_cells": k})
        print(f"    keeping >= {t:.0%} of the water: {k:,} usable")
    wf = water_f[usable]
    print(f"  water fraction in usable cells: median {np.median(wf):.3f}, "
          f"{int((wf < 0.02).sum())} cells under 2 %, {int((wf > 0.5).sum())} over 50 %")

    vprof = {"driver": "GTiff", "height": ny, "width": nx, "count": 1, "crs": vg["crs"],
             "transform": from_origin(vg["x0"], vg["y1"], res, res), "compress": "deflate"}
    tifs = [("raw", raw_f)] + ([("shifted", water_f)] if (east or north) else [])
    for name, arr in tifs:
        pth = os.path.join(out_dir, f"{date}_waterfrac_375m_{name}.tif")
        with rasterio.open(pth, "w", **vprof, dtype="float32", nodata=None) as d:
            d.write(arr.astype(np.float32), 1)
        open(pth[:-4] + ".qml", "w", encoding="utf-8").write(FRAC_QML)
    status = np.zeros((ny, nx), np.uint8)
    status[usable] = 1
    status[candidate & (reason == "geoloc")] = 2
    status[candidate & ~usable & (reason != "geoloc")] = 3
    pth = os.path.join(out_dir, f"{date}_cell_status_375m.tif")
    with rasterio.open(pth, "w", **vprof, dtype="uint8", nodata=None) as d:
        d.write(status, 1)
    open(pth[:-4] + ".qml", "w", encoding="utf-8").write(STATUS_QML)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, "runs", stamp)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["keep_frac", "usable_cells"])
        w.writeheader(); w.writerows(rows_out)
    json.dump({"script": "select_cells.py", "date": date, "keep_frac": keep_frac,
               "max_unsure": max_unsure, "correction_east_m": east,
               "correction_north_m": north, "uncertainty_e_m": unc_e,
               "uncertainty_n_m": unc_n, "r_i2_raw": round(r_raw, 4),
               "r_i2_corrected": round(r_cor, 4),
               "base": os.path.basename(base_path),
               "base_variant": open(os.path.join(out_dir, "base_variant.txt")).readline().strip(),
               "candidates": int(candidate.sum()), "usable": int(usable.sum())},
              open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"), indent=1)
    print(f"  run recorded in runs/{stamp}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pick usable VIIRS cells for one date.")
    p.add_argument("--date", required=True)
    p.add_argument("--keep-frac", type=float, default=0.5,
                   help="share of the cell's water that must stay in it (default 0.5)")
    p.add_argument("--no-correct", action="store_true",
                   help="ignore <date>_geoloc_offset.json")
    p.add_argument("--unc-e", type=float, default=None,
                   help="override remaining E-W uncertainty, metres")
    p.add_argument("--unc-n", type=float, default=None,
                   help="override remaining N-S uncertainty, metres")
    p.add_argument("--max-unsure", type=float, default=0.05)
    a = p.parse_args()
    main(a.date, a.keep_frac, a.max_unsure, not a.no_correct, (a.unc_e, a.unc_n))
