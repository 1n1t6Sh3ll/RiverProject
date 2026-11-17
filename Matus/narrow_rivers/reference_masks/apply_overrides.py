"""
Final mask for one date = base + your QGIS overrides, exported as a dated snapshot,
plus the per-VIIRS-cell class fractions that become the U-Net target.

Priority (highest wins):  override polygons  >  hand labels  >  base (model)
The base already has the hand labels burnt in (make_base.py / RF), so in practice
it is: overrides on top of the base. Provenance is kept per pixel so every cell
can report how much of it is hand label, override or model.

Optional "done" layer: done/done_<date>.shp, polygons you draw around cells you
have finished checking (any class_id). Cells whose centre lies inside are flagged
done=1 -- the U-Net trains on done cells only. Without the layer, done = 0.

Read only: base, override shapefile, label shapefile, done shapefile.
A snapshot is never overwritten: a second export on the same day gets _HHMM.

Outputs (in reference_masks/<date>/final/):
  <date>_final_<today>.tif          class map (1 river/water, 2 ice, 3 snow, 4 land, 9 unsure)
  <date>_final_<today>.qml
  <date>_provenance_<today>.tif     1 hand label, 2 override, 3 model
  <date>_fractions_375m_<today>.tif bands: f1 f2 f3 f4, unsure, hand, override,
                                    model, usable, done
  <date>_fractions_375m_<today>.csv one row per usable cell
  <date>_export_<today>.json        what went in, counts

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\apply_overrides.py --date 2024-06-22
"""

import os, sys, csv, json, argparse, datetime
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
sys.path.insert(0, os.path.join(NR_ROOT, "classify"))
import patch_config as C
import extract_samples as EX          # label_rasters (read-only)
import make_fractions as MF           # _read_dbf
import build_patches as B             # _labels_qml colours
from rf_fractions import _to_fine, SUB

CLASSES = (1, 2, 3, 4)
UNSURE = 9
HAND, OVERRIDE, MODEL = 1, 2, 3


def class_qml():
    rows = [f'        <paletteEntry value="{c}" color="{C.CLASS_COLOURS[c]}" alpha="255" '
            f'label="{c} {C.CLASSES[c]}"/>' for c in CLASSES]
    rows.append('        <paletteEntry value="9" color="#9e9e9e" alpha="140" label="9 unsure"/>')
    return ("<!DOCTYPE qgis>\n<qgis version=\"3.28\">\n  <pipe>\n"
            "    <rasterrenderer type=\"paletted\" band=\"1\" opacity=\"1\" nodataColor=\"\">\n"
            "      <colorPalette>\n" + "\n".join(rows) + "\n      </colorPalette>\n"
            "    </rasterrenderer>\n  </pipe>\n</qgis>\n")


def read_polys(shp, shape, transform):
    """Class raster from a hand-drawn layer; empty layer -> zeros. Read-only."""
    if not os.path.exists(shp) or not MF._read_dbf(shp[:-4] + ".dbf"):
        return np.zeros(shape, np.uint8), 0
    y, _, polys = EX.label_rasters(shp, shape, transform)
    return y, len(polys)


def main(date):
    out_dir = os.path.join(HERE, date)
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    base_path = os.path.join(out_dir, f"{date}_base_water.tif")
    with rasterio.open(base_path) as s:
        base, profile, tf = s.read(1), s.profile, s.transform
    shape = base.shape

    ov, n_ov = read_polys(os.path.join(out_dir, "overrides", f"overrides_{date}.shp"), shape, tf)
    lab, n_lab = read_polys(os.path.join(patch, "labels", f"labels_{date}.shp"), shape, tf)
    done_r, n_done = read_polys(os.path.join(out_dir, "done", f"done_{date}.shp"), shape, tf)

    final = base.copy()
    prov = np.full(shape, MODEL, np.uint8)
    hand = np.isin(lab, CLASSES)
    final[hand] = lab[hand]
    prov[hand] = HAND
    o = np.isin(ov, CLASSES)
    conflict = int((o & hand & (ov != lab)).sum())
    changed = int((o & (ov != final)).sum())
    final[o] = ov[o]
    prov[o] = OVERRIDE
    print(f"Export — {date}: {n_lab} label polygons, {n_ov} override polygons "
          f"({int(o.sum()):,} px, {changed:,} changed the base), {n_done} done polygon(s)")
    if conflict:
        print(f"  NOTE: {conflict:,} override px disagree with a hand label — override kept")

    today = datetime.date.today().isoformat()
    fdir = os.path.join(out_dir, "final")
    os.makedirs(fdir, exist_ok=True)
    tag = today
    if os.path.exists(os.path.join(fdir, f"{date}_final_{tag}.tif")):
        tag = f"{today}_{datetime.datetime.now():%H%M}"
    if os.path.exists(os.path.join(fdir, f"{date}_final_{tag}.tif")):
        raise SystemExit(f"ERROR: {date}_final_{tag}.tif exists — wait a minute and rerun")

    p8 = dict(profile, dtype="uint8", count=1, nodata=None, compress="deflate")
    fpath = os.path.join(fdir, f"{date}_final_{tag}.tif")
    with rasterio.open(fpath, "w", **p8) as d:
        d.write(final, 1)
    open(fpath[:-4] + ".qml", "w", encoding="utf-8").write(class_qml())
    with rasterio.open(os.path.join(fdir, f"{date}_provenance_{tag}.tif"), "w", **p8) as d:
        d.write(prov, 1)

    # ── per-VIIRS-cell fractions (exact overlap, same as rf_fractions.py) ────
    vg = C.viirs_grid()
    ny, nx, res = vg["H"], vg["W"], vg["res"]
    fine = _to_fine(final, tf.c, tf.f, vg, np.uint8(0))
    fprov = _to_fine(prov, tf.c, tf.f, vg, np.uint8(0))
    n = SUB * SUB

    def frac(m):
        return m.reshape(ny, SUB, nx, SUB).sum(axis=(1, 3)) / n

    fr = {c: frac(fine == c) for c in CLASSES}
    f_unsure = frac(fine == UNSURE)
    f_hand, f_ov, f_model = (frac(fprov == k) for k in (HAND, OVERRIDE, MODEL))
    with rasterio.open(os.path.join(out_dir, f"{date}_cell_status_375m.tif")) as s:
        usable = s.read(1) == 1
    # done: cell centre inside a done polygon
    rows_c = ((vg["y1"] - (np.arange(ny) + 0.5) * res - tf.f) / tf.e).astype(int)
    cols_c = (((vg["x0"] + (np.arange(nx) + 0.5) * res) - tf.c) / tf.a).astype(int)
    rr, cc = np.meshgrid(np.clip(rows_c, 0, shape[0] - 1), np.clip(cols_c, 0, shape[1] - 1),
                         indexing="ij")
    done = (done_r[rr, cc] > 0) & usable

    bands = [fr[1], fr[2], fr[3], fr[4], f_unsure, f_hand, f_ov, f_model,
             usable.astype(float), done.astype(float)]
    names = ["f_river", "f_ice", "f_snow", "f_land", "f_unsure", "f_hand",
             "f_override", "f_model", "usable", "done"]
    vpath = os.path.join(fdir, f"{date}_fractions_375m_{tag}.tif")
    with rasterio.open(vpath, "w", driver="GTiff", height=ny, width=nx, count=len(bands),
                       dtype="float32", crs=vg["crs"], transform=from_origin(vg["x0"], vg["y1"], res, res),
                       compress="deflate") as d:
        for i, (b, nm) in enumerate(zip(bands, names), 1):
            d.write(b.astype(np.float32), i)
            d.set_band_description(i, nm)
    with open(vpath[:-4] + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "alaska_row", "alaska_col", "lat", "lon"] + names[:-2] + ["done"])
        for r, c in zip(*np.nonzero(usable)):
            w.writerow([f"{vg['row0'] + r}_{vg['col0'] + c}", vg["row0"] + r, vg["col0"] + c,
                        f"{vg['y1'] - (r + 0.5) * res:.6f}", f"{vg['x0'] + (c + 0.5) * res:.6f}"]
                       + [f"{b[r, c]:.4f}" for b in bands[:8]] + [int(done[r, c])])

    summ = {"date": date, "exported": tag, "base": os.path.basename(base_path),
            "base_variant": open(os.path.join(out_dir, "base_variant.txt")).readline().strip(),
            "label_polygons": n_lab, "override_polygons": n_ov, "override_px": int(o.sum()),
            "override_px_changed": changed, "override_vs_label_conflicts": conflict,
            "done_polygons": n_done, "usable_cells": int(usable.sum()), "done_cells": int(done.sum()),
            "class_px": {int(c): int((final == c).sum()) for c in (*CLASSES, UNSURE)}}
    json.dump(summ, open(os.path.join(fdir, f"{date}_export_{tag}.json"), "w",
                         encoding="utf-8"), indent=1)
    print(f"  wrote final/{os.path.basename(fpath)}, provenance, fractions ({int(usable.sum())} "
          f"usable cells, {int(done.sum())} marked done)")
    u = usable
    print(f"  usable cells: mean hand {f_hand[u].mean():.2f}, override {f_ov[u].mean():.3f}, "
          f"model {f_model[u].mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Base + overrides -> dated final mask + fractions.")
    p.add_argument("--date", required=True)
    main(p.parse_args().date)
