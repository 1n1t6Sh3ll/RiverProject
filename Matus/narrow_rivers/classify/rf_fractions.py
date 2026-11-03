"""
Per-VIIRS-cell class fractions from the Random Forest class map.

This is the project's actual training target: for every 375 m VIIRS cell, what
fraction of it is river, ice, snow and land. make_fractions.py does the same job
from the hand-drawn polygons alone, which only ever covers part of the box; this
version starts from the RF class raster, so every cell gets a value.

PROVENANCE IS KEPT SEPARATE. Each cell reports frac_hand (how much of it you drew
by hand) and frac_model (how much the Random Forest filled in), so results can be
reported on hand-labelled cells alone and the model-labelled cells treated as an
expanded secondary set. Without this the two can never be told apart again.

Grid arithmetic (exact, no interpolation):
  VIIRS cell   = SHARED_RES            = 50 fine cells per side
  Landsat cell = SHARED_RES / 12.5     =  4 fine cells per side
A VIIRS cell is 12.5 Landsat cells, so VIIRS boundaries fall halfway through every
other Landsat cell -- which is why the fine grid exists. Each Landsat cell is
repeated 4 x 4 onto it, and 50 x 50 fine cells are then summed per VIIRS cell.

Outputs (in patches/<patch>/rf/):
  <date>_fractions_375m.tif   bands: river, ice, snow, land, hand, model, nodata
  <date>_fractions_375m.csv   one row per VIIRS cell, with alaska_row/col and lat/lon
  <date>_fractions_qc.png     quick look

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\rf_fractions.py --dates 2023-10-04
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\rf_fractions.py --dates 2023-10-04 2024-06-22
"""

import os, sys, csv, math, argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
sys.path.insert(0, HERE)
import patch_config as C
import extract_samples as EX

SUB = 50                              # fine cells per VIIRS cell, per side
FINE_RES = C.SHARED_RES / SUB
LS_TO_FINE = int(round(C.LS_RES / FINE_RES))        # 4
OUT_SUBDIR = "rf"


def _to_fine(arr, ls_x0, ls_y1, vg, fill):
    """Landsat-grid array -> the VIIRS fine grid, by exact 4x4 repetition.

    The Landsat and VIIRS grids are snapped separately, so their origins differ
    by a whole number of fine cells; that offset is applied here.
    """
    big = np.repeat(np.repeat(arr, LS_TO_FINE, axis=0), LS_TO_FINE, axis=1)
    fine_h, fine_w = vg["H"] * SUB, vg["W"] * SUB
    out = np.full((fine_h, fine_w), fill, dtype=arr.dtype)

    # offset of the Landsat origin inside the fine grid (may be negative)
    dx = int(round((ls_x0 - vg["x0"]) / FINE_RES))
    dy = int(round((vg["y1"] - ls_y1) / FINE_RES))
    sy0, sx0 = max(0, -dy), max(0, -dx)
    dy0, dx0 = max(0, dy), max(0, dx)
    h = min(big.shape[0] - sy0, fine_h - dy0)
    w = min(big.shape[1] - sx0, fine_w - dx0)
    out[dy0:dy0 + h, dx0:dx0 + w] = big[sy0:sy0 + h, sx0:sx0 + w]
    return out


def process(date, min_labelled):
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    out_dir = os.path.join(patch, OUT_SUBDIR)
    cls_path = os.path.join(out_dir, f"{date}_rf_class.tif")
    if not os.path.exists(cls_path):
        raise SystemExit(f"ERROR: {cls_path} missing — run train_rf.py first")

    vg = C.viirs_grid()
    with rasterio.open(cls_path) as src:
        cls = src.read(1)
        ls_x0, ls_y1 = src.transform.c, src.transform.f
        ls_shape = (src.height, src.width)

    print("=" * 66)
    print(f"Fractions from the RF map — {date}")
    print("  " + C.describe(vg, "VIIRS: "))
    print(f"  Landsat cell -> {LS_TO_FINE} x {LS_TO_FINE} fine cells, "
          f"VIIRS cell -> {SUB} x {SUB}")

    # which Landsat cells are hand-drawn (provenance)
    shp = os.path.join(patch, "labels", f"labels_{date}.shp")
    hand_ras, _, polys = EX.label_rasters(shp, ls_shape,
                                          from_origin(ls_x0, ls_y1, C.LS_RES, C.LS_RES))
    print(f"  {len(polys)} hand polygons; "
          f"{int((hand_ras > 0).sum()):,} hand-labelled Landsat cells")

    fine_cls = _to_fine(cls, ls_x0, ls_y1, vg, np.uint8(C.NODATA_CLASS))
    fine_hand = _to_fine((hand_ras > 0).astype(np.uint8), ls_x0, ls_y1, vg, np.uint8(0))

    ny, nx = vg["H"], vg["W"]
    blocks = fine_cls.reshape(ny, SUB, nx, SUB)
    hblocks = fine_hand.reshape(ny, SUB, nx, SUB)
    n = SUB * SUB
    frac = {c: (blocks == c).sum(axis=(1, 3)) / n for c in C.CLASSES}
    nodata_f = (blocks == C.NODATA_CLASS).sum(axis=(1, 3)) / n
    hand_f = hblocks.sum(axis=(1, 3)) / n
    classified = sum(frac.values())
    model_f = np.clip(classified - hand_f, 0, 1)

    usable = (classified >= min_labelled) & (nodata_f == 0)
    print(f"\n  {nx*ny:,} VIIRS cells: {int(usable.sum()):,} usable "
          f"(>= {min_labelled:.0%} classified, no nodata)")
    print(f"  fully hand-labelled cells      : {int((hand_f >= 0.999).sum()):,}")
    print(f"  cells with any hand label      : {int((hand_f > 0).sum()):,}")
    print(f"  cells the model alone filled   : {int(((hand_f == 0) & usable).sum()):,}")
    print("\n  mean fraction over usable cells")
    for c, name in C.CLASSES.items():
        print(f"    {name:6} {frac[c][usable].mean():.4f}   "
              f"cells > 0.5: {int((frac[c] > 0.5).sum()):,}")
    riv = frac[1]
    print(f"\n  river-bearing cells (> 0.05): {int((riv > 0.05).sum()):,}; "
          f"of those, {int(((riv > 0.05) & (hand_f > 0)).sum()):,} have hand labels")

    # ── rasters ──────────────────────────────────────────────────────────────
    names = [C.CLASSES[c] for c in C.CLASSES] + ["hand", "model", "nodata"]
    stack = np.stack([frac[c] for c in C.CLASSES] +
                     [hand_f, model_f, nodata_f]).astype(np.float32)
    out_tif = os.path.join(out_dir, f"{date}_fractions_375m.tif")
    with rasterio.open(out_tif, "w", driver="GTiff", height=ny, width=nx,
                       count=len(names), dtype="float32", crs=vg["crs"],
                       transform=from_origin(vg["x0"], vg["y1"], vg["res"], vg["res"]),
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(stack)
        for i, nm in enumerate(names, start=1):
            dst.set_band_description(i, nm)
    print(f"\n    {os.path.basename(out_tif):36} {os.path.getsize(out_tif)/1e6:5.2f} MB")

    # ── csv ──────────────────────────────────────────────────────────────────
    out_csv = os.path.join(out_dir, f"{date}_fractions_375m.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "alaska_row", "alaska_col", "lat", "lon"] +
                   [f"frac_{C.CLASSES[c]}" for c in C.CLASSES] +
                   ["frac_hand", "frac_model", "frac_nodata", "usable"])
        for r in range(ny):
            lat = vg["y1"] - (r + 0.5) * vg["res"]
            for c in range(nx):
                lon = vg["x0"] + (c + 0.5) * vg["res"]
                w.writerow([date, vg["row0"] + r, vg["col0"] + c,
                            f"{lat:.6f}", f"{lon:.6f}"] +
                           [f"{frac[k][r, c]:.4f}" for k in C.CLASSES] +
                           [f"{hand_f[r, c]:.4f}", f"{model_f[r, c]:.4f}",
                            f"{nodata_f[r, c]:.4f}", int(usable[r, c])])
    print(f"    {os.path.basename(out_csv):36} {os.path.getsize(out_csv)/1e6:5.2f} MB")

    # ── quick look ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lat_mid = vg["y1"] - ny * vg["res"] / 2
        aspect = 110.574 / (111.320 * math.cos(math.radians(lat_mid)))
        fig, ax = plt.subplots(1, 5, figsize=(16, 7))
        for a, (c, name) in zip(ax, C.CLASSES.items()):
            im = a.imshow(frac[c], vmin=0, vmax=1, cmap="viridis", aspect=aspect)
            a.set_title(f"{name}\nmean {frac[c][usable].mean():.3f}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        im = ax[4].imshow(hand_f, vmin=0, vmax=1, cmap="magma", aspect=aspect)
        ax[4].set_title("hand-labelled\nshare of cell", fontsize=10)
        ax[4].set_xticks([]); ax[4].set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.suptitle(f"{date} — per-VIIRS-cell fractions from the RF map "
                     f"({int(usable.sum()):,} usable cells)")
        png = os.path.join(out_dir, f"{date}_fractions_qc.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"    {os.path.basename(png):36} {os.path.getsize(png)/1e6:5.2f} MB")
    except Exception as e:                       # a figure is not worth failing over
        print(f"    (quick look skipped: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VIIRS-cell fractions from the RF map.")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--min-labelled", type=float, default=1.0,
                   help="a cell counts as usable at this classified share (default 1.0)")
    a = p.parse_args()
    for d in a.dates:
        process(d, a.min_labelled)
