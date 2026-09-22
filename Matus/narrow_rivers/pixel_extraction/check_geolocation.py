"""
VIIRS geolocation check against a near-simultaneous Landsat scene.

VIIRS I1/I2/I3 have Landsat 8/9 spectral twins (B4 red, B5 NIR, B6 SWIR1).
For every VIIRS pixel in a clear test region:
  1. Average Landsat over a ~375 m box centred on the pixel's own GITCO
     lat/lon (in UTM metres — no degree grid involved).
  2. Correlate VIIRS with that Landsat average, per band pair.
  3. Repeat with the VIIRS positions shifted east/north in 30 m steps.
The shift with the highest correlation is the geolocation offset.
Peak at ~0 m -> geolocation is fine.

Landsat is pulled once per scene via ee.data.computePixels (no Drive export),
on its native 30 m UTM grid.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\pixel_extraction\\check_geolocation.py --scene 2024-06-12-npp
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from extract_river_pixels import SCENES, NR_ROOT, EE_PROJECT, _load_nodes, DEFAULT_RIVER
from viirs_training_loader import load_viirs_training_pair

# Clear test regions (lat_min, lat_max, lon_min, lon_max) per scene.
# 27 May: north is cloudy, so only the southern corridor.
TEST_REGIONS = {
    "2024-06-12-npp": (68.70, 70.20, -148.95, -147.95),
    "2024-06-12-j02": (68.70, 70.20, -148.95, -147.95),
    "2024-05-27-j01": (68.70, 69.60, -148.95, -148.40),
}

UTM_CRS     = "EPSG:32606"   # UTM zone 6N — the Sagavanirktok
LS_RES      = 30.0
BOX_PX      = 13             # 13 × 30 m = 390 m ≈ one near-nadir VIIRS I-band pixel
MAX_SHIFT_M = 750.0          # ± 2 VIIRS pixels
SHIFT_STEP  = 30.0
MARGIN_M    = MAX_SHIFT_M + BOX_PX * LS_RES   # keep shifted boxes inside the patch
MIN_VALID   = 0.9            # fraction of valid Landsat pixels required in a box

BAND_PAIRS = [("I1", "SR_B4"), ("I2", "SR_B5"), ("I3", "SR_B6")]
TILE_ROWS  = 1500            # computePixels request size limit


# ── Landsat ───────────────────────────────────────────────────────────────────

def _fetch_landsat(product_id, ee_collection, x0, y1, width, height):
    """Landsat SR B4/B5/B6 as reflectance (NaN = fill) on a 30 m UTM grid with
    top-left corner (x0, y1). Fetched in row tiles to stay under the
    computePixels size limit."""
    import ee
    ee.Initialize(project=EE_PROJECT)
    img = (ee.ImageCollection(ee_collection)
             .filter(ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id))
             .first()
             .select([b for _, b in BAND_PAIRS]))

    out = {b: np.full((height, width), np.nan, dtype=np.float32) for _, b in BAND_PAIRS}
    for r0 in range(0, height, TILE_ROWS):
        h = min(TILE_ROWS, height - r0)
        arr = ee.data.computePixels({
            "expression": img,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": width, "height": h},
                "affineTransform": {
                    "scaleX": LS_RES, "shearX": 0, "translateX": x0,
                    "shearY": 0, "scaleY": -LS_RES, "translateY": y1 - r0 * LS_RES,
                },
                "crsCode": UTM_CRS,
            },
        })
        for _, b in BAND_PAIRS:
            raw = arr[b].astype(np.float32)
            refl = raw * 0.0000275 - 0.2
            refl[raw == 0] = np.nan          # fill
            out[b][r0:r0 + h] = refl
        print(f"  Landsat rows {r0}-{r0 + h} of {height}")
    return out


def _integral(a):
    """Summed-area tables of values and of valid-pixel counts (1-px zero pad)."""
    valid = np.isfinite(a)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    n = np.zeros_like(s)
    s[1:, 1:] = np.where(valid, a, 0).cumsum(0).cumsum(1)
    n[1:, 1:] = valid.cumsum(0).cumsum(1)
    return s, n


def _box_mean(s, n, rows, cols, half):
    """Mean over the (2*half+1)^2 box centred at integer (rows, cols)."""
    r0, r1 = rows - half, rows + half + 1
    c0, c1 = cols - half, cols + half + 1
    tot = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    cnt = n[r1, c1] - n[r0, c1] - n[r1, c0] + n[r0, c0]
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    ok = cnt >= MIN_VALID * (2 * half + 1) ** 2
    return np.where(ok, mean, np.nan)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ── main ──────────────────────────────────────────────────────────────────────

def main(scene_key):
    scene  = SCENES[scene_key]
    region = TEST_REGIONS[scene_key]
    out_dir = os.path.join(NR_ROOT, scene["output_dir"], "geolocation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"VIIRS geolocation check — {scene_key}")
    print(f"Landsat: {scene['landsat_scene']}")
    print(f"Test region: lat {region[0]}–{region[1]}  lon {region[2]}–{region[3]}")
    print("=" * 60)

    # 1. VIIRS pixels (swath geometry, own lat/lon)
    lat, lon, bands, angles, meta = load_viirs_training_pair(
        os.path.join(NR_ROOT, scene["gitco"]),
        os.path.join(NR_ROOT, scene["gimgo"]),
        clip_bbox=region)
    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat > -90)
    for vb, _ in BAND_PAIRS:
        ok &= np.isfinite(bands[vb])
    vx, vy = to_utm.transform(lon[ok], lat[ok])
    vvals = {vb: bands[vb][ok] for vb, _ in BAND_PAIRS}
    vza = angles["VZA"][ok]
    print(f"\nVIIRS pixels with valid I1–I3: {ok.sum():,}  "
          f"(VZA {np.nanmin(vza):.1f}–{np.nanmax(vza):.1f}°)")

    # 2. Landsat patch on its 30 m UTM grid, covering region + shift margin
    rx, ry = to_utm.transform([region[2], region[3], region[2], region[3]],
                              [region[0], region[0], region[1], region[1]])
    x0 = math.floor((min(rx) - MARGIN_M) / LS_RES) * LS_RES
    x1 = math.ceil((max(rx) + MARGIN_M) / LS_RES) * LS_RES
    y0 = math.floor((min(ry) - MARGIN_M) / LS_RES) * LS_RES
    y1 = math.ceil((max(ry) + MARGIN_M) / LS_RES) * LS_RES
    W, H = int((x1 - x0) / LS_RES), int((y1 - y0) / LS_RES)
    print(f"\nFetching Landsat {W} × {H} px (30 m) via Earth Engine...")
    ls = _fetch_landsat(scene["landsat_scene"], scene["ee_collection"], x0, y1, W, H)
    tables = {b: _integral(ls[b]) for _, b in BAND_PAIRS}

    # Keep VIIRS pixels whose every shifted box stays inside the patch
    half = BOX_PX // 2
    inside = ((vx - MAX_SHIFT_M - half * LS_RES > x0) & (vx + MAX_SHIFT_M + half * LS_RES < x1) &
              (vy - MAX_SHIFT_M - half * LS_RES > y0) & (vy + MAX_SHIFT_M + half * LS_RES < y1))
    vx, vy = vx[inside], vy[inside]
    vvals = {k: v[inside] for k, v in vvals.items()}
    print(f"VIIRS pixels used: {vx.size:,}")

    # 3. Correlation for every shift
    shifts = np.arange(-MAX_SHIFT_M, MAX_SHIFT_M + 1, SHIFT_STEP)
    R = {vb: np.full((shifts.size, shifts.size), np.nan) for vb, _ in BAND_PAIRS}
    print(f"\nTesting {shifts.size}×{shifts.size} shifts (±{MAX_SHIFT_M:.0f} m, {SHIFT_STEP:.0f} m steps)...")
    for i, dy in enumerate(shifts):
        rows = np.floor((y1 - (vy + dy)) / LS_RES).astype(int)
        for j, dx in enumerate(shifts):
            cols = np.floor(((vx + dx) - x0) / LS_RES).astype(int)
            for vb, lb in BAND_PAIRS:
                s, n = tables[lb]
                R[vb][i, j] = _corr(vvals[vb], _box_mean(s, n, rows, cols, half))
    Rmean = np.nanmean(np.stack([R[vb] for vb, _ in BAND_PAIRS]), axis=0)

    zi = int(np.argmin(np.abs(shifts)))
    lines = [f"VIIRS geolocation check — {scene_key}",
             f"Landsat {scene['landsat_scene']}",
             f"Region lat {region[0]}–{region[1]} lon {region[2]}–{region[3]}, "
             f"{vx.size} VIIRS pixels, box {BOX_PX * LS_RES:.0f} m",
             "",
             f"{'band pair':14} {'r at 0 m':>9} {'best r':>8} {'best shift east, north (m)':>28}"]
    for vb, lb in BAND_PAIRS + [("mean", "")]:
        Rb = Rmean if vb == "mean" else R[vb]
        bi, bj = np.unravel_index(np.nanargmax(Rb), Rb.shape)
        name = "mean of 3" if vb == "mean" else f"{vb} vs {lb}"
        lines.append(f"{name:14} {Rb[zi, zi]:9.3f} {Rb[bi, bj]:8.3f} "
                     f"{shifts[bj]:+13.0f}, {shifts[bi]:+.0f}")
    bi, bj = np.unravel_index(np.nanargmax(Rmean), Rmean.shape)
    best_dx, best_dy = shifts[bj], shifts[bi]
    lines += ["", f"Best overall shift: {best_dx:+.0f} m east, {best_dy:+.0f} m north "
                  f"({math.hypot(best_dx, best_dy):.0f} m total)"]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out_dir, "geolocation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 4. Figures: correlation surface + VIIRS vs Landsat side by side
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(Rmean, origin="lower", cmap="viridis",
                   extent=[shifts[0], shifts[-1], shifts[0], shifts[-1]])
    ax.axhline(0, color="white", lw=0.6); ax.axvline(0, color="white", lw=0.6)
    ax.plot(best_dx, best_dy, "r+", markersize=14, markeredgewidth=2)
    ax.set_xlabel("VIIRS shift east (m)"); ax.set_ylabel("VIIRS shift north (m)")
    ax.set_title(f"Mean correlation VIIRS vs Landsat (I1–I3)\n{scene_key} — red + = best shift")
    plt.colorbar(im, ax=ax, label="r")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_vs_shift.png"), dpi=150)
    plt.close()

    rows0 = np.floor((y1 - vy) / LS_RES).astype(int)
    cols0 = np.floor((vx - x0) / LS_RES).astype(int)
    s, n = tables["SR_B5"]
    ls_at_viirs = _box_mean(s, n, rows0, cols0, half)
    nodes = _load_nodes(DEFAULT_RIVER)
    nx, ny = to_utm.transform([p["lon"] for p in nodes], [p["lat"] for p in nodes])

    fig, axes = plt.subplots(1, 2, figsize=(10, 12), sharex=True, sharey=True)
    for ax, vals, title in [
            (axes[0], vvals["I2"], "VIIRS I2 (NIR) at GITCO positions"),
            (axes[1], ls_at_viirs, "Landsat B5 (NIR), 390 m box mean\nat the same positions")]:
        lo, hi = np.nanpercentile(vals, [2, 98])
        ax.scatter(vx / 1000, vy / 1000, c=vals, s=1.2, marker="s",
                   cmap="gray", vmin=lo, vmax=hi, linewidths=0)
        ax.plot(np.array(nx) / 1000, np.array(ny) / 1000, ".", color="yellow", markersize=0.8)
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xlabel("UTM 6N easting (km)")
    axes[0].set_ylabel("UTM 6N northing (km)")
    fig.suptitle(f"{scene_key} — SWORD centerline in yellow", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viirs_vs_landsat_nir.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VIIRS geolocation check vs Landsat.")
    p.add_argument("--scene", default="2024-06-12-npp", choices=list(TEST_REGIONS.keys()))
    main(p.parse_args().scene)
