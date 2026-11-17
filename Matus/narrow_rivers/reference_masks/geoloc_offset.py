"""
Measure the VIIRS-Landsat geolocation offset of one date, WITH its direction.

patches/add_viirs.py runs the same search but stores only the length of the best
shift (patch_tracking.csv: geoloc_offset_m), on 50 m steps. This script repeats it
read-only -- nothing in the patch folder is written -- and refines it on 10 m
steps around the coarse best.

Convention (same as add_viirs.py): the best (east, north) is where the VIIRS
pixels should be MOVED to agree with Landsat, i.e. a VIIRS cell on our grid
actually sees the ground (east, north) metres away from where the grid puts it.

Output: reference_masks/<date>/<date>_geoloc_offset.json

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\geoloc_offset.py --date 2024-06-22
"""

import os, sys, json, math, argparse
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(os.path.dirname(HERE), "patches")
sys.path.insert(0, PATCHES)
import patch_config as C
import add_viirs as AV          # _granule, _integral, _box_mean, _corr, BAND_PAIRS, PASSES


def main(date, fine_step, fine_radius):
    sat, t_code, _, _ = AV.PASSES[date]
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    lg = C.landsat_grid()
    with rasterio.open(os.path.join(patch, f"{date}_sr.tif")) as s:
        sr = s.read().astype(np.float32)
    lat_mid = lg["y1"] - lg["H"] * lg["res"] / 2
    m_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    m_lat = 110_574.0
    hx = max(1, int(round(0.5 * C.RESOLUTION_M / (lg["res"] * m_lon))))
    hy = max(1, int(round(0.5 * C.RESOLUTION_M / (lg["res"] * m_lat))))
    tabs = {vb: AV._integral(sr[i]) for vb, i in AV.BAND_PAIRS}

    gitco, gimgo = AV._granule(sat, t_code)
    pad = 0.25
    clip = (C.BOX[1] - pad, C.BOX[3] + pad, C.BOX[0] - pad, C.BOX[2] + pad)
    lat, lon, bands, _, _ = AV.load_viirs_training_pair(gitco, gimgo, clip_bbox=clip)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat > -90)
    for vb, _ in AV.BAND_PAIRS:
        ok &= np.isfinite(bands[vb])
    ok &= ((lon > C.BOX[0] - 0.02) & (lon < C.BOX[2] + 0.02) &
           (lat > C.BOX[1] - 0.02) & (lat < C.BOX[3] + 0.02))
    vlon, vlat = lon[ok], lat[ok]
    vv = {vb: bands[vb][ok] for vb, _ in AV.BAND_PAIRS}

    def r_at(dx, dy):
        rows = np.floor((lg["y1"] - (vlat + dy / m_lat)) / lg["res"]).astype(int)
        cols = np.floor(((vlon + dx / m_lon) - lg["x0"]) / lg["res"]).astype(int)
        return float(np.nanmean([AV._corr(vv[vb], AV._box_mean(tabs[vb], rows, cols, hy, hx))
                                 for vb, _ in AV.BAND_PAIRS]))

    def search(cx, cy, radius, step):
        steps = np.arange(-radius, radius + 1e-6, step)
        grid = {(cx + dx, cy + dy): r_at(cx + dx, cy + dy) for dy in steps for dx in steps}
        return max(grid, key=grid.get), grid

    print(f"Geolocation offset — {date} ({sat} {t_code}), {ok.sum()} VIIRS pixels over the box")
    (cx, cy), coarse = search(0.0, 0.0, AV.MAX_SHIFT_M, AV.SHIFT_STEP_M)
    print(f"  coarse ({AV.SHIFT_STEP_M:.0f} m steps): best {cx:+.0f} m E, {cy:+.0f} m N, "
          f"r={coarse[(cx, cy)]:.4f}  (r at 0,0 = {coarse[(0.0, 0.0)]:.4f})")
    (bx, by), fine = search(cx, cy, fine_radius, fine_step)
    rb = fine[(bx, by)]
    print(f"  fine   ({fine_step:.0f} m steps):  best {bx:+.0f} m E, {by:+.0f} m N, r={rb:.4f}, "
          f"length {math.hypot(bx, by):.0f} m")
    # how sharp is the peak: shifts whose r is within 0.002 of the best
    near = [k for k, v in fine.items() if v >= rb - 0.002]
    ex = max(abs(k[0] - bx) for k in near)
    ny = max(abs(k[1] - by) for k in near)
    print(f"  peak width (r within 0.002 of best): ±{ex:.0f} m E-W, ±{ny:.0f} m N-S")

    out = {"date": date, "granule": f"{sat}_{t_code}",
           "east_m": bx, "north_m": by, "length_m": round(math.hypot(bx, by), 1),
           "r_best": round(rb, 4), "r_zero": round(coarse[(0.0, 0.0)], 4),
           "coarse_east_m": cx, "coarse_north_m": cy, "fine_step_m": fine_step,
           "peak_halfwidth_e_m": ex, "peak_halfwidth_n_m": ny,
           "meaning": "a VIIRS cell sees the ground east_m/north_m away from its grid position"}
    path = os.path.join(HERE, date, f"{date}_geoloc_offset.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"  wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VIIRS-Landsat offset with direction.")
    p.add_argument("--date", required=True)
    p.add_argument("--fine-step", type=float, default=10.0)
    p.add_argument("--fine-radius", type=float, default=100.0)
    a = p.parse_args()
    main(a.date, a.fine_step, a.fine_radius)
