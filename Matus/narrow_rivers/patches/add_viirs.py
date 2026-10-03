"""
Add the chosen VIIRS pass to each patch folder.

Per date:
  1. Geolocation check against that date's Landsat patch. VIIRS I1/I2/I3 have
     Landsat twins (B4/B5/B6); each VIIRS pixel is compared with the Landsat
     mean over a ~375 x 375 m ground box at its GITCO position, over a grid of
     shifts. The best shift should be near zero.
  2. Resample I1..I5 + SZA/SAA/VZA/VAA onto the patch's 375 m grid
     (EPSG:4326, Alaska grid) -> <date>_viirs_375m.tif, the model input.
  3. Update README.txt and patch_tracking.csv.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\add_viirs.py
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\add_viirs.py --dates 2024-06-22
"""

import os, sys, csv, math, glob, argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import patch_config as C

NR_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(NR_ROOT))          # Matus/ for the loader
from viirs_training_loader import load_viirs_training_pair

# date -> (satellite, t-code, gap minutes vs Landsat, mean VZA over the box)
PASSES = {
    "2023-07-24": ("j02", "t2219233", 68.6, 12.0),
    "2023-10-04": ("npp", "t2156572", 44.3, 10.2),
    "2024-04-21": ("j02", "t2220285", 68.6, 12.8),
    "2024-06-22": ("j01", "t2208041", 44.4, 0.9),
    "2026-03-26": ("j01", "t2133484", 21.4, 28.6),
}

BAND_PAIRS = [("I1", 3), ("I2", 4), ("I3", 5)]   # VIIRS band -> band index in _sr.tif
BANDS  = ["I1", "I2", "I3", "I4", "I5"]
ANGLES = ["SZA", "SAA", "VZA", "VAA"]
MAX_SHIFT_M, SHIFT_STEP_M = 750.0, 50.0
MIN_VALID = 0.9


def _granule(sat, t_code):
    gitco = glob.glob(os.path.join(NR_ROOT, "data", "viirs", f"GITCO_{sat}_*_{t_code}_*.h5"))
    gimgo = glob.glob(os.path.join(NR_ROOT, "data", "viirs", f"GIMGO-*_{sat}_*_{t_code}_*.h5"))
    if not gitco or not gimgo:
        raise SystemExit(f"ERROR: granule {sat} {t_code} not found in data/viirs")
    return gitco[0], gimgo[0]


def _integral(a):
    valid = np.isfinite(a)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    n = np.zeros_like(s)
    s[1:, 1:] = np.where(valid, a, 0).cumsum(0).cumsum(1)
    n[1:, 1:] = valid.cumsum(0).cumsum(1)
    return s, n


def _box_mean(tab, rows, cols, hy, hx):
    s, n = tab
    r0, r1 = np.clip(rows - hy, 0, s.shape[0] - 1), np.clip(rows + hy + 1, 0, s.shape[0] - 1)
    c0, c1 = np.clip(cols - hx, 0, s.shape[1] - 1), np.clip(cols + hx + 1, 0, s.shape[1] - 1)
    tot = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    cnt = n[r1, c1] - n[r0, c1] - n[r1, c0] + n[r0, c0]
    full = (2 * hy + 1) * (2 * hx + 1)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return np.where(cnt >= MIN_VALID * full, mean, np.nan)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 50 else np.nan


def _resample(lat, lon, data, g):
    from pyresample import geometry, kd_tree
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(data)
    swath = geometry.SwathDefinition(lons=np.ma.masked_array(lon, mask=~valid),
                                     lats=np.ma.masked_array(lat, mask=~valid))
    area = geometry.AreaDefinition(
        "patch", "patch", "patch", {"proj": "longlat", "datum": "WGS84"},
        g["W"], g["H"],
        (g["x0"], g["y1"] - g["H"] * g["res"], g["x0"] + g["W"] * g["res"], g["y1"]))
    out = kd_tree.resample_nearest(
        swath, np.ma.masked_array(data.astype(np.float32), mask=~valid), area,
        radius_of_influence=600, epsilon=0.1, fill_value=np.nan)
    return np.asarray(np.ma.filled(out, np.nan), dtype=np.float32)


def process(date):
    sat, t_code, gap, vza_nom = PASSES[date]
    patch_dir = os.path.join(NR_ROOT, "patches", f"nenana_{date.replace('-', '')}")
    sr_path = os.path.join(patch_dir, f"{date}_sr.tif")
    if not os.path.exists(sr_path):
        raise SystemExit(f"ERROR: {sr_path} missing — run build_patches.py first")

    lg, vg = C.landsat_grid(), C.viirs_grid()
    print(f"\n{date}  {sat} {t_code}  (gap {gap:+.0f} min)")

    with rasterio.open(sr_path) as src:
        sr = src.read().astype(np.float32)
    lat_mid = lg["y1"] - lg["H"] * lg["res"] / 2
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    m_per_deg_lat = 110_574.0
    # a ~375 x 375 m ground box, in Landsat cells (13 x 30 m here)
    hx = max(1, int(round(0.5 * C.RESOLUTION_M / (lg["res"] * m_per_deg_lon))))
    hy = max(1, int(round(0.5 * C.RESOLUTION_M / (lg["res"] * m_per_deg_lat))))
    tabs = {vb: _integral(sr[i]) for vb, i in BAND_PAIRS}

    gitco, gimgo = _granule(sat, t_code)
    pad = 0.25
    clip = (C.BOX[1] - pad, C.BOX[3] + pad, C.BOX[0] - pad, C.BOX[2] + pad)
    lat, lon, bands, angles, meta = load_viirs_training_pair(gitco, gimgo, clip_bbox=clip)

    ok = np.isfinite(lat) & np.isfinite(lon) & (lat > -90)
    for vb, _ in BAND_PAIRS:
        ok &= np.isfinite(bands[vb])
    ok &= ((lon > C.BOX[0] - 0.02) & (lon < C.BOX[2] + 0.02) &
           (lat > C.BOX[1] - 0.02) & (lat < C.BOX[3] + 0.02))
    vlon, vlat = lon[ok], lat[ok]
    vvals = {vb: bands[vb][ok] for vb, _ in BAND_PAIRS}
    print(f"  VIIRS pixels over the box: {ok.sum()}")

    # geolocation: shift the VIIRS positions and see where agreement peaks
    steps = np.arange(-MAX_SHIFT_M, MAX_SHIFT_M + 1, SHIFT_STEP_M)
    best = (-2, 0.0, 0.0)
    r_zero = None
    for dy in steps:
        rows = np.floor((lg["y1"] - (vlat + dy / m_per_deg_lat)) / lg["res"]).astype(int)
        for dx in steps:
            cols = np.floor(((vlon + dx / m_per_deg_lon) - lg["x0"]) / lg["res"]).astype(int)
            rs = [_corr(vvals[vb], _box_mean(tabs[vb], rows, cols, hy, hx))
                  for vb, _ in BAND_PAIRS]
            r = float(np.nanmean(rs))
            if dx == 0 and dy == 0:
                r_zero = r
            if r > best[0]:
                best = (r, dx, dy)
    off = math.hypot(best[1], best[2])
    print(f"  geolocation: r={r_zero:.3f} at 0 m, best r={best[0]:.3f} at "
          f"{best[1]:+.0f} m E, {best[2]:+.0f} m N ({off:.0f} m)")
    if off > 375:
        print("  WARNING: offset larger than one VIIRS pixel — check before using this date")

    # resample onto the patch's 375 m grid
    stack, names = [], []
    for name in BANDS + ANGLES:
        src = bands.get(name, angles.get(name))
        stack.append(_resample(lat, lon, src, vg))
        names.append(name)
    stack = np.stack(stack)
    out = os.path.join(patch_dir, f"{date}_viirs_375m.tif")
    with rasterio.open(out, "w", driver="GTiff", height=vg["H"], width=vg["W"],
                       count=len(names), dtype="float32", crs=vg["crs"],
                       transform=from_origin(vg["x0"], vg["y1"], vg["res"], vg["res"]),
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(stack)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
    filled = int(np.isfinite(stack[0]).sum())
    print(f"  {os.path.basename(out)}  {filled}/{vg['W']*vg['H']} cells, "
          f"{os.path.getsize(out)/1e6:.1f} MB")

    # README + tracking
    readme_path = os.path.join(patch_dir, "README.txt")
    text = open(readme_path, encoding="utf-8").read()
    viirs_block = (f"  Pass      : {sat.upper()} {t_code}, {meta['date'][:4]}-{meta['date'][4:6]}-"
                   f"{meta['date'][6:]}, {gap:+.0f} min vs Landsat\n"
                   f"  View angle: ~{vza_nom:.0f} deg over the box\n"
                   f"  Geolocation check vs this Landsat scene: best shift {off:.0f} m "
                   f"(r={best[0]:.2f})\n"
                   f"  Grid file : {date}_viirs_375m.tif — I1..I5 + SZA/SAA/VZA/VAA\n"
                   f"              on the same 375 m cells the fractions will use.\n"
                   f"  The granule itself stays out of this folder (~230 MB).\n")
    text = text.split("VIIRS\n")[0] + "VIIRS\n" + viirs_block
    open(readme_path, "w", encoding="utf-8").write(text)
    return {"date": date, "viirs_granule": f"{sat}_{t_code}",
            "viirs_gap_min": round(gap, 1), "viirs_vza": vza_nom,
            "geoloc_offset_m": round(off)}


def main(dates):
    results = [process(d) for d in dates]
    track = os.path.join(NR_ROOT, "patches", "patch_tracking.csv")
    rows = {r["date"]: r for r in csv.DictReader(open(track, newline="", encoding="utf-8"))}
    cols = list(next(iter(rows.values())).keys())
    for r in results:
        rows[r["date"]].update({k: v for k, v in r.items() if k != "date"})
        rows[r["date"]]["status"] = "ready to label"
    with open(track, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in sorted(rows):
            w.writerow(rows[d])
    print(f"\ntracking table updated: {track}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add the chosen VIIRS pass to each patch.")
    p.add_argument("--dates", nargs="+", default=sorted(PASSES), choices=sorted(PASSES))
    main(p.parse_args().dates)
