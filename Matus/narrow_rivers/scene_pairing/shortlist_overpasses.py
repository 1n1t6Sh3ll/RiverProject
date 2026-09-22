"""
Shortlist VIIRS overpasses for a Landsat scene over one SWORD river corridor.

Metadata only — no granule is downloaded. For every GITCO granule of the date
(NPP, NOAA-20, NOAA-21 buckets, 17:00–24:00 UTC) it reads, over S3 byte-range:
  - granule attributes (day/night, ascending/descending, lat bounds, times)
  - SCPosition: spacecraft ECEF position per scan (48 × 3 floats)
and computes, for every SWORD node of the river:
  - which scan saw it (closest approach of the spacecraft)
  - scan angle (must be within the VIIRS ±56.06° swath)
  - view zenith angle (angle between the local vertical and the line of sight)

Output: one row per granule that sees any part of the corridor, ranked by
coverage (full first), then mean VZA. Time gap to Landsat is reported.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\scene_pairing\\shortlist_overpasses.py
        --landsat LC08_L2SP_073011_20240612_20240628_02_T1
"""

import os, sys, csv, math, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import boto3
from botocore import UNSIGNED
from botocore.client import Config

NR_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # narrow_rivers/
PIPELINE_ROOT = os.path.dirname(NR_ROOT)                                      # Matus/
sys.path.insert(0, PIPELINE_ROOT)

from auto_pipeline import (BUCKETS, S3RangeReader, list_gitco_keys,
                           parse_time_window, in_time_window,
                           GITCO_ATTR_PATH, DEFAULT_TIME_WINDOW)

NODES_CSV     = os.path.join(NR_ROOT, "data", "sword_nodes_75_300m_csv.csv")
OUTPUT_DIR    = os.path.join(NR_ROOT, "scene_pairing", "output")
EE_PROJECT    = "noaa-river-ice"

DEFAULT_RIVER = "Sagavanirktok River"
MAX_SCAN_DEG  = 56.06        # VIIRS I-band swath half-angle
MAX_WORKERS   = 16



class _SmallBlockReader(S3RangeReader):
    """auto_pipeline's reader with a 64 KB read-ahead instead of 1 MB — the
    header + SCPosition need only a few small reads (~0.4 s vs ~4 s)."""
    BLOCK = 64 * 1024


WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_nodes(river):
    lats, lons = [], []
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["river_name"] == river:
                lats.append(float(row["lat"]))
                lons.append(float(row["lon"]))
    return np.array(lats), np.array(lons)


def _ecef(lat_deg, lon_deg):
    """Ground points (height 0) to ECEF metres + local vertical unit vectors."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    xyz = np.column_stack([
        n * np.cos(lat) * np.cos(lon),
        n * np.cos(lat) * np.sin(lon),
        n * (1 - WGS84_E2) * np.sin(lat),
    ])
    up = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])
    return xyz, up


def _landsat_info(product_id):
    """Footprint polygon (lon, lat) and acquisition datetime (UTC) via EE."""
    import ee
    ee.Initialize(project=EE_PROJECT)
    coll = ("LANDSAT/LC09/C02/T1_L2" if product_id.startswith("LC09")
            else "LANDSAT/LC08/C02/T1_L2")
    img = (ee.ImageCollection(coll)
             .filter(ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id))
             .first())
    props = img.getInfo()["properties"]
    footprint = props["system:footprint"]["coordinates"]
    t = props["SCENE_CENTER_TIME"][:8]            # "HH:MM:SS.xxxxxxZ"
    dt = datetime.strptime(f"{props['DATE_ACQUIRED']} {t}", "%Y-%m-%d %H:%M:%S")
    return footprint, dt


def _read_granule(s3, sat, bucket, key, size, lat_rng):
    """Granule attributes + per-scan spacecraft position. None if not a
    daytime ascending granule reaching the corridor's latitude band."""
    try:
        with h5py.File(_SmallBlockReader(s3, bucket, key, size), "r") as h5:
            a = h5[GITCO_ATTR_PATH].attrs
            day  = a["N_Day_Night_Flag"][0][0].decode()
            asc  = int(a["Ascending/Descending_Indicator"][0][0]) == 0
            s    = float(a["South_Bounding_Coordinate"][0][0])
            n    = float(a["North_Bounding_Coordinate"][0][0])
            if day == "Night" or not asc or n < lat_rng[0] or s > lat_rng[1]:
                return None
            d0 = a["Beginning_Date"][0][0].decode()
            t0 = a["Beginning_Time"][0][0].decode()[:6]
            t1 = a["Ending_Time"][0][0].decode()[:6]
            pos = h5["All_Data/VIIRS-IMG-GEO-TC_All/SCPosition"][:].astype(np.float64)
        n_scans = pos.shape[0]
        ok = np.all(np.isfinite(pos), axis=1) & (np.linalg.norm(pos, axis=1) > 6.5e6)
        return {
            "satellite": sat, "bucket": bucket, "key": key,
            "start": datetime.strptime(d0 + t0, "%Y%m%d%H%M%S"),
            "end":   datetime.strptime(d0 + t1, "%Y%m%d%H%M%S"),
            "pos": pos, "pos_ok": ok, "n_scans": n_scans,
        }
    except Exception as exc:
        print(f"  [warn] {os.path.basename(key)}: {exc}")
        return None


def _corridor_geometry(g, node_xyz, node_up):
    """Per node: covered?, VZA, scan index of closest approach."""
    idx_ok = np.where(g["pos_ok"])[0]
    if idx_ok.size < 3:
        return None
    pos = g["pos"][idx_ok]                                   # (S, 3)
    d = np.linalg.norm(node_xyz[:, None, :] - pos[None, :, :], axis=2)  # (N, S)
    j = np.argmin(d, axis=1)
    sat = pos[j]                                             # (N, 3)
    los = sat - node_xyz
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    vza = np.degrees(np.arccos(np.clip(np.sum(los * node_up, axis=1), -1, 1)))
    # Scan angle at the spacecraft: between nadir (toward Earth centre) and node
    to_node = node_xyz - sat
    to_node /= np.linalg.norm(to_node, axis=1, keepdims=True)
    nadir = -sat / np.linalg.norm(sat, axis=1, keepdims=True)
    scan = np.degrees(np.arccos(np.clip(np.sum(to_node * nadir, axis=1), -1, 1)))
    # Closest approach at the first/last scan means the node lies before/after
    # this granule along-track.
    inside_track = (j > 0) & (j < idx_ok.size - 1)
    covered = inside_track & (scan <= MAX_SCAN_DEG)
    return covered, vza, idx_ok[j]


# ── main ──────────────────────────────────────────────────────────────────────

def main(landsat_id, river, time_window):
    node_lat, node_lon = _load_nodes(river)
    if node_lat.size == 0:
        print(f"ERROR: No SWORD nodes for river '{river}'")
        sys.exit(1)
    node_xyz, node_up = _ecef(node_lat, node_lon)

    print("=" * 60)
    print("VIIRS overpass shortlist (metadata only)")
    print(f"River: {river}  ({node_lat.size} SWORD nodes)")
    print(f"Landsat: {landsat_id}")

    footprint, ls_dt = _landsat_info(landsat_id)
    from matplotlib.path import Path
    inside_ls = Path(np.array(footprint)).contains_points(
        np.column_stack([node_lon, node_lat]))
    print(f"Landsat acquired: {ls_dt:%Y-%m-%d %H:%M:%S} UTC")
    print(f"SWORD nodes inside Landsat footprint: {inside_ls.sum()}/{node_lat.size}")
    print("=" * 60)

    date = ls_dt.strftime("%Y-%m-%d")
    lat_rng = (node_lat.min(), node_lat.max())
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                          max_pool_connections=MAX_WORKERS))

    jobs = []
    for sat, bucket in BUCKETS.items():
        keys = [(k, s) for k, s in list_gitco_keys(s3, bucket, date)
                if in_time_window(k, *time_window)]
        print(f"{sat}: {len(keys)} granules in {time_window[0]:04d}-{time_window[1]:04d} UTC")
        jobs += [(sat, bucket, k, s) for k, s in keys]

    print(f"\nReading headers + spacecraft positions of {len(jobs)} granules...")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        granules = [g for g in ex.map(
            lambda j: _read_granule(s3, *j, lat_rng), jobs) if g is not None]
    print(f"  {len(granules)} daytime ascending granules reach the corridor latitudes "
          f"({time.time() - t_start:.0f} s)")

    rows = []
    for g in granules:
        geo = _corridor_geometry(g, node_xyz, node_up)
        if geo is None:
            continue
        covered, vza, scan_idx = geo
        if not covered.any():
            continue
        # Observation time over the corridor from the scan index
        scan_dt = (g["end"] - g["start"]) / g["n_scans"]
        t_obs = g["start"] + scan_dt * float(np.median(scan_idx[covered]) + 0.5)
        gap = (t_obs - ls_dt).total_seconds() / 60.0
        name = os.path.basename(g["key"])
        rows.append({
            "river": river,
            "landsat_product_id": landsat_id,
            "landsat_datetime_utc": ls_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "satellite": g["satellite"],
            "viirs_t_code": name.split("_")[3],
            "viirs_datetime_utc": t_obs.strftime("%Y-%m-%d %H:%M:%S"),
            "gap_minutes": round(gap, 1),
            "coverage_pct": round(100.0 * covered.mean(), 1),
            "mean_vza_over_corridor": round(float(vza[covered].mean()), 1),
            "max_vza_over_corridor": round(float(vza[covered].max()), 1),
            "gitco_file": name,
        })

    rows.sort(key=lambda r: (-r["coverage_pct"], r["mean_vza_over_corridor"]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"candidates_{landsat_id}.csv")
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'sat':4} {'t-code':9} {'obs UTC':9} {'gap min':>8} {'cover %':>8} "
          f"{'mean VZA':>9} {'max VZA':>8}")
    for r in rows:
        print(f"{r['satellite']:4} {r['viirs_t_code']:9} {r['viirs_datetime_utc'][11:]:9} "
              f"{r['gap_minutes']:8.1f} {r['coverage_pct']:8.1f} "
              f"{r['mean_vza_over_corridor']:9.1f} {r['max_vza_over_corridor']:8.1f}")
    print(f"\nSaved: {out}" if rows else "\nNo granule sees the corridor.")
    print("Partial-coverage rows: the rest of the corridor is in the adjacent "
          "granule of the same pass.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Shortlist VIIRS overpasses (metadata only).")
    p.add_argument("--landsat", default="LC08_L2SP_073011_20240612_20240628_02_T1",
                   help="Landsat LANDSAT_PRODUCT_ID")
    p.add_argument("--river", default=DEFAULT_RIVER, help="SWORD river_name")
    p.add_argument("--time-window", default=DEFAULT_TIME_WINDOW,
                   help="UTC HHMM-HHMM filter on granule start time")
    a = p.parse_args()
    main(a.landsat, a.river, parse_time_window(a.time_window))
