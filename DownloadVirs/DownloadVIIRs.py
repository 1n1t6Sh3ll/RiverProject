#!/usr/bin/env python3
"""
Download a NOAA CLASS-style aggregated VIIRS granule from NOAA's PDS S3 buckets.

Usage:
    python DownloadVIIRs.py SVI02_j01_d20240904_t2357037_e0002437_b35223_c20240905002744224000
    python DownloadVIIRs.py REF1 REF2
    python DownloadVIIRs.py --granule-file granules.txt

Each input ref names an AGGREGATED granule (start time + end time spans
multiple individual S3 granules, each ~85 s long). The script:
    1. Lists every individual S3 granule in the orbit between tstart and tend
    2. Downloads GITCO + SVI01..SVI05 for each
    3. Vertically concatenates them into one aggregated GITCO file and one
       aggregated GIMGO-SVI01-...-SVI05 file -- exactly the format
       data/viirs/<YYYYMMDD_tTSTART>/{GITCO_*.h5, GIMGO-SVI*.h5} that
       viirs_training_loader.py expects.
    4. Writes coverage_map.html in the same folder as the script

The XML AOI is informational: it overlays the AOI on the map and prints a
YES/NO overlap warning. The leading product token (SVI02 / GITCO / etc.)
in the ref is ignored.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import math
import re
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import folium
import h5py
import numpy as np
import s3fs
from botocore import UNSIGNED
from botocore.config import Config


HERE        = Path(__file__).resolve().parent
DEFAULT_OUT = HERE
DEFAULT_XML = HERE / "VIIRS_SDR.xml"

SAT_TO_BUCKET = {
    "j01": "noaa-nesdis-n20-pds",
    "j02": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}

PRODUCTS = [
    ("SVI01", "VIIRS-I1-SDR"),
    ("SVI02", "VIIRS-I2-SDR"),
    ("SVI03", "VIIRS-I3-SDR"),
    ("SVI04", "VIIRS-I4-SDR"),
    ("SVI05", "VIIRS-I5-SDR"),
    ("GITCO", "VIIRS-IMG-GEO-TC"),
]

GRANULE_RE = re.compile(
    r"(j01|j02|npp)_d(\d{8})_t(\d{7})_e(\d{7})_b(\d+)(?:_c(\d+))?"
)
WORKERS = 12


# ── parsing & time helpers ────────────────────────────────────────────────────

def parse_ref(ref):
    """Parse aggregated ref. End time (e...) is REQUIRED here -- it's the span."""
    m = GRANULE_RE.search(ref)
    if not m:
        raise ValueError(
            f"Cannot parse VIIRS ref (need t<start>_e<end>_b<orbit>): {ref!r}"
        )
    sat, date, tstart, tend, orbit, creation = m.groups()
    return {
        "sat":      sat,
        "date":     date,
        "tstart":   tstart,
        "tend":     tend,
        "orbit":    orbit,
        "creation": creation,
        "sig":      f"{sat}_d{date}_t{tstart}_e{tend}_b{orbit}",
        "subdir":   f"{date}_t{tstart}",
    }


def parse_aoi(xml_path):
    root  = ET.parse(xml_path).getroot()
    items = {}
    for item in root.findall("item"):
        items.setdefault(item.get("group"), (item.text or "").strip())
    return {
        "lat_min": float(items["slat"]),
        "lat_max": float(items["nlat"]),
        "lon_min": float(items["wlon"]),
        "lon_max": float(items["elon"]),
    }


def to_dt(date_str, t_str):
    """date='20240904', t='2357037' -> datetime(..., microsecond=...)."""
    return (datetime.strptime(f"{date_str}_{t_str[:6]}", "%Y%m%d_%H%M%S")
            + timedelta(microseconds=int(t_str[6]) * 100_000))


def next_date(yyyymmdd):
    return (datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


# ── S3 ────────────────────────────────────────────────────────────────────────

def list_individual_granules(s3, bucket, agg, pad=0):
    """Return individual granule dicts for the given aggregate's orbit.

    First lists every individual granule in the orbit on agg.date and the
    next day (orbit may cross midnight). Then keeps the contiguous run
    [first_in_window - pad ... last_in_window + pad], so `pad` extra
    granules are added on each side of [agg.tstart, agg.tend]."""
    range_start = to_dt(agg["date"], agg["tstart"])
    end_date    = next_date(agg["date"]) if agg["tend"] < agg["tstart"] else agg["date"]
    range_end   = to_dt(end_date, agg["tend"])

    orbit_tok = f"_b{agg['orbit']}_"
    seen, all_refs = set(), []
    for d in (agg["date"], next_date(agg["date"])):
        prefix = (f"VIIRS-IMG-GEO-TC/{d[:4]}/{d[4:6]}/{d[6:8]}/"
                  f"GITCO_{agg['sat']}_d{d}_")
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if not (k.endswith(".h5") and orbit_tok in k):
                    continue
                m = re.search(r"d(\d{8})_t(\d{7})_e(\d{7})_b(\d+)", k)
                if not m:
                    continue
                this_d, this_t, this_e, this_orb = m.groups()
                key = (this_d, this_t)
                if key in seen:
                    continue
                seen.add(key)
                all_refs.append({
                    "sat":       agg["sat"],
                    "date":      this_d,
                    "tstart":    this_t,
                    "tend":      this_e,
                    "orbit":     this_orb,
                    "subdir":    f"{this_d}_t{this_t}",
                    "gitco_key": k,
                    "_dt":       to_dt(this_d, this_t),
                })
    all_refs.sort(key=lambda r: r["_dt"])

    in_window = [i for i, r in enumerate(all_refs)
                 if range_start <= r["_dt"] <= range_end]
    if not in_window:
        return []
    lo = max(0, in_window[0] - pad)
    hi = min(len(all_refs), in_window[-1] + 1 + pad)
    refs = all_refs[lo:hi]
    for r in refs:
        r.pop("_dt", None)
    return refs


def find_key(s3, bucket, product, file_prefix, g):
    d   = g["date"]
    pre = (f"{product}/{d[:4]}/{d[4:6]}/{d[6:8]}/"
           f"{file_prefix}_{g['sat']}_d{d}_t{g['tstart']}")
    orbit_tok = f"_b{g['orbit']}_"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pre):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".h5") and orbit_tok in k:
                return k
    return None


def download(s3, bucket, key, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))
    return dest


def parallel_download(s3, bucket, plan):
    paths = []
    if not plan:
        return paths
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(download, s3, bucket, k, p): k for k, p in plan}
        for fut in as_completed(futs):
            try:
                paths.append(fut.result())
            except Exception as e:
                print(f"  err {futs[fut]}: {e}")
    return paths


# ── HDF5 ──────────────────────────────────────────────────────────────────────

def _valid_mask(lat, lon):
    return (np.isfinite(lat) & np.isfinite(lon) &
            (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180))


def _aoi_test(lat, lon, aoi):
    valid = _valid_mask(lat, lon)
    if not np.any(valid):
        return False
    return bool(np.any(
        (lat[valid] >= aoi["lat_min"]) & (lat[valid] <= aoi["lat_max"]) &
        (lon[valid] >= aoi["lon_min"]) & (lon[valid] <= aoi["lon_max"])
    ))


def stream_overlaps_aoi(fs, bucket, key, aoi):
    with fs.open(f"{bucket}/{key}", "rb") as f, h5py.File(f, "r") as h:
        g = h["All_Data/VIIRS-IMG-GEO-TC_All"]
        return _aoi_test(g["Latitude"][:], g["Longitude"][:], aoi)


def aggregate_h5(src_paths, out_path):
    """Stack same-named datasets from src files vertically (axis 0).
    Sources are time-sorted by their filename's d/t/e fields, so per-band
    files (SVI01..SVI05) and per-granule files all combine correctly:
    each dataset is filled only by the files that actually contain it."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(p):
        m = re.search(r"_d(\d{8})_t(\d{7})_", Path(p).name)
        return (m.group(1), m.group(2)) if m else ("", "")
    src_paths = sorted(src_paths, key=sort_key)

    srcs = [h5py.File(p, "r") for p in src_paths]
    try:
        # Only process datasets under All_Data/ (Data_Products/ holds HDF5
        # reference arrays the training loader doesn't need and that h5py
        # can't write back via create_dataset(data=...))
        all_paths = set()
        for s in srcs:
            if "All_Data" not in s:
                continue
            def collect(name, obj, _grp="All_Data"):
                if isinstance(obj, h5py.Dataset):
                    all_paths.add(f"{_grp}/{name}")
            s["All_Data"].visititems(collect)

        with h5py.File(out_path, "w") as dst:
            for ds_path in sorted(all_paths):
                arrays = [s[ds_path][...] for s in srcs if ds_path in s]
                if not arrays or arrays[0].dtype == object:
                    continue
                try:
                    combined = np.concatenate(arrays, axis=0)
                except (ValueError, TypeError):
                    combined = arrays[0]   # can't stack -- keep first
                # Match NOAA CLASS storage: gzip-5 + shuffle, chunk on per-granule rows
                kw = {}
                if combined.ndim >= 2 and combined.shape[0] >= len(arrays):
                    rows_per_chunk = combined.shape[0] // len(arrays)
                    kw["chunks"]           = (rows_per_chunk,) + combined.shape[1:]
                    kw["compression"]      = "gzip"
                    kw["compression_opts"] = 5
                    kw["shuffle"]          = True
                dst.create_dataset(ds_path, data=combined, **kw)
    finally:
        for s in srcs:
            s.close()


def aggregated_filename(prefix, agg, src_paths):
    """Build aggregated filename: <prefix>_<sat>_d<date>_t<tstart>_e<tend>_b<orbit>_c<latest>_<suffix>.h5"""
    latest_c = ""
    suffix   = "oeac_ops"
    for p in src_paths:
        m = re.search(r"_c(\d+)_(\w+)\.h5$", Path(p).name)
        if m:
            if m.group(1) > latest_c:
                latest_c = m.group(1)
            suffix = m.group(2)
    if not latest_c:
        latest_c = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "000"
    return (f"{prefix}_{agg['sat']}_d{agg['date']}"
            f"_t{agg['tstart']}_e{agg['tend']}_b{agg['orbit']}"
            f"_c{latest_c}_{suffix}.h5")


def already_done(granule_dir):
    if not granule_dir.is_dir():
        return False
    names = [p.name for p in granule_dir.iterdir()]
    return (any(n.startswith("GITCO_")       for n in names) and
            any(n.startswith("GIMGO-SVI01-") for n in names))


# ── coverage map (folium / Leaflet) ──────────────────────────────────────────

def _swath_boundary(lat, lon, n=400):
    """Trace the four edges of the valid swath in array index order.

    For each row, find leftmost/rightmost valid columns. Walk:
      top row left -> right, right edge top -> bottom (per-row right col),
      bottom row right -> left, left edge bottom -> top (per-row left col).
    Per-row sampling on the long along-track sides keeps the edges tight
    against the actual swath. Cross-track top/bottom rows are sampled
    densely enough to follow dateline curves cleanly."""
    valid = _valid_mask(lat, lon)
    if not np.any(valid):
        return []
    rows_any = np.where(valid.any(axis=1))[0]
    if len(rows_any) == 0:
        return []
    r0, r1 = int(rows_any[0]), int(rows_any[-1])

    cols_top = np.where(valid[r0])[0]
    cols_bot = np.where(valid[r1])[0]
    if not len(cols_top) or not len(cols_bot):
        return []

    sample_cols_top = np.linspace(cols_top[0], cols_top[-1], n).astype(int)
    sample_cols_bot = np.linspace(cols_bot[0], cols_bot[-1], n).astype(int)
    step = max(1, (r1 - r0 + 1) // n)
    sample_rows = list(range(r0, r1 + 1, step))
    if sample_rows[-1] != r1:
        sample_rows.append(r1)

    pts = []
    # top edge: row r0, left -> right
    for c in sample_cols_top:
        if valid[r0, c]:
            pts.append((float(lat[r0, c]), float(lon[r0, c])))
    # right edge: per-row rightmost valid col, top -> bottom (skip r0 already included)
    for r in sample_rows[1:]:
        cols = np.where(valid[r])[0]
        if not len(cols):
            continue
        c = int(cols[-1])
        pts.append((float(lat[r, c]), float(lon[r, c])))
    # bottom edge: row r1, right -> left (skip first col already included)
    for c in sample_cols_bot[::-1][1:]:
        if valid[r1, c]:
            pts.append((float(lat[r1, c]), float(lon[r1, c])))
    # left edge: per-row leftmost valid col, bottom -> top (skip both ends)
    for r in sample_rows[::-1][1:-1]:
        cols = np.where(valid[r])[0]
        if not len(cols):
            continue
        c = int(cols[0])
        pts.append((float(lat[r, c]), float(lon[r, c])))
    return pts


def _split_dateline(pts):
    """Unwrap polygon longitudes into a continuous frame, then split at every
    lon = (2k+1)*180 boundary so each output piece lies entirely within a
    single [-180, 180] copy of the world. Each crossing inserts a vertex on
    the boundary on both pieces (with interpolated latitude) so each piece
    renders as a closed polygon."""
    if not pts or len(pts) < 3:
        return []

    unwrapped = [pts[0]]
    for la, lo in pts[1:]:
        prev_lo = unwrapped[-1][1]
        while lo - prev_lo > 180:
            lo -= 360
        while lo - prev_lo < -180:
            lo += 360
        unwrapped.append((la, lo))

    closed = unwrapped + [unwrapped[0]]
    pieces = [[]]
    for i in range(len(closed) - 1):
        la1, lo1 = closed[i]
        la2, lo2 = closed[i + 1]
        pieces[-1].append((la1, lo1))
        if lo1 == lo2:
            continue
        lo_lo, lo_hi = (lo1, lo2) if lo1 < lo2 else (lo2, lo1)
        k_min = math.floor((lo_lo / 180.0 - 1) / 2) + 1
        k_max = math.ceil((lo_hi / 180.0 - 1) / 2) - 1
        crossings = []
        for k in range(k_min, k_max + 1):
            bound = (2 * k + 1) * 180
            t = (bound - lo1) / (lo2 - lo1)
            la_b = la1 + t * (la2 - la1)
            crossings.append((t, bound, la_b))
        crossings.sort()
        for _, bound, la_b in crossings:
            pieces[-1].append((la_b, bound))
            pieces.append([(la_b, bound)])

    # The polygon's closure connects pieces[-1] back to pieces[0]; if both
    # land in the same world copy, fuse them into one piece.
    if len(pieces) >= 2 and pieces[0] and pieces[-1]:
        avg0 = sum(p[1] for p in pieces[0]) / len(pieces[0])
        avgL = sum(p[1] for p in pieces[-1]) / len(pieces[-1])
        if math.floor((avg0 + 180) / 360) == math.floor((avgL + 180) / 360):
            pieces[0] = pieces[-1] + pieces[0]
            pieces.pop()

    out = []
    for piece in pieces:
        if len(piece) < 3:
            continue
        avg = sum(p[1] for p in piece) / len(piece)
        shift = 0.0
        while avg + shift > 180:
            shift -= 360
        while avg + shift < -180:
            shift += 360
        out.append([(la, lo + shift) for la, lo in piece])
    return out


def write_coverage_map(gitco_paths, aoi, html_path):
    """Build coverage_map.html showing one or more GITCO swaths plus AOI."""
    if not gitco_paths:
        return None

    lat_c = (aoi["lat_min"] + aoi["lat_max"]) / 2
    lon_c = (aoi["lon_min"] + aoi["lon_max"]) / 2
    m = folium.Map(location=[lat_c, lon_c], zoom_start=4,
                   tiles="OpenStreetMap", world_copy_jump=True)

    folium.Rectangle(
        bounds=[(aoi["lat_min"], aoi["lon_min"]),
                (aoi["lat_max"], aoi["lon_max"])],
        color="red", weight=2, fill=False, tooltip="AOI (XML)",
    ).add_to(m)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, gp in enumerate(gitco_paths):
        gp = Path(gp)
        try:
            with h5py.File(gp, "r") as f:
                g   = f["All_Data/VIIRS-IMG-GEO-TC_All"]
                lat = g["Latitude"][:]
                lon = g["Longitude"][:]
        except Exception as e:
            print(f"  map: skip {gp.name}: {e}")
            continue
        color, label = palette[i % len(palette)], gp.parent.name
        for piece in _split_dateline(_swath_boundary(lat, lon)):
            folium.Polygon(locations=piece, color=color, weight=2,
                           fill=True, fill_opacity=0.18,
                           tooltip=label).add_to(m)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(html_path))
    return html_path


# ── per-ref orchestration ─────────────────────────────────────────────────────

def process(ref_str, out_root, aoi, s3_clients, fs, make_map=True, pad=0):
    agg         = parse_ref(ref_str)
    bucket      = SAT_TO_BUCKET[agg["sat"]]
    granule_dir = out_root / agg["subdir"]
    staging     = granule_dir / "_staging"
    map_html    = granule_dir / "coverage_map.html"

    print(f"\n=== {agg['sig']} -> {granule_dir.name}/ ===")

    if already_done(granule_dir):
        print("  already complete, skipping download")
        if make_map and not map_html.exists():
            gitco = next(granule_dir.glob("GITCO_*.h5"), None)
            if gitco:
                write_coverage_map([gitco], aoi, map_html)
                print(f"  map -> {map_html.name}")
        return True

    s3 = s3_clients.setdefault(
        bucket, boto3.client("s3", config=Config(signature_version=UNSIGNED))
    )

    # 1. find every individual granule whose tstart falls in [agg.tstart, agg.tend],
    #    plus `pad` extra granules on each side of that window
    print(f"Step 1: listing individual granules in range (pad={pad})")
    individuals = list_individual_granules(s3, bucket, agg, pad=pad)
    print(f"  {len(individuals)} individual granules:")
    for r in individuals:
        print(f"    {r['subdir']}  (t{r['tstart']}_e{r['tend']})")
    if not individuals:
        print("  none found, skipping")
        return False

    # 2. AOI overlap check (informational)
    sample = individuals[len(individuals) // 2]
    try:
        in_aoi = stream_overlaps_aoi(fs, bucket, sample["gitco_key"], aoi)
        print(f"  AOI overlap (middle granule): {'YES' if in_aoi else 'NO'}")
    except Exception as e:
        print(f"  AOI check failed: {e}")

    # 3. download all 6 files per individual granule into a staging folder
    plan = []
    for r in individuals:
        plan.append((r["gitco_key"], staging / Path(r["gitco_key"]).name))
        for fp, prod in PRODUCTS:
            if fp == "GITCO":
                continue
            k = find_key(s3, bucket, prod, fp, r)
            if k:
                plan.append((k, staging / Path(k).name))
            else:
                print(f"  warn {r['subdir']}: missing {fp} on S3")
    print(f"Step 2: downloading {len(plan)} files")
    parallel_download(s3, bucket, plan)

    # 4. aggregate GITCOs into one
    gitco_paths = sorted(staging.glob("GITCO_*.h5"))
    final_gitco = granule_dir / aggregated_filename("GITCO", agg, gitco_paths)
    print(f"Step 3: aggregating {len(gitco_paths)} GITCOs -> {final_gitco.name}")
    aggregate_h5(gitco_paths, final_gitco)

    # 5. aggregate all SVI files (5 bands x N granules) into one combined file
    svi_paths = sorted([p for p in staging.glob("SVI*.h5")])
    final_svi = granule_dir / aggregated_filename(
        "GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05", agg, svi_paths
    )
    print(f"Step 4: aggregating {len(svi_paths)} SVI files -> {final_svi.name}")
    aggregate_h5(svi_paths, final_svi)

    # 6. cleanup staging
    shutil.rmtree(staging, ignore_errors=True)
    print("Step 5: staging cleaned up")

    # 7. per-granule coverage map alongside the data
    if make_map:
        write_coverage_map([final_gitco], aoi, map_html)
        print(f"Step 6: map -> {map_html.name}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_refs(args):
    refs = list(args.granules or [])
    if args.granule_file:
        for line in Path(args.granule_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                refs.append(line)
    return refs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("granules", nargs="*",
                   help="Aggregated refs, e.g. SVI02_j01_d20240904_t2357037_e0002437_b35223_c20240905002744224000")
    p.add_argument("--granule-file",
                   help="Text file with one ref per line (# comments OK)")
    p.add_argument("--xml", default=str(DEFAULT_XML),
                   help=f"AOI XML (default: {DEFAULT_XML.name})")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output root (default: {DEFAULT_OUT})")
    p.add_argument("--no-map", action="store_true",
                   help="Skip writing per-granule coverage_map.html files")
    args = p.parse_args()

    aoi = parse_aoi(args.xml)
    print(f"AOI from {Path(args.xml).name}: "
          f"lat[{aoi['lat_min']},{aoi['lat_max']}]  "
          f"lon[{aoi['lon_min']},{aoi['lon_max']}]")

    refs = collect_refs(args)
    if not refs:
        p.error("Provide at least one granule ref or use --granule-file")

    out_root   = Path(args.out)
    s3_clients = {}
    fs         = s3fs.S3FileSystem(anon=True)
    ok = fail  = 0
    for ref in refs:
        try:
            if process(ref, out_root, aoi, s3_clients, fs, make_map=not args.no_map):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"ERROR {ref!r}: {e}")
            fail += 1

    print(f"\n{ok} ok, {fail} failed, of {len(refs)} input refs")


if __name__ == "__main__":
    main()
