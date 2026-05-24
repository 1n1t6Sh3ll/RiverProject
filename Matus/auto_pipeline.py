"""
auto_pipeline.py — End-to-end automation of the VIIRS-river-ice scene setup.

Given a Landsat scene ID and date, this script:
  1. Queries Earth Engine for the Landsat scene bbox.
  2. Lists all GITCO granules for the date in the J01, J02, and NPP buckets.
  3. Reads each granule's bounding-box attributes over S3 byte-range (no full download).
  4. Picks the granule with the highest bbox coverage (>= 65%).
  5. Downloads the full SVI01-05 + GITCO files for that granule.
  6. Stitches the SVI bands into a GIMGO H5.
  7. Rewrites the SCENES dict in extract_training_pixels_auto.py to point at the new files.
  8. Deletes any stale Alaska TIF for this date.

Usage:
    python auto_pipeline.py --scene LC90710162022296LGN01 --date 2022-10-23
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import boto3
import h5py
from botocore import UNSIGNED
from botocore.client import Config

import AWSExtract
import stitch_h5

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BUCKETS = {
    "j01": "noaa-nesdis-n20-pds",
    "j02": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}

COVERAGE_THRESHOLD = 0.65
MAX_WORKERS = 16

# Alaska ascending VIIRS overpasses occur in late UTC afternoon/evening.
# Pre-filter granules to this t-code window (HHMM start time) before doing
# the expensive bbox metadata read.
DEFAULT_TIME_WINDOW = "1700-2400"

GITCO_ATTR_PATH = "/Data_Products/VIIRS-IMG-GEO-TC/VIIRS-IMG-GEO-TC_Gran_0"
GITCO_PREFIX = "VIIRS-IMG-GEO-TC"


# ── 1. Landsat scene ID parsing ──────────────────────────────────────────────

def parse_scene_id(scene_id: str) -> tuple[int, int, int]:
    """Return (sensor_int=8|9, path, row) for a Landsat scene ID like LC90710162022296LGN01."""
    m = re.match(r"LC(\d)(\d{3})(\d{3})\d{7}[A-Z]{3}\d{2}", scene_id)
    if not m:
        raise ValueError(f"Could not parse Landsat scene ID: {scene_id}")
    sensor = int(m.group(1))
    path = int(m.group(2))
    row = int(m.group(3))
    if sensor not in (8, 9):
        raise ValueError(f"Only LC8/LC9 supported, got LC{sensor}")
    return sensor, path, row


# ── 2. GEE bbox lookup ───────────────────────────────────────────────────────

def get_landsat_bbox(sensor: int, path: int, row: int, date: str) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) for the Landsat scene."""
    import ee
    ee.Initialize(project="noaa-river-ice")
    collection_id = f"LANDSAT/LC0{sensor}/C02/T1_L2"
    next_day = datetime.strptime(date, "%Y-%m-%d").toordinal() + 1
    next_date = datetime.fromordinal(next_day).strftime("%Y-%m-%d")
    img = (
        ee.ImageCollection(collection_id)
        .filter(ee.Filter.eq("WRS_PATH", path))
        .filter(ee.Filter.eq("WRS_ROW", row))
        .filterDate(date, next_date)
        .first()
    )
    info = img.getInfo()
    if info is None:
        raise RuntimeError(f"No Landsat scene found for path={path} row={row} date={date}")
    coords = info["properties"]["system:footprint"]["coordinates"]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lats), max(lats), min(lons), max(lons)


# ── 3. S3 byte-range file-like wrapper ───────────────────────────────────────

class S3RangeReader(io.RawIOBase):
    """Minimal file-like wrapper around boto3 GetObject Range= requests.

    Buffers the most recent block so h5py's small adjacent reads don't each
    cost an HTTP round trip.
    """
    BLOCK = 1024 * 1024  # 1 MB read-ahead

    def __init__(self, s3, bucket, key, size):
        self._s3 = s3
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0
        self._buf_start = -1
        self._buf = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def _fill(self, want_start, want_len):
        block_start = (want_start // self.BLOCK) * self.BLOCK
        block_end = min(block_start + max(self.BLOCK, want_len) - 1, self._size - 1)
        resp = self._s3.get_object(
            Bucket=self._bucket, Key=self._key,
            Range=f"bytes={block_start}-{block_end}",
        )
        self._buf = resp["Body"].read()
        self._buf_start = block_start

    def read(self, n=-1):
        if n < 0 or self._pos + n > self._size:
            n = self._size - self._pos
        if n <= 0:
            return b""
        chunks = []
        remaining = n
        while remaining > 0:
            if (self._buf_start < 0
                    or self._pos < self._buf_start
                    or self._pos >= self._buf_start + len(self._buf)):
                self._fill(self._pos, remaining)
            offset = self._pos - self._buf_start
            available = len(self._buf) - offset
            take = min(remaining, available)
            chunks.append(self._buf[offset:offset + take])
            self._pos += take
            remaining -= take
        return b"".join(chunks)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# ── 4. List + score GITCO granules ───────────────────────────────────────────

@dataclass
class GranuleMatch:
    satellite: str
    bucket: str
    key: str
    t_code: str        # e.g. "t2053479"
    bbox: tuple[float, float, float, float]  # (S, N, W, E)  swath envelope
    nadir_bbox: tuple[float, float, float, float]  # (S, N, W, E) nadir track
    coverage: float    # fraction of Landsat bbox inside swath envelope
    nadir_distance: float  # degrees from Landsat bbox center to nadir track


def list_gitco_keys(s3, bucket: str, date: str) -> list[tuple[str, int]]:
    """Return [(key, size), ...] for all GITCO granules in bucket on date."""
    d = datetime.strptime(date, "%Y-%m-%d")
    prefix = f"{GITCO_PREFIX}/{d:%Y}/{d:%m}/{d:%d}/"
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append((obj["Key"], obj["Size"]))
    return out


def read_granule_bboxes(s3, bucket: str, key: str, size: int):
    """Return ((S,N,W,E) swath envelope, (S,N,W,E) nadir track) from a GITCO file."""
    try:
        reader = S3RangeReader(s3, bucket, key, size)
        with h5py.File(reader, "r") as h5:
            attrs = h5[GITCO_ATTR_PATH].attrs
            n = float(attrs["North_Bounding_Coordinate"][0][0])
            s = float(attrs["South_Bounding_Coordinate"][0][0])
            e = float(attrs["East_Bounding_Coordinate"][0][0])
            w = float(attrs["West_Bounding_Coordinate"][0][0])
            ns = float(attrs["N_Nadir_Latitude_Min"][0][0])
            nn = float(attrs["N_Nadir_Latitude_Max"][0][0])
            nw = float(attrs["N_Nadir_Longitude_Min"][0][0])
            ne = float(attrs["N_Nadir_Longitude_Max"][0][0])
        return (s, n, w, e), (ns, nn, nw, ne)
    except Exception as exc:
        print(f"  [warn] failed to read {key}: {exc}")
        return None, None


def coverage_pct(granule_bbox, landsat_bbox) -> float:
    """Fraction of the Landsat bbox covered by the granule bbox."""
    g_s, g_n, g_w, g_e = granule_bbox
    l_lat_min, l_lat_max, l_lon_min, l_lon_max = landsat_bbox
    overlap_lat = max(0.0, min(g_n, l_lat_max) - max(g_s, l_lat_min))
    overlap_lon = max(0.0, min(g_e, l_lon_max) - max(g_w, l_lon_min))
    bbox_area = (l_lat_max - l_lat_min) * (l_lon_max - l_lon_min)
    if bbox_area <= 0:
        return 0.0
    return (overlap_lat * overlap_lon) / bbox_area


def nadir_distance_to_bbox(nadir_bbox, landsat_bbox) -> float:
    """Euclidean degree-distance from Landsat bbox center to nadir track rectangle.

    The nadir bbox describes where the satellite's sub-point traveled during the
    granule. If the Landsat bbox center is inside the nadir bbox the distance is 0;
    otherwise it's the closest-point distance.
    """
    n_s, n_n, n_w, n_e = nadir_bbox
    l_lat_min, l_lat_max, l_lon_min, l_lon_max = landsat_bbox
    cx = (l_lat_min + l_lat_max) / 2.0  # center lat
    cy = (l_lon_min + l_lon_max) / 2.0  # center lon
    dlat = max(0.0, max(n_s - cx, cx - n_n))
    dlon = max(0.0, max(n_w - cy, cy - n_e))
    return (dlat * dlat + dlon * dlon) ** 0.5


def extract_t_code(key: str) -> str:
    """From '.../GITCO_j01_d20221023_t2053479_e2055106_...' return 't2053479'."""
    name = os.path.basename(key)
    m = re.search(r"_t(\d{7})_", name)
    if not m:
        raise ValueError(f"Could not extract t-code from {name}")
    return f"t{m.group(1)}"


def parse_time_window(spec: str) -> tuple[int, int]:
    """Parse 'HHMM-HHMM' (UTC) into (start_int, end_int) HHMM values."""
    a, b = spec.split("-")
    return int(a), int(b)


def in_time_window(key: str, start_hhmm: int, end_hhmm: int) -> bool:
    """True if the granule's t-prefix (HHMM) falls in [start, end)."""
    name = os.path.basename(key)
    m = re.search(r"_t(\d{4})\d{3}_", name)
    if not m:
        return False
    hhmm = int(m.group(1))
    return start_hhmm <= hhmm < end_hhmm


def find_best_granule(s3, landsat_bbox, date: str,
                      time_window: tuple[int, int] | None = None
                      ) -> tuple[GranuleMatch | None, list[GranuleMatch]]:
    """Search all 3 satellite buckets, return best match + top-5 list."""
    all_candidates: list[tuple[str, str, str, int]] = []  # (sat, bucket, key, size)
    for sat, bucket in BUCKETS.items():
        print(f"\nListing GITCO granules in {bucket} for {date}...")
        keys = list_gitco_keys(s3, bucket, date)
        print(f"  {len(keys)} granules found")
        if time_window:
            filtered = [(k, s) for (k, s) in keys
                        if in_time_window(k, *time_window)]
            print(f"  {len(filtered)} granules after time filter "
                  f"{time_window[0]:04d}-{time_window[1]:04d} UTC")
            keys = filtered
        for key, size in keys:
            all_candidates.append((sat, bucket, key, size))

    print(f"\nReading bbox attrs from {len(all_candidates)} granules via S3 byte-range...")
    matches: list[GranuleMatch] = []
    t0 = time.time()

    def _read_one(sat, bucket, key, size):
        bbox, nadir = read_granule_bboxes(s3, bucket, key, size)
        if bbox is None or nadir is None:
            return None
        cov = coverage_pct(bbox, landsat_bbox)
        ndist = nadir_distance_to_bbox(nadir, landsat_bbox)
        return GranuleMatch(
            satellite=sat, bucket=bucket, key=key,
            t_code=extract_t_code(key),
            bbox=bbox, nadir_bbox=nadir,
            coverage=cov, nadir_distance=ndist,
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_read_one, *c) for c in all_candidates]
        for i, fut in enumerate(as_completed(futures), 1):
            m = fut.result()
            if m is not None:
                matches.append(m)
            if i % 25 == 0:
                print(f"  ... {i}/{len(futures)} done ({time.time()-t0:.1f}s)")

    # Rank: granules whose swath envelope covers the bbox AND whose nadir
    # passes closest to the bbox win. Sort primarily by nadir distance,
    # tiebreak by coverage descending. Coverage threshold still applies.
    matches.sort(key=lambda m: (m.nadir_distance, -m.coverage))
    qualified = [m for m in matches if m.coverage >= COVERAGE_THRESHOLD]
    top5 = qualified[:5] if qualified else matches[:5]
    best = qualified[0] if qualified else None
    return best, top5


# ── 5. Auto-edit extract_training_pixels_auto.py SCENES dict ─────────────────

SCENES_BLOCK_RE = re.compile(
    r"SCENES\s*=\s*\{.*?^\}",
    re.DOTALL | re.MULTILINE,
)
DATE_DEFAULT_RE = re.compile(
    r'"--date",\s*required=False,\s*default="[^"]+"',
)


def rewrite_scenes(scene_entry: dict, date: str):
    path = os.path.join(PROJECT_ROOT, "extract_training_pixels_auto.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    new_block = (
        "SCENES = {\n"
        f'    "{date}": {{\n'
        f'        "gitco":         "{scene_entry["gitco"]}",\n'
        f'        "gimgo":         "{scene_entry["gimgo"]}",\n'
        f'        "landsat_scene": "{scene_entry["landsat_scene"]}",\n'
        f'        "landsat_bbox":  {scene_entry["landsat_bbox"]},\n'
        f'        "output_dir":    "{scene_entry["output_dir"]}",\n'
        f'        "ee_collection": "{scene_entry["ee_collection"]}",\n'
        f'        "viirs_tif":     "{scene_entry["viirs_tif"]}",\n'
        "    },\n"
        "}"
    )
    if not SCENES_BLOCK_RE.search(src):
        raise RuntimeError("Could not locate SCENES = {...} block in extract_training_pixels_auto.py")
    src = SCENES_BLOCK_RE.sub(new_block, src, count=1)

    src, n = DATE_DEFAULT_RE.subn(
        f'"--date", required=False, default="{date}"', src, count=1
    )
    if n == 0:
        print("  [warn] Could not find --date default argparse entry to update")

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  -> Rewrote SCENES + --date default in extract_training_pixels_auto.py")


# ── 6. Orchestration ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect, download, and wire up VIIRS data for a Landsat scene."
    )
    parser.add_argument("--scene", required=True, help="Landsat scene ID, e.g. LC90710162022296LGN01")
    parser.add_argument("--date", required=True, help="Scene date YYYY-MM-DD")
    parser.add_argument("--time-window", default=DEFAULT_TIME_WINDOW,
                        help=f"UTC time-of-day filter for granules, "
                             f"HHMM-HHMM (default {DEFAULT_TIME_WINDOW}). "
                             f"Pass 'all' to disable.")
    parser.add_argument("--no-download", action="store_true",
                        help="Stop after granule selection, print top-5 and exit.")
    args = parser.parse_args()

    time_window = None if args.time_window.lower() == "all" else parse_time_window(args.time_window)

    sensor, path, row = parse_scene_id(args.scene)
    print(f"Scene: {args.scene}  ->  LC0{sensor}  Path={path}  Row={row}")

    print(f"\nQuerying Earth Engine for Landsat footprint...")
    landsat_bbox = get_landsat_bbox(sensor, path, row, args.date)
    print(f"  Landsat bbox: lat {landsat_bbox[0]:.4f}–{landsat_bbox[1]:.4f}  "
          f"lon {landsat_bbox[2]:.4f}–{landsat_bbox[3]:.4f}")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    best, top5 = find_best_granule(s3, landsat_bbox, args.date, time_window=time_window)

    print(f"\nTop-5 candidates (ranked by nadir distance, then coverage):")
    for i, m in enumerate(top5, 1):
        print(f"  {i}. {m.satellite.upper()}  {m.t_code}  "
              f"nadir_dist={m.nadir_distance:.2f}°  "
              f"coverage={m.coverage*100:.1f}%  "
              f"nadir bbox=S{m.nadir_bbox[0]:.2f} N{m.nadir_bbox[1]:.2f} "
              f"W{m.nadir_bbox[2]:.2f} E{m.nadir_bbox[3]:.2f}")

    if best is None:
        print(f"\nNo granule reached the {COVERAGE_THRESHOLD*100:.0f}% threshold. Aborting.")
        sys.exit(1)

    print(f"\nBEST: {best.satellite.upper()}  {best.t_code}  "
          f"nadir_dist={best.nadir_distance:.2f}°  coverage={best.coverage*100:.1f}%")
    print(f"      s3://{best.bucket}/{best.key}")

    if args.no_download:
        print("\n--no-download set; stopping after selection.")
        return

    # ── Download full granule (SVI01-05 + GITCO) ─────────────────────────
    print(f"\nDownloading {best.satellite.upper()} granule {best.t_code}...")
    original_bucket = AWSExtract.BUCKET_NAME
    AWSExtract.BUCKET_NAME = best.bucket
    try:
        AWSExtract.download_viirs_data(
            target_date_str=args.date,
            target_utc_hour=f"_{best.t_code}_",
            download_dir=os.path.join(PROJECT_ROOT, "viirs_data"),
            max_files_per_band=1,
        )
    finally:
        AWSExtract.BUCKET_NAME = original_bucket

    # ── Stitch SVI bands ────────────────────────────────────────────────
    print(f"\nStitching SVI bands...")
    viirs_dir = os.path.join(PROJECT_ROOT, "viirs_data")
    gimgo_path = stitch_h5.stitch_svi_files(
        viirs_dir, satellite=best.satellite, t_code=best.t_code,
    )
    if not gimgo_path:
        print("ERROR: stitching failed.")
        sys.exit(1)
    gimgo_name = os.path.basename(gimgo_path)

    # ── Find the GITCO filename we just downloaded ──────────────────────
    import glob
    gitco_matches = glob.glob(
        os.path.join(viirs_dir, f"GITCO_{best.satellite}_d{args.date.replace('-', '')}_{best.t_code}*.h5")
    )
    if not gitco_matches:
        print("ERROR: Could not locate downloaded GITCO file.")
        sys.exit(1)
    gitco_name = os.path.basename(gitco_matches[0])

    # ── Build SCENES entry and rewrite ──────────────────────────────────
    output_dir = args.date.replace("-", "")
    scene_entry = {
        "gitco":         f"viirs_data/{gitco_name}",
        "gimgo":         f"viirs_data/{gimgo_name}",
        "landsat_scene": args.scene,
        "landsat_bbox":  landsat_bbox,
        "output_dir":    output_dir,
        "ee_collection": f"LANDSAT/LC0{sensor}/C02/T1_L2",
        "viirs_tif":     f"{output_dir}/viirs_alaska_{output_dir}.tif",
    }
    rewrite_scenes(scene_entry, args.date)

    # ── Delete any stale TIF ────────────────────────────────────────────
    stale_tif = os.path.join(PROJECT_ROOT, output_dir, f"viirs_alaska_{output_dir}.tif")
    if os.path.exists(stale_tif):
        os.remove(stale_tif)
        print(f"  -> Deleted stale TIF: {stale_tif}")

    print("\n" + "=" * 60)
    print("DONE. Now run:")
    print("    python extract_training_pixels_auto.py")
    print("(optionally with --region TL|TM|TR|ML|MM|MR|BL|BM|BR for sub-region runs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
