"""
verify_water_filter.py — Flag confirmed training pixels that sit on high-
occurrence or permanent-water cells in the JRC rasters.

Read-only: does NOT modify any CSV.

Usage:
    python training/verify_water_filter.py
"""

import csv
import os
import pathlib
import sys

import numpy as np
import rasterio

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

OCC_PATH = PROJECT_ROOT / "outputs" / "alaska_occ_375m.tif"
SEA_PATH = PROJECT_ROOT / "outputs" / "alaska_sea_375m.tif"

ALASKA_H = 5328
ALASKA_W = 12432

OCC_THRESHOLD = 0.90
SEA_THRESHOLD = 1.0


def _find_csvs() -> list[pathlib.Path]:
    """Discover all training candidate CSVs (exclude _enriched, _n*)."""
    patterns = [
        "training/viirs_d*/outputs/training_candidates_????-??-??.csv",
        "training/20*/training_candidates_????-??-??.csv",
    ]
    found = []
    for pat in patterns:
        found.extend(PROJECT_ROOT.glob(pat))
    return sorted(set(found))


def main() -> None:
    print("Loading occurrence raster...")
    with rasterio.open(OCC_PATH) as src:
        occ = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        occ[occ == nodata] = np.nan

    print("Loading seasonality raster...")
    with rasterio.open(SEA_PATH) as src:
        sea = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        sea[sea == nodata] = np.nan

    csvs = _find_csvs()
    if not csvs:
        print("No training candidate CSVs found.")
        sys.exit(0)

    print(f"\nFound {len(csvs)} CSV(s):\n")

    total_confirmed = 0
    total_flagged = 0
    total_clean = 0

    header_printed = False

    for csv_path in csvs:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if r.get("ground_truth_class", "").strip()]

        if not rows:
            continue

        scene_label = csv_path.relative_to(PROJECT_ROOT)

        for row in rows:
            r = int(row["row_shared_grid"])
            c = int(row["col_shared_grid"])

            if not (0 <= r < ALASKA_H and 0 <= c < ALASKA_W):
                continue

            occ_val = float(occ[r, c]) if np.isfinite(occ[r, c]) else np.nan
            sea_val = float(sea[r, c]) if np.isfinite(sea[r, c]) else np.nan

            flagged = False
            reasons = []
            if np.isfinite(occ_val) and occ_val >= OCC_THRESHOLD:
                flagged = True
                reasons.append(f"occ>={OCC_THRESHOLD}")
            if np.isfinite(sea_val) and sea_val >= SEA_THRESHOLD:
                flagged = True
                reasons.append(f"sea>={SEA_THRESHOLD}")

            total_confirmed += 1
            if flagged:
                total_flagged += 1
            else:
                total_clean += 1

            if not header_printed:
                print(f"{'scene':<60s}  {'lat':>10s}  {'lon':>11s}  "
                      f"{'occ_wf':>7s}  {'sea_wf':>7s}  {'FLAG':>6s}  "
                      f"{'class':<45s}")
                print("-" * 160)
                header_printed = True

            flag_str = " | ".join(reasons) if flagged else ""
            print(f"{str(scene_label):<60s}  {row['lat']:>10s}  {row['lon']:>11s}  "
                  f"{occ_val:7.4f}  {sea_val:7.4f}  "
                  f"{'YES' if flagged else '':>6s}  "
                  f"{row['ground_truth_class']:<45s}"
                  f"{'  ' + flag_str if flag_str else ''}")

    print("\n" + "=" * 80)
    print(f"Confirmed pixels total : {total_confirmed}")
    print(f"  Flagged              : {total_flagged}")
    print(f"  Clean                : {total_clean}")


if __name__ == "__main__":
    main()
