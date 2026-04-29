"""
enrich_modis.py — Enrich training candidate CSVs with MODIS NDVI and land cover.

Reads training_candidates_{DATE}.csv files produced by extract_training_pixels.py,
samples MODIS MOD13Q1 NDVI (250 m) and MCD12Q1 land cover (500 m) at
Landsat-confirmed pixel coordinates via the Earth Engine Python API, and
writes training_candidates_{DATE}_enriched.csv alongside the originals.

Only rows where LS_SR_B2 is not null (Landsat-confirmed) are sent to EE.
All other rows are preserved in the output with NaN MODIS columns.

Usage:
    python training/enrich_modis.py --date 2022-10-05   # single scene
    python training/enrich_modis.py                      # all scenes

Dependencies (all already in .venv):
    earthengine-api  — pip install earthengine-api
    pandas           — pip install pandas
    numpy            — installed with pandas
    pathlib, csv, datetime, argparse, sys, os  — stdlib
"""

import argparse
import csv as _csv
import os
import sys
from datetime import datetime
import pathlib

import numpy as np
import pandas as pd

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

IGBP_LOOKUP = {
    0:   'Water Bodies',
    1:   'Evergreen Needleleaf Forests',
    2:   'Evergreen Broadleaf Forests',
    3:   'Deciduous Needleleaf Forests',
    4:   'Deciduous Broadleaf Forests',
    5:   'Mixed Forests',
    6:   'Closed Shrublands',
    7:   'Open Shrublands',
    8:   'Woody Savannas',
    9:   'Savannas',
    10:  'Grasslands',
    11:  'Permanent Wetlands',
    12:  'Croplands',
    13:  'Urban and Built-up Lands',
    14:  'Cropland/Natural Vegetation Mosaics',
    15:  'Permanent Snow and Ice',
    16:  'Barren',
    17:  'Unclassified',
    255: 'Fill/NoData',
}

# ── I/O HELPERS ───────────────────────────────────────────────────────────────

def _read_csv_robust(path: pathlib.Path) -> pd.DataFrame:
    """
    Read training CSV, tolerant of unquoted commas in the trailing notes field.

    The C pandas engine rejects rows where a field (always notes, the last
    column) contains a literal comma that was not quoted on write.  We handle
    this at the csv-module level: if a row has more fields than the header,
    all extra fields are re-joined into the last column with a comma.
    """
    with open(path, newline='', encoding='utf-8') as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        n_cols = len(header)
        rows = []
        for raw in reader:
            if len(raw) > n_cols:
                # notes field contained an unquoted comma — merge extras back
                merged = raw[:n_cols - 1] + [','.join(raw[n_cols - 1:])]
                rows.append(merged)
            else:
                rows.append(raw)
    return pd.DataFrame(rows, columns=header)


def _is_confirmed(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask: True for rows that have Landsat-sampled values.

    A row is Landsat-confirmed when:
      1. The CSV contains the LS_SR_B2 column (newer format only), AND
      2. LS_SR_B2 is not empty / NaN, AND
      3. The notes column does not contain the string "Landsat null".
    CSVs that pre-date the LS_* column addition (18-col format) produce all-False.
    """
    if 'LS_SR_B2' not in df.columns:
        return pd.Series(False, index=df.index)

    has_ls = df['LS_SR_B2'].replace('', np.nan).notna()

    notes_col = df.get('notes', pd.Series('', index=df.index))
    no_null_note = ~notes_col.astype(str).str.contains('Landsat null', na=False)

    return has_ls & no_null_note


# ── EARTH ENGINE SAMPLING ─────────────────────────────────────────────────────

def _sample_modis(ee, confirmed_df: pd.DataFrame, scene_date: str) -> dict:
    """
    Sample MOD13Q1 NDVI and MCD12Q1 LC_Type1 at confirmed pixel locations.

    Parameters
    ----------
    ee           : earthengine module (already initialised)
    confirmed_df : rows to sample — must have lat, lon columns;
                   DataFrame index is used as the join key (row_idx)
    scene_date   : YYYY-MM-DD string for this scene

    Returns
    -------
    dict  {row_idx (int): {'modis_ndvi': float|None, 'modis_lc_type1': int|None}}
    All entries are pre-populated with None so callers can detect EE misses.

    filterDate windows (per spec):
      NDVI : advance(-16, 'day') → advance(+1, 'day')    [16-day composites]
      LC   : year_start → year_end                        [annual product]
    """
    dt = datetime.strptime(scene_date, '%Y-%m-%d')
    year_start = f'{dt.year}-01-01'
    year_end   = f'{dt.year + 1}-01-01'

    # Build FeatureCollection — store DataFrame index as 'row_idx' so we can
    # join results back without floating-point lat/lon comparisons
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row['lon']), float(row['lat'])]),
            {'row_idx': int(idx)}
        )
        for idx, row in confirmed_df.iterrows()
    ]
    fc = ee.FeatureCollection(features)

    # Pre-populate results with None (EE miss = stays None → NaN in output)
    results = {
        int(idx): {'modis_ndvi': None, 'modis_lc_type1': None}
        for idx in confirmed_df.index
    }

    # ── NDVI — MOD13Q1 (250 m, 16-day composite) ─────────────────────────
    try:
        ndvi_img = (
            ee.ImageCollection('MODIS/061/MOD13Q1')
              .filterDate(
                  ee.Date(scene_date).advance(-16, 'day'),   # 16-day lookback
                  ee.Date(scene_date).advance(1,  'day'),    # up to +1 day
              )
              .sort('system:time_start', False)              # most-recent first
              .first()
              .select('NDVI')
        )
        ndvi_sampled = ndvi_img.sampleRegions(
            collection=fc,
            scale=250,
            projection='EPSG:4326',
            geometries=True,
        ).getInfo()

        for feat in ndvi_sampled.get('features', []):
            props = feat['properties']
            idx   = int(props['row_idx'])
            raw   = props.get('NDVI')
            if raw is not None:
                # Apply MODIS NDVI scale factor: integer * 0.0001
                results[idx]['modis_ndvi'] = round(float(raw) * 0.0001, 6)

    except Exception as exc:
        print(f"  WARNING: NDVI sampling failed for {scene_date}: {exc}")

    # ── Land Cover — MCD12Q1 (500 m, annual) ─────────────────────────────
    try:
        lc_img = (
            ee.ImageCollection('MODIS/061/MCD12Q1')
              .filterDate(year_start, year_end)
              .first()
              .select('LC_Type1')
        )
        lc_sampled = lc_img.sampleRegions(
            collection=fc,
            scale=500,
            projection='EPSG:4326',
            geometries=True,
        ).getInfo()

        for feat in lc_sampled.get('features', []):
            props = feat['properties']
            idx   = int(props['row_idx'])
            raw   = props.get('LC_Type1')
            if raw is not None:
                results[idx]['modis_lc_type1'] = int(raw)

    except Exception as exc:
        print(f"  WARNING: Land cover sampling failed for {scene_date}: {exc}")

    return results


# ── PER-CSV PROCESSING ────────────────────────────────────────────────────────

def process_csv(csv_path: pathlib.Path, ee) -> None:
    """
    Process one training CSV: add MODIS columns and write _enriched.csv.

    Output file: same directory, stem + '_enriched.csv'.
    Original CSV is never modified.
    """
    out_path = csv_path.parent / (csv_path.stem + '_enriched.csv')

    if out_path.exists():
        print(f"\n{'='*60}")
        print(f"SKIP: {out_path.relative_to(PROJECT_ROOT)} already exists")
        return

    print(f"\n{'='*60}")
    print(f"Input:  {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Output: {out_path.relative_to(PROJECT_ROOT)}")

    df = _read_csv_robust(csv_path)
    n_total = len(df)

    # Add MODIS columns initialised to NaN / None
    df['modis_ndvi']     = np.nan          # float; NaN written as empty in CSV
    df['modis_lc_type1'] = None            # object; None written as empty
    df['modis_lc_name']  = None            # object; None written as empty

    confirmed_mask = _is_confirmed(df)
    n_confirmed    = int(confirmed_mask.sum())
    confirmed_df   = df[confirmed_mask].copy()

    print(f"  Confirmed mask: {n_confirmed} True, "
          f"{n_total - n_confirmed} False (Landsat-null or missing LS_SR_B2)")
    print(f"  Rows total: {n_total}  |  Landsat-confirmed: {n_confirmed}"
          f"  |  skipped (Landsat-null): {n_total - n_confirmed}")

    n_modis_returned  = 0
    n_join_mismatches = 0

    if n_confirmed == 0:
        print("  No Landsat-confirmed pixels — skipping EE sampling")
    elif ee is None:
        print("  EE unavailable — MODIS values will be NaN for all rows")
    else:
        # Group by viirs_date (typically one date per CSV)
        for scene_date, grp in confirmed_df.groupby('viirs_date'):
            print(f"  Sampling MODIS for {scene_date} ({len(grp)} points)...")
            modis = _sample_modis(ee, grp, str(scene_date))

            for row_idx, vals in modis.items():
                # Verify row_idx exists in the full DataFrame
                if row_idx not in df.index:
                    print(f"  WARNING: join mismatch — row_idx={row_idx} "
                          f"not found in DataFrame index")
                    n_join_mismatches += 1
                    continue

                # Cross-check lat/lon to guard against index collisions
                exp_lat = float(df.at[row_idx, 'lat'])
                exp_lon = float(df.at[row_idx, 'lon'])
                got_lat = float(confirmed_df.at[row_idx, 'lat'])
                got_lon = float(confirmed_df.at[row_idx, 'lon'])
                if abs(exp_lat - got_lat) > 1e-4 or abs(exp_lon - got_lon) > 1e-4:
                    print(f"  WARNING: lat/lon mismatch at row_idx={row_idx}: "
                          f"expected ({exp_lat}, {exp_lon}), "
                          f"got ({got_lat}, {got_lon})")
                    n_join_mismatches += 1

                ndvi = vals.get('modis_ndvi')
                lc   = vals.get('modis_lc_type1')

                if ndvi is not None:
                    df.at[row_idx, 'modis_ndvi'] = ndvi
                    n_modis_returned += 1
                if lc is not None:
                    df.at[row_idx, 'modis_lc_type1'] = lc
                    df.at[row_idx, 'modis_lc_name']  = IGBP_LOOKUP.get(
                        lc, f'Unknown({lc})')

    # Write enriched output — original CSV untouched
    df.to_csv(out_path, index=False)

    # Per-CSV summary
    mismatch_flag = '  ← WARNING' if n_join_mismatches else ''
    print(f"\n  Summary:")
    print(f"    Rows total              : {n_total}")
    print(f"    Landsat-confirmed       : {n_confirmed}")
    print(f"    MODIS values returned   : {n_modis_returned}")
    print(f"    Join mismatches         : {n_join_mismatches}{mismatch_flag}")
    print(f"    Enriched CSV written    : {out_path.name}")


# ── CSV DISCOVERY ─────────────────────────────────────────────────────────────

def _find_csvs(date: str | None) -> list[pathlib.Path]:
    """Return sorted list of training candidate CSVs to process.

    Searches both training/viirs_d*/outputs/ and training/20*/ folders.
    Excludes _enriched.csv and _n*.csv files.
    """
    patterns = [
        'training/viirs_d*/outputs/training_candidates_{slug}.csv',
        'training/20*/training_candidates_{slug}.csv',
    ]
    if date:
        found = []
        for pat in patterns:
            found.extend(PROJECT_ROOT.glob(pat.format(slug=date)))
        if not found:
            print(f"ERROR: No CSV found for date {date}")
            sys.exit(1)
        return sorted(set(found))

    found = []
    for pat in patterns:
        found.extend(PROJECT_ROOT.glob(pat.format(slug='????-??-??')))
    return sorted(set(found))


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Enrich training CSVs with MODIS NDVI and land cover via EE.')
    parser.add_argument(
        '--date', metavar='YYYY-MM-DD', default=None,
        help='Process only the CSV for this scene date. '
             'Omit to process all scenes.')
    parser.add_argument(
        '--file', metavar='PATH', default=None,
        help='Process a single CSV at this explicit path, bypassing '
             'the automatic discovery logic. Path is relative to PROJECT_ROOT '
             'or absolute.')
    args = parser.parse_args()

    # Initialise Earth Engine once — degrade gracefully if unavailable
    try:
        import ee
        ee.Initialize(project='noaa-river-ice')
        print("Earth Engine initialised (project=noaa-river-ice)")
    except Exception as exc:
        print(f"WARNING: Earth Engine init failed: {exc}")
        print("MODIS values will be NaN for all rows — "
              "check 'earthengine authenticate' and project access.")
        ee = None

    if args.file:
        explicit = PROJECT_ROOT / args.file if not pathlib.Path(args.file).is_absolute() \
                   else pathlib.Path(args.file)
        if not explicit.exists():
            print(f"ERROR: File not found: {explicit}")
            sys.exit(1)
        csvs = [explicit]
    else:
        csvs = _find_csvs(args.date)
    if not csvs:
        print("No training candidate CSVs found.")
        sys.exit(0)

    print(f"\nFound {len(csvs)} CSV(s) to process:")
    for p in csvs:
        print(f"  {p.relative_to(PROJECT_ROOT)}")

    for csv_path in csvs:
        process_csv(csv_path, ee)

    print(f"\n{'='*60}")
    print("All CSVs processed.")


if __name__ == '__main__':
    main()
