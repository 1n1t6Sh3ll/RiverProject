# VIIRS River Ice Training Pipeline — AI Skill

> **How to use this file:** At the start of any new conversation, the user will reference this file.
> Read it in full, confirm you understand the pipeline, and wait for the user to give you a
> Landsat scene ID and date. Then decide whether to use the **automated path** (`auto_pipeline.py`,
> preferred) or the **manual path** (steps 1–4 below).

---

## Project Overview

This project builds a **training dataset for a VIIRS satellite river ice classifier** covering Alaska.
Each training pixel is labeled as one of four classes based on the simultaneous state of the river and surrounding land:

| Class | River | Land |
|-------|-------|------|
| `ice_free_river_snow_free_land` | Liquid water | No snow |
| `ice_free_river_snow_covered_land` | Liquid water | Snow-covered |
| `ice_covered_river_snow_covered_land` | Frozen | Ice/snow-covered |
| `ice_covered_river_snow_free_land` | Frozen | No snow |

`ground_truth_class` is **auto-filled** by the extract scripts from Landsat 30m thermal and
NDSI (`ST_B10 < 273K` → ice, `NDSI > 0.4` → snow). The GEE step (STEP 6) is *verification of an
existing label*, not classification from scratch.

> **Naming history:** the snow-covered class was originally emitted as `ice_free_river_snow_land`.
> It is now `ice_free_river_snow_covered_land`, matching the `{ice_free|ice_covered}_river_{snow_free|snow_covered}_land`
> pattern used by the other three. No row carries the old name any more — the single legacy row
> lived in the 2023-09-25 scene, which has since been removed. If the old name ever reappears in a
> CSV it came from a pre-rename run and should be corrected.

**GEE project:** `noaa-river-ice`
**Python venv:** `.venv` (Python 3.13) — always run scripts with `.venv\Scripts\python.exe`
**Workspace root:** the project directory containing this file

---

## Shared Alaska Grid (all outputs co-registered)

```
Origin:     top-left = (lon -171.0, lat 72.0)
Resolution: 375m ≈ 0.003378°/pixel
Size:       12432 × 5328 pixels
CRS:        EPSG:4326
```

---

## Satellite Buckets (for AWSExtract.py)

| Satellite | AWS Bucket | Designation |
|-----------|------------|-------------|
| NOAA-20   | `noaa-nesdis-n20-pds` | J01 |
| NOAA-21   | `noaa-nesdis-n21-pds` | J02 |
| Suomi NPP | `noaa-nesdis-snpp-pds` | NPP |

---

## Dual-Mask Candidate Filter (applied in extract_training_pixels[_auto].py)

Candidate pixels must satisfy ALL of:
- `0.05 < occurrence < 0.90` — mixed land-water, not open ocean
- `seasonality < 1.0` — not permanent water (excludes lakes)
- `VIIRS I1 is valid` — not NaN (pixel must be within swath)

---

## Key Files

| Script | What it does | What changes per scene |
|--------|-------------|------------------------|
| `auto_pipeline.py` | **End-to-end automated setup** — given `--scene` and `--date`, finds the best VIIRS granule, downloads it, stitches, rewrites SCENES in `extract_training_pixels_auto.py` | Nothing — just pass CLI args |
| `extract_training_pixels_auto.py` | Same logic as the manual extractor; this is the file `auto_pipeline.py` rewrites | Never edit by hand |
| `extract_training_pixels.py` | Manual/fallback extractor with hand-maintained SCENES dict | Add a new entry to `SCENES` |
| `enrich_modis.py` | Adds MODIS NDVI + land cover via GEE | `--date YYYY-MM-DD` (+ optional `--force`, `--region`) |
| `get_landsat_footprint.py` | Standalone helper: prints bbox for a Landsat scene via GEE | Path, Row, date, collection |
| `AWSExtract.py` | Downloads VIIRS H5 granules from AWS S3 (used by `auto_pipeline.py`, or directly in manual mode) | `BUCKET_NAME`, `TARGET_DATE`, `TARGET_HOUR` |
| `stitch_h5.py` | Stitches SVI01-05 into one GIMGO H5 (used by `auto_pipeline.py` or manually) | Glob pattern `SVI0{i}_<sat>_*_t<HHMM>*.h5` |
| `viirs_training_loader.py` | H5 parser library used by extract | Never modify — library code |

**Note on the two extract scripts:** `extract_training_pixels.py` and `extract_training_pixels_auto.py`
have **identical downstream code**. The only intentional difference is the `SCENES` dict (the
`_auto` file is rewritten by `auto_pipeline.py`, the non-auto file is hand-maintained). When changing
the pipeline logic (region handling, GEE template, etc.), **make the edit in both files**.

**Data masks (project root, never modify):**
- `alaska_occ_375m.tif` — JRC water occurrence (198 MB)
- `alaska_sea_375m.tif` — JRC water seasonality (198 MB)

---

## Output Structure (per scene)

Each processed scene creates a `YYYYMMDD/` folder. When using `--region`, outputs land in `YYYYMMDD/<REGION>/`:

| File | Description |
|------|-------------|
| `training_candidates_YYYY-MM-DD.csv` | Up to 25 candidate pixels with VIIRS + Landsat values, `ground_truth_class` auto-filled. Often pruned by hand to only the confirmed rows |
| `training_candidates_YYYY-MM-DD_enriched.csv` | Same + MODIS NDVI and land cover columns |
| `viirs_alaska_YYYYMMDD.tif` | 9-band Alaska domain GeoTIFF (I1-I5, SZA, SAA, VZA, VAA). Acts as a fast-path cache — if present, the extract script reuses it instead of resampling from H5 |
| `viirs_vs_watermask_YYYY-MM-DD.png` | Side-by-side visual: red × = Landsat-confirmed, gray ○ = outside swath |
| `gee_inspector_YYYYMMDD.js` | GEE script for visual pixel verification — includes HYBRID basemap + a toggleable Sentinel-2 10m layer (least-cloudy pass within ±7 days of the scene date) |

---

## Sub-Region Selection (extract scripts `--region` flag)

Both extract scripts accept `--region` to restrict candidates to a portion of the bbox.
Useful when interesting rivers cluster in one part of the scene.

| Code | Meaning |
|------|---------|
| `T` / `M` / `B` | Full row (top / middle / bottom third), spans entire longitude |
| `TL` `TM` `TR` `ML` `MM` `MR` `BL` `BM` `BR` | Single 3×3 sub-cell |

Each run still selects up to 25 candidate pixels — they're just spread over the selected region.
Outputs go to `YYYYMMDD/<REGION>/...`.

### Root folder = finished pixels

Agreed with the professor and the team: **the root `YYYYMMDD/` folder holds the pixels that are
good to go.** Sub-region folders (`YYYYMMDD/<REGION>/`) are working runs. Once a sub-region's
pixels are verified, promote them by copying the good rows into the root
`training_candidates_YYYY-MM-DD.csv` and its `_enriched.csv`. Only root CSVs count toward the
dataset.

---

## Currently Processed Scenes

See the `20*` folders in the project root for the authoritative list. 12 scenes have outputs:

```
20221023 (+B, ML, MR)   20230415   20230519   20231026   20240314   20240427
20241001 (M, TR)   20241002 (+B, BL)   20241018   20250420   20250515
20251011 (+M, MR)
```

`(+...)` means the date has a root folder of finished pixels plus sub-region runs; `(...)` alone
means sub-region runs only, so nothing from that date is in the dataset yet. The 2023-09-25 scene
was deliberately removed.

### Registry status — by design

`SCENES` holds **only the scenes currently being processed** — in both extract scripts. Finished
scenes are removed. Only **3 of the 12** scenes have an entry right now:

| File | Registered dates |
|------|------------------|
| `extract_training_pixels.py` | 2025-04-20 |
| `extract_training_pixels_auto.py` | 2024-10-01, 2024-10-02 (rewritten by `auto_pipeline.py`) |

`auto_pipeline.py`'s `rewrite_scenes()` replaces the whole `SCENES` block with a single entry
for the date just processed. **This is deliberate — do not "fix" it into a merge.** The registry
is kept minimal so it is obvious which scene a bare run targets, and the operator checks
manually that an existing scene folder is not about to be overwritten before running.
`extract_training_pixels.py` is pruned by hand to the same standard.

**When pruning, keep the `--date` default pointing at a registered date.** argparse does not
check defaults against `choices`, so a default naming a removed date only fails later, as a
`KeyError` inside `main()`. (`auto_pipeline.py` updates the `_auto` default itself.)

Consequence to accept: 20221023, 20230415, 20230519, 20231026, 20240314, 20240427, 20241018,
20250515 and 20251011 have outputs but no entry. To re-run one, rebuild its entry from the scene
folder plus the `landsat_scene` column of its CSV.

Note that entry count does not affect anything else: `SCENES[date]` selects exactly one scene
per run and `N_CANDIDATES = 25` applies to that scene alone, so extra entries can neither
collide nor change the pixel count. The overwrite protection comes from the manual pre-run check.

### `viirs_data/` is currently empty

Every `gitco`/`gimgo` path in both `SCENES` dicts points at a granule that is not on disk.
All 12 remaining scenes do have their cached `viirs_alaska_*.tif`, so the extractor's fast path still works.
Anything that needs to resample from H5 — a new granule, or any step that says to delete the TIF —
will fail until those granules are re-downloaded with `AWSExtract.py`.

### CSV row convention

The extractor writes up to 25 rows *including* unconfirmed ones (blank `ground_truth_class`).
By convention most CSVs are then **pruned by hand to only the Landsat-confirmed rows**, which is
why row counts vary (2-16). Only `20241002/BL/` still holds the full 25.
Pruning is optional — `enrich_modis.py` filters to confirmed rows on its own. Root CSVs hold
finished pixels only (see *Root folder = finished pixels*).

### Class balance — finished dataset (88 pixels)

Counted from the **root** `training_candidates_*.csv` of each date (not `_enriched`, not
sub-regions), across 11 dates:

| Count | Class |
|-------|-------|
| 36 | `ice_free_river_snow_free_land` |
| 28 | `ice_covered_river_snow_free_land` |
| 19 | `ice_covered_river_snow_covered_land` |
| 5 | `ice_free_river_snow_covered_land` |

All four classes are represented and only these four names occur. `ice_free_river_snow_covered_land`
is by far the thinnest at 5 and is the one worth targeting — best months are April (spring
snowmelt) or November.

**Not yet promoted:** 22 labeled pixels sit in sub-region folders without a copy in their root —
`20241001/M` (4), `20241001/TR` (4), `20241002/BL` (5), `20251011/M` (4), `20251011/MR` (5).
Ten of them are `ice_free_river_snow_covered_land`. If they pass verification, promoting them
would triple the scarce class.

---

# AUTOMATED PATH — `auto_pipeline.py` (preferred)

For most new scenes, this is the only setup script you need. Given a Landsat scene ID and a date, it:

1. Parses the scene ID → sensor (LC8/LC9), Path, Row.
2. Queries Earth Engine for the Landsat scene's bbox.
3. Lists every GITCO granule for that date in the J01, J02, and NPP buckets.
4. Pre-filters by UTC time window (default `1700-2400`, when Alaska ascending passes occur).
5. Reads each granule's bbox attrs over S3 **byte-range** (no full download — fast).
6. Ranks granules by `(nadir_distance asc, coverage desc)` and picks the best with coverage ≥ 65%.
7. Downloads the full SVI01–05 + GITCO files for that granule.
8. Stitches the SVI bands into a GIMGO H5.
9. Rewrites the `SCENES` dict in `extract_training_pixels_auto.py` to point at the new files.
10. Deletes any stale Alaska TIF for this date (so the next extract resamples fresh).

### Usage

```bash
.venv\Scripts\python.exe auto_pipeline.py --scene LC90710162022296LGN01 --date 2022-10-23
```

Optional flags:
- `--time-window HHMM-HHMM` — UTC time-of-day filter for granules (default `1700-2400`; pass `all` to disable).
- `--no-download` — Stop after selection, print top-5 candidates and exit. Useful for diagnosing bad picks.

After it finishes, run the extractor + enrichment:

```bash
.venv\Scripts\python.exe extract_training_pixels_auto.py
.venv\Scripts\python.exe enrich_modis.py --date YYYY-MM-DD
```

Then jump to **STEP 6 — Visual verification in GEE** and **STEP 7 — MODIS enrichment** below.

---

# MANUAL PATH — Step-by-Step Pipeline

Use this when `auto_pipeline.py` can't find a good granule, or when you want fine-grained control
over which granule is used. Steps 1–4 replace what the auto pipeline does; steps 5–7 are shared.

---

### STEP 1 — Get the Landsat bbox (YOU run this)

Update `get_landsat_footprint.py` with the correct collection, Path, Row, and date, then run it:

```python
# Landsat 8 → 'LANDSAT/LC08/C02/T1_L2'   (ID starts with LC8)
# Landsat 9 → 'LANDSAT/LC09/C02/T1_L2'   (ID starts with LC9)
# Parse Path and Row from the scene ID:
#   LC8  073  014  2024292LGN00
#        ^^^  ^^^
#       Path  Row
```

```bash
.venv\Scripts\python.exe get_landsat_footprint.py
```

Record the **Bounding Box** output (Lat min/max, Lon min/max).

---

### STEP 2 — Find the VIIRS overpass (USER does this)

Tell the user:
> "Here are the Landsat bounding box coordinates:
> - Lat: {min} to {max}
> - Lon: {min} to {max}
>
> Please go to https://www.class.noaa.gov/ and search for an **ascending** VIIRS pass
> (J01, J02, or NPP) on {DATE} that overlaps this bounding box.
> Look at the SVI01 product. When you find a good candidate, paste back:
> - The satellite (J01, J02, or NPP)
> - The start time (the `t` code, e.g. `t2108`)
> - The full dataset filename from NOAA CLASS"

**Wait for the user to come back with the overpass information.**

#### Critical: Granule vs. Full Pass
NOAA CLASS shows the *entire* orbit pass (6+ minutes), but AWS stores data in small **~1.5-minute granules**. The granule the user picks in NOAA CLASS may not be the one that actually covers the Landsat bbox.

- Once you know the `t` time code, download that granule and run `extract_training_pixels.py`.
- If the output PNG shows the VIIRS data only covering part of the scene (missing left or right side), the granule is off. Try the adjacent granule — the one 1-2 minutes earlier or later. Check the bbox overlap from the VIIRS loader output (`[clip] Lat/Lon` line).
- Never delete the old Alaska TIF if you're switching granules — delete it explicitly so the script resamples fresh data.

---

### STEP 3 — Download VIIRS granules (YOU run this)

Update `AWSExtract.py`:
```python
BUCKET_NAME = 'noaa-nesdis-n21-pds'   # J02 example — change per satellite
# ...
TARGET_DATE = 'YYYY-MM-DD'
TARGET_HOUR = '_tHHMM'                 # e.g. '_t2108'
```

```bash
.venv\Scripts\python.exe AWSExtract.py
```

This downloads: `SVI01`, `SVI02`, `SVI03`, `SVI04`, `SVI05`, and `GITCO` into `./viirs_data/`.

---

### STEP 4 — Stitch the SVI bands (YOU run this)

Update the glob pattern in `stitch_h5.py` line 11:
```python
pattern = os.path.join(target_dir, f"SVI0{i}_<sat>_*_t<HHMM>*.h5")
# e.g.: f"SVI0{i}_j02_*_t2108*.h5"
```

```bash
.venv\Scripts\python.exe stitch_h5.py
```

Note the exact stitched filename printed (e.g. `GIMGO-SVI01-...-t2108..._stitched.h5`).

---

### STEP 5 — Update extract_training_pixels.py (YOU edit, USER runs)

1. Add an entry for the new scene to the `SCENES` dict at the top of the file. `SCENES` holds only the scenes currently being processed, so remove entries for finished scenes, and keep the `--date` default pointing at a registered date (see *Registry status*):

```python
"YYYY-MM-DD": {
    "gitco":         "viirs_data/GITCO_<sat>_d<DATE>_t<TIME>_..._oebc_ops.h5",
    "gimgo":         "viirs_data/GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_<sat>_d<DATE>_t<TIME>_..._stitched.h5",
    "landsat_scene": "<LANDSAT_SCENE_ID>",
    "landsat_bbox":  (<lat_min>, <lat_max>, <lon_min>, <lon_max>),
    "output_dir":    "YYYYMMDD",
    "ee_collection": "LANDSAT/LC08/C02/T1_L2",   # or LC09
    "viirs_tif":     "YYYYMMDD/viirs_alaska_YYYYMMDD.tif",
},
```

2. If re-processing a scene with a different granule, **delete the old TIF** so the script resamples from scratch:
```powershell
Remove-Item YYYYMMDD\viirs_alaska_YYYYMMDD.tif
```

3. Tell the user:
> "Everything is set up. Please run:
> ```
> .venv\Scripts\python.exe extract_training_pixels.py --date YYYY-MM-DD
> ```
> Optionally add `--region B` (or any of T/M/B/TL/.../BR) to scope candidates to a sub-region.
> Let me know what the output shows and whether the PNG looks correct (VIIRS data should cover the full Landsat bounding box on the left panel)."

**Do NOT run `extract_training_pixels.py` yourself — always hand off to the user so they can see the live terminal output.**

---

### STEP 6 — Visual verification in GEE (USER does this)

After the user runs the extractor and confirms the PNG looks good:

Tell the user:
> "Paste `YYYYMMDD/gee_inspector_YYYYMMDD.js` into https://code.earthengine.google.com/.
> The script sets the basemap to HYBRID (Google satellite imagery) and includes a toggleable
> Sentinel-2 10m layer for the nearest cloud-free pass within ±7 days.
> Click the Inspector tab and then click each marker to verify the auto-labeled class.
> If any pixel's auto-label looks wrong (e.g. labeled ice but Landsat shows open water),
> update `ground_truth_class` in the CSV directly."

Wait for the user to confirm verification is done.

---

### STEP 7 — MODIS enrichment (YOU run this)

After GEE verification, enrich the CSV with MODIS NDVI and land cover:

```bash
.venv\Scripts\python.exe enrich_modis.py --date YYYY-MM-DD
```

Useful flags:
- `--force` — overwrite an existing `_enriched.csv` (otherwise it's skipped).
- `--region <code>` — restrict to one subfolder. Examples: `--region B` (the full bottom row),
  `--region BL` (single cell), `--region ROOT` (only the top-level folder, skipping subregions).
  Omit to enrich the top-level folder **and** every sub-region folder for that date.
- `--file PATH` — enrich one explicit CSV, bypassing date/region discovery entirely.
  Path is relative to the project root, or absolute.
- `--date` may be omitted to process every scene.

The script reads `YYYYMMDD[/<REGION>]/training_candidates_YYYY-MM-DD.csv` (only the Landsat-confirmed rows),
queries MODIS MOD13Q1 NDVI (250m, 16-day composite) and MCD12Q1 land cover (500m, annual)
via Earth Engine, and writes `…_enriched.csv` alongside.

The original CSV is never modified.

**Expected output columns added:**
- `modis_ndvi` — float, scaled (raw × 0.0001); NaN if pixel outside MODIS coverage
- `modis_lc_type1` — IGBP integer class (e.g. 1=Evergreen Needleleaf, 17=Unclassified)
- `modis_lc_name` — human-readable IGBP label

---

## Known Pitfalls

| Issue | What happened | Fix |
|-------|--------------|-----|
| VIIRS data missing on left side of PNG | The granule's swath edge doesn't reach the west side of the Landsat bbox | Try the next granule 1-2 min earlier or later; re-download, re-stitch, delete old TIF, re-run. With `auto_pipeline.py`, try `--time-window all` or inspect the top-5 via `--no-download` |
| `ValueError: No VIIRS pixels within bbox` | The granule doesn't overlap the bbox at all | Wrong granule — go back to NOAA CLASS (manual) or re-check the auto-pipeline top-5 |
| `auto_pipeline.py` reports "No granule reached the 65% threshold" | No granule's swath envelope covered enough of the bbox | Widen `--time-window all`, or fall back to the manual path |
| `UnicodeDecodeError` reading a training CSV | The extract scripts write UTF-8, but a CSV re-saved by Excel can come back as cp1252 (a literal em-dash in `notes` is the usual culprit) | Handled: `_read_csv_robust()` tries `utf-8-sig`, falls back to cp1252 and prints a NOTE. All CSVs currently on disk are UTF-8 |
| A pixel labeled `ice_free_river_snow_land` | Legacy class name from before the rename. No row on disk uses it today | Verify the pixel in GEE, then set it to `ice_free_river_snow_covered_land` or `ice_free_river_snow_free_land` |
| Auto-label says snow but the ground is bare | `NDSI > 0.4` over-calls snow in shoulder seasons — it picks up cloud and open water. In `20230519` all 9 `snow_land` auto-calls were rejected during GEE verification (mid-May, 0-2°C) | Trust the GEE verification over the auto-label; correct `ground_truth_class` in the CSV |
| `enrich_modis.py` skips a CSV | `_enriched.csv` already exists | Pass `--force` to overwrite |
| Edit applied only to one extract script | The two extract scripts share downstream code but are separate files | Always apply pipeline-logic edits to **both** `extract_training_pixels.py` and `extract_training_pixels_auto.py` |

---

## Landsat Scene ID Parsing Reference

```
LC8  073  014  2024  292  LGN00
│    │    │    │     │    └─ Ground station version
│    │    │    │     └─ Day-of-year (Julian)
│    │    │    └─ Year
│    │    └─ WRS Row (3 digits)
│    └─ WRS Path (3 digits)
└─ Sensor (LC8=Landsat8, LC9=Landsat9)
```

Use `LC08/C02/T1_L2` for Landsat 8, `LC09/C02/T1_L2` for Landsat 9 in both GEE Python and JS.
