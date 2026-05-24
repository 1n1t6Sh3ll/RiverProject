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
| `ice_free_river_snow_land` | Liquid water | Snow-covered |
| `ice_covered_river_snow_covered_land` | Frozen | Ice/snow-covered |
| `ice_covered_river_snow_free_land` | Frozen | No snow |

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
| `training_candidates_YYYY-MM-DD.csv` | 25 candidate pixels with VIIRS + Landsat values |
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

Each run still produces exactly 25 candidate pixels — they're just spread over the selected region.
Outputs go to `YYYYMMDD/<REGION>/...`.

---

## Currently Processed Scenes

See the `20*` folders in the project root for the authoritative list. Notable recent scenes:
- `20221023/` — LC90710162022296LGN01 (J01) — `B/`, `ML/`, `MR/` sub-region runs
- `20240427/` — LC80710142024118LGN00 (J02)
- `20250420/` — LC90730162025110LGN00 (NPP)

(Older scenes: 20230415, 20230925, 20231026, 20240314, 20241018.)

**Class still needed:** `ice_free_river_snow_land` — best months: April (spring snowmelt) or November.

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

1. Add a new entry to the `SCENES` dict at the top of the file (do NOT remove old entries — selection is by `--date`):

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
| `UnicodeEncodeError` writing GEE .js or CSV | Special characters in output strings (already fixed) | `open(..., encoding="utf-8")` in the write calls |
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
