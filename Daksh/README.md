# Reprojection — VIIRS + Landsat river-ice training data (Alaska)

Builds training-candidate pixels for river-ice / snow classification by
putting VIIRS (375 m) and Landsat (30 m) on a common Alaska EPSG:4326 grid,
sampling mixed river pixels, enriching them with MODIS NDVI / land cover,
and checking every point against Landsat in Google Earth Engine with an
HTML report.

```
DownloadVirs/DownloadVIIRs.py        download VIIRS granules  -> data/viirs/<date_t...>/
        (Landsat scenes you put in)                           -> data/landsat/<YYYYMMDD>/
                │
main_reproject.daksh.py              reproject + 25 candidates -> output/<YYYYMMDD>/
                │                    (evenly spaced, MODIS-enriched CSV)
extracting_randomized_pixels.Daksh.py  random 25 candidates     -> output_pixels/<YYYYMMDD>/
                │                    (MODIS-enriched CSV + automatic GEE report)
output/report_main.py                GEE check + HTML report  -> <csv>_report.html, <csv>_gee_results.csv
output/report_map_server.py          live Earth Engine map for the reports (http://127.0.0.1:8765)
```

---

## Setup (once)

```
pip install numpy pandas rasterio pyproj pyresample h5py matplotlib earthengine-api folium boto3 s3fs
earthengine authenticate
```

Earth Engine project used everywhere: **`noaa-river-ice`**
(`modis_enrich.EE_PROJECT`). If Earth Engine can't initialise, the pipeline
still runs — MODIS columns are just left empty.

Required inputs:

| Path | What |
|---|---|
| `data/viirs/<YYYYMMDD_tHHMMSSS>/` | `GITCO_*.h5` + `GIMGO-SVI01-...-SVI05_*.h5` per granule |
| `data/landsat/<YYYYMMDD>/` | USGS Collection 2 L1 scene files (`*_B1..B7.TIF`, `*_B10.TIF`, angle TIFs, `*_MTL.txt`) — several scenes per date is fine |
| `data/alaska_occ_375m.tif`, `data/alaska_sea_375m.tif` | water-occurrence / sea masks on the 375 m grid |

---

## 1. Download VIIRS — `DownloadVirs/DownloadVIIRs.py`

Downloads NOAA-CLASS-style **aggregated** granules from the public NOAA PDS S3
buckets and writes them in the same format as `data/viirs/`.

```
python DownloadVirs/DownloadVIIRs.py SVI02_j01_d20240904_t2357037_e0002437_b35223_c20240905002744224000
python DownloadVirs/DownloadVIIRs.py --granule-file granules.txt
```

Options: `--xml` (AOI file, default `DownloadVirs/VIIRS_SDR.xml`), `--out`, `--no-map`.
Writes a `coverage_map.html` per granule (swath outlines + AOI box).

## 2. Reproject + candidates — `main_reproject.daksh.py`

```
python main_reproject.daksh.py
```

For every date that has both VIIRS and Landsat:

1. VIIRS → full-Alaska 9-band GeoTIFF (`I1–I5, SZA, SAA, VZA, VAA`)
2. Each Landsat scene → `landsat_<date>_<N>.tif` (scene extent, aligned to the Alaska grid)
3. Alignment check + sample tiles
4. 25 evenly spaced mixed-pixel candidates → `training_candidates_<date>_<tag>.csv` + PNGs
5. MODIS enrichment (see below)

Settings at the top of the file: `OVERWRITE`, `ENABLE_MODIS`, `EE_PROJECT`.
Dates (`viirs_date`, `landsat_date`) are written as `YYYY-MM-DD`, same as step 3. Log: `output/pipeline_run.log`.

## 3. Random candidates + report — `extracting_randomized_pixels.Daksh.py`

Runs **after** step 2 (needs the TIFs in `output/<date>/`). Picks 25 random
mixed-pixel candidates per VIIRS × Landsat pair, adds MODIS, then
**automatically** runs the GEE report on each CSV.

```
python extracting_randomized_pixels.Daksh.py                 # all dates
python extracting_randomized_pixels.Daksh.py 20241126        # one date (or several)
python extracting_randomized_pixels.Daksh.py 20241126 --view # + open reports with live map
```

| Flag | Effect |
|---|---|
| `--overwrite` | regenerate existing CSV/PNG/report |
| `--seed N` | reproducible selection |
| `--uniform` / `--pure-random` | evenly spaced / fully random (default: stratified random) |
| `--region top\|middle\|bottom`, `--side left\|right` | restrict where candidates are picked |
| `--no-modis` | skip MODIS columns |
| `--no-report` | don't run the GEE report |
| `--view` / `--no-view` | open reports + start map server without asking / never ask (default: asks at the end) |

Output in `output_pixels/<YYYYMMDD>/`:

```
training_candidates_<date>_v<V>_ls<N>.csv                 candidates (+ MODIS)
training_candidates_<date>_v<V>_ls<N>_report.html         GEE report
training_candidates_<date>_v<V>_ls<N>_gee_results.csv     GEE values per point
viirs_vs_watermask_<date>_v<V>_ls<N>.png, panels/
```

`landsat_scene` holds the real USGS scene ID (e.g. `LC08_L1TP_074019_...`),
taken from that scene's MTL — the report needs it to find the scene in Earth Engine.

## 4. GEE report — `output/report_main.py`

Works on **any** candidates CSV (from step 2 or step 3; needs `lat`, `lon`, `landsat_scene`).

```
python output/report_main.py output/20241130/training_candidates_2024-11-30_ls0.csv --project noaa-river-ice
```

Per point: samples every Landsat band, indices, ST_B10 and the full QA cloud
flags; classifies it (CLASS 0–4); compares with `ground_truth_class`
(MATCH / MISMATCH / CLOUD / NO DATA); adds MODIS; downloads image chips.

Options: `--views true_color,cloud_qa`, `--no-chips` (fast), `--no-open`, `--no-modis`.

## 5. Live map — `output/report_map_server.py`

The report's live map (pan/zoom, switch band combos) needs this small local server.

```
python output/report_map_server.py --project noaa-river-ice --root output_pixels
```

It walks `--root` (default: current folder) for every `*_report.html` and
prints a link for each, e.g.
`http://127.0.0.1:8765/20241126/training_candidates_2024-11-26_v0_ls3_report.html`.
Open reports **through that link** — double-clicking the HTML works for the
table/chips, but the live map panel usually won't load from `file://`.
Use `--root output` for reports next to step-2 CSVs, `--root .` for all. Ctrl+C to stop.
`--view` in step 3 does all of this for you.

---

## MODIS enrichment — `modis_enrich.py`

Shared by steps 2, 3 and 4 so they behave identically. Adds three columns:

| Column | Source |
|---|---|
| `modis_ndvi` | MOD13Q1 NDVI, 250 m, latest 16-day composite up to the scene date (scaled −1…1) |
| `modis_lc_type1` | MCD12Q1 `LC_Type1` (IGBP), 500 m, scene year — or up to 2 years earlier if not yet published |
| `modis_lc_name` | IGBP class name |

Only **Landsat-confirmed** rows are sampled; rows whose `notes` say
"Landsat null" keep empty MODIS cells. Empty NDVI on a confirmed row means
MODIS itself has no value there (common in winter / low sun).
The report reuses MODIS values already in the CSV and only fills gaps.

## Maps

All HTML maps (reports, coverage maps) use **Esri World Imagery** tiles.
OpenStreetMap tiles return *403 / Access blocked* when a page is opened from
disk, so don't switch back to them.

## Other

- `steps/` — modular version of the step-2 pipeline (`python -m steps.pipeline`),
  with its own `config.py`, `data/` and `output/`. Its MODIS sampling is its own
  copy (`steps/step2_output.py`), not `modis_enrich.py`.
- Candidate selection rule (all versions): `0.05 < water_fraction_occ < 0.90`,
  `water_fraction_sea < 1.0`, VIIRS I1 valid, Landsat valid = **any** band
  valid (OR across bands). Keep it that way.
- `check.md` — notes on dates skipped (clouds / no Landsat).
