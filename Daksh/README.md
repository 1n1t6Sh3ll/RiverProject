# Daksh — VIIRS/Landsat River-Ice Training Pipeline

Scripts and data for building a labeled training set of Alaskan river-ice / snow surface conditions from VIIRS and Landsat imagery, and for training/evaluating classifiers on it. This folder covers reprojection and pixel extraction; the `CNN/` subfolder contains the model training code (see [`CNN/README.md`](CNN/README.md)).

## Pipeline

1. **Reprojection** — `main_reproject.daksh.py`
   Reads raw VIIRS H5 granules (GITCO/GIMGO: bands I1–I5, angles SZA/SAA/VZA/VAA) and Landsat GeoTIFF scenes (B1–B6, thermal B10, MTL metadata), and reprojects both to a common Alaska grid in EPSG:4326:
   - VIIRS → 375 m grid (12,432 × 5,328 px), nearest-neighbor swath resampling via `pyresample`.
   - Landsat → 30 m grid aligned to the same domain origin, Lanczos for spectral bands / nearest-neighbor for angles.
   Validates CRS/bounds/grid alignment between the two, then writes georeferenced GeoTIFFs per date to `output/<YYYYMMDD>/`.

   `training_main.Daksh.py` is an earlier/alternate version of this same reprojection + candidate-sampling workflow (also produces 128×128 sample tiles and comparison PNGs directly).

2. **Candidate pixel extraction** — `extracting_randomized_pixels.Daksh.py`
   Scans `output/<YYYYMMDD>/` for reprojected VIIRS/Landsat pairs and selects up to 25 "mixed pixel" candidates per date — locations where water occurrence is between 0.05 and 0.90 and seasonality < 1.0 (see the OR-across-Landsat-bands selection rule; do not simplify this to AND — it under-selects valid candidates). Supports `--uniform`, `--pure-random`, or the default stratified spatial sampling, plus `--region`/`--side` filters and `--seed` for reproducibility.
   For each candidate it pulls VIIRS bands/angles, Landsat bands, NDSI, and (if raw B10 + MTL are available) brightness temperature, then auto-assigns a preliminary `ground_truth_class`. Outputs a CSV plus a 3-panel comparison PNG (VIIRS / water mask / Landsat) with candidates marked.

3. **GEE cross-check** — `candidatePixelGEELookup.js`
   Google Earth Engine script for spot-checking a single candidate point/date against independent Landsat 8/9 TOA imagery: computes NDSI/NDWI/thermal at the point, flags cloud contamination, and classifies into the same 4-class (+cloud) scheme, with an interactive map for manual inspection.

4. **Evaluation** — `evaluate_dice_iou.py`
   Loads a trained CNN (`cnn1d_combined.pt`) plus its label encoder/scaler, re-runs the stratified 80/20 test split, and reports per-class and macro-averaged Dice and IoU from the confusion matrix. Writes `outputs/dice_iou_report.txt`.

## Data

- `2024-06-09.csv`, `2024-09-23.csv`, `2024-11-18.csv`, `2024-11-21.csv`, `2024-11-28.csv`, `2024-11-29.csv` — sample candidate-pixel tables produced by the extraction step, one per VIIRS/Landsat date pair. Columns: `viirs_date, landsat_scene, row_shared_grid, col_shared_grid, lat, lon, water_fraction_occ, I1–I5, SZA, SAA, VZA, VAA, ground_truth_class, notes, modis_ndvi, modis_lc_type1, modis_lc_name`.

## Classification scheme

All stages share the same 4-class labeling (plus cloud, in the GEE script):

| Class | Meaning |
|---|---|
| 1 | Ice-free river / snow-free land |
| 2 | Ice-covered river / snow-covered land |
| 3 | Ice-covered river / snow-free land |
| 4 | Ice-free river / snow-covered land |

Rules of thumb used across scripts: NDSI > 0.4 → snow-covered; B10 brightness temperature < 273 K → ice-covered.

## Requirements

Python with `numpy`, `rasterio`, `pyresample`, `h5py`, `matplotlib`, `scikit-learn`; `earthengine-api` if MODIS enrichment is enabled in `main_reproject.daksh.py` (requires `earthengine authenticate`, project `noaa-river-ice`). The `.js` file runs in the [Google Earth Engine Code Editor](https://code.earthengine.google.com/), not locally.

## Running

All Python scripts here are configured via hardcoded constants (paths, Alaska domain bounds, resampling parameters) rather than CLI flags, except `extracting_randomized_pixels.Daksh.py`:

```bash
python main_reproject.daksh.py

- python extracting_randomized_pixels.Daksh.py [dates] [--overwrite] [--seed N] [--uniform | --pure-random] [--region {top|bottom|middle}] [--side {left|right}]
- python extracting_randomized_pixels.Daksh.py 20241118 --overwrite --region middle --side right --scene 3


python evaluate_dice_iou.py
```

See `CNN/README.md` for model training and results.
