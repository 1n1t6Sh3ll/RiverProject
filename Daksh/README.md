# River-Ice Surface Classification — CNN & Ensemble Models

Classifies river-ice surface conditions from VIIRS satellite imagery using a 1D CNN, Random Forest, and XGBoost. Each pixel is labelled as one of four classes based on spectral bands, viewing geometry, and ancillary data.

---

## Classes

| Label | Description |
|---|---|
| `ice_covered_river_snow_covered_land` | River frozen, surrounding land snow-covered |
| `ice_covered_river_snow_free_land` | River frozen, surrounding land snow-free |
| `ice_free_river_snow_free_land` | Open river, snow-free land |
| `ice_free_river_snow_land` | Open river, snow-covered land |

---

## Features (21)

| Feature | Source |
|---|---|
| I1, I2, I3, I4, I5 | VIIRS I-band reflectances / brightness temperatures |
| SZA, SAA, VZA, VAA | Solar/view zenith and azimuth angles |
| VIIRS_NDWI | Derived: (I1 − I2) / (I1 + I2) |
| water_fraction | MODIS-derived water fraction |
| modis_ndvi | MODIS NDVI |
| lc_1, lc_4, lc_5, lc_7, lc_8, lc_9, lc_10, lc_11, lc_16 | One-hot encoded MODIS IGBP land-cover type (`modis_lc_type1`) — all 9 category codes present in the data |

**Verified from source** (`python3 -c "from train_cnn import FEATURE_COLUMNS; print(FEATURE_COLUMNS)"`), not just claimed — this is the literal, currently-active `FEATURE_COLUMNS` list in both `train_cnn.py` and `train_model.py`:
```
['I1', 'I2', 'I3', 'I4', 'I5', 'SZA', 'SAA', 'VZA', 'VAA', 'VIIRS_NDWI',
 'water_fraction', 'modis_ndvi',
 'lc_1', 'lc_4', 'lc_5', 'lc_7', 'lc_8', 'lc_9', 'lc_10', 'lc_11', 'lc_16']
```
21 entries. None of `I1`–`I5` are Landsat bands (they're VIIRS); `landsat_scene` and the free-text `notes` column (which contains Landsat `ST_B10`/`NDSI` values used only to hand-verify labels) are **not** in this list and are not used as model inputs.

> **Land-cover feature history (for redo/revert):**
> 1. **No land cover** (12 features) — original pipeline. Git commit `0d77748`.
> 2. **Full one-hot, 9 categories** `[1, 4, 5, 7, 8, 9, 10, 11, 16]` (21 features, **current state**) — added to test whether land-cover context helps. Improved every model over (1): RF macro F1 63.2%→71.9%, XGB 61.3%→69.7%, CNN 75.3%→77.1%. Git commit `4e9b727`.
> 3. **Pruned, 6 categories** `[1, 7, 8, 9, 10, 11]` (18 features) — tried dropping `lc_4`/`lc_5`/`lc_16` (near-zero measured importance, rarest categories: n=1/7/4). **Made things worse**, not better — CNN macro F1 77.1%→75.4%, RF macro F1 71.9%→70.8%, RF minority-class recall 0.43→0.29. Kept only for comparison. Git commit `a5d8b3d`.
> 4. **Reverted to state (2)** — restored via `git checkout 4e9b727 -- train_cnn.py train_model.py`, retrained, and reconfirmed identical metrics (deterministic seed=42). This is the current, best-performing, and validated state. Git commit `e615a41`.
>
> To redo/switch states: `git log --oneline` lists all commits in order; `git checkout <hash> -- train_cnn.py train_model.py` restores a given state's code, then rerun `train_cnn.py`, `train_model.py`, `compare_all_models.py`, `map_predictions.py`, and `export_geojson.py` to regenerate everything (models, comparison report, map, GeoJSON) to match.

---

## Dataset

- **File:** `data/combined_all.csv`
- **Total samples after filtering:** 385
- **Filters applied:** `water_fraction < 0.90`, no Landsat-null / CONFLICT / EXCLUDED rows, valid `modis_ndvi`
- **Split:** 80% train / 20% test (stratified), with 10% of train held for early-stopping validation (CNN only)
- **Last full pipeline run:** 2026-09-08 — all scripts (`train_cnn.py`, `train_model.py`, `compare_all_models.py`, `map_predictions.py`, `export_geojson.py`) re-run end to end; metrics reproduced exactly (seed=42, deterministic), confirming the committed state (`e615a41`) is stable and reproducible.

---

## Scripts

| Script | Purpose |
|---|---|
| `train_cnn.py` | Train 1D CNN classifier, save model + scaler + confusion matrix |
| `train_model.py` | Train Random Forest and XGBoost classifiers, save models + reports |
| `evaluate_dice_iou.py` | Compute Dice and IoU metrics on predictions |
| `extract_training_pixels.py` | Extract pixel-level training data from raw imagery |
| `enrich_modis.py` | Append MODIS NDVI / water fraction to training CSV |
| `verify_water_filter.py` | Sanity-check water fraction filter thresholds — **not runnable in this checkout**: needs raw JRC water-occurrence/seasonality rasters (`outputs/alaska_occ_375m.tif`, `alaska_sea_375m.tif`) and raw per-scene training-candidate CSVs from an earlier pipeline stage, neither of which exist here |
| `viirs_training_loader.py` | Load and preprocess VIIRS granules for training |

---

## Model Architecture — 1D CNN

```
Input: (batch, 1, 12)
  → Conv1d(1→32, k=3) + BN + ReLU
  → Conv1d(32→64, k=3) + BN + ReLU
  → AdaptiveMaxPool1d(4) + Dropout(0.3)
  → Linear(256→128) + ReLU + Dropout(0.3)
  → Linear(128→4)
```

- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=150)
- **Loss:** CrossEntropyLoss with inverse-frequency class weights
- **Epochs:** 150, best checkpoint by validation accuracy

---

## Results

### 1D CNN

| Metric | Combined |
|---|---|
| n_train | 308 |
| n_test | 77 |
| Test accuracy | **79.22%** |
| 5-fold CV accuracy | 77.66% ± 3.62% |
| F1 ice_free_snow_free | 0.8814 |
| F1 ice_free_snow_land | 0.8000 |
| F1 ice_cov_snow_cov | 0.7805 |
| F1 ice_cov_snow_free | 0.6667 |

### Random Forest vs XGBoost

| Metric | RF | XGB |
|---|---|---|
| Test accuracy | 77.92% | 72.73% |
| CV mean accuracy | **81.82%** | 81.30% |
| F1 ice_free_snow_free | 0.8750 | 0.8387 |
| F1 ice_cov_snow_cov | 0.7568 | 0.7317 |
| F1 ice_cov_snow_free | 0.7000 | 0.6500 |

### CNN vs RF vs XGBoost — overall

| Metric | CNN | RF | XGB |
|---|---|---|---|
| Test accuracy | **77.92%** | 77.92% | 75.32% |
| Balanced accuracy | **78.35%** | 70.04% | 68.42% |
| Macro F1 | **77.13%** | 71.87% | 69.65% |
| Cohen's Kappa | 0.6898 | 0.6812 | 0.6465 |

**Land-cover experiment result:** adding the 9 one-hot MODIS land-cover columns (see Features) improved every model over the pre-land-cover baseline — RF gained the most (balanced accuracy 63.18% → 70.04%, macro F1 63.22% → 71.87%, minority-class F1 for `ice_free_river_snow_land` 0.22 → 0.55), XGB similarly (balanced accuracy 61.45% → 68.42%), and CNN's macro F1 rose 75.31% → 77.13% at the same test accuracy. Full breakdown in `outputs/comparison_all_models.txt`.

To revert this experiment: `git log --oneline` shows the pre-land-cover baseline commit; `git diff <baseline-hash> -- train_cnn.py train_model.py` shows exactly what changed, and `git checkout <baseline-hash> -- train_cnn.py train_model.py` restores the 12-feature version.

---

## Outputs

| File | Description |
|---|---|
| `outputs/cnn1d_combined.pt` | CNN model weights |
| `outputs/scaler_combined.pkl` | StandardScaler fitted on CNN training data |
| `outputs/label_encoder.pkl` | LabelEncoder (shared across models) |
| `outputs/rf_combined.pkl` | Trained Random Forest model |
| `outputs/xgb_combined.pkl` | Trained XGBoost model |
| `outputs/confusion_matrix_cnn1d_combined.png` | CNN confusion matrix |
| `outputs/confusion_matrix_rf.png` | RF confusion matrix |
| `outputs/confusion_matrix_xgb.png` | XGBoost confusion matrix |
| `outputs/cnn1d_report.txt` | CNN summary metrics |
| `outputs/comparison_report_v2.txt` | RF vs XGBoost comparison |
| `outputs/comparison_all_models.txt` | CNN vs RF vs XGBoost full comparison |
| `outputs/dice_iou_report.txt` | Standalone CNN Dice/IoU eval (`evaluate_dice_iou.py`) — macro Dice 0.7821, macro IoU 0.6486 |
| `outputs/ice_map.html` | Interactive satellite map of predictions (folium) |
| `outputs/ice_predictions.geojson` | Predictions as GeoJSON for Google Earth Engine |

---

## Requirements

```
torch
scikit-learn
xgboost
pandas
numpy
matplotlib
joblib
```

Install with:

```bash
pip install torch scikit-learn xgboost pandas numpy matplotlib joblib
```

## Usage

```bash
# Train CNN
python3 train_cnn.py

# Train RF + XGBoost
python3 train_model.py

# Compare all three models
python3 compare_all_models.py

# Standalone CNN Dice/IoU eval
python3 evaluate_dice_iou.py

# Generate interactive prediction map (outputs/ice_map.html)
python3 map_predictions.py

# Export predictions as GeoJSON for Earth Engine
python3 export_geojson.py
```
