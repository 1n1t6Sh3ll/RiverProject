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

## Features (12)

| Feature | Source |
|---|---|
| I1, I2, I3, I4, I5 | VIIRS I-band reflectances / brightness temperatures |
| SZA, SAA, VZA, VAA | Solar/view zenith and azimuth angles |
| VIIRS_NDWI | Derived: (I1 − I2) / (I1 + I2) |
| water_fraction | MODIS-derived water fraction |
| modis_ndvi | MODIS NDVI |

---

## Dataset

- **File:** `data/combined_all.csv`
- **Total samples after filtering:** 364
- **Filters applied:** `water_fraction < 0.90`, no Landsat-null / CONFLICT / EXCLUDED rows, valid `modis_ndvi`
- **Split:** 80% train / 20% test (stratified), with 10% of train held for early-stopping validation (CNN only)

---

## Scripts

| Script | Purpose |
|---|---|
| `train_cnn.py` | Train 1D CNN classifier, save model + scaler + confusion matrix |
| `train_model.py` | Train Random Forest and XGBoost classifiers, save models + reports |
| `evaluate_dice_iou.py` | Compute Dice and IoU metrics on predictions |
| `extract_training_pixels.py` | Extract pixel-level training data from raw imagery |
| `enrich_modis.py` | Append MODIS NDVI / water fraction to training CSV |
| `verify_water_filter.py` | Sanity-check water fraction filter thresholds |
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
| n_train | 291 |
| n_test | 73 |
| Test accuracy | **76.71%** |
| 5-fold CV accuracy | 75.27% ± 4.67% |
| F1 ice_free_snow_free | 0.8148 |
| F1 ice_free_snow_land | 0.7273 |
| F1 ice_cov_snow_cov | 0.8000 |
| F1 ice_cov_snow_free | 0.6857 |

### Random Forest vs XGBoost

| Metric | RF | XGB |
|---|---|---|
| Test accuracy | 73.68% | 73.68% |
| CV mean accuracy | **80.70%** | 78.95% |
| F1 ice_free_snow_free | 0.8772 | 0.8852 |
| F1 ice_cov_snow_cov | 0.7273 | 0.6875 |
| F1 ice_cov_snow_free | 0.3529 | 0.3750 |

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
```
