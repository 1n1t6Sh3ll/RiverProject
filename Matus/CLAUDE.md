# Matus/ — river ice / narrow-rivers work

## Environment
- Run scripts from `Matus/` with `.venv\Scripts\python.exe`.
- Rasters (*.tif), *.csv and *.h5 are gitignored; scripts regenerate them.
  Exception: `narrow_rivers/classify/samples_2024-06-22.csv` (v3) and
  `narrow_rivers/classify/qc/experiments/samples_2024-06-22_v1.csv` are force-added (`git add -f`).

## Hand labels — never break these
- Label shapefiles in `narrow_rivers/patches/nenana_*/labels/` are edited only by hand in QGIS.
  Scripts open them read-only; never write, reproject or "fix" them.
- `poly_id` is positional: deleting a polygon renumbers the rest. Re-run
  `narrow_rivers/classify/extract_samples.py` after every QGIS label edit.
- 2025-07-04 is the held-out test date: never label it, never train on it.
- Trust only our own labels, not other teammates' label sets.

## Random Forest gotchas
- Do NOT use the Landsat QA_PIXEL cloud mask: it flags silty water / river ice
  as cloud (77.7 % of the main channel on 2023-07-24). Scenes are checked by eye.
- Cross-date prediction must use anchoring (subtract each feature's median over
  JRC=0 dry land, per date, in training and prediction). The cause is haze, not sun angle.
- `class_weight="balanced"` inflates river area ~3x; keep `class_weight=None`.
- Compare label versions on matched pixels / same folds, never raw CV alone.

## Where things are
- Method history and numbers: `narrow_rivers/classify/qc/NOTES.md`.
