# Handoff — label accuracy work, 2026-10-27

Read this to pick up with no prior context. Full method history is in
`NOTES.md` beside this file (v1 → v5); this is the summary and the next step.

## The project

Sub-pixel river-ice/water mapping on the **Nenana River**, Alaska. VIIRS (375 m)
is the target sensor; Landsat (30 m) hand labels are aggregated into per-VIIRS-cell
**fractions** of river / ice / snow / land. Sentinel-2 (10 m) is a digitising aid
and a QC reference only — never a model input, because only 2 of 10 dates have a
coincident scene.

Repo root: `C:\Users\matus\dev\UG Research Assistant\RiverProject` (work happens
in `Matus/`). Branch `matus/narrow-rivers`. Python via `.venv\Scripts\python.exe`,
run from `Matus/`.

Ten patches in `narrow_rivers/patches/nenana_<YYYYMMDD>/`, all on **one identical
grid** (1635 × 995, EPSG:4326, origin −149.33783783783784 / 64.5354054054), so any
mask transfers pixel-for-pixel between dates. Tracking: `patches/patch_tracking.csv`.

Label classes (`patches/patch_config.py`): `1 river, 2 ice, 3 snow, 4 land,
9 nodata`. Labels are **polygons** in `<patch>/labels/labels_<date>.shp`.

## Where accuracy stands

Random Forest, train-and-predict within one date, 5-fold `StratifiedGroupKFold`
grouped by polygon (`classify/train_rf.py`, 300 trees, `min_samples_leaf=5`,
`class_weight=None`).

| 2024-06-22 (summer) | recall | precision | | 2023-10-04 (freeze-up) | recall | precision |
|---|---|---|---|---|---|---|
| river | 84.4% | 92.7% | | river | 70.8% | 81.1% |
| land | 98.9% | 97.5% | | ice | 80.5% | 75.6% |
| **overall** | **96.9%** | | | snow | **45.3%** | 60.4% |
| | | | | land | 98.6% | 97.3% |
| | | | | **overall** | **93.4%** | |

**Always report per-class recall** — overall is carried by land (83–86% of pixels).
The professor wants ~95%; it is still unclear whether she means the labels or the
model (unanswered question, see below).

`class_weight="balanced"` is rejected: it inflated October river area to 37.9 km²
against 11.9 km² without (June reference 16.0 km²).

## What was settled today

**The labels are good.** An earlier check claimed 658 river pixels (10.8%) held no
water; that was wrong — it thresholded Sentinel NDWI at 0, which misses narrow
channels because a 30–50% water pixel never gets green above NIR. The real figure
is **~97 pixels (1.6%)**. So **label correction is not the route to 95%** — the
model's ~950 June river errors are mostly real model errors.

**Gravel islands were the big false positive.** Polygons 39 and 20 are dry gravel
islands in the main channel (confirmed by eye). A two-endmember water↔land unmix
called them 85% water. Fixed by spectral unmixing against **four** endmembers:

| | green | red | NIR | SWIR1 | SWIR2 |
|---|---|---|---|---|---|
| water | 0.074 | 0.075 | 0.054 | 0.012 | 0.009 |
| gravel | 0.071 | 0.076 | 0.088 | 0.083 | 0.064 |
| dark forest | 0.026 | 0.015 | 0.213 | 0.093 | 0.040 |
| vegetation | 0.036 | 0.022 | 0.266 | 0.133 | 0.058 |

SWIR separates gravel from water (0.083 vs 0.012); NIR separates vegetation.
False "land is water" flags fell **1,992 → 34** of 39,275.

**Unmixing and NDWI are mutually necessary.** Gravel fools NDWI (it reads 0.935
water on polygon 39); dark vegetation fools the unmix (0.70 water at NDWI −0.75).
So `f_mix = min(f_unmix, f_ndwi)` — a pixel is only as wet as both allow.

**Polygon 89 is dry.** Disputed with the professor, who believed it a wet channel.
It is dry on NDWI, MNDWI, NIR, SWIR1, SWIR2, thermal and the unmix, across four
summer dates (2023-07-24, 2024-06-22, 2025-07-04, 2026-08-01), each judged against
its own local references. It is a real channel *form* in the braid belt that
carried no water on any date we hold. **Resolved 2026-10-27:** the professor
reviewed it and judged it a real water channel (visible on Sentinel-2), so it is
included in training as river.

**Labelling rule (decided 2026-10-28):** if any single Sentinel-2 pixel inside a
Landsat pixel contains water, the whole Landsat pixel is labelled water — it is
already a partial pixel.

**Other facts worth keeping:** MNDWI beats NDWI at 30 m (gap 0.616 vs 0.506) but
ties at 10 m (B11 is natively 20 m). Rounding 30 m cells to wet/dry instead of
averaging fractions costs only −0.7% on water area at VIIRS scale, so the mixed
class matters for the *classifier*, not for fraction arithmetic.

## What is NOT settled

**Partial water (bank edges, narrow braid threads) has no trustworthy reference.**
Two Sentinel-derived definitions of water disagree by a factor of 1.9 on total
area (9.55 km² vs 5.14 km²), so the A/B/C experiment below was inconclusive:

| variant | water recall | MAE vs unmix frac | MAE vs threshold frac |
|---|---|---|---|
| A current 2 classes | 84.3% | 0.1556 | **0.0384** |
| B 3 classes with `mixed` | 80.5% | **0.1008** | 0.1899 |
| C partial withheld | 84.1% | 0.1543 | **0.0352** |

The two truths pick opposite winners. B's water recall is also *worse* than A's.
**Verdict: the `mixed` class is not justified by this evidence — but that is a
negative result about the reference, not about the class.**

Attempts that failed, so don't repeat them:
- Fetching Sentinel B11/B12 as real reflectance (done, on disk) did **not** help.
- A hand-drawn dark-forest polygon for Sentinel made it **worse**: the dark-forest
  spectrum sits 0.032 from vegetation while water is 0.242 away, condition number
  1142 vs 52 for the earlier rule. At 10 m dark forest and ordinary vegetation are
  not separable enough to be distinct endmembers, however carefully drawn.
  Sentinel is pinned at three endmembers.

## Agreed next step

**Invert the order: hand-label `mixed` first, then measure.** Draw three or four
small polygons on bank edges and braid threads where partial water is visible,
then test whether the RF can learn them. Enough to test, not enough to waste if it
fails. Proposed scheme (not implemented):

- `5 = mixed` — trainable; also contributes partial weight in `rf_fractions.py`
- `8 = subpixel_channel` — excluded from training (spectrally identical to land by
  construction); its job is to mark channel location for the ice/snow split. Should
  carry a provenance note, since it cannot be checked against the imagery.
  Polygon 89 is the candidate case.

## Open questions for the professor

1. Does "ice" mean **river ice** (split geometrically by a summer channel mask) or
   any frozen surface? This changes the product. A geometric split would also let
   the RF stop being asked an impossible question — ice and snow are spectrally
   near-identical, which is why snow recall is 45%.
2. Is the **95% target** on the hand labels or on the model?

## Tooling

```
.venv\Scripts\python.exe narrow_rivers\classify\extract_samples.py --dates 2024-06-22
.venv\Scripts\python.exe narrow_rivers\classify\qc\label_check.py --date 2024-06-22
.venv\Scripts\python.exe narrow_rivers\classify\train_rf.py --date 2024-06-22
.venv\Scripts\python.exe narrow_rivers\patches\make_composites.py --date 2024-06-22
.venv\Scripts\python.exe narrow_rivers\patches\add_sentinel.py --dates 2024-06-22 --extras
```

`label_check.py` writes, into `classify/qc/`: a report `.txt`, a flagged-pixel
`.geojson` + `.qml`, and single-band rasters `f_mix` (**the estimate to use**),
`f_unmix`, `f_ndwi`, `f_gravel`, `resid`, `s2_frac`, `s2_frac_10m`. Each has a
`.qml` so QGIS styles it on load.

Viewing layers added to `nenana_20240622/`: `_ndwi.tif`, `_432.tif` (true colour),
`_543.tif` (colour infrared — Landsat 543 = Sentinel 843, same wavelengths),
`_mndwi.tif`, `_s2_843_10m.tif`, `_s2_mndwi_10m.tif`, `_s2_swir_refl_10m.tif`.

## Traps

- **`poly_id` is positional** — a running counter over kept polygons, so deleting
  one renumbers every polygon after it. `GRAVEL_POLYS` and `DARK_VEG_POLYS` in
  `label_check.py` are keyed on it. A guard aborts if the gravel endmember stops
  looking like gravel, but re-verify after editing labels.
- **Re-run `extract_samples.py` after any QGIS label edit** — `samples_<date>.csv`
  goes stale silently and every statistic is then keyed to old geometry.
- **All `*.tif` are gitignored** (`.gitignore:21`, "large data, kept locally"), so
  new rasters never appear in `git status`. Imagery is shared via OneDrive by hand.
- `add_sentinel.py --extras` deliberately does not update `README.txt`; the new
  Sentinel lines need adding by hand before a commit.
- October (2023-10-04) is **not** ready for `label_check.py`: ice and snow also
  darken NIR and brighten green, so they need their own endmember. Breakup dates
  (2023-05-05, 2024-04-21) are refused outright — the Sentinel scene is minutes
  away and floes drift >1 m/s, so disagreement measures drift, not labelling.
- The user reviews plans, diffs and commit messages before anything is changed,
  committed or pushed.

## Uncommitted state

Branch `matus/narrow-rivers`. Modified: `patches/add_sentinel.py`,
`requirements.txt`, and the label shapefiles for both worked dates. Untracked:
`narrow_rivers/classify/` (the whole folder, including `qc/`),
`patches/add_ndwi.py`, `patches/make_composites.py`, the per-patch `rf/` folders,
`export/` folders, and the new `.qml` files. Nothing has been committed today.
