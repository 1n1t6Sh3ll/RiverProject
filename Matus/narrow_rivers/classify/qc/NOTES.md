# Label QC — method log

Running record of how the label check works, why it changed, and what it has
measured. Append new entries at the bottom; don't rewrite history, because the
corrections are the useful part.

Scope: Nenana patches. Reference dates 2024-06-22 (summer, river/land) and
2023-10-04 (freeze-up, four classes) — the only two with Sentinel-2.

---

## Classifier accuracy (Random Forest, Mode A, 5-fold grouped by polygon)

| 2024-06-22 | recall | precision | | 2023-10-04 | recall | precision |
|---|---|---|---|---|---|---|
| river | 84.4% | 92.7% | | river | 70.8% | 81.1% |
| land | 98.9% | 97.5% | | ice | 80.5% | 75.6% |
| **overall** | **96.9%** | | | snow | **45.3%** | 60.4% |
| | | | | land | 98.6% | 97.3% |
| | | | | **overall** | **93.4%** | |

Overall % is carried by land (83–86% of pixels) — **always report per-class
recall**. Snow→ice is the worst cell: 238 of 521 snow pixels called ice.

`class_weight="balanced"` rejected: inflated October river area to 37.9 km²
against 11.9 km² without it (June reference 16.0 km²). Good for rare-class
recall, wrong for an area/fraction product.

---

## v1 — Sentinel NDWI thresholded at 0 (superseded)

Water = 10 m NDWI > 0, block-averaged 3×3 to the Landsat grid. Flagged river
labels with 0/9 water and land labels with ≥6/9.

Reported **658 river pixels (10.8%) with no water at all**.

**Why it was wrong.** A 30–50% water pixel never gets green above NIR, so a
genuinely wet narrow channel reads as completely dry. Re-examining those 658 by
NIR unmixing: **79% carry real water, median 45.7%**. Only ~100 were genuinely
dry, and 67 of those were in one polygon. The sensor was never the limit — the
binary cut was.

Threshold sensitivity at the time: −0.10 → 451 flags, 0.00 → 658, +0.10 → 1,240.

## v1b — single-band NIR unmix (rejected before use)

Unmixing NIR alone is sensitive enough to see a 3 m channel, but **over-calls
water anywhere dark**: 4.80% of the box above 90% water (river ≈1.5%), and 61%
of those cells contradicted NDWI — shadow and dark conifer. Calibrating an NDWI
curve against it collapsed the curve (topped out at NDWI −0.47 where confirmed
water sits at +0.156), so that whole run's flag counts were void.

Also found: `_scale8` writes NaN as 0, so **Sentinel nodata reads as 100%
water** — 2.24% of the 10 m grid, along the top edge where the scene doesn't
reach. Now excluded explicitly.

## v2 — two-band unmix in (NIR, green) — current

Water is **darker in NIR but brighter in green**; shadow darkens **both**. So
the two move in different directions in that plane and are separable, which no
single band can do.

- `f_mix` — least-squares position along the land→water mixing line. Main estimate.
- `resid` — distance off that line.
- `f_nir` — the old single-band estimate, kept only for contrast.
- `s2_frac` — same unmix at 10 m, averaged over the 3×3 cells that nest exactly.

Endmembers come from each date's own pixels, so nothing is carried between
scenes. 2024-06-22: Landsat water NIR 0.054 / green 0.074, land 0.267 / 0.036.

Sanity: `f_mix` ≥0.90 covers **1.77%** of the box against a river of ~1.5%
(single band gave 1.96%, and 4.80% at 10 m).

### Results, 2024-06-22 (43,229 labelled pixels)

| class | n | f_mix | s2_frac | f_nir | resid |
|---|---|---|---|---|---|
| river | 6,110 | 0.978 | 0.981 | 0.977 | 0.0036 |
| land | 37,119 | 0.046 | 0.000 | 0.046 | 0.0180 |

| flag | n | |
|---|---|---|
| `river_dry` | **68** of 6,110 (1.1%) | labelled river, no water found |
| `land_wet` | **1,992** of 37,119 (5.4%) | labelled land, mostly water |
| `shadow_suspect` | 0 | — see caveats |
| `off_line` | 1,826 (4.2%) | — see caveats |
| `disagree` | 769 (1.8%) | the estimates disagree |

All 68 new `river_dry` are a subset of v1's 658. **Your river labels are ~99%
sound** — far better than v1 claimed.

### Polygons to act on

| polygon | n | f_mix | reading |
|---|---|---|---|
| 39 (land) | 172 | **0.879** | 100% flagged — genuine mislabel, it's water |
| 20 (land) | 297 | — | 71.7% flagged |
| 89 (river) | 80 | **0.000** (p95 0.236) | dry on every test — `subpixel_channel` candidate |
| 102 (class 9) | 1,694 | **0.187** (p95 0.865) | real partial water — `mixed` candidate |

---

## Established

- **Partial water is strongly detectable at 30 m.** Binned by the independent
  Sentinel fraction, Cohen's *d* against dry land: NIR −2.30, NDWI +2.64, green
  +2.57, SWIR1 −1.97, thermal −1.67. A pixel only 11% water (≈3.3 m across a
  30 m cell) is already separable. So a **trainable `mixed` class is viable**,
  and the ≥10% labelling rule sits almost exactly at the detectability limit.
- **MNDWI beats NDWI at 30 m** (water/land gap 0.616 vs 0.506; water median
  +0.72 vs +0.16) but is **a tie at 10 m** (0.602 vs 0.585), because B11 is
  natively 20 m and resampling gives the gain back.
- **Rounding is not the problem for the fractions.** Rounding each 30 m cell to
  wet/dry instead of averaging true fractions costs mean |error| 0.008, max
  0.042, and −0.7% on total water area at VIIRS scale. Errors cancel over ~144
  cells. The `mixed` class earns its place through the **classifier**, not
  through fraction arithmetic.
- **Polygon 89 is dry**, by NDWI, MNDWI, NIR, SWIR1, SWIR2, thermal and the
  two-band unmix, on four separate summer dates (2023-07-24, 2024-06-22,
  2025-07-04, 2026-08-01), each judged against its own local references. Its
  wettest cells are *rarer* than the wettest cells of hand-labelled land. In
  March it reads as open snow-covered ground while the surrounding halo is
  forest — so it is a real channel **form** in the braid belt that carried no
  water on any summer date we hold. Google Maps showing water there in 2026 is
  consistent with this: braided channels switch between years.
- **Label correction is not the route to 95%.** With ~68–100 genuinely bad river
  pixels, the model's 952 June river errors are mostly real model errors. The
  `mixed` class and spatial context features are the path.

## Caveats on the current tool

- **`shadow_suspect` fired 0 times, and that is not the extra conditions
  blocking it** — the gap `f_nir − f_mix` never exceeds 0.041 over labelled
  pixels. So there is essentially no shadow/water confusion in a June scene at
  48.6° sun elevation. The guard is therefore **untested**, and October (21.1°
  sun, known shadow problems) is where it will actually be exercised.
- **`off_line` mostly measures land heterogeneity, not anomalies.** Land's
  residual median is 0.0180 with p90 0.0727, so a 0.030 cut catches the upper
  third of ordinary land — forest, gravel and tundra are not one endmember. Treat
  this flag as weak until there are multiple land endmembers.
- **`land_wet` at 5.4% is high and unverified.** Probably sloughs and ponds
  inside large land polygons; needs eyes before it's believed.
- No DEM hillshade, by choice. Shadow detection is spectral only, and shadowed
  green is lifted by blue/green-rich skylight, towards water.
- SWIR deliberately unused for the shadow test: skylight is very weak there, so
  shadowed ground goes nearly black and mimics water better than anywhere else.
- Ice-free dates only. Ice and snow also darken NIR and brighten green. Breakup
  dates (2023-05-05, 2024-04-21) are refused: the Sentinel scene is minutes to
  tens of minutes away and floes drift >1 m/s.

## Open decisions

1. **Professor:** does "ice" mean river ice (geometric definition, split by a
   summer channel mask) or any frozen surface? Changes the product.
2. **Professor:** is the 95% target on the hand labels or the model?
3. Adopt `mixed` (trainable) and `subpixel_channel` (excluded from training,
   feeds the channel mask) as label classes? Not yet implemented.

## Next

1. Check polygons 39 and 20 in QGIS — likely genuine land→water mislabels.
2. Re-cut polygon 102: threads as `mixed`, the bars between as `land`. 40.6% of
   its cells are genuinely dry, so one class for the whole box is wrong either
   way. Leaving it as class 9 blanks part of several VIIRS cells in
   `rf_fractions.py` and teaches the model nothing.
3. Run the check on 2023-10-04 — needs a visible-brightness term for
   water-vs-frozen, and this is where the shadow guard gets its real test.
4. Implement `mixed` / `subpixel_channel` across `patch_config.py`,
   `extract_samples.py`, `rf_fractions.py`.
5. Save out-of-fold predictions in `train_rf.py`; cross-tab model error against
   `f_mix`.
6. Re-sample the blind audit stratified by predicted class (~50/class, area
   weights recorded) — the current 200/date gave only 3–4 river points.
7. Out-of-sample test on a teammate's dates, per the 2026-10-27 meeting.

## Files

`narrow_rivers/classify/qc/` — `label_check.py` (current), `s2_label_check.py`
(v1, superseded, safe to delete), and per-date `label_check_<date>.{txt,geojson,qml}`
plus `waterfrac_<date>.tif` (bands `f_mix`, `resid`, `f_nir`, `s2_frac`).

Supporting layers added to the patches: `add_ndwi.py` → `<date>_ndwi.tif`;
`make_composites.py` → `<date>_432.tif`, `<date>_543.tif`, `<date>_mndwi.tif`;
`add_sentinel.py --extras` → `<date>_s2_843_10m.tif`, `<date>_s2_mndwi_10m.tif`.
Landsat "432/543" are the same wavelengths as Sentinel "432/843".

---

## v3 — three, then four endmembers (2026-10-27)

**What forced it.** Checking polygons 39 and 20 by eye showed they are **dry
gravel islands** in the main channel, cut through by channels too small to see at
30 m — not water. v2 reported them at **f_mix 0.85**. Gravel is nearly identical
to water in green (0.071 vs 0.074) and only slightly brighter in NIR, so a
water↔land line had nowhere else to put it. SWIR is what separates them: gravel
0.083 against water 0.012.

Root error: the scene was modelled with **two** endmembers. It has at least four.

| | green | red | NIR | SWIR1 | SWIR2 |
|---|---|---|---|---|---|
| water | 0.074 | 0.075 | 0.054 | 0.012 | 0.009 |
| gravel | 0.071 | 0.076 | 0.088 | 0.083 | 0.064 |
| dark forest | 0.024 | 0.015 | 0.192 | 0.078 | 0.034 |
| vegetation | 0.036 | 0.022 | 0.267 | 0.134 | 0.058 |

Adding gravel fixed the islands (0.85 → 0.01) but exposed the mirror fault:
7,531 pixels at `f_unmix` 0.70 with NDWI −0.75 — **dark vegetation**, darker than
the single vegetation endmember, so the unmix explained the darkness by adding
water. 161 clusters, concentrated in the north of the box; the largest (4,541 px
at 64.5127, −149.1395) was confirmed as dark dense forest by eye and became the
fourth endmember. Clusters 2, 5 and 6 lie within 0.011–0.025 of it.

**The two methods are mutually necessary**, which is the real lesson:
- gravel fools **NDWI** (`f_ndwi` 0.935 on polygon 39); the gravel endmember catches it
- dark vegetation fools the **unmix**; NDWI catches it

So `f_mix = min(f_unmix, f_ndwi)` — a pixel is only as wet as both allow.

### Results, 2024-06-22

| class | n | f_mix | f_gravel | f_dark | s2_frac | resid |
|---|---|---|---|---|---|---|
| river | 6,110 | **0.874** | 0.016 | 0.002 | 0.911 | 0.0103 |
| land | 37,119 | **0.000** | 0.000 | 0.000 | 0.047 | 0.0826 |

`f_mix` ≥ 0.90 covers 0.89% of the box (unmix alone 0.95%, NDWI alone 3.55%,
NIR alone 1.96%) against a river of ~1.5%.

| flag | v1 | v2 | **v3** |
|---|---|---|---|
| `river_dry` | 658 | 68 | **81** (1.3%) |
| `land_wet` | 629 | 1,992 | **345** (0.9%) |
| `off_line` | — | 1,826 | **10** |

### Still broken: the partial range

The NDWI calibration curve is **degenerate** — flat at −0.69 from fraction 0.04
to 0.79, then jumping to +0.15. It is calibrated against `s2_frac`, and Sentinel
carries only green, red, NIR and a SWIR reconstructed from MNDWI at 20 m. Four
endmembers on that basis made the 10 m abundances unstable (land came out at
0.61 water), so Sentinel stays at three — and therefore still has the
dark-vegetation fault that Landsat no longer has. Its mid-range is polluted, so
the curve cannot be fitted there, so `f_ndwi` acts as a near-binary gate.

Consequence: **binary water/land is now good; partial water is not trustworthy.**
Polygon 102's braid threads read median 0.000 (p95 0.640) where v2 gave 0.187,
so narrow channels are probably now *under*-called.

**Concrete fix:** fetch Sentinel B11 and B12 as real reflectance (one GEE call,
extend `add_sentinel.py --extras`). That gives Sentinel five independent bands,
enough for a fourth endmember, which should unpollute the mid-range and let the
NDWI curve calibrate properly.

### Retraction

The A/B/C mixed-class experiment run earlier on 2026-10-27 is **void**. It
defined `mixed` from the gravel-blind v2 `s2_frac`, and 8,426 of its 9,723
"partial" pixels were drawn as land — largely the gravel islands. Variant B's
apparent win (water-fraction MAE 0.082 vs 0.105) was substantially "learn to
recognise gravel", not "learn partial water". Rerun only once the partial range
is trustworthy. **The mixed-class decision is not yet answerable.**

---

## v4 — the NDWI curve fixed, and the A/B/C experiment rerun (2026-10-27, later)

**Labels re-extracted first.** `samples_2024-06-22.csv` was a day and a half
stale: polygon 102 (the class-9 braid box) had been deleted and more land
polygons added. Now 44,056 rows — river 6,110 unchanged, land 37,946 (+827).
Note `poly_id` is a *positional* counter over kept polygons, so deleting one
renumbers every polygon after it. Only record 101 was deleted, which is after
39, so `GRAVEL_POLYS = {39, 20}` survived — by luck. A guard now checks the
gravel endmember really looks like a dry bright bar and aborts if not.

**What actually fixed the degenerate NDWI curve: the calibration reference, not
Sentinel's band count.** B11 and B12 were fetched as real reflectance (so
Sentinel has five independent bands, same as Landsat) and it made no difference
— a fourth endmember still destabilised it, because dark forest has no polygon
at 10 m and must be chosen by rule. Every rule landed on *mildly*-dark forest:
0.067 from the vegetation endmember where water is 0.242 away, condition number
52, land coming out at 0.36–0.61 water.

The real fault was calibrating the NDWI curve against `s2_frac` — the unmix's
own mid-range errors went in as truth. Calibrating against the **threshold**
fraction (share of 10 m cells with NDWI > 0) instead fixes it: that definition
misses narrow channels but gravel sits at NDWI −0.106 and dark forest well below,
so neither pollutes it. Biased in *coverage*, not in the *relationship* — and
the relationship is all that is being fitted.

| water fraction | 0.04 | 0.12 | 0.21 | 0.38 | 0.54 | 0.79 | 0.96 |
|---|---|---|---|---|---|---|---|
| median Landsat NDWI | −0.71 | −0.21 | −0.20 | −0.13 | −0.07 | −0.02 | +0.13 |

Monotonic and spread, against flat at −0.68 before. `f_ndwi` is now a graded
fraction: 50.6 % of the box strictly between 0 and 1.

### Results

| class | n | f_mix | f_gravel | s2_frac |
|---|---|---|---|---|
| river | 6,110 | 0.847 | 0.016 | 0.919 |
| land | 37,946 | 0.000 | 0.000 | 0.022 |

| flag | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| `river_dry` | 658 | 68 | 81 | **95** (1.6 %) |
| `land_wet` | 629 | 1,992 | 345 | **36 (0.1 %)** |
| `off_line` | — | 1,826 | 10 | **0** |

The gravel islands are fully resolved, and the newly added land polygons
produced no new false water.

### A/B/C rerun — INCONCLUSIVE, and that is the finding

A = current 2 classes; B = 3 classes with `mixed` carved out by Sentinel;
C = 2 classes with the partial pixels withheld. Identical features, folds, seed
and RF settings; `StratifiedGroupKFold` by polygon.

| variant | water recall | MAE vs `s2_frac` | MAE vs `s2_thresh` | implied water |
|---|---|---|---|---|
| A current | 84.3 % | 0.1556 | **0.0384** | 4.98 km² |
| B mixed | 80.5 % | **0.1008** | 0.1899 | 10.86 km² |
| C withheld | 84.1 % | 0.1543 | **0.0352** | 4.83 km² |

**The two truths pick opposite winners.** Against `s2_frac`, B wins; against
`s2_thresh`, A and C win by a wide margin. The cause is visible in the last
column: `s2_frac` says the labelled area holds **9.55 km²** of water,
`s2_thresh` says **5.14 km²** — a factor of 1.9 apart. We do not know the true
water area to within 2x, so the comparison cannot be adjudicated.

`s2_frac` is the suspect one: it places **40.2 %** of labelled pixels in the
partial range (17,722, of which 15,050 were drawn land), up from 22.5 % under
v2, while its median on land is 0.022. That shape — most land near zero, a large
minority smeared across 0.1–0.9 — is the unfixed dark-forest fault at 10 m.

Note too that B's water recall is *worse* than A's (80.5 % vs 84.3 %), so there
is no reading of this where B is clearly better.

**Verdict: the mixed class is not justified by this evidence, and the question
stays open.** It is not a negative result about the class; it is a negative
result about our partial-water reference.

### The one thing that would settle it

Sentinel needs a **hand-drawn dark-forest polygon** so its fourth endmember has
a real spectrum instead of a threshold rule. The Landsat equivalent took one
confirmation and fixed the Landsat side completely. Target: the cluster at
64.5127, −149.1395 (4,541 Landsat pixels, confirmed dark dense forest by eye).
Draw it, note its `poly_id`, and it plugs in exactly as the gravel polygons did.

Until then `s2_frac`'s mid-range is unreliable, so no experiment that uses it as
truth can decide the mixed-class question.

---

## v5 — hand-drawn dark-forest polygons: a measured dead end (2026-10-27)

Polygons 104 and 105 were drawn over the confirmed dark forest and wired in as
the dark-forest endmember for both sensors (`DARK_VEG_POLYS`), replacing a
hardcoded spectrum and a threshold rule. Samples re-extracted: 45,385 rows.

**Landsat: fine.** Dark forest green 0.026 / NIR 0.213 / SWIR1 0.093. `river_dry`
97, `land_wet` 34 of 39,275, `off_line` 0.

**Sentinel: worse, and the reason is structural.**

| dark-forest source | distance from vegetation | cond | s2_frac land median |
|---|---|---|---|
| threshold rule | 0.067 | 52 | 0.022 |
| **drawn polygons 104+105** | **0.032** | **1142** | **0.180** |

Water sits 0.242 from vegetation. So a *better* polygon moved the endmember
*closer* to vegetation and the system became more ill-conditioned, not less. At
10 m with green, red, NIR and SWIR, dark forest and ordinary vegetation are not
separable enough to be distinct endmembers — the conclusion holds regardless of
how carefully the polygon is drawn. Sentinel is therefore pinned at three
endmembers, and `s2_frac`'s mid-range cannot be cleaned this way.

**Consequence for the mixed class:** Sentinel cannot serve as the partial-water
reference, so the A/B/C experiment cannot be made decidable by this route. The
remaining path inverts the order — hand-label some `mixed` polygons along banks
and braid threads, then test whether the model learns them, rather than defining
the class from Sentinel and measuring against it.

Reverted to Sentinel-3 endmembers: river `s2_frac` 0.919, land 0.029.

---

## v6 — context features for narrow channels: no gain (2026-11-02)

**Question.** River errors on 2024-06-22 concentrate in narrow channels: the worst
5 of 49 river polygons (83, 101, 89, 91, 92) hold 68 % of river errors, and river
polygons < 20 px have recall 0.21 against 0.86 for larger ones. Missed river pixels
have median NDWI −0.39 (correct ones +0.13), i.e. they look like land per pixel.
Can neighbourhood context make them learnable?

**Test.** Same samples (45,385, labels unchanged since 2026-10-27), folds, seed and RF
settings as `train_rf.py`; only the feature set varies. Context features, all from the
Landsat image itself: NDWI max over 3×3 and 5×5, NDWI and NIR relative to their 7×7
mean, NDWI std over 5×5, distance to clear water (NDWI > 0.1, capped at 50 px). JRC:
occurrence at the pixel and its 3×3 max.

| variant | river recall | narrow polys | river polys < 20 px | land → river | scene river area |
|---|---|---|---|---|---|
| A baseline (10 features) | 0.845 | 0.666 | 0.210 | 382 | 11.94 km² |
| B + image context | 0.851 | 0.684 | 0.215 | 373 | 12.50 km² |
| C + JRC | 0.839 | 0.653 | 0.210 | 387 | 11.60 km² |
| D context + JRC | 0.847 | 0.670 | 0.221 | 376 | 12.26 km² |

Polygon 89 stays at 0.00 in every variant.

**Verdict: no meaningful gain (< 1 point, within noise).** Context helps only when
something nearby is visibly wet; in these channels the whole thread reads dry at 30 m,
so the neighbours do too. The water is real (visible on Sentinel-2 at 10 m) but below
what Landsat resolves. No feature built from Landsat pixels fixes this, and a 2D CNN is
unlikely to either, since the channel shape is not visible to learn from.

**Consequences.**
- June river recall of ~85 % is close to the ceiling for Landsat under the
  "any wet Sentinel pixel → water" labelling rule. Report it split: ~0.86 on visible
  channels, ~0.21 on narrow ones. June overall is 97.1 %.
- More June river labelling, a separate narrow/mixed class, or context features
  will not move it. Effort is better spent on October snow/ice and on the
  out-of-sample test on a teammate's dates.
- Applying the new labelling rule will add more land-looking pixels to river, so
  measured river recall is expected to *drop* while the labels become more correct.

Script (scratch, not in the repo): `context_test.py`.

## v7 — QA mask dropped, anchoring, cross-date transfer (2026-11-02)

**QA_PIXEL mask dropped.** CFMask flags silty summer water and river ice as cloud or
shadow: 77.7 % of the main channel on 2023-07-24, yet those pixels sit at river
temperature (~298 K), not cold like cloud tops, and no date has pixels > 25 K below the
scene median. An early transfer score of 98.9 / 96.3 % river recall was inflated because
masked pixels were excluded from scoring; with them counted it was 22 / 25 %. Spotted by
eye in QGIS (water missing on 2023-07-24). `train_rf.py` now predicts every finite pixel;
scenes are checked for cloud by eye. Training still keeps QA-flagged rows (`--drop-qa`
is opt-in).

**Anchoring.** Train on 2024-06-22, predict the other summer dates. Raw features do not
transfer: on 2025-07-04 they gave 88 km² of false river (real ≈ 13 km²). Fix: on every
date, subtract each feature's median over dry land (JRC occurrence = 0), in training and
in prediction. Dry-land medians:

| date | sun elev | B2 | B3 | B6 | thermal | NDWI |
|---|---|---|---|---|---|---|
| 2024-06-22 (train) | 48.6° | 0.015 | 0.035 | 0.119 | 300.5 K | −0.708 |
| 2023-07-24 | 44.9° | 0.021 | 0.045 | 0.126 | 306.4 K | −0.666 |
| 2025-07-04 | 47.9° | 0.024 | 0.054 | 0.122 | 302.0 K | −0.605 |
| 2026-08-01 | 43.0° | 0.020 | 0.040 | 0.122 | 300.1 K | −0.687 |

**The cause is not sun angle:** 2025-07-04 has nearly June's sun but the largest offset,
and surface reflectance already normalises flat-terrain illumination. Blue/green rise
40–60 % while SWIR stays flat, which points to residual haze/smoke (wildfire season). Thermal is absolute and
follows the weather (+6 K on 2023-07-24). Phenology adds a little. Limits: additive offset
only, one offset for the whole box, assumes dry land itself is stable.

**Transfer scoring (no ground truth on these dates):** land recall on the June land
polygons; "main-channel recall" on June river pixels with JRC ≥ 80; share of JRC ≥ 90
pixels called river. Result, QA mask dropped, anchored, all 10 features:

| date | land recall | main-channel recall | river area |
|---|---|---|---|
| 2024-06-22 (own date) | 0.999 | 0.995 | 12.43 km² |
| 2023-07-24 | 0.960 | 0.972 | 14.10 km² |
| 2025-07-04 (held-out test date, stays unlabelled) | 0.960 | 0.959 | 13.19 km² |
| 2026-08-01 | 0.978 | 0.619 | 9.80 km² |

2026-08-01 is unresolved (silty water or channel migration — to inspect).
Outputs: `patches/nenana_<date>/rf_transfer/<date>_rf_from_20240622_anchored_{class,confidence}.tif`.
The non-anchored `<date>_rf_from_20240622_{class,confidence}.tif` there are stale
(made with the QA mask).

## v8 — labelling wide visible channels (2026-11-02 / 03)

**Round 1 (v2 labels):** +15 polygons (106–120, 10,957 px): 14 land (incl. 5 small
in-channel islands) and one river, 117 (916 px, a wide visible channel). The previous
model already called the land polygons land at ~100 % confidence (redundant); it missed
60 % of 117. **Round 2 (v3 labels):** +4 wide-channel river polygons 121–124 (356 px),
117 trimmed to 906 px. Now 56,688 px / 124 polygons. No polygon IDs shifted.

Compared on identical pixels and folds (the original 103 polygons, original fold split):

| labels | river recall | river precision | land recall | June river area |
|---|---|---|---|---|
| v1 original | 0.845 | 0.931 | 0.990 | 12.43 km² |
| v2 +117 + land | 0.862 | 0.902 | 0.985 | 13.74 km² |
| v3 +121–124 | 0.863 | 0.895 | 0.984 | 15.03 km² |

**Main result — leave-one-polygon-out recall of each wide channel:**

| polygon | v1 | v2 | v3 |
|---|---|---|---|
| 117 | 0.41 | 0.39 | 0.42 |
| 121 | 0.06 | 0.09 | **0.88** |
| 122 | 0.09 | 0.14 | **0.75** |
| 123 | 0.63 | 0.72 | **0.75** |
| 124 | 0.03 | 0.20 | 0.24 |

Several examples of a type teach each other; one example does not. This corrects v6's
"more June river labelling will not move it" — that holds for *narrow* sub-pixel
channels, not for *wide* visible ones that were simply never labelled.

`train_rf.py` CV on all v3 polygons: river recall 0.807, precision 0.860, land 0.980
(land → river errors 537 → 968). Worst land polygon 107 (bright hot bare ground, B6
0.195, 310 K, JRC 0, centred 64.288 N −149.191 E): 18 % called river when held out.
The extra 2.91 km² of June river: 78 % where JRC > 0, 49 % JRC ≥ 50, 59 % within 2 px of
the v1 river (bank widening, consistent with the any-wet-S2-pixel rule); ~1/5 likely false.

Transfer, same pixels, v1 → v3:

| date | main-channel recall | JRC ≥ 90 called river | area |
|---|---|---|---|
| 2023-07-24 | 0.972 → 0.972 | 76.9 → 80.0 % | 14.10 → 14.93 km² |
| 2025-07-04 | 0.959 → 0.951 | 66.4 → 71.4 % | 13.19 → 14.05 km² |
| 2026-08-01 | 0.619 → 0.637 | 42.5 → 48.9 % | 9.80 → 11.47 km² |

**Trade-off:** more channel found, river precision falling. **Next labels: land** —
bare/bright ground like 107, and islands where the map now calls gravel river.
Lesson for labelling: draw where the model is wrong; check a new polygon against the
current map before drawing more of the same kind.

**Accuracy is measured three ways** (state the limits in the write-up): (1) grouped CV
on hand labels; (2) transfer scored on stable proxies (June land, JRC ≥ 80 river);
(3) sanity checks (area, SWORD width, JRC agreement). All use pixels chosen by the
labeller, so they are likely optimistic; the unbiased number would be the blind
200-point audit (`rf/2024-06-22_rf_audit.geojson`), not yet done.

**Viewing layers for the out-of-sample demo, 2023-07-24:** `_432`, `_543`, `_ndwi`,
`_mndwi` (Landsat) and `_s2_truecolour/_swir/_ndwi/_843/_mndwi/_swir_refl_10m` from the
same-day Sentinel-2 scene (21:47 UTC, +0.5 h, 2.3 % cloud; `add_sentinel.py --max-days 0`,
because the default picked 07-23 for 0.3 % less cloud).

Experiment scripts and the v1/v2 sample snapshots: `qc/experiments/`.

**Figures (2026-11-03):** `qc/figures/01`–`06b` + `figure_numbers.txt`, regenerated by
`qc/experiments/make_figures.py` (reads labels read-only, writes only PNGs). Raw-vs-anchored
transfer with v3 labels (area / main-channel recall / land recall): 07-24 raw 21.4 km² /
0.951 / 0.936 vs anchored 14.9 / 0.940 / 0.964; 07-04 raw 90.8 / 0.970 / 0.862 vs 14.0 /
0.897 / 0.966; 08-01 raw 13.7 / 0.724 / 0.970 vs 11.5 / 0.638 / 0.978. Raw scores higher
main-channel recall only by over-calling river; on 2026-08-01 raw is not clearly worse — an
honest caveat. Zoom on 2023-07-24 (06b) shows the model missing braid threads that are
visible on Sentinel-2 — more wide-channel labels apply. `classify/accuracy_2024-06-22_2023-10-04.png`
(2026-10-26) is outdated.

**Feature importance (2026-11-03):** `figures/07_feature_importance.png` (script
`experiments/make_importance_figure.py`, reads `rf/<date>_rf_cv.json`, no retraining).
June (v3): NDSI 27 %, SWIR1 20 %, NDWI 13 %, NIR 10 %, thermal 10 %, NDVI 8 %, SWIR2 5 %,
visible bands 2 % each. October: NDWI 25 %, NDVI 19 %, green 14 %, red 10 %, blue 8 %, NIR 7 %,
thermal 6 %, NDSI 5 %, SWIR 3 % each. The two dates rely on different inputs (summer: SWIR
separates water from land; freeze-up: visible/NDWI for ice and snow), so a single model across
seasons would likely transfer poorly. Impurity-based importance; correlated inputs share credit.
