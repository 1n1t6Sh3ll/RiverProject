# Narrow rivers — sub-pixel river-ice workstream

New workstream, separate from the main pipeline in `Matus/`: classifying river ice on
**narrow Alaskan rivers (75–300 m wide)**, where every VIIRS 375 m pixel is a mix of river
and bank. The plan is a 2D (patch-based) model trained on **per-pixel class fractions**
computed from a near-simultaneous Landsat 30 m scene (~156 Landsat pixels per VIIRS pixel).

This first stage checks whether a river is usable at all. Current scope: the
**Sagavanirktok River** only.

Nothing in the existing pipeline was modified — the scripts here import
`viirs_training_loader`, `AWSExtract`, `stitch_h5` and the JRC masks from `Matus/`.

---

## Contents

```
narrow_rivers/
  README.md
  get_landsat_footprint.py        # Landsat footprint + how much of each SWORD river it covers
  data/
    sword_nodes_75_300m_csv.csv   # SWORD nodes, 200 m spacing (1586 nodes)
    sword_reaches_75_300m_csv.csv # SWORD reaches, ~10 km each (26 reaches)
    sword_nodes_killik_sagavanirktok.geojson
    sword_reaches_75_300m.geojson
    viirs/                        # VIIRS granules (.h5) — not in git, see "Re-downloading VIIRS"
  pixel_extraction/
    extract_river_pixels.py       # corridor-based candidate pixel extractor
    check_geolocation.py          # VIIRS geolocation check against Landsat
    output/<date>_<satellite>/    # results per scene pair
```

## Data

SWORD (SWOT River Database) reaches and nodes for rivers 75–300 m wide, exported via GEE:
[Google Drive folder](https://drive.google.com/drive/folders/1d39DFKyq8OF1tI5SkhrEULhLUbCG47qu?usp=drive_link)

| River         | Reaches | Nodes (CSV) | Lat         | Lon                |
| ------------- | ------- | ----------- | ----------- | ------------------ |
| Sagavanirktok | 19      | 1131        | 68.71–70.21 | −148.86 to −147.98 |
| Killik        | 7       | 455         | 68.14–68.98 | −154.17 to −153.41 |

Note: the node GeoJSON is a superset of the CSV — it has all 1586 CSV nodes plus 712 more
(1432 Sagavanirktok / 866 Killik in total). The extra nodes all belong to reaches outside
the 75–300 m width selection (median width 63 m). The scripts use the CSV.

The rivers are ~5° of longitude apart and treated as separate corridors. The Killik lies
outside Landsat path/row 073/011 and is on hold.

## Scene pairs so far

All with Landsat path/row **073/011**, which covers all 19 Sagavanirktok reaches.

| Scene key        | Landsat                                    | VIIRS granule      | Time vs Landsat | VZA over candidates | Geolocation offset | Auto-labels (25 px)                                                      |
| ---------------- | ------------------------------------------ | ------------------ | --------------- | ------------------- | ------------------ | ------------------------------------------------------------------------ |
| `2024-06-12-npp` | `LC08_L2SP_073011_20240612_20240628_02_T1` | NPP `t2132308`     | −2 min          | 2–10°               | 42 m               | 19 ice-free/snow-free, 6 ice-free/snow                                   |
| `2024-06-12-j02` | same                                       | NOAA-21 `t2247556` | +73 min         | 39–43°              | 60 m               | same as above                                                            |
| `2024-05-27-j01` | `LC08_L2SP_073011_20240527_20240611_02_T1` | NOAA-20 `t2157530` | +24 min         | 8–15°               | 67 m               | 12 ice-covered/snow, 6 ice-free/snow, 5 ice-free/snow-free, 2 no Landsat |

- **Auto-labels are not verified yet.** They come from the same ST_B10 < 273 K / NDSI > 0.4
  rule as the main pipeline and need the usual GEE check. North of ~69.6° N the 27 May
  scene is partly cloudy, so its northern "ice" labels are suspect.
- 12 June is after breakup (river ice-free); 27 May is during breakup.
- The two 12 June runs use the same 25 locations on purpose: same pixels, different
  viewing angle. Band values differ by up to 0.15 reflectance / 8 K between them.

## Findings

1. **The whole Sagavanirktok fits in one Landsat scene** (073/011, all 1131 nodes).
2. **Landsat's cloud mask (QA_PIXEL / CFMask) flags river ice and snow as cloud.** On 27 May
   it reports ~30 % cloud over the corridor, but visually much of that is ice. Scene-wide
   cloud filters (e.g. EarthExplorer's) hid this scene entirely (44.7 % scene cloud).
3. **One VIIRS granule was geolocated ~3 km off.** NPP `t2132184` on 2024-05-27 was shifted
   by a rigid 3 km (1350 m W, 2700 m N — identical in every quarter of the test area).
   NOAA-20 and NOAA-21 granules from the same day, and all 12 June granules, are within
   ~0–70 m. It is not related to the `oeac`/`oebc` code in the file names.
   `check_geolocation.py` catches this in a couple of minutes, so it is now run on every pair.
   **The main pipeline never checks geolocation against Landsat**, so a granule like this
   could exist in the current dataset unnoticed.
4. **View angle matters.** Measured VIIRS pixel spacing over the river is ~400 × 380 m near
   nadir (VZA 8–10°) but ~430 × 470 m at VZA 40° (along-scan × along-track, from GITCO);
   prefer near-nadir passes (VZA < ~20–25°).
5. **The degree grid cell is not square on the ground.** The shared grid uses the same
   step (0.003378°) in latitude and longitude, so at 69.5° N a cell is ~375 m × ~130 m.
   Fine for placing values, but Landsat fractions have to be computed on an equal-area
   grid which is the reason for the planned EPSG:3338 reprojection (next section).

## Proposed change: reproject to EPSG:3338 (Alaska Albers)

**The problem with the current grid.** The shared grid is EPSG:4326 (lat/lon) with the same
step, 0.003378° (= 375 m of latitude), in both directions. A degree of longitude shrinks
with latitude, so the cells are only 375 m tall; their width depends on where they are:

| Latitude                | Cell size on the ground (N–S × E–W) | Landsat 30 m pixels per cell |
| ----------------------- | ----------------------------------- | ---------------------------- |
| 55° N                   | 375 m × 216 m                       | ~90                          |
| 65° N                   | 375 m × 159 m                       | ~66                          |
| 69.5° N (Sagavanirktok) | 375 m × 132 m                       | ~55                          |
| 72° N                   | 375 m × 116 m                       | ~48                          |

Consequences:

- **Fractions would be computed over the wrong area.** A VIIRS pixel covers ~375 × 375 m
  (~156 Landsat pixels). Counting Landsat pixels inside a 375 × 132 m cell gives the class mix
  of about a third of the ground VIIRS actually measured.
- **VIIRS values are duplicated.** Nearest-neighbour resampling copies each ~375 m VIIRS pixel
  into ~3 neighbouring cells east–west at 69.5° N. The grid looks like 375 m but carries
  coarser information.
- **Patches are not comparable.** A 2D model sees patches of N × N cells. On this grid a patch
  is ~3× taller than wide at 69.5° N, and its ground size changes with latitude, so the same
  pattern (a 150 m channel) looks different depending on where it is. Convolutions assume
  square, evenly spaced pixels.

**Why EPSG:3338.** NAD83 / Alaska Albers is an equal-area projection designed for Alaska
(standard parallels 55° N and 65° N) and a common choice for statewide Alaska datasets.

- Units are metres, and every 375 m cell covers the same ground area anywhere in Alaska.
  Shape stays close to square: ~370 × 380 m at the Sagavanirktok (1.4 % off), ~3 % off at
  72° N, exact along 55° N and 65° N.
- One grid covers the whole state. UTM would split Alaska across several zones with seams
  between them.
- Landsat 30 m nests into it cleanly: 375 / 30 = 12.5 Landsat pixels per side, ~156 per cell.

**Proposed approach** (open for discussion):

1. **VIIRS:** resample directly from the swath to the EPSG:3338 grid (no lat/lon step
   in between), nearest neighbour for all bands, including the angle layers.
2. **Landsat:** export the 6 native bands + QA_PIXEL from GEE for the chosen scenes only,
   clipped to the corridor. NDSI/NDWI are derived locally. Export via Google Drive or Cloud
   Storage is still open.
3. **Fractions:** area-weighted overlap between each Landsat pixel and each VIIRS cell. Since
   375 / 30 = 12.5, counting Landsat pixel centres would give 144, 156 or 169 pixels per
   cell depending on alignment; area weighting avoids that.
4. **Order:** VIIRS half first, tested on the clear part of 27 May and on 12 June; then Landsat.

**What it does not fix.** Reprojection changes how pixels are placed, not what VIIRS
measured: off-nadir pixels stay larger than 375 m (finding 4), and a mislocated granule stays
mislocated (finding 3). Those are handled by choosing near-nadir passes and by
`check_geolocation.py`.

**Scope.** This applies to the narrow-river workstream. The main pipeline's dataset stays
on its current grid unless the team decides otherwise.

## Requirements

- The `Matus/` virtual environment (`.venv`, Python 3.13) with `Matus/requirements.txt`.
  No extra packages: everything used here (`numpy`, `h5py`, `boto3`, `rasterio`,
  `pyresample`, `pykdtree`, `pyproj`, `matplotlib`, `earthengine-api`) is already pinned
  there.
- **Google Earth Engine** access to the `noaa-river-ice` project, authenticated once with
  `earthengine authenticate`. Used for Landsat footprints, sampling and the geolocation check.
- **JRC water masks** `alaska_occ_375m.tif` and `alaska_sea_375m.tif` in `Matus/` (same files
  the main extractor uses).
- **No AWS credentials** — the NOAA VIIRS buckets are public and read anonymously.

## How to run

All commands from `Matus/`.

**1. Landsat footprint and SWORD coverage**

```
.venv\Scripts\python.exe narrow_rivers\get_landsat_footprint.py --landsat LC08_L2SP_073011_20240527_20240611_02_T1
```

**2. Download + stitch a VIIRS granule** (bucket per satellite: `npp` → `noaa-nesdis-snpp-pds`,
`j01` → `noaa-nesdis-n20-pds`, `j02` → `noaa-nesdis-n21-pds`)

```python
import AWSExtract, stitch_h5
AWSExtract.BUCKET_NAME = 'noaa-nesdis-n20-pds'
AWSExtract.download_viirs_data('2024-05-27', '_t2157530', download_dir='narrow_rivers/data/viirs')
stitch_h5.stitch_svi_files('narrow_rivers/data/viirs', satellite='j01', t_code='t2157530')
```

**3. Register the pair** in `SCENES` in `pixel_extraction/extract_river_pixels.py`
(key = `<date>-<satellite>`), and add a clear test region for it in `TEST_REGIONS` in
`check_geolocation.py`.

**4. Check geolocation** — expect a best shift near 0 m:

```
.venv\Scripts\python.exe narrow_rivers\pixel_extraction\check_geolocation.py --scene 2024-05-27-j01
```

**5. Extract candidate pixels**

```
.venv\Scripts\python.exe narrow_rivers\pixel_extraction\extract_river_pixels.py --scene 2024-05-27-j01
```

Outputs go to `pixel_extraction/output/<date>_<satellite>/`:

- `training_candidates_<date>.csv` — 25 candidates with VIIRS I1–I5 + angles, Landsat 30 m
  values, SWORD node/reach, and an auto-label
- `viirs_vs_watermask_<date>.png` — VIIRS false colour vs JRC water fraction, with the
  SWORD centerline and the ±2 km corridor
- `gee_inspector_<date>.js` — paste into the GEE Code Editor to verify the pixels
  (true colour, NIR, SWIR, thermal, NDSI, Sentinel-2, cloud mask)
- `geolocation/` — report + correlation plot from `check_geolocation.py`

**How candidates are picked:** same JRC filter as the main extractor (occurrence
0.05–0.90, seasonality < 1), restricted to cells within 2 km of a Sagavanirktok SWORD
node, then 25 spread evenly along the river.

## Re-downloading VIIRS

The `.h5` files are not in git (~230 MB per granule). Granules used:

| Scene key        | Satellite | Date       | t-code     |
| ---------------- | --------- | ---------- | ---------- |
| `2024-06-12-npp` | npp       | 2024-06-12 | `t2132308` |
| `2024-06-12-j02` | j02       | 2024-06-12 | `t2247556` |
| `2024-05-27-j01` | j01       | 2024-05-27 | `t2157530` |

## Open items / next steps

1. **GEE-verify** the 27 May candidates (SWIR layer to separate ice from cloud).
2. **Reprojection to EPSG:3338**, VIIRS half first, then Landsat. Open decision: Landsat
   export via Google Drive or Cloud Storage. The team wants local Landsat copies for the
   segmentation (corridor clip, 6 bands + QA_PIXEL, chosen scenes only).
3. **Other seasons:** a fully frozen scene (April) and freeze-up (late Sept–early Oct).
   Landsat Level-2 is not produced when the sun is > 76° from zenith: over the river, all
   364 Landsat 8 L2 scenes (2014–2025) fall in March–October (lowest sun 14.2° above the
   horizon), so November–February is out.
4. **Candidate picking:** replace the flat 2 km buffer with a per-node buffer
   (half of SWORD `max_width` + ~200 m); some current candidates are 1–1.7 km from a
   single-thread channel and are ponds or tundra, not river. For the patch-based model
   the JRC filter will be dropped, since patches need pure land cells too.
