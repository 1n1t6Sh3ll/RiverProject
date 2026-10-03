# Narrow rivers - sub-pixel river-ice workstream

Classifying river ice on **narrow Alaskan rivers**, where a VIIRS 375 m pixel never
sees the river alone: it always mixes channel, bank and floodplain. The plan is a
2D (patch-based) model that estimates, for every VIIRS cell, **what fraction of it
is river, ice, snow and land**, trained on fractions measured from a Landsat 30 m scene.

Current study area: the **Nenana River** (median SWORD node width 74 m; 931 of its
993 nodes are narrower than a single VIIRS pixel).

Nothing in the existing pipeline is modified - these scripts import
`viirs_training_loader`, `AWSExtract` and `stitch_h5` from `Matus/`.

---

## Status

|                      |                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------- |
| Study box            | lon −149.33766 … −149.06919, lat 64.09373 … 64.53537 (13 × 49 km, 321 SWORD nodes) |
| Scene pairs prepared | 5 dates, see below                                                                 |

### Scene pairs

| Date       | Season    | Landsat      | VIIRS pass   | Gap     | VZA  | Geoloc offset |
| ---------- | --------- | ------------ | ------------ | ------- | ---- | ------------- |
| 2023-07-24 | summer    | LC09 069/015 | J02 t2219233 | +69 min | 12°  | 50 m          |
| 2024-06-22 | summer    | LC09 071/015 | J01 t2208041 | +44 min | 0.9° | 71 m          |
| 2024-04-21 | breakup   | LC09 069/015 | J02 t2220285 | +69 min | 13°  | 0 m           |
| 2026-03-26 | ice       | LC09 069/015 | J01 t2133484 | +21 min | 29°  | 50 m          |
| 2023-10-04 | freeze-up | LC08 069/015 | NPP t2156572 | +44 min | 10°  | 50 m          |

`patches/patch_tracking.csv` holds the same table plus status, and is the file to
keep in step with the shared document.

---

## Layout

```
narrow_rivers/
  README.md                       this file
  fetch_sword_river.py            SWORD reaches + nodes for any river, from GEE
  get_landsat_footprint.py        Landsat footprint + how much of each river it covers
  data/
    sword_nodes_nenana.csv        993 nodes, 200 m spacing      <- used by the scripts
    sword_reaches_nenana.csv      14 reaches
    sword_*_75_300m*.csv/geojson  Killik + Sagavanirktok (earlier study area)
    viirs/                        downloaded granules (.h5) — not in git
  patches/                        THE CURRENT PIPELINE
    patch_config.py               box, scenes, grids, class codes — single source of truth
    build_patches.py              per-date QGIS folder: Landsat rasters + styles + labels
    add_viirs.py                  geolocation check + VIIRS on the patch grid
    make_viirs_wide.py            wide-area VIIRS for judging cloud
    make_fractions.py             polygons -> per-VIIRS-cell class fractions
    patch_tracking.csv            one row per date: scene, pass, gap, VZA, status
    nenana_<yyyymmdd>/            one folder per date, shared via OneDrive
  scene_pairing/
    shortlist_overpasses.py       which VIIRS passes see the river, and how well
  pixel_extraction/               earlier Sagavanirktok work (see "History")
    extract_river_pixels.py       1D candidate-pixel extractor along a SWORD corridor
    check_geolocation.py          standalone geolocation check
```

---

## Grids and classes

Everything is **EPSG:4326 on the main pipeline's Alaska grid**, so our cells coincide
with the existing dataset's cells.

```
Alaska grid  lat 54..72, lon -171..-129, 12432 x 5328 cells
```

At 64° N this means cells are **rectangular on the ground**: about 13 × 30 m for
Landsat and 163 × 374 m for VIIRS. That is accepted.

**Class codes:** `1` river (open water), `2` ice, `3` snow, `4` land, `9` no data.

---

## Workflow

All commands run from `Matus/`.

**1. River geometry** (once per river)

```
.venv\Scripts\python.exe narrow_rivers\fetch_sword_river.py --river "Nenana River"
```

**2. Pick a Landsat scene.** For professor's cloud free Landsat scenes. To check how much of the river a scene covers:

```
.venv\Scripts\python.exe narrow_rivers\get_landsat_footprint.py --landsat <PRODUCT_ID>
```

**3. Build the patch folder** (Landsat + water mask + styles + empty label layer).
Add the date to `SCENES` in `patch_config.py` first.

```
.venv\Scripts\python.exe narrow_rivers\patches\build_patches.py --dates 2024-06-22
```

**4. Find the VIIRS passes** that see the river, ranked by view angle and time gap

```
.venv\Scripts\python.exe narrow_rivers\scene_pairing\shortlist_overpasses.py ^
    --landsat <PRODUCT_ID> --river "Nenana River" ^
    --nodes narrow_rivers\data\sword_nodes_nenana.csv
```

Coverage in that output is over the **whole river**; a partly covering granule can
still cover the box.

**5. Download and stitch the chosen granule**

```python
import AWSExtract, stitch_h5
AWSExtract.BUCKET_NAME = 'noaa-nesdis-n20-pds'       # npp -> snpp, j01 -> n20, j02 -> n21
AWSExtract.download_viirs_data('2024-06-22', '_t2208041',
                               download_dir='narrow_rivers/data/viirs')
stitch_h5.stitch_svi_files('narrow_rivers/data/viirs', satellite='j01', t_code='t2208041')
```

**6. Add VIIRS to the patch** (geolocation check, then resample to the 375 m grid).
Record the pass in `PASSES` in `add_viirs.py` first.

```
.venv\Scripts\python.exe narrow_rivers\patches\add_viirs.py --dates 2024-06-22
```

Expect an offset well under one VIIRS pixel; stop and investigate if it approaches
375 m.

**7. Wide-area VIIRS for cloud checking**

```
.venv\Scripts\python.exe narrow_rivers\patches\make_viirs_wide.py --dates 2024-06-22
```

**8. Digitise in QGIS.** Drag the patch folder's `.tif` files and the empty label
layer into QGIS; styles load automatically. Draw polygons, set `class_id`, and keep
classes from overlapping (Project → Snapping Options → Avoid overlap on active
layer). A VIIRS cell is only usable when it is labelled edge to edge.

---

## What we learned the hard way

- **The NOAA AWS VIIRS archive starts ~2022-08 (S-NPP) and ~2023-02 (NOAA-21)**,
  and NOAA-20 has gaps before 2023. Landsat dates before February 2023 generally
  cannot be paired from that source. NASA's VNP02/VNP03 go back to 2012 if needed.
- **View angle matters.** Pixel spacing measured from GITCO: ~400 × 380 m near
  nadir, ~430 × 470 m at 40°. Prefer near-nadir passes.

---

## Requirements

- The `Matus/` virtual environment (`.venv`, Python 3.13) and `requirements.txt`.
  No extra packages.
- **Earth Engine** access to the `noaa-river-ice` project (`earthengine authenticate`).
- **JRC water masks** `alaska_occ_375m.tif` / `alaska_sea_375m.tif` in `Matus/`,
  and the 30 m tiles in `Source/EarthEngineExports/` for the patch water mask.
- No AWS credentials: the NOAA VIIRS buckets are public.
- Not in git: VIIRS granules (`data/viirs/*.h5`, ~230 MB each) and the patch rasters
  (`*.tif`). Both are reproducible from the scripts; the patch folders are shared
  through OneDrive.
