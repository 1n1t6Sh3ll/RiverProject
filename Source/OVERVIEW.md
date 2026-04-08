# Reproject & Overlay — Landsat 30 m over VIIRS 375 m

## What This Notebook Does

Takes raw VIIRS (375 m) and Landsat (30 m) satellite imagery over Alaska, reprojects both onto common UTM grids, merges all Landsat scenes into one mosaic, overlays them, exports to GeoTIFF and KMZ, and extracts co-located pixel samples for river-ice classification training.

---

## Input Data

### VIIRS (data/viirs/)
- **GITCO files** — Terrain-corrected geolocation (lat, lon, solar/satellite angles). These provide the most accurate pixel positions because they account for terrain elevation.
- **GIMGO-SVI files** — Combined I-band SDR (Science Data Record) files containing the actual measurements:
  - **I1, I2, I3** — Reflectance bands (visible/near-IR). Raw integer values scaled by a reflectance factor.
  - **I4, I5** — Brightness temperature bands (thermal IR). Raw integers scaled by a BT factor + offset.
- Files are paired by orbit timestamp (date + time extracted from filenames like `_d20240113_t1234567`).
- Fill values (63999, 65533, >65529) indicate invalid/missing pixels and are masked to NaN.

### Landsat (data/landsat/)
- Individual scene directories, each containing band GeoTIFFs (B1–B5).
- Scenes may come from Landsat 8 (LC08) or Landsat 9 (LC09).
- Each scene covers a different footprint (path/row), so multiple scenes are needed to cover the study area.
- **All scenes must have the same bands** — if one scene is missing a band, the merge will fail.

---

## Cell-by-Cell Walkthrough

### Cell 0 — Imports & Parameters
- Loads libraries: numpy, h5py, rasterio (for GeoTIFF I/O and reprojection), pyresample (for swath-to-grid), pyproj (for CRS transforms), matplotlib, tqdm.
- Sets directory paths: `data/viirs/`, `data/landsat/`, `outputs/`.
- Defines the **Alaska bounding box**: lat 54°–72°N, lon 171°–129°W. This is the full processing domain for VIIRS.
- Sets grid resolutions: VIIRS = 375 m, Landsat = 30 m.
- Landsat is NOT reprojected to the full Alaska domain at 30 m (that would be ~20 GB/band). Instead, each scene keeps its own footprint.

### Cell 1 — Helper Functions

**`choose_utm_crs(lon0, lat0)`**
- Given a centre point, picks the correct UTM zone and returns a pyproj CRS. For central Alaska (~-150° lon), this is UTM zone 5N (EPSG:32605).

**`read_viirs_geo(h5_path)`**
- Opens an HDF5 file and reads geolocation arrays from `All_Data/VIIRS-IMG-GEO-TC_All` (GITCO, terrain-corrected) or `All_Data/VIIRS-IMG-GEO_All` (GIMGO, non-TC).
- Returns a dict with: Latitude, Longitude, SatelliteAzimuthAngle, SatelliteZenithAngle, SolarAzimuthAngle, SolarZenithAngle.
- All returned as float32 numpy arrays with the same shape as the swath (rows × scan pixels).

**`read_viirs_sdr(h5_path, band)`**
- Reads the SDR data for a specific I-band (1–5) from `All_Data/VIIRS-I{band}-SDR_All`.
- For bands 1–3: reads Reflectance + ReflectanceFactors.
- For bands 4–5: reads BrightnessTemperature + BrightnessTemperatureFactors.
- Returns raw integer arrays — scaling is applied separately.

**`apply_viirs_scaling(raw, factors)`**
- Converts raw uint16 to float32.
- Masks fill values: pixels with values 63999, 65533, or >65529 are set to NaN. These are VIIRS-defined fill/error codes.
- Applies scale factor: `scaled = raw * factors[0]`. If a second factor exists, adds offset: `scaled = raw * factors[0] + factors[1]`.
- For reflectance bands: result is dimensionless (0–1 range).
- For BT bands: result is in Kelvin.

**`viirs_swath_to_grid(lat, lon, data, crs, resolution, bbox)`**
- The core reprojection function. VIIRS data comes as a **swath** (irregular grid following the satellite orbit). This function resamples it onto a **regular UTM grid**.
- Uses pyresample's `ImageContainerNearest` — nearest-neighbour resampling with a 7 km radius of influence.
- The angles (solar zenith, satellite azimuth, etc.) are NOT used to control the reprojection. They are simply additional 2D arrays that get reprojected onto the same grid, just like the band data. The reprojection is purely geometric: lat/lon → UTM (x, y).
- Returns: (2D grid array, affine transform).

**`has_viirs_sdr(h5_path)`** / **`has_viirs_geo(h5_path)`**
- Quick checks to see what's inside an HDF5 file without reading all data.

**`viirs_orbit_key(filename)`**
- Extracts a `YYYYMMDD_tHHMMSSS` key from VIIRS filenames for matching GITCO geo files with GIMGO-SVI band files from the same orbit pass.

### Cell 2 — Data Inventory
- Scans `data/viirs/` for all `.h5` files.
- Classifies each as GITCO (geolocation) or GIMGO-SVI (bands) based on filename prefix.
- Groups by orbit key so geo and band files from the same pass are paired.
- For each orbit: checks latitude range to see if it covers the Alaska bbox.
- Selects the **best orbit** — the one covering Alaska with the most I-bands available.
- Also scans Landsat directories and catalogues available scenes and bands.

### Cell 3 — Compute Target UTM Grid
- Takes the Alaska bbox corners (lat/lon) and transforms them to UTM coordinates.
- Computes grid dimensions: `nx = ceil((x_max - x_min) / 375)`, `ny = ceil((y_max - y_min) / 375)`.
- This defines the target grid that ALL VIIRS data (bands + angles) will be reprojected onto.
- Typical size: ~8000×5000 pixels at 375 m.

### Cell 4 — Reproject VIIRS → 375 m UTM
- Reads geolocation from the selected GITCO file (terrain-corrected lat/lon + angles).
- Cleans geolocation: sets out-of-range values (lat outside ±90°, lon outside ±180°) to NaN.
- **Reprojects angles first:** SolarZenithAngle, SatelliteZenithAngle, SolarAzimuthAngle, SatelliteAzimuthAngle. Each is an independent 2D array that gets nearest-neighbour resampled from swath to the UTM grid. Out-of-range angle values (>360° or <-360°) are masked.
- **Reprojects I-bands:** Reads each band from the GIMGO-SVI file, applies scaling (fill masking + scale factor), then reprojects.
- If the GIMGO-SVI file has its own geolocation (different shape from GITCO), it uses that for band reprojection. Otherwise uses the more accurate GITCO geo.
- All results are stored in `viirs_375` dict: keys like `"SolZen"`, `"SatZen"`, `"I1"`, `"I2"`, etc. All share the same grid and affine transform (`viirs_tf`).

### Cell 5 — Reproject Landsat → 30 m UTM
- **Cleans old output first:** Deletes `outputs/landsat_scene_chunks/` to avoid stale files from previous runs with different band counts.
- Loops over each Landsat scene:
  - Reads each band GeoTIFF.
  - Reprojects from the scene's native CRS to the project UTM CRS at 30 m resolution using `rasterio.warp`.
  - Saves as a per-scene multi-band GeoTIFF in `outputs/landsat_scene_chunks/`.
- **Merges all scenes** into a single mosaic using `rasterio.merge`. This stitches overlapping scenes together, preferring valid data over nodata. The result is one large Landsat image covering the union of all scene footprints.
- Saves the mosaic as `outputs/landsat_30m_mosaic.tif`.
- Creates a **downsampled version** (`mosaic_ds`) for in-notebook display to avoid memory issues. The downsample factor targets ~2000 px on the longest side.
- **Important:** All scenes must have the same number of bands. If one scene is missing a band, the merge fails with `DatasetIOShapeError`.

### Cell 6 — Landsat True-Colour RGB
- Displays the merged Landsat mosaic as a true-colour composite: B4 (red) → R, B3 (green) → G, B2 (blue) → B.
- Uses percentile stretching (2%–98%) per channel for contrast.
- Uses the downsampled mosaic to keep memory reasonable.

### Cell 7 — Per-Scene Landsat Thumbnails
- Reads each per-scene GeoTIFF back from disk and displays as an RGB thumbnail.
- Helps visually verify that individual scenes reprojected correctly before trusting the mosaic.

### Cell 8 — VIIRS I-Band Imagery
- Displays each reprojected VIIRS I-band as a separate panel.
- Reflectance bands (I1–I3) use grayscale colourmap.
- Thermal bands (I4–I5) use inferno colourmap.
- Full Alaska domain at 375 m.

### Cell 9 — Overlay: Landsat over VIIRS
- Three-panel figure:
  1. **VIIRS I1** (full Alaska, 375 m)
  2. **Landsat RGB** (mosaic footprint, 30 m downsampled)
  3. **Overlay** — VIIRS cropped to Landsat region with Landsat RGB on top at partial transparency
- The overlay shows how the 30 m Landsat detail sits within the coarser 375 m VIIRS coverage.

### Cell 10 — Summary Figure: 2×2
- Four-panel overview: VIIRS visible, VIIRS thermal, Landsat RGB, and overlay with AOI rectangle.

### Cell 10 (export) — Export GeoTIFFs
- **VIIRS:** Writes `outputs/viirs_375m_<date>.tif` — a multi-band GeoTIFF with all reprojected bands and angles. Each band is tagged with its name (I1, I2, SolZen, etc.).
- **Landsat:** Already saved as `outputs/landsat_30m_mosaic.tif` in Cell 5.

### Cell 10b — Export KMZ (Google Earth)
- Creates a KMZ file (`outputs/viirs_landsat_overlay_<date>.kmz`) containing two ground overlay layers:
  - **VIIRS I1** as a grayscale PNG, georeferenced to its lat/lon bounding box.
  - **Landsat RGB** (all scenes merged) as a RGBA PNG. Areas with no Landsat data are fully transparent (alpha=0) so the VIIRS base shows through. Valid Landsat pixels have alpha=200 (semi-transparent) so you can see both layers.
- The KMZ is a ZIP file containing `doc.kml` + two PNG images.
- Open in Google Earth → both layers appear as toggleable folders. You can turn Landsat on/off to compare against VIIRS underneath.
- Coordinates are converted from UTM back to WGS84 lat/lon for KML compatibility.

### Cell 11 — Extract Sample Data
- Computes the Landsat mosaic extent in UTM coordinates from the affine transform.
- Creates a 20×15 grid of sample points spaced across the Landsat footprint (with 5 km inset from edges).
- For each sample point:
  - Maps UTM (x, y) → VIIRS pixel (row, col) using the VIIRS affine transform.
  - Maps UTM (x, y) → Landsat pixel (row, col) using the Landsat affine transform (accounting for downsample factor).
  - Skips points where either dataset has no valid data.
  - Converts UTM → lat/lon for the output table.
  - Reads VIIRS angles (SolZen, SatZen, SolAzi, SatAzi) at that pixel.
  - Reads VIIRS I-bands (I1–I5) at that pixel.
  - Reads Landsat bands (B1–B5) at that pixel.
  - Sets **overlap flags**: `Has_VIIRS` (any I-band valid), `Has_Landsat` (any Landsat band valid), `Has_Both` (both true).
- Outputs a pandas DataFrame with all columns.
- Prints overlap summary: how many pixels have both datasets, VIIRS-only, or Landsat-only.
- Displays the full table and summary statistics.

### Cell 11b — Overlap Coverage Map
- **Left panel: Coverage map** — Builds a 2D mask on the VIIRS grid:
  - 0 (dark) = no data from either sensor
  - 1 (blue) = VIIRS data only
  - 2 (orange) = Landsat data only
  - 3 (green) = both VIIRS and Landsat have valid data
  - Samples every 3rd column for speed (the VIIRS grid can be large).
- **Right panel: Sample points** — The extracted pixels from Cell 11 plotted on a faded VIIRS I1 background, colour-coded by overlap status (same blue/orange/green scheme).
- Prints pixel counts for each coverage category.

### Cell 12 — Per-Class Training Tables
- Provides 4 empty template DataFrames, one per river-ice class:
  1. **Ice-free river + snow-free land**
  2. **Ice-free river + snow-covered land**
  3. **Ice-covered river + snow-free land**
  4. **Ice-covered river + snow-covered land**
- Columns match the extraction table: acquisition date, pixel coords, angles, bands, vegetation type, NDVI.
- The user manually assigns pixels from the Cell 11 extraction table into these classes based on visual inspection of the Landsat RGB overlay.

---

## Data Flow

```
GITCO (HDF5)              GIMGO-SVI (HDF5)           Landsat (per-scene GeoTIFFs)
  │ lat, lon                 │ I1–I5 raw SDR             │ B1–B5
  │ SolZen, SatZen           │ ReflectanceFactors         │
  │ SolAzi, SatAzi           │ BrightnessTemperatureFactors│
  └──────┬───────────────────┘                             │
         │                                                 │
         │ pyresample nearest-neighbour                    │ rasterio.warp
         │ (swath lat/lon → UTM grid)                      │ (scene CRS → UTM)
         ▼                                                 ▼
   VIIRS 375 m UTM grid                          Per-scene 30 m UTM GeoTIFFs
   ┌─────────────────┐                           ┌──────────────────────────┐
   │ I1, I2, I3 (refl)│                          │ scene_LC08_072017.tif    │
   │ I4, I5 (BT)      │                          │ scene_LC09_071017.tif    │
   │ SolZen, SatZen    │                          │ ...                      │
   │ SolAzi, SatAzi    │                          └──────────┬───────────────┘
   └────────┬──────────┘                                     │
            │                                     rasterio.merge (stitch all scenes)
            │                                                │
            │                                                ▼
            │                                     Landsat 30 m mosaic GeoTIFF
            │                                     (single image, all scenes merged)
            │                                                │
            └──────────────┬─────────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
         GeoTIFF       KMZ (Google      Overlay +
         export        Earth overlay)   Sample Extraction
                                             │
                                    ┌────────┼────────┐
                                    ▼        ▼        ▼
                              Extraction  Coverage  Training
                              table w/    map       tables
                              overlap     (4-class  (4 ice
                              flags       colour)   classes)
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/viirs_375m_<date>.tif` | Reprojected VIIRS — all I-bands + angles as multi-band GeoTIFF |
| `outputs/landsat_30m_mosaic.tif` | Merged Landsat mosaic — all scenes stitched, multi-band GeoTIFF |
| `outputs/landsat_scene_chunks/` | Individual per-scene Landsat GeoTIFFs (intermediate) |
| `outputs/viirs_landsat_overlay_<date>.kmz` | KMZ for Google Earth — VIIRS base + Landsat RGB overlay |
| In-notebook extraction table | Sample pixels with VIIRS bands, angles, Landsat bands, overlap flags |
| In-notebook coverage map | Colour-coded map showing VIIRS/Landsat/both coverage |
| In-notebook training tables | Empty templates for 4 river-ice classes |

## Key Technical Notes

- **Reprojection is geometry-only.** Solar/satellite angles do not influence the reprojection. They are passthrough data arrays that get resampled onto the same grid as the bands.
- **VIIRS uses nearest-neighbour resampling** (pyresample) with 7 km radius of influence. This preserves original pixel values without interpolation artefacts.
- **Landsat uses rasterio.warp** for reprojection, which supports bilinear/nearest resampling.
- **Date matching is not enforced.** For meaningful ice classification, VIIRS and Landsat data should ideally be from the same day or within 1–2 days. Ice conditions change rapidly.
- **The full-res Landsat mosaic is deleted from memory** after saving to disk. Only the downsampled version is kept for display and sampling. This is necessary because the full 30 m mosaic can exceed available RAM.
