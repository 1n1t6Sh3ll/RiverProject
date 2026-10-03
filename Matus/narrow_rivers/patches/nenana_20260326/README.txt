Nenana River patch — 2026-03-26 (ice)
==========================================================

Landsat scene : LC90690152026085LGN00
                LC09_L2SP_069015_20260326_20260330_02_T1
                acquired 2026-03-26 21:12:56 UTC, cloud 0.02 % (whole scene), sun 27.7 deg
Box           : lon -149.33766 .. -149.06919, lat 64.09373 .. 64.53537
Grid          : EPSG:4326 on the pipeline's Alaska grid
                Landsat 995 x 1635 cells of 0.00027027 deg
                VIIRS   80 x 132 cells of 0.00337838 deg
                Pixels are rectangular on the ground at this latitude.

HOW TO START (QGIS)
  1. Drag every .tif from this folder into QGIS. Styles load automatically.
  2. Drag labels/labels_2026-03-26.shp in as well — it is empty and ready to draw on.
  3. Digitise polygons and set class_id on each:
        1 = river (open water)     2 = ice
        3 = snow                   4 = land
        9 = no data
     Do not let polygons of different classes overlap
     (Project > Snapping Options > Avoid overlap on active layer).
  4. A VIIRS cell only becomes usable when it is labelled edge to edge.

LAYERS
  2026-03-26_sr.tif                SR_B2..B7 surface reflectance (float)
                               default style = true colour (B4/B3/B2, 0..0.4)
                               alternative   = 2026-03-26_sr_swir.qml
                               (Layer Properties > Style > Load Style)
  2026-03-26_thermal.tif           ST_B10, kelvin
  2026-03-26_qa_pixel.tif          QA_PIXEL bit flags (CFMask; it often marks river
                               ice as cloud, so treat it as a hint only)
  2026-03-26_water_occurrence.tif  JRC water occurrence 0-100 %, a guide for where
                               the channel usually is
  2026-03-26_truecolour.tif        8-bit quicklooks, fixed scaling, for a fast look
  2026-03-26_swir.tif              snow/ice = cyan, water = black, cloud = white

VIIRS
  Pass      : J01 t2133484, 2026-03-26, +21 min vs Landsat
  View angle: ~29 deg over the box
  Geolocation check vs this Landsat scene: best shift 50 m (r=0.93)
  Grid file : 2026-03-26_viirs_375m.tif — I1..I5 + SZA/SAA/VZA/VAA
              on the same 375 m cells the fractions will use.
  The granule itself stays out of this folder (~230 MB).

CLOUD CHECK (wide-area VIIRS)
  2026-03-26_viirs_wide_falsecolour.tif  I3/I2/I1 over ~110 x 270 km around the box
                                         cloud = white (bright in SWIR as well)
                                         snow/ice = cyan (dark in SWIR)
                                         water = black, land = green/brown
  2026-03-26_viirs_wide_thermal.tif      I5 brightness temperature (K);
                                         cloud tops are colder than the surface
  2026-03-26_viirs_wide_375m.tif         I1..I5 + SZA/SAA/VZA/VAA over the wide area
  2026-03-26_study_box.geojson           the study box, red outline, no fill
