Nenana River patch — 2024-06-22 (summer)
==========================================================

Landsat scene : LC90710152024174LGN00
                LC09_L2SP_071015_20240622_20240623_02_T1
                acquired 2024-06-22 21:24:24 UTC, cloud 6.9 % (whole scene), sun 48.6 deg
Box           : lon -149.33766 .. -149.06919, lat 64.09373 .. 64.53537
Grid          : EPSG:4326 on the pipeline's Alaska grid
                Landsat 995 x 1635 cells of 0.00027027 deg
                VIIRS   80 x 132 cells of 0.00337838 deg
                Pixels are rectangular on the ground at this latitude.

HOW TO START (QGIS)
  1. Drag every .tif from this folder into QGIS. Styles load automatically.
  2. Drag labels/labels_2024-06-22.shp in as well — it is empty and ready to draw on.
  3. Digitise polygons and set class_id on each:
        1 = river (open water)     2 = ice
        3 = snow                   4 = land
        9 = no data
     Do not let polygons of different classes overlap
     (Project > Snapping Options > Avoid overlap on active layer).
  4. A VIIRS cell only becomes usable when it is labelled edge to edge.

LAYERS
  2024-06-22_sr.tif                SR_B2..B7 surface reflectance (float)
                               default style = true colour (B4/B3/B2, 0..0.4)
                               alternative   = 2024-06-22_sr_swir.qml
                               (Layer Properties > Style > Load Style)
  2024-06-22_thermal.tif           ST_B10, kelvin
  2024-06-22_qa_pixel.tif          QA_PIXEL bit flags (CFMask; it often marks river
                               ice as cloud, so treat it as a hint only)
  2024-06-22_water_occurrence.tif  JRC water occurrence 0-100 %, a guide for where
                               the channel usually is
  2024-06-22_truecolour.tif        8-bit quicklooks, fixed scaling, for a fast look
  2024-06-22_swir.tif              snow/ice = cyan, water = black, cloud = white

VIIRS
  Pass      : J01 t2208041, 2024-06-22, +44 min vs Landsat
  View angle: ~1 deg over the box
  Geolocation check vs this Landsat scene: best shift 71 m (r=0.35)
  Grid file : 2024-06-22_viirs_375m.tif — I1..I5 + SZA/SAA/VZA/VAA
              on the same 375 m cells the fractions will use.
  The granule itself stays out of this folder (~230 MB).

CLOUD CHECK (wide-area VIIRS)
  2024-06-22_viirs_wide_falsecolour.tif  I3/I2/I1 over ~110 x 270 km around the box
                                         cloud = white (bright in SWIR as well)
                                         snow/ice = cyan (dark in SWIR)
                                         water = black, land = green/brown
  2024-06-22_viirs_wide_thermal.tif      I5 brightness temperature (K);
                                         cloud tops are colder than the surface
  2024-06-22_viirs_wide_375m.tif         I1..I5 + SZA/SAA/VZA/VAA over the wide area
  2024-06-22_study_box.geojson           the study box, red outline, no fill
