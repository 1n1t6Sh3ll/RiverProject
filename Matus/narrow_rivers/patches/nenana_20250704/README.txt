Nenana River patch — 2025-07-04 (summer)
==========================================================

Landsat scene : LC90700152025185LGN00
                LC09_L2SP_070015_20250704_20250705_02_T1
                acquired 2025-07-04 21:18:55 UTC, cloud 9.09 % (whole scene), sun 47.9 deg
Box           : lon -149.33766 .. -149.06919, lat 64.09373 .. 64.53537
Grid          : EPSG:4326 on the pipeline's Alaska grid
                Landsat 995 x 1635 cells of 0.00027027 deg
                VIIRS   80 x 132 cells of 0.00337838 deg
                Pixels are rectangular on the ground at this latitude.

HOW TO START (QGIS)
  1. Drag every .tif from this folder into QGIS. Styles load automatically.
  2. Drag labels/labels_2025-07-04.shp in as well — it is empty and ready to draw on.
  3. Digitise polygons and set class_id on each:
        1 = river (open water)     2 = ice
        3 = snow                   4 = land
        9 = no data
     Do not let polygons of different classes overlap
     (Project > Snapping Options > Avoid overlap on active layer).
  4. A VIIRS cell only becomes usable when it is labelled edge to edge.

LAYERS
  2025-07-04_sr.tif                SR_B2..B7 surface reflectance (float)
                               default style = true colour (B4/B3/B2, 0..0.4)
                               alternative   = 2025-07-04_sr_swir.qml
                               (Layer Properties > Style > Load Style)
  2025-07-04_thermal.tif           ST_B10, kelvin
  2025-07-04_qa_pixel.tif          QA_PIXEL bit flags (CFMask; it often marks river
                               ice as cloud, so treat it as a hint only)
  2025-07-04_water_occurrence.tif  JRC water occurrence 0-100 %, a guide for where
                               the channel usually is
  2025-07-04_truecolour.tif        8-bit quicklooks, fixed scaling, for a fast look
  2025-07-04_swir.tif              snow/ice = cyan, water = black, cloud = white

VIIRS
  Pass      : NPP t2216388, 2025-07-04, +59 min vs Landsat
  View angle: ~7 deg over the box
  Geolocation check vs this Landsat scene: best shift 50 m (r=0.42)
  Grid file : 2025-07-04_viirs_375m.tif — I1..I5 + SZA/SAA/VZA/VAA
              on the same 375 m cells the fractions will use.
  The granule itself stays out of this folder (~230 MB).
