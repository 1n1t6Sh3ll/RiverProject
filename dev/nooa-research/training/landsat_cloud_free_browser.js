// =============================================================
// Landsat cloud-free scene browser — Valentina training data
// Paste into code.earthengine.google.com
//
// TARGET: Interior Alaska river system (Yukon/Tanana confluence)
// BBOX:   lon -157 to -151, lat 63 to 65
//
// SEARCH WINDOWS — uncomment ONE block at a time:
//   WINDOW A: Jan  (ice_covered_river_snow_covered_land)
//   WINDOW B: Sept (ice_free_river_snow_free_land)
//   WINDOW C: Apr–May (ice_free_river_snow_land)
//
// HOW TO USE:
//   1. Uncomment the target SEARCH WINDOW below
//   2. Run the script — scenes print in Console, sorted by cloud cover
//   3. Copy a scene ID into sceneId at the bottom to inspect it
//   4. Click on map to get pixel-level Landsat band values
//   5. When you find a good scene, note its ID + date for CLASS order
// =============================================================

// ── TARGET AOI — interior Alaska, same system as Oct reference scene ──────────

var aoi = ee.Geometry.Rectangle([-157, 63, -151, 65]);
Map.centerObject(aoi, 7);
Map.addLayer(ee.Image().paint(aoi, 0, 2),
  {palette: ['cyan']}, 'Target AOI (interior Alaska)', true);

// ── SEARCH WINDOW — uncomment ONE block ──────────────────────────────────────
// Only one window should be active at a time.

// WINDOW A: January — ice_covered_river_snow_covered_land
var start = '2020-01-01';
var end   = '2023-02-28';
var windowLabel = 'WINDOW A — Jan (ice_covered_river_snow_covered_land)';

// WINDOW B: Late summer — ice_free_river_snow_free_land
// var start = '2020-08-01';
// var end   = '2023-09-30';
// var windowLabel = 'WINDOW B — Aug–Sept (ice_free_river_snow_free_land)';

// WINDOW C: Spring transition — ice_free_river_snow_land
// var start = '2020-04-15';
// var end   = '2023-05-31';
// var windowLabel = 'WINDOW C — Apr–May (ice_free_river_snow_land)';

print('=== ACTIVE SEARCH ===');
print(windowLabel);
print('AOI: lon -157 to -151, lat 63 to 65');

// ── SCENE SEARCH ──────────────────────────────────────────────────────────────

var datasetL9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(aoi)
  .filterDate(start, end)
  .filterMetadata('CLOUD_COVER', 'less_than', 20)
  .sort('CLOUD_COVER');

var datasetL8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(aoi)
  .filterDate(start, end)
  .filterMetadata('CLOUD_COVER', 'less_than', 20)
  .sort('CLOUD_COVER');

var dataset = datasetL9.merge(datasetL8).sort('CLOUD_COVER');

// ── SCALE + MASK ──────────────────────────────────────────────────────────────

function applyScaleFactors(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(optical, null, true)
              .addBands(thermal, null, true);
}

function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).or(qa.bitwiseAnd(1 << 4));
  return image.updateMask(cloud.not());
}

var scaled = dataset.map(applyScaleFactors).map(maskClouds);
var count  = dataset.size().getInfo();
print('Total scenes found (L8+L9, cloud < 20%):', count);

if (count === 0) {
  print('No scenes found — widen date range or raise cloud threshold');
} else {
  var list = dataset.toList(Math.min(count, 30));
  print('--- Top scenes (sorted by cloud cover) ---');
  for (var i = 0; i < Math.min(count, 30); i++) {
    var img   = ee.Image(list.get(i));
    var date  = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo();
    var cloud = ee.Number(img.get('CLOUD_COVER')).round().getInfo();
    var id    = img.get('LANDSAT_SCENE_ID').getInfo();
    var sat   = id.substring(0, 4);   // LC08 or LC09
    print('Index: ' + i + '  Date: ' + date + '  Cloud: ' + cloud +
          '%  ID: ' + id);

    // Add each scene as a hidden layer — toggle in Layers panel
    Map.addLayer(
      applyScaleFactors(img).select(['SR_B4', 'SR_B3', 'SR_B2']),
      {min: 0.0, max: 0.3},
      i + ' | ' + date + ' | ' + cloud + '% | ' + sat,
      false
    );
  }

  // Median composite for context
  var median = scaled.median();
  Map.addLayer(
    median.select(['SR_B4', 'SR_B3', 'SR_B2']),
    {min: 0.0, max: 0.3},
    'Median composite (' + windowLabel + ')',
    true
  );

  // ── SELECTED SCENE — paste a scene ID here to inspect ──────────────────────
  //
  // Change sceneId to the ID printed in Console for the scene you want.
  // Leave as null to skip individual scene inspection.
  //
  var sceneId = 'LC90730152022070LGN01';
  // Example — uncomment and fill in after finding a candidate:
  // var sceneId = 'LC09_L2SP_073015_20220105_20230404_02_T1';

  if (sceneId !== null) {
    var selected = dataset.filter(
      ee.Filter.eq('LANDSAT_SCENE_ID', sceneId)).first();

    var selectedScaled = applyScaleFactors(selected);

    Map.addLayer(
      selectedScaled.select(['SR_B4', 'SR_B3', 'SR_B2']),
      {min: 0.0, max: 0.3},
      'SELECTED RGB — ' + sceneId,
      true
    );
    Map.addLayer(
      selectedScaled.select(['SR_B5', 'SR_B4', 'SR_B3']),
      {min: 0.0, max: 0.3},
      'SELECTED False Color NIR — ' + sceneId,
      false
    );

    // NDWI — water index (blue = water, brown = land)
    var ndwi = selectedScaled.normalizedDifference(['SR_B3', 'SR_B5']);
    Map.addLayer(ndwi,
      {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
      'SELECTED NDWI — ' + sceneId,
      false
    );

    // NDSI — snow index (blue = snow/ice, brown = no snow)
    // Uses SR_B3 (green) and SR_B6 (SWIR1)
    var ndsi = selectedScaled.normalizedDifference(['SR_B3', 'SR_B6']);
    Map.addLayer(ndsi,
      {min: -0.5, max: 0.8, palette: ['brown', 'white', 'cyan']},
      'SELECTED NDSI — snow/ice — ' + sceneId,
      false
    );

    // Thermal band — cold = purple, warm = yellow
    // Ice-covered river: ST_B10 < 273K
    Map.addLayer(
      selectedScaled.select('ST_B10'),
      {min: 260, max: 300, palette: ['purple', 'blue', 'white', 'yellow']},
      'SELECTED Thermal ST_B10 (K) — ' + sceneId,
      false
    );

    print('=== SELECTED SCENE ===');
    print('ID:', sceneId);
    var selDate = ee.Date(selected.get('system:time_start'))
      .format('YYYY-MM-dd').getInfo();
    var selCloud = selected.get('CLOUD_COVER').getInfo();
    print('Date:', selDate);
    print('Cloud:', selCloud + '%');
    print('Tip: toggle NDWI for water, NDSI for snow/ice, Thermal for freeze state');

    // ── PIXEL INSPECTOR — click on map ────────────────────────────────────────
    // Prints lat/lon + band values at click point.
    // Use these to verify ground truth class at each training candidate.
    Map.onClick(function(coords) {
      print('--- Click ---');
      print('lat:', coords.lat.toFixed(5), 'lon:', coords.lon.toFixed(5));
      var pt = ee.Geometry.Point([coords.lon, coords.lat]);
      var vals = selectedScaled
        .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ST_B10'])
        .reduceRegion({
          reducer:  ee.Reducer.first(),
          geometry: pt,
          scale:    30
        }).getInfo();
      print('SR_B2 (blue):', vals['SR_B2']);
      print('SR_B3 (green):', vals['SR_B3']);
      print('SR_B4 (red):', vals['SR_B4']);
      print('SR_B5 (NIR):', vals['SR_B5']);
      print('SR_B6 (SWIR1):', vals['SR_B6']);
      print('ST_B10 thermal (K):', vals['ST_B10']);
      print('NDWI (water):', vals['SR_B3'] !== null && vals['SR_B5'] !== null
        ? ((vals['SR_B3'] - vals['SR_B5']) /
           (vals['SR_B3'] + vals['SR_B5'])).toFixed(4)
        : 'null');
      print('NDSI (snow):', vals['SR_B3'] !== null && vals['SR_B6'] !== null
        ? ((vals['SR_B3'] - vals['SR_B6']) /
           (vals['SR_B3'] + vals['SR_B6'])).toFixed(4)
        : 'null');
      print('Ice on river? ST_B10 < 273K:', vals['ST_B10'] !== null
        ? (vals['ST_B10'] < 273 ? 'YES — frozen' : 'NO — liquid')
        : 'null');
      print('Snow on land? NDSI > 0.4:', vals['SR_B3'] !== null
        ? (((vals['SR_B3'] - vals['SR_B6']) /
            (vals['SR_B3'] + vals['SR_B6'])) > 0.4
           ? 'YES — snow' : 'NO — snow-free')
        : 'null');
    });
  } else {
    print('No scene selected — set sceneId variable to inspect a specific scene');
    print('Copy an ID from the Console list above');
  }
}
