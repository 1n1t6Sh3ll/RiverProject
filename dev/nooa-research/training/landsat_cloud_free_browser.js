// =============================================================
// Landsat cloud-free scene browser — Valentina training data
// Paste into code.earthengine.google.com
// Current working scene: LC90730152022278LGN01, 2022-10-05
// =============================================================

var aoi = ee.Geometry.Rectangle([-171, 54, -129, 66.17]);
Map.centerObject(aoi, 4);

var start = '2022-09-01';
var end   = '2022-11-30';

var datasetL9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(aoi).filterDate(start, end)
  .filterMetadata('CLOUD_COVER', 'less_than', 50)
  .sort('CLOUD_COVER');

var datasetL8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(aoi).filterDate(start, end)
  .filterMetadata('CLOUD_COVER', 'less_than', 50)
  .sort('CLOUD_COVER');

var dataset = datasetL9.merge(datasetL8).sort('CLOUD_COVER');

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
print('Total scenes found (L8+L9):', count);

if (count === 0) {
  print('No scenes found — try wider date range or higher cloud threshold');
} else {
  var list = dataset.toList(count);
  for (var i = 0; i < Math.min(count, 20); i++) {
    var img   = ee.Image(list.get(i));
    var date  = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo();
    var cloud = img.get('CLOUD_COVER').getInfo();
    var id    = img.get('LANDSAT_SCENE_ID').getInfo();
    print(i + ' | ' + date + ' | Cloud: ' + cloud + '% | ' + id);
    Map.addLayer(
      applyScaleFactors(img).select(['SR_B4','SR_B3','SR_B2']),
      {min:0.0, max:0.3},
      i + ' | ' + date + ' | ' + cloud + '%', false);
  }

  var median = scaled.median();
  Map.addLayer(median.select(['SR_B4','SR_B3','SR_B2']),
    {min:0.0, max:0.3}, 'Median composite', true);
  var ndvi = median.normalizedDifference(['SR_B5','SR_B4']);
  Map.addLayer(ndvi,
    {min:-0.3, max:0.8, palette:['brown','white','green']}, 'NDVI', false);

  // === SELECTED SCENE — change sceneId to inspect any scene ===
  var sceneId   = 'LC90730152022278LGN01';  // 2022-10-05, cloud 0.03%
  var sceneBbox = ee.Geometry.Rectangle(
    [-156.53049, 63.09720, -151.47357, 65.33383]);

  var selected = dataset.filter(
    ee.Filter.eq('LANDSAT_SCENE_ID', sceneId)).first();

  Map.centerObject(sceneBbox, 9);
  Map.addLayer(
    applyScaleFactors(selected).select(['SR_B4','SR_B3','SR_B2']),
    {min:0.0, max:0.3}, 'SELECTED RGB ' + sceneId, true);
  Map.addLayer(
    applyScaleFactors(selected).select(['SR_B5','SR_B4','SR_B3']),
    {min:0.0, max:0.3}, 'SELECTED False Color (NIR)', false);

  var ndwi = applyScaleFactors(selected)
    .normalizedDifference(['SR_B3','SR_B5']);
  Map.addLayer(ndwi,
    {min:-0.5, max:0.5, palette:['brown','white','blue']},
    'SELECTED NDWI (water)', false);

  Map.addLayer(ee.Image().paint(sceneBbox, 0, 2),
    {palette:['yellow']}, 'Scene bbox', true);

  print('=== SELECTED SCENE ===');
  print('ID:', sceneId);
  print('Date: 2022-10-05');
  print('Cloud: 0.03%');
  print('BBox: lon -156.53 to -151.47, lat 63.10 to 65.33');

  // === PIXEL INSPECTOR — click on map to get lat/lon ===
  Map.onClick(function(coords) {
    print('Clicked lat/lon:', coords.lat.toFixed(6), coords.lon.toFixed(6));
    var pt = ee.Geometry.Point([coords.lon, coords.lat]);
    var vals = applyScaleFactors(selected).select(
      ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5'])
      .reduceRegion({reducer:ee.Reducer.first(),
                     geometry:pt, scale:30}).getInfo();
    print('Landsat band values at click:', vals);
  });
}
