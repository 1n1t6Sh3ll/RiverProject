/*
 * GEE Visual Inspector - 2022-10-23
 * Landsat scene: LC90710162022296LGN01
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90710162022296LGN01'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Map layers ──────────────────────────────────────────────────────────────
Map.addLayer(full, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}, 'RGB (True Color)', true);
Map.addLayer(full, {bands: ['SR_B5', 'SR_B4', 'SR_B3'], min: 0, max: 0.4}, 'False Color NIR', false);
Map.addLayer(full, {bands: ['ST_B10'], min: 250, max: 290, palette: ['purple','blue','cyan','yellow','red']}, 'Thermal (ST_B10)', false);
Map.addLayer(ndsi, {min: -0.5, max: 1.0, palette: ['brown','white','cyan']}, 'NDSI', false);

var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
Map.addLayer(cloudMask.selfMask(), {palette: ['red']}, 'Cloud Mask', false);

// ── Candidate pixels ────────────────────────────────────────────────────────
var candidates = [
  {id: 1, lat: 63.13734, lon: -150.1018, auto_class: 'ice_covered_river_snow_covered_land', st: 261.6033, ndsi: 0.9814},
  {id: 2, lat: 63.09005, lon: -150.31802, auto_class: 'ice_covered_river_snow_covered_land', st: 258.9782, ndsi: 0.9153},
  {id: 3, lat: 63.03599, lon: -150.0714, auto_class: 'ice_covered_river_snow_covered_land', st: 264.2078, ndsi: 0.9134},
  {id: 4, lat: 62.90086, lon: -150.83491, auto_class: 'ice_covered_river_snow_covered_land', st: 259.696, ndsi: 0.9351},
  {id: 5, lat: 62.8468, lon: -150.61869, auto_class: 'ice_covered_river_snow_covered_land', st: 261.1897, ndsi: 0.9079},
  {id: 6, lat: 62.78261, lon: -150.63559, auto_class: 'ice_covered_river_snow_covered_land', st: 260.342, ndsi: 0.9438},
  {id: 7, lat: 62.76234, lon: -151.06464, auto_class: 'ice_covered_river_snow_covered_land', st: 262.9021, ndsi: 0.9433},
  {id: 8, lat: 62.74545, lon: -150.54099, auto_class: 'ice_covered_river_snow_covered_land', st: 263.2336, ndsi: 0.9429},
  {id: 9, lat: 62.68126, lon: -150.79437, auto_class: 'ice_covered_river_snow_covered_land', st: 269.2904, ndsi: 0.6866},
  {id: 10, lat: 62.6441, lon: -150.36194, auto_class: 'ice_covered_river_snow_free_land', st: 272.6435, ndsi: -0.7657}
];

var iceSnow = [], iceNoSnow = [], freeNoSnow = [], freeSnow = [];
candidates.forEach(function(c) {
  var feat = ee.Feature(ee.Geometry.Point([c.lon, c.lat]),
    {'Pixel': c.id, 'Auto_Class': c.auto_class, 'ST_B10_K': c.st, 'NDSI': c.ndsi});
  if (c.auto_class === 'ice_covered_river_snow_covered_land') iceSnow.push(feat);
  else if (c.auto_class === 'ice_covered_river_snow_free_land') iceNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_free_land') freeNoSnow.push(feat);
  else freeSnow.push(feat);
});

if (iceSnow.length > 0)   Map.addLayer(ee.FeatureCollection(iceSnow),   {color: 'FF0000'}, 'ice_covered+snow_covered (' + iceSnow.length + ')', true);
if (iceNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(iceNoSnow), {color: 'FF8800'}, 'ice_covered+snow_free (' + iceNoSnow.length + ')', true);
if (freeNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(freeNoSnow), {color: '00FF00'}, 'ice_free+snow_free (' + freeNoSnow.length + ')', true);
if (freeSnow.length > 0)  Map.addLayer(ee.FeatureCollection(freeSnow),  {color: '0088FF'}, 'ice_free+snow (' + freeSnow.length + ')', true);

Map.setCenter(-150.2, 62.9, 8);

print('2022-10-23 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
