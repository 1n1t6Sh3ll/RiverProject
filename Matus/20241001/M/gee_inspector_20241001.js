/*
 * GEE Visual Inspector - 2024-10-01
 * Landsat scene: LC90740122024275LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90740122024275LGN00'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-154.44942490496624, 67.1273222036833, -148.36261906395418, 69.41365118989921]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2024-09-24', '2024-10-08')
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

// ── Map layers ──────────────────────────────────────────────────────────────
Map.addLayer(full, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}, 'RGB (True Color)', true);
Map.addLayer(full, {bands: ['SR_B5', 'SR_B4', 'SR_B3'], min: 0, max: 0.4}, 'False Color NIR', false);
Map.addLayer(full, {bands: ['ST_B10'], min: 250, max: 290, palette: ['purple','blue','cyan','yellow','red']}, 'Thermal (ST_B10)', false);
Map.addLayer(ndsi, {min: -0.5, max: 1.0, palette: ['brown','white','cyan']}, 'NDSI', false);
Map.addLayer(s2, {bands: ['B4','B3','B2'], min: 0, max: 3000}, 'Sentinel-2 10m (nearest pass)', false);

var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
Map.addLayer(cloudMask.selfMask(), {palette: ['red']}, 'Cloud Mask', false);

// ── Candidate pixels ────────────────────────────────────────────────────────
var candidates = [
  {id: 1, lat: 68.63156, lon: -153.23828, auto_class: 'ice_covered_river_snow_covered_land', st: 267.1404, ndsi: 0.9563},
  {id: 2, lat: 68.58088, lon: -153.47138, auto_class: 'ice_covered_river_snow_covered_land', st: 264.7205, ndsi: 0.9735},
  {id: 3, lat: 68.52683, lon: -150.05247, auto_class: 'ice_covered_river_snow_covered_land', st: 262.7483, ndsi: 0.9616},
  {id: 4, lat: 68.48291, lon: -149.46801, auto_class: 'ice_covered_river_snow_covered_land', st: 268.4598, ndsi: 0.9878},
  {id: 5, lat: 68.45926, lon: -151.20787, auto_class: 'ice_covered_river_snow_covered_land', st: 264.4094, ndsi: 0.9641},
  {id: 6, lat: 68.43899, lon: -149.92071, auto_class: 'ice_covered_river_snow_covered_land', st: 262.5159, ndsi: 0.9927},
  {id: 7, lat: 68.4221, lon: -149.34976, auto_class: 'ice_covered_river_snow_covered_land', st: 269.7484, ndsi: 0.9918},
  {id: 8, lat: 68.38493, lon: -149.35652, auto_class: 'ice_covered_river_snow_covered_land', st: 270.2269, ndsi: 0.9688},
  {id: 9, lat: 68.06399, lon: -150.17747, auto_class: 'ice_covered_river_snow_covered_land', st: 264.7547, ndsi: 0.4723},
  {id: 10, lat: 67.9998, lon: -153.12679, auto_class: 'ice_covered_river_snow_free_land', st: 271.9496, ndsi: -0.2738},
  {id: 11, lat: 67.92885, lon: -149.83963, auto_class: 'ice_covered_river_snow_free_land', st: 263.6096, ndsi: 0.1256}
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

Map.setCenter(-151.5, 68.3, 8);

print('2024-10-01 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
