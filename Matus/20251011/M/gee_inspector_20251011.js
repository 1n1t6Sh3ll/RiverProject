/*
 * GEE Visual Inspector - 2025-10-11
 * Landsat scene: LC90750122025284LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90750122025284LGN00'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-156.0248, 67.1274, -149.9376, 69.4136]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2025-10-04', '2025-10-18')
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
  {id: 1, lat: 68.61461, lon: -155.10081, auto_class: 'ice_covered_river_snow_covered_land', st: 263.6301, ndsi: 0.8735},
  {id: 2, lat: 68.48286, lon: -151.57716, auto_class: 'ice_covered_river_snow_covered_land', st: 266.8362, ndsi: 0.9098},
  {id: 3, lat: 68.43218, lon: -151.41838, auto_class: 'ice_covered_river_snow_covered_land', st: 268.7298, ndsi: 0.7682},
  {id: 4, lat: 68.41191, lon: -151.43527, auto_class: 'ice_covered_river_snow_covered_land', st: 268.3573, ndsi: 0.6432},
  {id: 5, lat: 68.38826, lon: -153.05689, auto_class: 'ice_covered_river_snow_covered_land', st: 269.6493, ndsi: 0.5101},
  {id: 6, lat: 68.33421, lon: -151.46568, auto_class: 'ice_covered_river_snow_covered_land', st: 268.8529, ndsi: 0.4736},
  {id: 7, lat: 68.2734, lon: -154.06365, auto_class: 'ice_covered_river_snow_free_land', st: 270.326, ndsi: -0.5765},
  {id: 8, lat: 68.24299, lon: -154.07716, auto_class: 'ice_covered_river_snow_free_land', st: 270.6302, ndsi: -0.9511},
  {id: 9, lat: 68.18556, lon: -154.13122, auto_class: 'ice_covered_river_snow_free_land', st: 270.3295, ndsi: -0.542},
  {id: 10, lat: 68.15178, lon: -152.79, auto_class: 'ice_covered_river_snow_free_land', st: 269.5262, ndsi: 0.3167},
  {id: 11, lat: 68.11799, lon: -151.10081, auto_class: 'ice_covered_river_snow_covered_land', st: 265.9373, ndsi: 0.7042},
  {id: 12, lat: 68.07069, lon: -155.73257, auto_class: 'ice_covered_river_snow_covered_land', st: 264.2454, ndsi: 0.8362},
  {id: 13, lat: 68.04029, lon: -152.18189, auto_class: 'ice_covered_river_snow_free_land', st: 268.2103, ndsi: -0.3333},
  {id: 14, lat: 67.99637, lon: -152.91838, auto_class: 'ice_covered_river_snow_covered_land', st: 264.7171, ndsi: 0.9534}
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

Map.setCenter(-154.0, 68.3, 8);

print('2025-10-11 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
