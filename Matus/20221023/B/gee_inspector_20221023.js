/*
 * GEE Visual Inspector - 2022-10-23
 * Landsat scene: LC90710162022296LGN01
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

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

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-154.46161174807895, 61.733758606360816, -149.68068983997256, 63.95661083361867]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2022-10-16', '2022-10-30')
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
  {id: 1, lat: 62.47181, lon: -154.22681, auto_class: 'ice_covered_river_snow_covered_land', st: 264.6863, ndsi: 0.7343},
  {id: 2, lat: 62.44141, lon: -151.91263, auto_class: 'ice_free_river_snow_free_land', st: 273.2484, ndsi: -0.138},
  {id: 3, lat: 62.41438, lon: -151.98695, auto_class: 'ice_covered_river_snow_free_land', st: 272.4896, ndsi: -0.6014},
  {id: 4, lat: 62.39073, lon: -151.1829, auto_class: 'ice_covered_river_snow_free_land', st: 272.4247, ndsi: -0.6458},
  {id: 5, lat: 62.34681, lon: -153.11533, auto_class: 'ice_covered_river_snow_free_land', st: 267.0379, ndsi: -0.9276},
  {id: 6, lat: 62.19816, lon: -152.28425, auto_class: 'ice_free_river_snow_free_land', st: 274.3285, ndsi: -0.9343},
  {id: 7, lat: 62.161, lon: -152.19303, auto_class: 'ice_free_river_snow_free_land', st: 275.0976, ndsi: -0.5294},
  {id: 8, lat: 62.07654, lon: -150.84844, auto_class: 'ice_free_river_snow_free_land', st: 273.6552, ndsi: -0.5682},
  {id: 9, lat: 61.97181, lon: -151.94641, auto_class: 'ice_free_river_snow_free_land', st: 273.5902, ndsi: -0.1617},
  {id: 10, lat: 61.9583, lon: -151.27073, auto_class: 'ice_free_river_snow_free_land', st: 273.6142, ndsi: -0.6053},
  {id: 11, lat: 61.93803, lon: -151.75046, auto_class: 'ice_free_river_snow_free_land', st: 273.7714, ndsi: -0.1695},
  {id: 12, lat: 61.91776, lon: -151.67614, auto_class: 'ice_free_river_snow_free_land', st: 274.2568, ndsi: -0.1798}
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

Map.setCenter(-153.0, 62.2, 8);

print('2022-10-23 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
