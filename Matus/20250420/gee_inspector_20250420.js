/*
 * GEE Visual Inspector - 2025-04-20
 * Landsat scene: LC90730162025110LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90730162025110LGN00'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-157.61019607961842, 61.73524864911403, -152.82892400172676, 63.95595546251622]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2025-04-13', '2025-04-27')
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
  {id: 1, lat: 63.37656, lon: -152.89905, auto_class: 'ice_free_river_snow_land', st: 274.831, ndsi: 0.6139},
  {id: 2, lat: 63.21102, lon: -154.04094, auto_class: 'ice_free_river_snow_land', st: 275.5864, ndsi: 0.643},
  {id: 3, lat: 63.12656, lon: -153.7504, auto_class: 'ice_free_river_snow_land', st: 274.602, ndsi: 0.8492},
  {id: 4, lat: 63.07927, lon: -153.43621, auto_class: 'ice_free_river_snow_land', st: 275.4804, ndsi: 0.9073},
  {id: 5, lat: 63.03535, lon: -154.36864, auto_class: 'ice_free_river_snow_free_land', st: 282.5386, ndsi: -0.3943},
  {id: 6, lat: 63.00494, lon: -154.22675, auto_class: 'ice_free_river_snow_free_land', st: 280.1289, ndsi: -0.4793},
  {id: 7, lat: 62.97791, lon: -155.22675, auto_class: 'ice_free_river_snow_land', st: 274.8583, ndsi: 0.9072},
  {id: 8, lat: 62.94075, lon: -155.61864, auto_class: 'ice_free_river_snow_free_land', st: 275.5078, ndsi: 0.363},
  {id: 9, lat: 62.88332, lon: -154.74702, auto_class: 'ice_free_river_snow_free_land', st: 281.4654, ndsi: -0.5134},
  {id: 10, lat: 62.82589, lon: -155.62878, auto_class: 'ice_free_river_snow_land', st: 276.2255, ndsi: 0.8692},
  {id: 11, lat: 62.76508, lon: -153.93621, auto_class: 'ice_free_river_snow_land', st: 274.7968, ndsi: 0.9689},
  {id: 12, lat: 62.65359, lon: -155.74364, auto_class: 'ice_free_river_snow_free_land', st: 277.2099, ndsi: 0.1476},
  {id: 13, lat: 62.57251, lon: -155.74026, auto_class: 'ice_free_river_snow_land', st: 277.08, ndsi: 0.8582},
  {id: 14, lat: 62.40021, lon: -154.7977, auto_class: 'ice_covered_river_snow_covered_land', st: 272.7494, ndsi: 0.8683},
  {id: 15, lat: 62.32927, lon: -153.74026, auto_class: 'ice_covered_river_snow_covered_land', st: 269.8954, ndsi: 0.8734},
  {id: 16, lat: 62.23129, lon: -153.96661, auto_class: 'ice_free_river_snow_land', st: 273.3066, ndsi: 0.8098},
  {id: 17, lat: 62.10291, lon: -156.19972, auto_class: 'ice_free_river_snow_land', st: 273.1186, ndsi: 0.8861},
  {id: 18, lat: 61.95764, lon: -154.45648, auto_class: 'ice_free_river_snow_land', st: 275.1694, ndsi: 0.8931}
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

Map.setCenter(-153.7, 62.7, 8);

print('2025-04-20 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
