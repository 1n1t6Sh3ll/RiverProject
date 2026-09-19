/*
 * GEE Visual Inspector - 2024-10-02
 * Landsat scene: LC80730112024276LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC80730112024276LGN00'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-151.3919, 68.4479, -144.8754, 70.7527]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2024-09-25', '2024-10-09')
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
  {id: 1, lat: 69.21385, lon: -151.00508, auto_class: 'ice_free_river_snow_free_land', st: 273.245, ndsi: -0.2516},
  {id: 2, lat: 69.15979, lon: -150.86318, auto_class: 'ice_free_river_snow_free_land', st: 273.6518, ndsi: -0.5946},
  {id: 3, lat: 69.10912, lon: -150.8294, auto_class: 'ice_free_river_snow_free_land', st: 275.8017, ndsi: -0.6114},
  {id: 4, lat: 69.06182, lon: -147.86656, auto_class: 'ice_covered_river_snow_free_land', st: 270.0902, ndsi: -0.2331},
  {id: 5, lat: 69.03817, lon: -147.92737, auto_class: 'ice_covered_river_snow_free_land', st: 268.5999, ndsi: -0.4877},
  {id: 6, lat: 69.02466, lon: -147.35981, auto_class: 'ice_covered_river_snow_free_land', st: 268.6854, ndsi: -0.7862},
  {id: 7, lat: 68.99425, lon: -148.95102, auto_class: 'ice_covered_river_snow_covered_land', st: 266.9661, ndsi: 0.9539},
  {id: 8, lat: 68.87601, lon: -148.8294, auto_class: 'ice_covered_river_snow_covered_land', st: 266.5833, ndsi: 0.9853},
  {id: 9, lat: 68.80506, lon: -147.30575, auto_class: 'ice_covered_river_snow_covered_land', st: 271.3173, ndsi: 0.4586},
  {id: 10, lat: 68.7679, lon: -148.68075, auto_class: 'ice_covered_river_snow_covered_land', st: 266.3475, ndsi: 0.8673}
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

Map.setCenter(-149.8, 69.0, 8);

print('2024-10-02 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
