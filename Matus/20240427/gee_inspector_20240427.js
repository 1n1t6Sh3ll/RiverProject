/*
 * GEE Visual Inspector - 2024-04-27
 * Landsat scene: LC80710142024118LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC80710142024118LGN00'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var sceneBbox = ee.Geometry.Rectangle(
  [-152.42881658598267, 64.45474013450857, -147.0838276763541, 66.70519256447828]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sceneBbox)
  .filterDate('2024-04-20', '2024-05-04')
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
  {id: 1, lat: 66.2981, lon: -148.63659, auto_class: 'ice_free_river_snow_free_land', st: 274.872, ndsi: 0.1659},
  {id: 2, lat: 66.26094, lon: -149.01834, auto_class: 'ice_covered_river_snow_covered_land', st: 272.8041, ndsi: 0.9832},
  {id: 3, lat: 66.17648, lon: -148.43388, auto_class: 'ice_free_river_snow_land', st: 273.4296, ndsi: 0.9986},
  {id: 4, lat: 66.13931, lon: -148.17375, auto_class: 'ice_covered_river_snow_covered_land', st: 272.9374, ndsi: 0.9664},
  {id: 5, lat: 66.08864, lon: -148.94402, auto_class: 'ice_free_river_snow_land', st: 274.6977, ndsi: 0.6253},
  {id: 6, lat: 66.01769, lon: -149.13321, auto_class: 'ice_free_river_snow_land', st: 273.3817, ndsi: 0.98},
  {id: 7, lat: 65.75756, lon: -150.02172, auto_class: 'ice_covered_river_snow_covered_land', st: 272.4315, ndsi: 0.8206},
  {id: 8, lat: 65.17985, lon: -149.26834, auto_class: 'ice_free_river_snow_land', st: 273.5561, ndsi: 0.935},
  {id: 9, lat: 65.13256, lon: -152.15348, auto_class: 'ice_free_river_snow_free_land', st: 279.2026, ndsi: 0.3127},
  {id: 10, lat: 64.93661, lon: -149.74807, auto_class: 'ice_free_river_snow_free_land', st: 280.2554, ndsi: -0.0217},
  {id: 11, lat: 64.88256, lon: -150.71767, auto_class: 'ice_free_river_snow_free_land', st: 281.291, ndsi: -0.1999},
  {id: 12, lat: 64.68661, lon: -149.13659, auto_class: 'ice_free_river_snow_land', st: 279.2539, ndsi: 0.8}
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

Map.setCenter(-148.9, 65.5, 8);

print('2024-04-27 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
