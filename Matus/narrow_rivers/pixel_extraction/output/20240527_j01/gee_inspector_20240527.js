/*
 * GEE Visual Inspector - 2024-05-27 - Sagavanirktok River
 * Landsat scene: LC08_L2SP_073011_20240527_20240611_02_T1
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Basemap: Google high-res satellite (toggle via Map/Satellite top-right) ─
Map.setOptions('HYBRID');

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', 'LC08_L2SP_073011_20240527_20240611_02_T1'))
  .first();

var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Sentinel-2 10m (least-cloudy pass within +/- 7 days of scene date) ─────
var corridorBbox = ee.Geometry.Rectangle(
  [-148.91554054054055, 68.6891891891892, -147.92567567567568, 70.22972972972973]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(corridorBbox)
  .filterDate('2024-05-20', '2024-06-03')
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

// ── Map layers ──────────────────────────────────────────────────────────────
Map.addLayer(full, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}, 'RGB (True Color)', true);
Map.addLayer(full, {bands: ['SR_B5', 'SR_B4', 'SR_B3'], min: 0, max: 0.4}, 'False Color NIR', false);
Map.addLayer(full, {bands: ['SR_B6', 'SR_B5', 'SR_B4'], min: 0, max: 0.5}, 'SWIR false colour (snow/ice = cyan, cloud = white)', false);
Map.addLayer(full, {bands: ['ST_B10'], min: 250, max: 290, palette: ['purple','blue','cyan','yellow','red']}, 'Thermal (ST_B10)', false);
Map.addLayer(ndsi, {min: -0.5, max: 1.0, palette: ['brown','white','cyan']}, 'NDSI', false);
Map.addLayer(s2, {bands: ['B4','B3','B2'], min: 0, max: 3000}, 'Sentinel-2 10m (nearest pass)', false);

var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
Map.addLayer(cloudMask.selfMask(), {palette: ['red']}, 'Cloud Mask', false);

// ── Candidate pixels (with 375 m VIIRS cell outline) ────────────────────────
var candidates = [
  {id: 1, lat: 69.97128, lon: -148.68412, auto_class: 'ice_covered_river_snow_covered_land', st: 271.1361, ndsi: 0.7175, wf: 0.5582},
  {id: 2, lat: 69.92736, lon: -148.67061, auto_class: 'ice_free_river_snow_free_land', st: 277.7329, ndsi: 0.2008, wf: 0.1656},
  {id: 3, lat: 69.86993, lon: -148.71791, auto_class: 'ice_free_river_snow_covered_land', st: 275.7299, ndsi: 0.7508, wf: 0.752},
  {id: 4, lat: 70.07601, lon: -148.48818, auto_class: 'ice_covered_river_snow_covered_land', st: 269.1707, ndsi: 0.6089, wf: 0.2338},
  {id: 5, lat: 70.04899, lon: -148.57601, auto_class: 'ice_covered_river_snow_covered_land', st: 270.3021, ndsi: 0.4322, wf: 0.695},
  {id: 6, lat: 70.08277, lon: -148.33277, auto_class: 'ice_covered_river_snow_covered_land', st: 270.4559, ndsi: 0.8773, wf: 0.1248},
  {id: 7, lat: 69.99831, lon: -148.67399, auto_class: 'ice_covered_river_snow_covered_land', st: 271.8881, ndsi: 0.662, wf: 0.1858},
  {id: 8, lat: 69.7348, lon: -148.64696, auto_class: 'ice_covered_river_snow_covered_land', st: 271.9462, ndsi: 0.8625, wf: 0.454},
  {id: 9, lat: 70.20101, lon: -147.98818, auto_class: 'ice_covered_river_snow_covered_land', st: 270.9652, ndsi: 0.8439, wf: 0.4216},
  {id: 10, lat: 70.17736, lon: -148.04223, auto_class: 'ice_covered_river_snow_covered_land', st: 270.3465, ndsi: 0.7627, wf: 0.1003},
  {id: 11, lat: 69.64696, lon: -148.65034, auto_class: 'ice_free_river_snow_free_land', st: 276.1709, ndsi: 0.0474, wf: 0.0862},
  {id: 12, lat: 70.11993, lon: -148.15709, auto_class: 'ice_covered_river_snow_covered_land', st: 269.3827, ndsi: 0.7898, wf: 0.0759},
  {id: 13, lat: 70.20101, lon: -148.07264, auto_class: 'ice_covered_river_snow_covered_land', st: 269.0067, ndsi: 0.6728, wf: 0.1263},
  {id: 14, lat: 70.1875, lon: -148.17399, auto_class: 'ice_covered_river_snow_covered_land', st: 271.5668, ndsi: 0.9022, wf: 0.1684},
  {id: 15, lat: 70.16047, lon: -148.23818, auto_class: 'ice_covered_river_snow_covered_land', st: 270.1517, ndsi: 0.7435, wf: 0.2642},
  {id: 16, lat: 69.45777, lon: -148.53547, auto_class: 'ice_free_river_snow_covered_land', st: 274.6874, ndsi: 0.8729, wf: 0.3082},
  {id: 17, lat: 69.40034, lon: -148.63007, auto_class: 'ice_free_river_snow_covered_land', st: 273.0844, ndsi: 0.9091, wf: 0.3288},
  {id: 18, lat: 69.30912, lon: -148.6875, auto_class: 'ice_free_river_snow_covered_land', st: 276.0375, ndsi: 0.5032, wf: 0.1077},
  {id: 19, lat: 69.20777, lon: -148.78209, auto_class: 'ice_free_river_snow_covered_land', st: 273.6278, ndsi: 0.8618, wf: 0.4495},
  {id: 20, lat: 69.09291, lon: -148.77196, auto_class: 'ice_free_river_snow_free_land', st: 288.0621, ndsi: -0.5245, wf: 0.1235},
  {id: 21, lat: 68.93074, lon: -148.82939, auto_class: 'ice_free_river_snow_covered_land', st: 274.9233, ndsi: 0.4867, wf: 0.2086},
  {id: 22, lat: 68.78547, lon: -148.75507, auto_class: 'ice_free_river_snow_free_land', st: 279.1958, ndsi: -0.1166, wf: 0.1391},
  {id: 23, lat: 68.70777, lon: -148.52534, auto_class: 'ice_free_river_snow_free_land', st: 286.8522, ndsi: -0.0314, wf: 0.0576}
];

var half = 0.0016891891891891893;
var iceSnow = [], iceNoSnow = [], freeNoSnow = [], freeSnow = [], unknown = [], cells = [];
candidates.forEach(function(c) {
  var feat = ee.Feature(ee.Geometry.Point([c.lon, c.lat]),
    {'Pixel': c.id, 'Auto_Class': c.auto_class, 'ST_B10_K': c.st, 'NDSI': c.ndsi, 'JRC_wf': c.wf});
  cells.push(ee.Feature(ee.Geometry.Rectangle([c.lon - half, c.lat - half, c.lon + half, c.lat + half])));
  if (c.auto_class === 'ice_covered_river_snow_covered_land') iceSnow.push(feat);
  else if (c.auto_class === 'ice_covered_river_snow_free_land') iceNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_free_land') freeNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_covered_land') freeSnow.push(feat);
  else unknown.push(feat);
});

Map.addLayer(ee.FeatureCollection(cells).style({color: 'FFFF00', fillColor: '00000000', width: 1}), {}, 'VIIRS 375 m cells', true);
if (iceSnow.length > 0)   Map.addLayer(ee.FeatureCollection(iceSnow),   {color: 'FF0000'}, 'ice_covered+snow_covered (' + iceSnow.length + ')', true);
if (iceNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(iceNoSnow), {color: 'FF8800'}, 'ice_covered+snow_free (' + iceNoSnow.length + ')', true);
if (freeNoSnow.length > 0) Map.addLayer(ee.FeatureCollection(freeNoSnow), {color: '00FF00'}, 'ice_free+snow_free (' + freeNoSnow.length + ')', true);
if (freeSnow.length > 0)  Map.addLayer(ee.FeatureCollection(freeSnow),  {color: '0088FF'}, 'ice_free+snow_covered (' + freeSnow.length + ')', true);
if (unknown.length > 0)   Map.addLayer(ee.FeatureCollection(unknown),   {color: 'AAAAAA'}, 'unlabeled / legacy class (' + unknown.length + ')', true);

Map.setCenter(-148.51, 69.71, 8);

print('2024-05-27 - Sagavanirktok River - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
