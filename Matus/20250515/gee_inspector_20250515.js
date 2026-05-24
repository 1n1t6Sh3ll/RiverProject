/*
 * GEE Visual Inspector - 2025-05-15
 * Landsat scene: LC90720142025135LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90720142025135LGN00'))
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
  {id: 1, lat: 66.55892, lon: -151.5786, auto_class: 'ice_free_river_snow_free_land', st: 280.1904, ndsi: -0.2211},
  {id: 2, lat: 66.48459, lon: -152.24752, auto_class: 'ice_free_river_snow_land', st: 276.5537, ndsi: 0.9133},
  {id: 3, lat: 66.29878, lon: -152.31171, auto_class: 'ice_free_river_snow_free_land', st: 280.7954, ndsi: -0.4811},
  {id: 4, lat: 66.19068, lon: -151.59211, auto_class: 'ice_free_river_snow_free_land', st: 281.7456, ndsi: -0.5054},
  {id: 5, lat: 65.98797, lon: -148.93333, auto_class: 'ice_free_river_snow_free_land', st: 276.3964, ndsi: 0.2851},
  {id: 6, lat: 65.79541, lon: -152.93671, auto_class: 'ice_free_river_snow_free_land', st: 283.1402, ndsi: -0.2543},
  {id: 7, lat: 65.47108, lon: -150.67319, auto_class: 'ice_free_river_snow_free_land', st: 277.9209, ndsi: 0.0957},
  {id: 8, lat: 65.17716, lon: -151.62927, auto_class: 'ice_free_river_snow_land', st: 273.7987, ndsi: 0.9308},
  {id: 9, lat: 65.15351, lon: -152.57184, auto_class: 'ice_free_river_snow_free_land', st: 285.4781, ndsi: -0.2828},
  {id: 10, lat: 65.13324, lon: -152.19008, auto_class: 'ice_free_river_snow_free_land', st: 289.5182, ndsi: -0.5386},
  {id: 11, lat: 65.10284, lon: -153.7036, auto_class: 'ice_free_river_snow_free_land', st: 283.0479, ndsi: 0.1721},
  {id: 12, lat: 64.99473, lon: -151.62589, auto_class: 'ice_free_river_snow_free_land', st: 284.3092, ndsi: -0.4701},
  {id: 13, lat: 64.93054, lon: -151.08873, auto_class: 'ice_free_river_snow_land', st: 282.9419, ndsi: 0.5108}
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

Map.setCenter(-151.3, 65.7, 8);

print('2025-05-15 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
