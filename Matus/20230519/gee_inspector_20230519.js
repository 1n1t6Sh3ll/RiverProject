/*
 * GEE Visual Inspector - 2023-05-19
 * Landsat scene: LC90710152023139LGN00
 *
 * Paste this into https://code.earthengine.google.com/
 * Click Inspector tab (top-right), then click any marker to see properties.
 */

// ── Load and scale Landsat scene ────────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90710152023139LGN00'))
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
  {id: 1, lat: 65.1927, lon: -151.46696, auto_class: 'ice_free_river_snow_land', st: 273.0297, ndsi: 0.525},
  {id: 2, lat: 65.16905, lon: -152.09871, auto_class: 'ice_free_river_snow_land', st: 274.7831, ndsi: 0.4936},
  {id: 3, lat: 65.12175, lon: -152.07506, auto_class: 'ice_free_river_snow_free_land', st: 286.2608, ndsi: -0.5108},
  {id: 4, lat: 65.08459, lon: -151.64939, auto_class: 'ice_free_river_snow_free_land', st: 295.6228, ndsi: -0.5678},
  {id: 5, lat: 64.98662, lon: -151.30479, auto_class: 'ice_free_river_snow_land', st: 277.3091, ndsi: 0.735},
  {id: 6, lat: 64.88864, lon: -151.18993, auto_class: 'ice_free_river_snow_land', st: 282.8292, ndsi: 0.8568},
  {id: 7, lat: 64.85824, lon: -151.31831, auto_class: 'ice_free_river_snow_land', st: 282.8223, ndsi: 0.8574},
  {id: 8, lat: 64.81094, lon: -149.83858, auto_class: 'ice_free_river_snow_land', st: 283.4683, ndsi: 0.8201},
  {id: 9, lat: 64.74337, lon: -149.00074, auto_class: 'ice_free_river_snow_free_land', st: 297.8548, ndsi: -0.692},
  {id: 10, lat: 64.6454, lon: -148.58182, auto_class: 'ice_free_river_snow_free_land', st: 300.5277, ndsi: -0.5279},
  {id: 11, lat: 64.5677, lon: -149.08858, auto_class: 'ice_free_river_snow_land', st: 287.8844, ndsi: 0.562},
  {id: 12, lat: 64.42243, lon: -151.01088, auto_class: 'ice_free_river_snow_free_land', st: 280.3579, ndsi: 0.0216},
  {id: 13, lat: 64.22986, lon: -150.62912, auto_class: 'ice_free_river_snow_free_land', st: 292.2526, ndsi: -0.4158},
  {id: 14, lat: 64.03729, lon: -151.64601, auto_class: 'ice_free_river_snow_free_land', st: 290.6974, ndsi: -0.668},
  {id: 15, lat: 63.95283, lon: -151.62574, auto_class: 'ice_free_river_snow_land', st: 287.2008, ndsi: 0.4573},
  {id: 16, lat: 63.90216, lon: -152.11898, auto_class: 'ice_free_river_snow_land', st: 280.3067, ndsi: 0.7084},
  {id: 17, lat: 63.80756, lon: -151.81155, auto_class: 'ice_free_river_snow_free_land', st: 289.9728, ndsi: -0.4793},
  {id: 18, lat: 63.64202, lon: -152.2102, auto_class: 'ice_free_river_snow_free_land', st: 297.8548, ndsi: -0.6324},
  {id: 19, lat: 63.55756, lon: -152.44331, auto_class: 'ice_free_river_snow_free_land', st: 290.8068, ndsi: -0.5125}
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

Map.setCenter(-152.0, 64.4, 8);

print('2023-05-19 - ' + candidates.length + ' confirmed pixels');
candidates.forEach(function(c) {
  print('#' + c.id + ' (' + c.lat + ', ' + c.lon + ') ' + c.auto_class);
});
