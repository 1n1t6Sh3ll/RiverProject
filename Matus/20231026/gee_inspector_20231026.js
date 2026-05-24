/*
 * GEE Visual Inspector — 2023-10-26
 * Landsat 9 scene: LC90710162023299LGN00 (Path 71, Row 16)
 *
 * Paste this into https://code.earthengine.google.com/
 * It will load the scene with RGB, False Color NIR, Thermal, and NDSI layers,
 * and place numbered markers at each candidate pixel for visual inspection.
 *
 * Click any marker → Inspector panel shows all band values at that point.
 */

// ── Load and scale Landsat 9 scene ──────────────────────────────────────────
var scene = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filter(ee.Filter.eq('LANDSAT_SCENE_ID', 'LC90710162023299LGN00'))
  .first();

// Apply scaling factors
var scaled = scene
  .select('SR_B.').multiply(0.0000275).add(-0.2)
  .addBands(scene.select('ST_B10').multiply(0.00341802).add(149.0));

// Derived indices
var ndsi = scaled.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var full = scaled.addBands(ndsi).addBands(ndwi);

// ── Add map layers ──────────────────────────────────────────────────────────
// True Color RGB
Map.addLayer(full, {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0, max: 0.3
}, 'RGB (True Color)', true);

// False Color NIR (vegetation=red, water=dark, snow/ice=cyan)
Map.addLayer(full, {
  bands: ['SR_B5', 'SR_B4', 'SR_B3'],
  min: 0, max: 0.4
}, 'False Color NIR', false);

// Thermal (ST_B10) — purple=cold/frozen, yellow=warm/liquid
Map.addLayer(full, {
  bands: ['ST_B10'],
  min: 250, max: 290,
  palette: ['purple', 'blue', 'cyan', 'yellow', 'red']
}, 'Thermal (ST_B10)', false);

// NDSI — high values = snow/ice
Map.addLayer(ndsi, {
  min: -0.5, max: 1.0,
  palette: ['brown', 'white', 'cyan']
}, 'NDSI', false);

// Cloud mask visualization (QA_PIXEL)
var qa = scene.select('QA_PIXEL');
var cloudMask = qa.bitwiseAnd(1 << 3).neq(0)  // cloud
  .or(qa.bitwiseAnd(1 << 4).neq(0));           // cloud shadow
Map.addLayer(cloudMask.selfMask(), {
  palette: ['red']
}, 'Cloud Mask (red = cloud/shadow)', false);

// ── Candidate pixels from training CSV ──────────────────────────────────────
var candidates = [
  {id: 1,  lat: 63.68074, lon: -152.11993, auto_class: 'ice_covered_river_snow_covered_land', st: 259.64, ndsi: 0.7164},
  {id: 2,  lat: 63.55236, lon: -151.33953, auto_class: 'ice_covered_river_snow_covered_land', st: 266.89, ndsi: 0.7813},
  {id: 3,  lat: 63.43412, lon: -151.09628, auto_class: 'ice_covered_river_snow_covered_land', st: 266.13, ndsi: 0.874},
  {id: 4,  lat: 63.36655, lon: -151.14020, auto_class: 'ice_covered_river_snow_covered_land', st: 268.53, ndsi: 0.9186},
  {id: 5,  lat: 62.86655, lon: -151.94426, auto_class: 'ice_covered_river_snow_covered_land', st: 268.37, ndsi: 0.8936},
  {id: 6,  lat: 62.79561, lon: -150.65709, auto_class: 'ice_covered_river_snow_covered_land', st: 268.56, ndsi: 0.9453},
  {id: 7,  lat: 62.72128, lon: -150.24831, auto_class: 'ice_free_river_snow_free_land',       st: 274.75, ndsi: -0.3035},
  {id: 8,  lat: 62.63682, lon: -150.43750, auto_class: 'ice_free_river_snow_free_land',       st: 274.48, ndsi: -0.6335},
  {id: 9,  lat: 62.21791, lon: -153.42736, auto_class: 'ice_free_river_snow_free_land',       st: 277.24, ndsi: -0.5012},
  {id: 10, lat: 62.07939, lon: -151.47466, auto_class: 'ice_free_river_snow_free_land',       st: 273.11, ndsi: -0.607},
  {id: 11, lat: 61.96791, lon: -152.24155, auto_class: 'ice_free_river_snow_free_land',       st: 275.26, ndsi: -0.5764},
];

// ── Place markers on map ────────────────────────────────────────────────────
// Add as raw FeatureCollection (NOT .style()) so Inspector click works.
// Separate layers per class for color coding.

var iceSnow = [];
var iceNoSnow = [];
var freeNoSnow = [];
var freeSnow = [];

candidates.forEach(function(c) {
  var feat = ee.Feature(
    ee.Geometry.Point([c.lon, c.lat]),
    {
      'Pixel': c.id,
      'Auto_Class': c.auto_class,
      'ST_B10_K': c.st,
      'NDSI': c.ndsi
    }
  );
  if (c.auto_class === 'ice_covered_river_snow_covered_land') iceSnow.push(feat);
  else if (c.auto_class === 'ice_covered_river_snow_free_land') iceNoSnow.push(feat);
  else if (c.auto_class === 'ice_free_river_snow_free_land') freeNoSnow.push(feat);
  else freeSnow.push(feat);
});

if (iceSnow.length > 0)
  Map.addLayer(ee.FeatureCollection(iceSnow), {color: 'FF0000'}, '🔴 ice_covered + snow_covered (' + iceSnow.length + ')', true);
if (iceNoSnow.length > 0)
  Map.addLayer(ee.FeatureCollection(iceNoSnow), {color: 'FF8800'}, '🟠 ice_covered + snow_free (' + iceNoSnow.length + ')', true);
if (freeNoSnow.length > 0)
  Map.addLayer(ee.FeatureCollection(freeNoSnow), {color: '00FF00'}, '🟢 ice_free + snow_free (' + freeNoSnow.length + ')', true);
if (freeSnow.length > 0)
  Map.addLayer(ee.FeatureCollection(freeSnow), {color: '0088FF'}, '🔵 ice_free + snow (' + freeSnow.length + ')', true);

// ── Center map on the candidates ────────────────────────────────────────────
Map.setCenter(-151.5, 63.0, 8);

// ── Print summary to console ────────────────────────────────────────────────
print('═══════════════════════════════════════');
print('Training Pixel Inspector — 2023-10-26');
print('Landsat 9: LC90710162023299LGN00');
print('═══════════════════════════════════════');
print('');
print('Layers available (toggle in Layers panel):');
print('  ✓ RGB (True Color) — ON by default');
print('  ○ False Color NIR — vegetation=red, water=dark');
print('  ○ Thermal (ST_B10) — purple=frozen, yellow=warm');
print('  ○ NDSI — white/cyan = snow/ice');
print('  ○ Cloud Mask — red blocks = clouds');
print('');
print('Click any × marker to see pixel info in Inspector panel.');
print('');
print('Marker colors:');
print('  🔴 Red    = ice_covered_river_snow_covered_land (auto)');
print('  🟠 Orange = ice_covered_river_snow_free_land (auto)');
print('  🟢 Green  = ice_free_river_snow_free_land (auto)');
print('  🔵 Blue   = ice_free_river_snow_land (auto)');
print('');
candidates.forEach(function(c) {
  print('Pixel #' + c.id + ': (' + c.lat + ', ' + c.lon + ') → ' +
        c.auto_class + ' | ST=' + c.st + 'K NDSI=' + c.ndsi);
});
