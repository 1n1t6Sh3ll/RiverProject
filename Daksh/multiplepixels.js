// ============================================================
// Inspect MANY candidate pixels on ONE Landsat scene — L8/L9 TOA
// Cloud label + cloud % + 4 river/land classes, for every point
// Paste into code.earthengine.google.com
// ============================================================
//
// ---- HOW TO USE ----
//
//   1. SCENE  : paste the Landsat file name exactly as downloaded,
//               e.g. 'LC09_L1TP_073018_20241127_20241128_02_T1'
//               (date, sensor and tier are read from it automatically)
//
//   2. POINTS : write each point as  [LAT, LON]
//                                     ^^^  ^^^
//                          LATITUDE FIRST, LONGITUDE SECOND
//
//      Example (Alaska):  [58.13155, -156.93412]
//          58.13155  = LATITUDE  (degrees north, positive)
//        -156.93412  = LONGITUDE (degrees west, NEGATIVE — keep the minus!)
//
//      - Comma after every point except the last one.
//      - Earth Engine internally uses [lon, lat]; the script flips it
//        for you, so do NOT flip it yourself.
//      - IDs are given automatically by list order (0, 1, 2 ...).
//      - Tip: click the map and the Console prints the clicked spot
//        already written as [LAT, LON], ready to paste into POINTS.
//
//   3. Run. Results print in the Console (one line per point).
//      CSV export: open the Tasks tab and click RUN.
// ============================================================

// ---- INPUTS ----
var SCENE = 'LC09_L1TP_073018_20241127_20241128_02_T1';

var POINTS = [
  // [LAT,      LON]
  [61.07250, -157.61926],   // 0
  [60.33264, -158.81520],   // 1
  [60.22115, -157.23412],   // 2
  [60.14007, -158.28818],   // 3
  [60.07926, -158.20372],   // 4
  [60.04209, -156.11588],   // 5
  [59.99142, -158.36250],   // 6
  [59.94412, -158.20372],   // 7
  [59.90696, -156.77128],   // 8
  [59.86304, -156.96047],   // 9
  [59.80223, -156.05169],   // 10
  [59.73804, -157.31182],   // 11
  [59.67385, -158.54155],   // 12
  [59.61642, -157.87939],   // 13
  [59.56912, -158.94358],   // 14
  [59.53534, -159.03818],   // 15
  [59.49818, -156.30169],   // 16
  [59.46101, -156.15304],   // 17
  [59.42723, -156.13615],   // 18
  [59.39007, -156.19358],   // 19
  [59.34615, -155.96385],   // 20
  [59.29885, -156.76453],   // 21
  [59.25831, -156.05507],   // 22
  [59.21101, -156.61250],   // 23
  [59.15696, -156.62264]    // 24
];

var BUFFER_M = 90;     // radius (m) for cloud % around each point
var RGB_MAX  = 1.5;    // display stretch; low winter sun needs a high max.
                       // Lower it (e.g. 0.4) for summer scenes.

// ============================================================
// Nothing below needs editing
// ============================================================

// ---- Build GEE asset ID + date from the file name ----
// LC09_L1TP_073018_20241127_20241128_02_T1
//  [0]  [1]   [2]     [3]      [4]    [5][6]
var parts   = SCENE.split('_');
var assetId = 'LANDSAT/' + parts[0] + '/C02/' + parts[6] + '_TOA/' +
              parts[0] + '_' + parts[2] + '_' + parts[3];
var DATE    = parts[3].slice(0, 4) + '-' + parts[3].slice(4, 6) + '-' +
              parts[3].slice(6, 8);

// ---- Class names ----
var CLASS_NAMES = {
  '0': 'CLASS 0 — CLOUD',
  '1': 'CLASS 1 — ice_free_river_snow_free_land',
  '2': 'CLASS 2 — ice_covered_river_snow_covered_land',
  '3': 'CLASS 3 — ice_covered_river_snow_free_land',
  '4': 'CLASS 4 — ice_free_river_snow_land'
};

// Helper for printing numbers that may be missing
function fmt(v, d) {
  return (v === null || v === undefined) ? 'NA' : Number(v).toFixed(d);
}

// ---- Sanity check: catch swapped lat/lon ----
POINTS.forEach(function(p, i) {
  if (Math.abs(p[0]) > 90) {
    print('⚠ Point ' + i + ': first number (' + p[0] + ') is not a valid ' +
          'latitude. Did you put LON first? Use [LAT, LON].');
  }
  if (p[1] > 0) {
    print('⚠ Point ' + i + ': longitude ' + p[1] + ' is positive (east). ' +
          'Alaska longitudes are negative — check the minus sign.');
  }
});

// ---- Points → FeatureCollection (flipped to [lon, lat] here) ----
var fc = ee.FeatureCollection(POINTS.map(function(p, i) {
  return ee.Feature(ee.Geometry.Point([p[1], p[0]]),
                    {id: i, lat: p[0], lon: p[1]});
}));

// ---- Load the scene (and stop cleanly if GEE doesn't have it) ----
var L = ee.Image(assetId);
var sceneOk = true;
try {
  var sunElev = L.get('SUN_ELEVATION').getInfo();
  print('Using scene:', assetId);
  print('Acquired:', DATE, '  Sun elevation:', fmt(sunElev, 1) + '°');
} catch (e) {
  sceneOk = false;
  print('❌ Scene not found in Earth Engine:', assetId);
  print('Check the SCENE file name for typos.');
}

if (sceneOk) {
  run();
}

// ============================================================
// MAIN
// ============================================================
function run() {

  Map.centerObject(fc, 8);

  // ---- Which points are inside the scene's real data footprint ----
  var footprint = L.geometry();
  print('Points INSIDE scene:',
    fc.filterBounds(footprint).aggregate_array('id'));
  print('Points OUTSIDE scene:',
    fc.filter(ee.Filter.bounds(footprint).not()).aggregate_array('id'));

  // ==========================================================
  // RGB VISUALS
  // ==========================================================
  Map.addLayer(L, {bands: ['B4', 'B3', 'B2'], min: 0.05, max: RGB_MAX},
    '1 — True Color RGB B4-B3-B2', false);
  Map.addLayer(L, {bands: ['B5', 'B4', 'B3'], min: 0.05, max: RGB_MAX},
    '2 — RGB False Color NIR B5-B4-B3', true);
  Map.addLayer(L, {bands: ['B7', 'B5', 'B3'], min: 0.05, max: RGB_MAX},
    '3 — RGB SWIR B7-B5-B3', false);
  Map.addLayer(L, {bands: ['B5', 'B5', 'B4'], min: 0.05, max: RGB_MAX},
    '4 — RGB 5-5-4 B5-B5-B4', false);

  var rgb221 = ee.Image.cat([L.select('B2'), L.select('B2'), L.select('B1')])
    .rename(['R', 'G', 'B']);
  Map.addLayer(rgb221,
    {bands: ['R', 'G', 'B'], min: 0.03, max: RGB_MAX, gamma: [1.4, 1.4, 1.2]},
    '5 — RGB 2-2-1 snow stretch', false);

  // ==========================================================
  // INDICES + THERMAL
  // ==========================================================
  var ndsi = L.normalizedDifference(['B3', 'B6']).rename('NDSI');
  var ndwi = L.normalizedDifference(['B3', 'B5']).rename('NDWI');

  Map.addLayer(ndsi, {min: -0.2, max: 0.8, palette: ['black', 'white', 'cyan']},
    '6 — NDSI snow/ice', false);
  Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
    '7 — NDWI water', false);
  Map.addLayer(L.select('B10'),
    {min: 220, max: 280, palette: ['blue', 'white', 'red']},
    '8 — Thermal B10 K', false);

  // ==========================================================
  // CLOUD MASK + CLOUD % (moving window, radius BUFFER_M)
  // ==========================================================
  // Simple cloud rule: cold thermal + bright blue reflectance
  var cloudMask = L.select('B10').lt(260)
    .and(L.select('B2').gt(0.2))
    .rename('cloud');

  // For every pixel: % of cloud pixels within BUFFER_M metres
  var cloudPct = cloudMask.focalMean(BUFFER_M, 'circle', 'meters')
    .multiply(100)
    .rename('cloud_pct');

  Map.addLayer(cloudMask.selfMask(), {palette: ['white']}, 'Cloud mask', false);

  // ==========================================================
  // CLASSIFICATION IMAGE — every pixel gets CLASS 0–4
  // ==========================================================
  var frozen = L.select('B10').lt(273).rename('frozen');   // river ice test
  var snow   = ndsi.gt(0.4).rename('snow');                // land snow test
  var water  = ndwi.gt(0).rename('water_like');

  var classImg = L.select('B10').multiply(0).add(1).toInt()   // CLASS 1 default
    .where(frozen.and(snow), 2)                                // CLASS 2
    .where(frozen.and(snow.not()), 3)                          // CLASS 3
    .where(frozen.not().and(snow), 4)                          // CLASS 4
    .where(cloudMask, 0)                                       // CLASS 0 wins
    .updateMask(ndsi.mask())
    .rename('class');

  Map.addLayer(classImg,
    {min: 0, max: 4, palette: ['gray', 'green', 'white', 'purple', 'cyan']},
    '10 — CLASS map (gray=0 green=1 white=2 purple=3 cyan=4)', false);

  // Everything we want to read at each point, in one image
  var stack = ee.Image.cat([
    L.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']),
    ndsi, ndwi, cloudMask, cloudPct, frozen, snow, water, classImg
  ]);

  // ==========================================================
  // SAMPLE ALL POINTS AT ONCE
  // ==========================================================
  var names = ee.Dictionary(CLASS_NAMES);

  var results = stack.reduceRegions({
    collection: fc,
    reducer: ee.Reducer.first(),
    scale: 30
  }).map(function(f) {
    var noData = ee.Algorithms.IsEqual(f.get('class'), null);
    return f.set({
      label: ee.Algorithms.If(noData,
        'NO DATA — outside scene or masked',
        names.get(ee.Number(f.get('class')).format('%d'))),
      date: DATE,
      scene: SCENE
    });
  });

  // ---- VIIRS I1 on the same day as the Landsat scene ----
  var day0 = ee.Date(DATE);
  var viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
    .filterBounds(fc)
    .filterDate(day0, day0.advance(1, 'day'));
  var hasViirs = viirs.size().getInfo() > 0;

  if (hasViirs) {
    var V = viirs.mosaic();
    Map.addLayer(V, {bands: ['I1', 'I1', 'I1'], min: 0, max: 6000},
      '9 — VIIRS I1 (same day)', false);
    results = V.select('I1').reduceRegions({
      collection: results,
      reducer: ee.Reducer.first().setOutputs(['VIIRS_I1']),
      scale: 375
    });
  } else {
    print('No VIIRS VNP09GA image on', DATE);
  }

  results = results.sort('id');

  // ---- Readable one-line-per-point summary ----
  print('=== RESULTS (one line per point) ===');
  results.evaluate(function(res, err) {
    if (err) { print('Error:', err); return; }
    res.features.forEach(function(f) {
      var p = f.properties;
      print('#' + p.id + '  [' + p.lat + ', ' + p.lon + ']  →  ' + p.label +
        '  | B10 ' + fmt(p.B10, 1) + ' K' +
        '  NDSI ' + fmt(p.NDSI, 3) +
        '  NDWI ' + fmt(p.NDWI, 3) +
        '  cloud ' + fmt(p.cloud_pct, 0) + '% within ' + BUFFER_M + ' m' +
        (hasViirs ? '  VIIRS_I1 ' + fmt(p.VIIRS_I1, 0) : ''));
    });
  });

  print('Full details per point (expand "features"):', results);

  // ---- Export as CSV (run it from the Tasks tab) ----
  var cols = ['id', 'lat', 'lon', 'date', 'scene', 'label', 'class',
              'cloud_pct', 'cloud', 'frozen', 'snow', 'water_like',
              'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'NDSI', 'NDWI'];
  if (hasViirs) cols.push('VIIRS_I1');

  Export.table.toDrive({
    collection: results,
    description: 'points_' + parts[0] + '_' + parts[2] + '_' + parts[3],
    fileFormat: 'CSV',
    selectors: cols
  });

  // ==========================================================
  // CLICK INSPECTOR — one pixel, prints it as [LAT, LON]
  // ==========================================================
  Map.onClick(function(c) {
    var clickPt = ee.Geometry.Point([c.lon, c.lat]);
    stack.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: clickPt,
      scale: 30
    }).evaluate(function(v) {
      var latlon = '[' + c.lat.toFixed(5) + ', ' + c.lon.toFixed(5) + ']';
      print('--- Click ---  ' + latlon + '   ← paste into POINTS as-is');
      if (!v || v['class'] === null || v['class'] === undefined) {
        print('NO DATA here (outside scene or masked)');
        return;
      }
      print('→ ' + CLASS_NAMES[String(v['class'])] +
        '  | B10 ' + fmt(v.B10, 1) + ' K' +
        '  NDSI ' + fmt(v.NDSI, 3) +
        '  NDWI ' + fmt(v.NDWI, 3) +
        '  cloud ' + fmt(v.cloud_pct, 0) + '% within ' + BUFFER_M + ' m');
      print('values:', v);
    });
  });

  // ==========================================================
  // FOOTPRINT + POINTS — drawn last so they stay on top
  // ==========================================================
  Map.addLayer(ee.FeatureCollection([ee.Feature(footprint)])
    .style({color: 'cyan', fillColor: '00000000', width: 2}),
    {}, 'Scene footprint', true);

  var buffers = fc.map(function(f) { return f.buffer(BUFFER_M); });
  Map.addLayer(buffers.style({color: 'red', fillColor: '00000000', width: 2}),
    {}, 'Point buffers (' + BUFFER_M + ' m)', true);
  Map.addLayer(fc.style({color: 'yellow', pointSize: 5}),
    {}, 'Candidate points', true);
}
