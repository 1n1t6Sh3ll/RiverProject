// ============================================================
// Inspect candidate pixel — FULL VERSION
// Added cloud label + cloud percentage
// ============================================================

// ---- Inputs ----
var LAT = 58.13155;
var LON = -156.93412;

var DATE = '2024-11-29';

var pt = ee.Geometry.Point([LON, LAT]);
var roi = pt.buffer(90);

Map.setCenter(LON, LAT, 12);

// ---- Date window ±15 days ----
var d0 = ee.Date(DATE).advance(-15, 'day');
var d1 = ee.Date(DATE).advance(15, 'day');

// ---- Build collection ----
function buildCol(geom) {
  return ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
    .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA'))
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T2_TOA'))
    .merge(ee.ImageCollection('LANDSAT/LC08/C02/T2_TOA'))
    .filterBounds(geom)
    .filterDate(d0, d1);
}

var col = buildCol(pt);
var n = col.size();

print('count at point:', n);

col = ee.ImageCollection(ee.Algorithms.If(
  n.gt(0),
  col,
  buildCol(pt.buffer(5000))
));

print('count with fallback:', col.size());
print('ids:', col.aggregate_array('system:index'));
print('dates:', col.aggregate_array('system:time_start')
  .map(function(t) {
    return ee.Date(t).format('YYYY-MM-dd');
  })
);

// ---- Pick closest scene ----
var target = ee.Date(DATE).millis();

var L = ee.Image(
  col.map(function(img) {
    return img.set(
      'dt',
      ee.Number(img.get('system:time_start')).subtract(target).abs()
    );
  }).sort('dt').first()
);

print('chosen scene:', L.get('system:index'), L.date().format('YYYY-MM-dd'));
print('band names:', L.bandNames());

// ============================================================
// RGB VISUALS
// ============================================================

Map.addLayer(
  L,
  {bands: ['B4', 'B3', 'B2'], min: 0.05, max: 0.35},
  '1 — True Color RGB B4-B3-B2',
  false
);

Map.addLayer(
  L,
  {bands: ['B5', 'B4', 'B3'], min: 0.05, max: 0.45},
  '2 — RGB False Color NIR B5-B4-B3',
  true
);

Map.addLayer(
  L,
  {bands: ['B7', 'B5', 'B3'], min: 0.05, max: 0.50},
  '3 — RGB SWIR B7-B5-B3',
  false
);

Map.addLayer(
  L,
  {bands: ['B5', 'B5', 'B4'], min: 0.05, max: 0.50},
  '4 — RGB 5-5-4 B5-B5-B4',
  false
);

var rgb221 = ee.Image.cat([
  L.select('B2'),
  L.select('B2'),
  L.select('B1')
]).rename(['R', 'G', 'B']);

Map.addLayer(
  rgb221,
  {
    bands: ['R', 'G', 'B'],
    min: 0.03,
    max: 0.35,
    gamma: [1.4, 1.4, 1.2]
  },
  '5 — RGB 2-2-1 fixed snow stretch',
  false
);

// ============================================================
// INDICES + THERMAL
// ============================================================

var ndsi = L.normalizedDifference(['B3', 'B6']).rename('NDSI');
var ndwi = L.normalizedDifference(['B3', 'B5']).rename('NDWI');

Map.addLayer(
  ndsi,
  {min: -0.2, max: 0.8, palette: ['black', 'white', 'cyan']},
  '6 — NDSI snow/ice',
  false
);

Map.addLayer(
  ndwi,
  {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
  '7 — NDWI water',
  false
);

Map.addLayer(
  L.select('B10'),
  {min: 220, max: 280, palette: ['blue', 'white', 'red']},
  '8 — Thermal B10 K',
  false
);

// ============================================================
// CLOUD MASK + CLOUD PERCENTAGE
// ============================================================

// Simple cloud rule:
// cold thermal + bright blue reflectance
var cloudMask = L.select('B10').lt(260)
  .and(L.select('B2').gt(0.2))
  .rename('cloud');

Map.addLayer(
  cloudMask.selfMask(),
  {palette: ['white']},
  'Cloud mask',
  false
);

// Cloud percentage inside candidate buffer
var cloudStats = cloudMask.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 30,
  maxPixels: 1e9
});

var cloudPercent = ee.Number(cloudStats.get('cloud')).multiply(100);

print('Cloud percentage inside 90 m candidate buffer:', cloudPercent);

// ============================================================
// SAMPLE PIXEL
// ============================================================

var vals = L
  .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'])
  .addBands(ndsi)
  .addBands(ndwi)
  .addBands(cloudMask)
  .reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: pt,
    scale: 30
  });

print('--- Pixel values ---');
print(vals);

// ============================================================
// CLASSIFICATION — CLOUD + 4 LAND/RIVER LABELS
// ============================================================

var b2 = ee.Number(vals.get('B2'));
var b10 = ee.Number(vals.get('B10'));
var ndsiVal = ee.Number(vals.get('NDSI'));
var ndwiVal = ee.Number(vals.get('NDWI'));

var isCloud = ee.Number(vals.get('cloud')).eq(1);

var riverFrozen = b10.lt(273);
var landSnow = ndsiVal.gt(0.4);

var class1 = riverFrozen.not().and(landSnow.not());
var class2 = riverFrozen.and(landSnow);
var class3 = riverFrozen.and(landSnow.not());
var class4 = riverFrozen.not().and(landSnow);

var label = ee.String(
  ee.Algorithms.If(isCloud,
    'CLASS 0 — CLOUD',
    ee.Algorithms.If(class1,
      'CLASS 1 — ice_free_river_snow_free_land',
      ee.Algorithms.If(class2,
        'CLASS 2 — ice_covered_river_snow_covered_land',
        ee.Algorithms.If(class3,
          'CLASS 3 — ice_covered_river_snow_free_land',
          ee.Algorithms.If(class4,
            'CLASS 4 — ice_free_river_snow_land',
            'UNKNOWN'
          )
        )
      )
    )
  )
);

print('==============================');
print('B2 reflectance:', b2);
print('B10 thermal K:', b10);
print('NDSI:', ndsiVal);
print('NDWI:', ndwiVal);
print('cloud? B10 < 260K and B2 > 0.2:', isCloud);
print('cloud percentage in 90 m buffer:', cloudPercent);
print('river frozen? B10 < 273K:', riverFrozen);
print('land snow? NDSI > 0.4:', landSnow);
print('water-like? NDWI > 0:', ndwiVal.gt(0));
print('FINAL CLASS:', label);
print('==============================');

// ============================================================
// VIIRS same-day layer
// ============================================================

var V = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
  .filterBounds(pt)
  .filterDate(ee.Date(DATE), ee.Date(DATE).advance(1, 'day'))
  .first();

Map.addLayer(
  V,
  {bands: ['I1', 'I1', 'I1'], min: 0, max: 6000},
  '9 — VIIRS I1 same day',
  false
);

print('VIIRS I1 at pt:', V.select('I1').reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: pt,
  scale: 375
}));

// ============================================================
// CLICK INSPECTOR
// ============================================================

Map.onClick(function(coords) {
  var clickPt = ee.Geometry.Point([coords.lon, coords.lat]);
  var clickRoi = clickPt.buffer(90);

  var cVals = L
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'])
    .addBands(ndsi)
    .addBands(ndwi)
    .addBands(cloudMask)
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: clickPt,
      scale: 30
    });

  var cB2 = ee.Number(cVals.get('B2'));
  var cB10 = ee.Number(cVals.get('B10'));
  var cNDSI = ee.Number(cVals.get('NDSI'));
  var cNDWI = ee.Number(cVals.get('NDWI'));

  var cCloud = ee.Number(cVals.get('cloud')).eq(1);

  var cFrozen = cB10.lt(273);
  var cSnow = cNDSI.gt(0.4);

  var cClass1 = cFrozen.not().and(cSnow.not());
  var cClass2 = cFrozen.and(cSnow);
  var cClass3 = cFrozen.and(cSnow.not());
  var cClass4 = cFrozen.not().and(cSnow);

  var cLabel = ee.String(
    ee.Algorithms.If(cCloud,
      'CLASS 0 — CLOUD',
      ee.Algorithms.If(cClass1,
        'CLASS 1 — ice_free_river_snow_free_land',
        ee.Algorithms.If(cClass2,
          'CLASS 2 — ice_covered_river_snow_covered_land',
          ee.Algorithms.If(cClass3,
            'CLASS 3 — ice_covered_river_snow_free_land',
            ee.Algorithms.If(cClass4,
              'CLASS 4 — ice_free_river_snow_land',
              'UNKNOWN'
            )
          )
        )
      )
    )
  );

  var cCloudPercent = ee.Number(
    cloudMask.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: clickRoi,
      scale: 30,
      maxPixels: 1e9
    }).get('cloud')
  ).multiply(100);

  print('--- Click ---');
  print('lat:', coords.lat.toFixed(5), 'lon:', coords.lon.toFixed(5));
  print('values:', cVals);
  print('B2 reflectance:', cB2);
  print('B10 K:', cB10);
  print('NDSI:', cNDSI);
  print('NDWI:', cNDWI);
  print('cloud?:', cCloud);
  print('cloud percentage in clicked 90 m buffer:', cCloudPercent);
  print('CLASS:', cLabel);
});

// ============================================================
// CANDIDATE POINT — DRAW LAST SO IT STAYS ON TOP
// ============================================================

Map.layers().add(
  ui.Map.Layer(
    roi,
    {color: 'red'},
    'candidate point buffer — TOP',
    true
  )
);

Map.layers().add(
  ui.Map.Layer(
    pt,
    {color: 'yellow'},
    'candidate exact point — TOP',
    true
  )
);
