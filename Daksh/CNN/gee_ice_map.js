// ============================================================
// River Ice Classification Map — Google Earth Engine
// ============================================================
// SETUP:
//   1. Upload ice_predictions.geojson as a GEE Table asset
//   2. Replace YOUR_USERNAME below with your GEE username
//   3. Paste this entire script into the GEE Code Editor and Run
// ============================================================

var predictions = ee.FeatureCollection('users/YOUR_USERNAME/ice_predictions');

// ── Class colours ────────────────────────────────────────────
var PALETTE = {
  0: '#1a6faf',   // ice_covered / snow_covered_land  → dark blue
  1: '#74c2e1',   // ice_covered / snow_free_land     → light blue
  2: '#2ecc71',   // ice_free / snow_free_land        → green
  3: '#f1c40f',   // ice_free / snow_land             → yellow
};

var CLASS_NAMES = [
  'Ice covered / Snow covered land',
  'Ice covered / Snow free land',
  'Ice free / Snow free land',
  'Ice free / Snow land',
];

// ── Satellite basemap (HYBRID = satellite imagery + place/river labels) ──
Map.setOptions('HYBRID');

// ── Style each point by predicted class ──────────────────────
function stylePoint(feature) {
  var classId = feature.get('class_id');
  var correct = feature.get('correct');

  // colour lookup
  var color = ee.Algorithms.If(ee.Number(classId).eq(0), PALETTE[0],
              ee.Algorithms.If(ee.Number(classId).eq(1), PALETTE[1],
              ee.Algorithms.If(ee.Number(classId).eq(2), PALETTE[2],
                                                          PALETTE[3])));

  // misclassified points get a red stroke
  var strokeColor = ee.Algorithms.If(ee.Number(correct).eq(1), '#000000', '#ff0000');

  return feature.set('style', {
    pointSize:   8,
    color:       strokeColor,
    fillColor:   color,
    width:       1.5,
  });
}

var styled = predictions.map(stylePoint);
Map.addLayer(styled.style({styleProperty: 'style'}), {}, 'Ice Classification');

// ── Centre map on the data ────────────────────────────────────
Map.centerObject(predictions, 8);

// ── Legend ───────────────────────────────────────────────────
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '10px',
    backgroundColor: 'white',
  }
});

legend.add(ui.Label({
  value: 'River Ice Classification',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'},
}));

var colors  = ['#1a6faf', '#74c2e1', '#2ecc71', '#f1c40f'];
CLASS_NAMES.forEach(function(name, i) {
  var row = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
  row.add(ui.Label({
    style: {
      backgroundColor: colors[i],
      padding: '8px',
      margin: '2px 6px 2px 0',
      border: '1px solid #999',
    }
  }));
  row.add(ui.Label({value: name, style: {margin: '4px 0'}}));
  legend.add(row);
});

legend.add(ui.Label({
  value: '─────────────────',
  style: {color: '#ccc', margin: '4px 0'},
}));
legend.add(ui.Label({
  value: '● Black border = correct',
  style: {fontSize: '11px', color: '#333'},
}));
legend.add(ui.Label({
  value: '● Red border = misclassified',
  style: {fontSize: '11px', color: '#cc0000'},
}));

Map.add(legend);

// ── Click inspector ───────────────────────────────────────────
Map.onClick(function(coords) {
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var nearby = predictions.filterBounds(point.buffer(5000)).limit(1);
  nearby.evaluate(function(fc) {
    if (fc.features.length === 0) return;
    var p = fc.features[0].properties;
    print('── Nearest point ──────────────');
    print('Predicted:', p.predicted);
    print('True class:', p.true_class);
    print('Correct:', p.correct === 1 ? 'Yes ✓' : 'No ✗');
    print('Land cover:', p.land_cover);
    print('Water fraction:', p.water_fraction);
    print('Date:', p.date);
  });
});

print('Total points:', predictions.size());
print('Correct predictions:', predictions.filter(ee.Filter.eq('correct', 1)).size());
