import ee

ee.Initialize(project='noaa-river-ice')

# Landsat 8 collection
collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

# Path 71, Row 16
img = collection.filter(ee.Filter.eq('WRS_PATH', 71)) \
                .filter(ee.Filter.eq('WRS_ROW', 16)) \
                .filterDate('2022-10-23', '2022-10-24') \
                .first()

info = img.getInfo()

if info is None:
    print("No scene found!")
else:
    footprint = info['properties']['system:footprint']['coordinates']

    print(f"Landsat Scene: LC90710162022296LGN01 (Path 71, Row 16)")
    print(f"Date: 2022-10-23")
    print(f"Cloud Cover: {info['properties'].get('CLOUD_COVER')}%")
    print()
    print("Footprint coordinates (Lon, Lat):")
    for coord in footprint:
        print(f"  Lon: {coord[0]:.4f}  Lat: {coord[1]:.4f}")
    
    lons = [c[0] for c in footprint]
    lats = [c[1] for c in footprint]
    print(f"\nBounding Box:")
    print(f"  Lat: {min(lats):.4f} to {max(lats):.4f}")
    print(f"  Lon: {min(lons):.4f} to {max(lons):.4f}")
    print(f"\nUse these coordinates to search NOAA for a VIIRS pass that covers this area.")
