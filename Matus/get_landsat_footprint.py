import ee

ee.Initialize(project='noaa-river-ice')

# Landsat 8 collection
collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

# Path 75, Row 12
img = collection.filter(ee.Filter.eq('WRS_PATH', 75)) \
                .filter(ee.Filter.eq('WRS_ROW', 12)) \
                .filterDate('2025-10-11', '2025-10-12') \
                .first()

info = img.getInfo()

if info is None:
    print("No scene found!")
else:
    footprint = info['properties']['system:footprint']['coordinates']

    print(f"Landsat Scene: LC90750122025284LGN00 (Path 75, Row 12)")
    print(f"Date: 2025-10-11")
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
