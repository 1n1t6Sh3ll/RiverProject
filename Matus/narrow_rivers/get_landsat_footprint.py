"""
Landsat footprint for the narrow-river workstream.

Same idea as Matus/get_landsat_footprint.py, but the scene is given by its
LANDSAT_PRODUCT_ID (no editing path/row/date), and it also reports how much of
each SWORD river falls inside the scene.

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\get_landsat_footprint.py --landsat LC08_L2SP_073011_20240527_20240611_02_T1
"""

import os, csv
from collections import defaultdict

import ee
import numpy as np
from matplotlib.path import Path

NR_ROOT    = os.path.dirname(os.path.abspath(__file__))
NODES_CSV  = os.path.join(NR_ROOT, "data", "sword_nodes_75_300m_csv.csv")
EE_PROJECT = "noaa-river-ice"


def main(product_id):
    ee.Initialize(project=EE_PROJECT)

    # Collection from the product ID: LC08_... -> Landsat 8, LC09_... -> Landsat 9
    collection = ("LANDSAT/LC09/C02/T1_L2" if product_id.startswith("LC09")
                  else "LANDSAT/LC08/C02/T1_L2")
    img = (ee.ImageCollection(collection)
             .filter(ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id))
             .first())
    info = img.getInfo()
    if info is None:
        print(f"No scene found for {product_id} in {collection}")
        return

    props = info["properties"]
    footprint = props["system:footprint"]["coordinates"]

    print(f"Landsat Scene: {product_id} "
          f"(Path {props['WRS_PATH']}, Row {props['WRS_ROW']})")
    print(f"Acquired: {props['DATE_ACQUIRED']} {props['SCENE_CENTER_TIME'][:8]} UTC")
    print(f"Cloud Cover (whole scene): {props.get('CLOUD_COVER')}%")
    print()
    print("Footprint coordinates (Lon, Lat):")
    for coord in footprint:
        print(f"  Lon: {coord[0]:.4f}  Lat: {coord[1]:.4f}")

    lons = [c[0] for c in footprint]
    lats = [c[1] for c in footprint]
    print(f"\nBounding Box:")
    print(f"  Lat: {min(lats):.4f} to {max(lats):.4f}")
    print(f"  Lon: {min(lons):.4f} to {max(lons):.4f}")

    # SWORD coverage: which rivers / reaches fall inside the footprint
    nodes = defaultdict(list)
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nodes[row["river_name"]].append(
                (float(row["lon"]), float(row["lat"]), row["reach_id"]))

    poly = Path(np.array(footprint))
    print(f"\nSWORD nodes inside the footprint:")
    for river, pts in sorted(nodes.items()):
        inside = poly.contains_points(np.array([(x, y) for x, y, _ in pts]))
        reaches = {r for (_, _, r), ok in zip(pts, inside) if ok}
        all_reaches = {r for _, _, r in pts}
        print(f"  {river}: {inside.sum()}/{len(pts)} nodes, "
              f"{len(reaches)}/{len(all_reaches)} reaches")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Landsat footprint + SWORD coverage.")
    p.add_argument("--landsat", default="LC08_L2SP_073011_20240527_20240611_02_T1",
                   help="Landsat LANDSAT_PRODUCT_ID")
    main(p.parse_args().landsat)
