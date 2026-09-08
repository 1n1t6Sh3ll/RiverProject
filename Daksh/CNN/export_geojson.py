"""
export_geojson.py

Runs the trained 1D CNN on all labeled pixels and exports predictions
as a GeoJSON file ready to upload to Google Earth Engine as an asset.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
import joblib
import torch
from pathlib import Path

from train_cnn import (
    RiverIceCNN1D, prepare_datasets, to_tensors,
    FEATURE_COLUMNS, N_FEATURES,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLASS_ID = {
    "ice_covered_river_snow_covered_land": 0,
    "ice_covered_river_snow_free_land":    1,
    "ice_free_river_snow_free_land":       2,
    "ice_free_river_snow_land":            3,
}


def main():
    device = torch.device("cpu")

    le     = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    scaler = joblib.load(OUTPUT_DIR / "scaler_combined.pkl")
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    datasets = prepare_datasets()
    df = datasets["Combined"].copy().reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_t, _ = to_tensors(X_scaled, np.zeros(len(X), dtype=int))

    model = RiverIceCNN1D(N_FEATURES, n_classes)
    model.load_state_dict(torch.load(OUTPUT_DIR / "cnn1d_combined.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(X_t).argmax(1).numpy()

    df["predicted_class"] = le.inverse_transform(preds)
    df["class_id"]        = df["predicted_class"].map(CLASS_ID)
    df["true_class"]      = df["ground_truth_class"]
    df["correct"]         = (df["predicted_class"] == df["true_class"]).astype(int)

    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])]
            },
            "properties": {
                "predicted":      row["predicted_class"],
                "class_id":       int(row["class_id"]),
                "true_class":     row["true_class"],
                "correct":        int(row["correct"]),
                "date":           str(row["viirs_date"]),
                "land_cover":     str(row.get("modis_lc_name", "Unknown")),
                "water_fraction": float(row["water_fraction"]),
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}

    out_path = OUTPUT_DIR / "ice_predictions.geojson"
    out_path.write_text(json.dumps(geojson, indent=2))
    print(f"Exported {len(features)} points → {out_path}")
    print("\nNext steps:")
    print("  1. Go to code.earthengine.google.com/assets")
    print("  2. Click 'New' → 'Table upload' → select ice_predictions.geojson")
    print("  3. Set asset ID e.g. users/yourname/ice_predictions")
    print("  4. Paste the GEE script (gee_ice_map.js) into the Code Editor")


if __name__ == "__main__":
    main()
