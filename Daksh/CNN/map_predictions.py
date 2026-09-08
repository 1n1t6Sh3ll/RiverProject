"""
map_predictions.py

Runs the trained 1D CNN on all labeled pixels from combined_all.csv,
then plots each point on an interactive satellite map colored by predicted class.
Output: outputs/ice_map.html  — open in any browser.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import joblib
import torch
import folium
import pandas as pd
from pathlib import Path

from train_cnn import (
    RiverIceCNN1D, prepare_datasets, to_tensors,
    FEATURE_COLUMNS, N_FEATURES, SEED,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLASS_COLORS = {
    "ice_covered_river_snow_covered_land": "#1a6faf",   # dark blue
    "ice_covered_river_snow_free_land":    "#74c2e1",   # light blue
    "ice_free_river_snow_free_land":       "#2ecc71",   # green
    "ice_free_river_snow_land":            "#f1c40f",   # yellow
}

CLASS_SHORT = {
    "ice_covered_river_snow_covered_land": "Ice covered / Snow covered land",
    "ice_covered_river_snow_free_land":    "Ice covered / Snow free land",
    "ice_free_river_snow_free_land":       "Ice free / Snow free land",
    "ice_free_river_snow_land":            "Ice free / Snow land",
}


def main():
    device = torch.device("cpu")

    le     = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    scaler = joblib.load(OUTPUT_DIR / "scaler_combined.pkl")
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    # Load and prepare all data
    datasets = prepare_datasets()
    df = datasets["Combined"].copy().reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_t, _ = to_tensors(X_scaled, np.zeros(len(X), dtype=int))

    # Load trained model
    model = RiverIceCNN1D(N_FEATURES, n_classes)
    model.load_state_dict(torch.load(OUTPUT_DIR / "cnn1d_combined.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(X_t).argmax(1).numpy()

    df["predicted_class"] = le.inverse_transform(preds)
    df["true_class"]      = df["ground_truth_class"]
    df["correct"]         = df["predicted_class"] == df["true_class"]

    # ── Build map ─────────────────────────────────────────────────────────────
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
    )

    # Satellite basemap (Esri)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Place-name / hydrography labels overlay (rivers, towns, etc.) — draws on top of imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Boundaries and Places",
        name="Place & River Labels",
        overlay=True,
        control=True,
        show=True,
    ).add_to(m)

    # One feature group per class so they can be toggled
    groups = {cls: folium.FeatureGroup(name=CLASS_SHORT[cls], show=True)
              for cls in class_names}

    for _, row in df.iterrows():
        cls   = row["predicted_class"]
        color = CLASS_COLORS.get(cls, "#888888")
        correct = row["correct"]

        lc_name = row.get("modis_lc_name", "Unknown")

        popup_html = f"""
        <b>Predicted:</b> {CLASS_SHORT.get(cls, cls)}<br>
        <b>True:</b> {CLASS_SHORT.get(row['true_class'], row['true_class'])}<br>
        <b>Match:</b> {'✓' if correct else '✗'}<br>
        <b>Land cover (MODIS):</b> {lc_name}<br>
        <b>Water fraction:</b> {row['water_fraction']:.3f}<br>
        <b>Date:</b> {row['viirs_date']}<br>
        <b>Lat/Lon:</b> {row['lat']:.4f}, {row['lon']:.4f}
        """

        tooltip_html = f"""
        <b>{CLASS_SHORT.get(cls, cls)}</b><br>
        Land cover: {lc_name}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="black" if not correct else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1 if correct else 2,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(groups[cls])

    for g in groups.values():
        g.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                border:1px solid #ccc; font-size:13px; line-height:1.8;">
      <b>River Ice Classification</b><br>
      <i style="background:#1a6faf;width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></i>Ice covered / Snow covered land<br>
      <i style="background:#74c2e1;width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></i>Ice covered / Snow free land<br>
      <i style="background:#2ecc71;width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></i>Ice free / Snow free land<br>
      <i style="background:#f1c40f;width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></i>Ice free / Snow land<br>
      <hr style="margin:6px 0;">
      <small>Black border = misclassified</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    out_path = OUTPUT_DIR / "ice_map.html"
    m.save(str(out_path))

    correct_total = df["correct"].sum()
    print(f"Total points: {len(df)}")
    print(f"Correct predictions: {correct_total}/{len(df)} ({100*correct_total/len(df):.1f}%)")
    print(f"\nMap saved → {out_path}")
    print("Open ice_map.html in your browser to view.")


if __name__ == "__main__":
    main()
