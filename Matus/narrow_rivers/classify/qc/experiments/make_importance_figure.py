r"""Feature importance (mean decrease in impurity / Gini importance) of the RF, June vs October.
Reads the importances train_rf.py saved in rf/<date>_rf_cv.json -- no retraining.
Run from Matus/:  .venv\Scripts\python.exe narrow_rivers\classify\qc\experiments\make_importance_figure.py"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(HERE, "..", "..", "..", "patches")
OUT = os.path.join(HERE, "..", "figures", "07_feature_importance.png")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

LABEL = {"B2": "blue (B2)", "B3": "green (B3)", "B4": "red (B4)", "B5": "NIR (B5)", "B6": "SWIR1 (B6)",
         "B7": "SWIR2 (B7)", "thermal": "thermal (B10)", "ndwi": "NDWI  green vs NIR",
         "ndsi": "NDSI  green vs SWIR1", "ndvi": "NDVI  NIR vs red"}
runs = [("2024-06-22 summer · river / land", "nenana_20240622/rf/2024-06-22_rf_cv.json", BLUE),
        ("2023-10-04 freeze-up · river / ice / snow / land", "nenana_20231004/rf/2023-10-04_rf_cv.json", ORANGE)]
imp = {name: json.load(open(os.path.join(PATCHES, path)))["importances"] for name, path, _ in runs}
feats = sorted(LABEL, key=lambda f: imp[runs[0][0]][f])          # ordered by June importance, largest on top

fig, ax = plt.subplots(figsize=(10, 6.2)); y = np.arange(len(feats)); h = 0.38
for k, (name, _, c) in enumerate(runs):
    v = [100 * imp[name][f] for f in feats]
    b = ax.barh(y + (0.5 - k) * h, v, h - 0.04, color=c, label=name)
    ax.bar_label(b, [f"{x:.0f} %" for x in v], fontsize=8, color=INK2, padding=3)
ax.set_yticks(y, [LABEL[f] for f in feats]); ax.grid(axis="y", visible=False)
ax.set_xlabel("share of the forest's decisions (importances sum to 100 % per date)")
ax.set_xlim(0, 32); ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Which inputs the Random Forest relies on (feature importance)\n"
             "Summer separates water by SWIR; freeze-up leans on NDWI, NDVI and green")
fig.text(0.01, -0.02, "Mean decrease in impurity, 300 trees, model trained on all labels of each date. "
         "Correlated inputs share credit (e.g. green also enters NDWI and NDSI).", fontsize=8, color=INK2)
fig.savefig(OUT, dpi=200, bbox_inches="tight"); print("wrote", os.path.abspath(OUT))
