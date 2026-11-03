r"""Figures for the report: label versions, wide-channel learning, anchoring, transfer, 2023-07-24 maps.
Run from Matus/:  .venv\Scripts\python.exe narrow_rivers\classify\qc\experiments\make_figures.py
Writes PNGs to narrow_rivers/classify/qc/figures/. Reads labels read-only; writes no model output."""
import os, sys, numpy as np, pandas as pd, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)
# transfer_anchored.py prelude: load(), occ, dry, ylab, stable_river, land_poly, EX, patch, TARGETS, PX_M2
exec(open(os.path.join(HERE, "transfer_anchored.py")).read().split("rows = [r")[0])
F = EX.FEATURES

# palette (dataviz reference, light) -- validated: blue / orange / aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VCOL = {"v1 original": BLUE, "v2 +117": ORANGE, "v3 +121–124": AQUA}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
def rf(): return RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight"); plt.close(fig); print("  wrote", name)

V = {"v1 original": pd.read_csv(os.path.join(HERE, "samples_2024-06-22_v1.csv")),
     "v2 +117": pd.read_csv(os.path.join(HERE, "samples_2024-06-22_v2.csv")),
     "v3 +121–124": pd.read_csv(os.path.join(CL, "samples_2024-06-22.csv"))}
V = {k: d[d.class_id.isin([1, 4])].reset_index(drop=True) for k, d in V.items()}
Xy = lambda d: (d[F].values.astype(np.float32), d.class_id.values, d.poly_id.values)

# ---- 1. confusion matrices, each version's own grouped CV (as train_rf.py) ----
print("1 confusion matrices")
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2)); fig.subplots_adjust(wspace=0.35)
for ax, (k, d) in zip(axs, V.items()):
    X, y, g = Xy(d); o = np.zeros_like(y)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        o[te] = rf().fit(X[tr], y[tr]).predict(X[te])
    M = np.array([[((y == a) & (o == b)).sum() for b in (1, 4)] for a in (1, 4)])
    R = M / M.sum(1, keepdims=True)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{R[i,j]:.1%}\n{M[i,j]:,} px", ha="center", va="center",
                    color="white" if R[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["river", "land"]); ax.set_yticks([0, 1], ["river", "land"])
    ax.set_xlabel("model"); ax.set_ylabel("hand label" if ax is axs[0] else "")
    prec = M[0, 0] / M[:, 0].sum()
    ax.set_title(f"{k}\nriver recall {R[0,0]:.3f} · precision {prec:.3f}", fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("June 2024-06-22 — 5-fold cross-validation grouped by polygon (rows sum to 100 %)",
             x=0.01, ha="left", fontweight="bold", y=1.04)
save(fig, "01_confusion_matrices_label_versions.png")

# ---- 2. leave-one-polygon-out recall of the wide channels ----
print("2 wide-channel leave-one-out")
NEWR = [117, 121, 122, 123, 124]; test = V["v3 +121–124"]
res = {}
for k, d in V.items():
    X, y, g = Xy(d); res[k] = []
    for p in NEWR:
        te = test[test.poly_id == p]
        pr = rf().fit(X[g != p], y[g != p]).predict(te[F].values.astype(np.float32))
        res[k].append((pr == 1).mean())
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.26; x = np.arange(len(NEWR))
for i, (k, vals) in enumerate(res.items()):
    b = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=VCOL[k], label=k)
    ax.bar_label(b, [f"{v:.2f}" for v in vals], fontsize=8, color=INK2, padding=2)
ax.set_xticks(x, [f"polygon {p}\n{(test.poly_id==p).sum()} px" for p in NEWR])
ax.set_ylim(0, 1.08); ax.set_ylabel("share of polygon called river"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("Wide visible channels: recall when each is held out of training\n"
             "With several examples (v3) they teach each other; one example (v2) does not")
save(fig, "02_wide_channel_leave_one_out.png")

# ---- 3. recall/precision trade-off on identical pixels and folds ----
print("3 trade-off on matched pixels")
ref = V["v1 original"]; Xr, yr, gr = Xy(ref)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xr, yr, gr))
keep = list(range(len(F))); Fs, _, _ = load(SRC, keep)
rec, prc, area = [], [], []
for k, d in V.items():
    X, y, g = Xy(d); o = np.zeros_like(yr)
    for tr, te in folds:
        m = ~np.isin(g, np.unique(gr[te])); o[te] = rf().fit(X[m], y[m]).predict(Xr[te])
    rec.append(((yr == 1) & (o == 1)).sum() / (yr == 1).sum()); prc.append(((yr == 1) & (o == 1)).sum() / (o == 1).sum())
    Fa, ok, _ = load(SRC, keep)
    mdl = rf().fit(Fs[:, d.row.values, d["col"].values].T, d.class_id.values)
    area.append((mdl.predict(Fa[:, ok].T) == 1).sum() * PX_M2 / 1e6)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
xs = np.arange(3); names = list(V)
for vals, lab, c in ((rec, "river recall", BLUE), (prc, "river precision", ORANGE)):
    a1.plot(xs, vals, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2)
    for xi, v in zip(xs, vals): a1.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    a1.annotate(lab, (xs[-1], vals[-1]), textcoords="offset points", xytext=(10, -3), color=INK, fontsize=9)
a1.set_xticks(xs, names); a1.set_xlim(-0.3, 2.9); a1.set_ylim(0.82, 0.95); a1.grid(axis="x", visible=False)
a1.set_title("Original 103 polygons, same folds")
b = a2.bar(xs, area, 0.6, color=[VCOL[n] for n in names]); a2.bar_label(b, [f"{v:.2f}" for v in area], fontsize=8, color=INK2, padding=2)
a2.set_xticks(xs, names, fontsize=8); a2.set_ylabel("km²"); a2.set_ylim(0, max(area) * 1.15); a2.grid(axis="x", visible=False)
a2.set_title("June river area")
fig.suptitle("More channel found, precision falling", x=0.01, ha="left", fontweight="bold", y=1.03)
save(fig, "03_recall_precision_area_tradeoff.png")

# ---- 4. anchoring evidence: dry-land medians per date ----
print("4 anchoring evidence")
raw = {}
for dte in TARGETS:
    P = patch(dte)
    with rasterio.open(f"{P}/{dte}_sr.tif") as s: sr = s.read().astype(np.float32)
    th = rasterio.open(f"{P}/{dte}_thermal.tif").read(1).astype(np.float32)
    Fx = EX.features(sr, th); ok = np.isfinite(Fx).all(0)
    raw[dte] = np.array([np.median(Fx[i][ok & dry]) for i in range(len(Fx))])
SUN = {"2024-06-22": 48.6, "2023-07-24": 44.9, "2025-07-04": 47.9, "2026-08-01": 43.0}
bands = ["B2", "B3", "B4", "B5", "B6", "B7"]; bi = [F.index(b) for b in bands]
lab = {"B2": "blue\nB2", "B3": "green\nB3", "B4": "red\nB4", "B5": "NIR\nB5", "B6": "SWIR1\nB6", "B7": "SWIR2\nB7"}
others = [d for d in TARGETS if d != SRC]; DC = dict(zip(others, [BLUE, ORANGE, AQUA]))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.6, 1]})
x = np.arange(len(bands)); w = 0.26
for i, dte in enumerate(others):
    pc = 100 * (raw[dte][bi] / raw[SRC][bi] - 1)
    b = a1.bar(x + (i - 1) * w, pc, w - 0.03, color=DC[dte], label=f"{dte}  (sun {SUN[dte]}°)")
    a1.bar_label(b, [f"{v:+.0f}" for v in pc], fontsize=7, color=INK2, padding=2)
a1.axhline(0, color=INK2, lw=0.8); a1.set_xticks(x, [lab[b] for b in bands]); a1.grid(axis="x", visible=False)
a1.set_ylabel("% vs June dry land"); a1.legend(frameon=False, fontsize=8, loc="upper right")
a1.set_title(f"Dry-land reflectance vs training date (June, sun {SUN[SRC]}°)\n"
             "Blue/green jump, SWIR flat → haze/smoke, not sun angle")
ti = F.index("thermal"); dk = [raw[d][ti] - raw[SRC][ti] for d in others]
b = a2.bar(range(3), dk, 0.6, color=[DC[d] for d in others]); a2.bar_label(b, [f"{v:+.1f} K" for v in dk], fontsize=8, color=INK2, padding=2)
a2.axhline(0, color=INK2, lw=0.8); a2.set_xticks(range(3), others, fontsize=8); a2.grid(axis="x", visible=False)
a2.set_title("Dry-land temperature\nvs June (weather)"); a2.set_ylabel("K")
save(fig, "04_anchoring_evidence.png")

# ---- 5. transfer with vs without anchoring (current v3 labels) ----
print("5 transfer raw vs anchored")
def load_raw(dte):
    P = patch(dte)
    with rasterio.open(f"{P}/{dte}_sr.tif") as s: sr = s.read().astype(np.float32)
    th = rasterio.open(f"{P}/{dte}_thermal.tif").read(1).astype(np.float32)
    Fx = EX.features(sr, th); return Fx, np.isfinite(Fx).all(0)
d3 = V["v3 +121–124"]; R3, C3 = d3.row.values, d3["col"].values
Fr0, _ = load_raw(SRC); m_raw = rf().fit(Fr0[:, R3, C3].T, d3.class_id.values)
m_anc = rf().fit(Fs[:, R3, C3].T, d3.class_id.values)
stats = {"raw features": [], "anchored": []}
for dte in TARGETS:
    for name, mdl, (Fx, ok) in (("raw features", m_raw, load_raw(dte)), ("anchored", m_anc, load(dte, keep)[:2])):
        cls = np.full((H, W), 9, np.uint8); cls[ok] = mdl.predict(Fx[:, ok].T)
        stats[name].append(((cls == 1).sum() * PX_M2 / 1e6, (cls[stable_river] == 1).mean(), (cls[land_poly] == 4).mean()))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8)); x = np.arange(len(TARGETS)); w = 0.36
for j, (ax, ttl, fmt) in enumerate(zip(axs, ["River area (km²)", "Main-channel recall\n(June river px, JRC ≥ 80)", "Land recall\n(June land polygons)"],
                                         ["{:.1f}", "{:.2f}", "{:.2f}"])):
    for i, (name, c) in enumerate((("raw features", ORANGE), ("anchored", BLUE))):
        vals = [s[j] for s in stats[name]]
        b = ax.bar(x + (i - 0.5) * w, vals, w - 0.03, color=c, label=name)
        ax.bar_label(b, [fmt.format(v) for v in vals], fontsize=7, color=INK2, padding=2)
    ax.set_xticks(x, [t + ("\n(train)" if t == SRC else "") for t in TARGETS], fontsize=8)
    ax.set_title(ttl); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(max(s[j] for s in stats[n]) for n in stats) * 1.15)
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Train on June, predict other summer dates: raw vs anchored features (v3 labels)",
             x=0.01, ha="left", fontweight="bold", y=1.03)
save(fig, "05_transfer_raw_vs_anchored.png")

# ---- 6. 2023-07-24 maps: Landsat, Sentinel-2, RF ----
print("6 maps 2023-07-24")
D = "2023-07-24"; P = patch(D)
ls = np.nan_to_num(np.clip(rasterio.open(f"{P}/{D}_432.tif").read() / 0.15, 0, 1).transpose(1, 2, 0))
s2 = rasterio.open(f"{P}/{D}_s2_truecolour_10m.tif").read().transpose(1, 2, 0)
cls = rasterio.open(f"{P}/rf_transfer/{D}_rf_from_20240622_anchored_class.tif").read(1)
gray = ls.mean(2, keepdims=True).repeat(3, 2) * 0.7 + 0.15
ov = gray.copy(); ov[cls == 1] = matplotlib.colors.to_rgb(BLUE)
asp = 1 / np.cos(np.radians(64.3))  # equal ground distance in x and y
r117 = V["v3 +121–124"].query("poly_id == 117")
rc, cc = int(r117.row.median()), int(r117["col"].median()); hw = 160
win = (slice(max(rc - int(hw / asp), 0), rc + int(hw / asp)), slice(max(cc - hw, 0), cc + hw))
def show(ax, img, ttl):
    ax.imshow(img, aspect=asp, interpolation="nearest"); ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
fig, axs = plt.subplots(1, 3, figsize=(9, 10.5)); fig.subplots_adjust(wspace=0.05)
show(axs[0], ls, "Landsat 9 true colour (30 m)")
show(axs[1], s2[::3, ::3], "Sentinel-2, same day +0.5 h")
show(axs[2], ov, "RF river (blue)")
for ax in axs:
    ax.add_patch(plt.Rectangle((win[1].start, win[0].start), win[1].stop - win[1].start,
                               win[0].stop - win[0].start, fill=False, ec=ORANGE, lw=1.5))
fig.suptitle(f"Out-of-sample {D}: no labels on this date; trained on 2024-06-22 only, anchored\n"
             "Study box ~13 × 49 km; orange box = zoom in figure 06b", x=0.02, ha="left", fontweight="bold", y=0.97)
save(fig, "06a_out_of_sample_2023-07-24_overview.png")
# zoom = orange box + 25 % margin on every side
mr, mc = int(0.25 * (win[0].stop - win[0].start)), int(0.25 * (win[1].stop - win[1].start))
zw = (slice(max(win[0].start - mr, 0), min(win[0].stop + mr, H)), slice(max(win[1].start - mc, 0), min(win[1].stop + mc, W)))
s2w = (slice(zw[0].start * 3, zw[0].stop * 3), slice(zw[1].start * 3, zw[1].stop * 3))
cir = np.nan_to_num(np.clip(rasterio.open(f"{P}/{D}_543.tif").read() / 0.45, 0, 1).transpose(1, 2, 0))
swir = rasterio.open(f"{P}/{D}_swir.tif").read().transpose(1, 2, 0)
ndwi = rasterio.open(f"{P}/{D}_ndwi.tif").read(1)
ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ndwi_qgis",   # same ramp as the QGIS .qml
    [(0, "#8b4513"), (0.35, "#d8b365"), (0.5, "#ffffff"), (0.65, "#67a9cf"), (1, "#0066ff")])
fig, axs = plt.subplots(2, 3, figsize=(13, 9.6)); fig.subplots_adjust(wspace=0.04, hspace=0.16)
show(axs[0, 0], ls[zw], "Landsat 9 true colour 432 (30 m)")
show(axs[0, 1], s2[s2w], "Sentinel-2 true colour (10 m), +0.5 h")
show(axs[0, 2], ov[zw], "RF river (blue) on Landsat grey")
show(axs[1, 0], cir[zw], "Landsat colour infrared 543\nwater dark, vegetation red")
show(axs[1, 1], swir[zw], "Landsat SWIR 654 (SWIR1/NIR/red)\nwater black, gravel bright")
im = axs[1, 2].imshow(ndwi[zw], cmap=ramp, vmin=-1, vmax=1, aspect=asp, interpolation="nearest")
axs[1, 2].set_xticks([]); axs[1, 2].set_yticks([]); axs[1, 2].grid(False)
for sp in axs[1, 2].spines.values(): sp.set_visible(False)
axs[1, 2].set_title("Landsat NDWI (green−NIR)/(green+NIR)\nsilty water stays below 0 on this date", fontsize=9)
cb = fig.colorbar(im, cax=axs[1, 2].inset_axes([1.02, 0, 0.035, 1])); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, colors=INK2)
for ax in axs.flat:
    k = 3 if ax is axs[0, 1] else 1   # Sentinel-2 panel has 3 x 3 cells per Landsat cell
    ax.add_patch(plt.Rectangle(((win[1].start - zw[1].start) * k - 0.5, (win[0].start - zw[0].start) * k - 0.5),
                 (win[1].stop - win[1].start) * k, (win[0].stop - win[0].start) * k, fill=False, ec=ORANGE, lw=1.2, ls="--"))
fig.suptitle(f"Out-of-sample {D}, zoom on the braided reach (dashed = orange box in 06a)",
             x=0.01, ha="left", fontweight="bold", y=0.95)
save(fig, "06b_out_of_sample_2023-07-24_zoom.png")

with open(os.path.join(OUT, "figure_numbers.txt"), "w", encoding="utf-8") as f:
    f.write("Numbers behind the figures (regenerate with make_figures.py)\n\n")
    f.write("2 leave-one-out recall:\n" + "\n".join(f"  {k}: " + ", ".join(f"{p}={v:.2f}" for p, v in zip(NEWR, vals)) for k, vals in res.items()))
    f.write("\n\n3 matched pixels: " + "; ".join(f"{n}: rec {r:.3f} prec {p:.3f} area {a:.2f} km2" for n, r, p, a in zip(names, rec, prc, area)))
    f.write("\n\n4 dry-land medians (" + ", ".join(F) + "):\n" + "\n".join(f"  {d}: " + " ".join(f"{v:.3f}" for v in raw[d]) for d in TARGETS))
    f.write("\n\n5 transfer (area km2, main-channel recall, land recall):\n" + "\n".join(
        f"  {n} {d}: {a:.2f} {r:.3f} {l:.3f}" for n in stats for d, (a, r, l) in zip(TARGETS, stats[n])))
print("done ->", os.path.abspath(OUT))
