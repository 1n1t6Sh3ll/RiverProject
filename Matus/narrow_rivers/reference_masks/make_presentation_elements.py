r"""Individual presentation elements (PNG + legends + notes) for the weekly meeting.

Each map is the study box exactly (no margin), same extent and size, no title baked in,
so the files drop into PowerPoint or an HTML deck as they are. Reads rasters only.

  01_true_colour.png              Landsat true colour, 2024-06-22
  02_jrc_occurrence.png (+legend) Valentina's mask: JRC water occurrence 0-100 %
  03_rf_water_mask.png            RF river mask (v3 labels)
  04_final_mask.png               final overridden mask (RF + Sentinel-2 union + hand fixes)
  04b_final_by_source.png         the same, coloured by where each water pixel came from
  05_swir.png                     SWIR false colour
  06_sword_map.png (+legend)      SWORD nodes/reaches over the box;  SWORD_notes.txt
  08_unet_error.png               U-Net pipeline test vs baselines (5 splits)
  09_unet_october_*.png           October: VIIRS input, reference, U-Net prediction

Run from Matus/:
  .venv\Scripts\python.exe narrow_rivers\reference_masks\make_presentation_elements.py --out <folder>
"""
import os, json, glob, argparse, statistics
import numpy as np, rasterio
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(NR, "patches", "nenana_20240622")
RM = os.path.join(NR, "reference_masks")
D = "2024-06-22"
SCALE = 2                                    # upscale: 1635 rows -> 3270 px tall
ASPECT = float(np.cos(np.radians(64.3145)))  # grid cells are square in degrees; on the ground E-W is cos(lat) shorter
BLUE, AQUA, ORANGE, RED = "#2a78d6", "#1baf7a", "#eb6834", "#d03b3b"
LAND, UNSURE, INK, INK2 = "#ecebe7", "#b4b3ae", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 13, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#d6d5d0"})

def hex2rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def read(path):
    with rasterio.open(path) as r: return r.read(), r.bounds
def save_rgb(arr, name, scale=SCALE, smooth=False):
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((round(im.width * scale * ASPECT), im.height * scale), Image.LANCZOS if smooth else Image.NEAREST)
    im.save(os.path.join(OUT, name), dpi=(300, 300)); print("  wrote", name, im.size)
def stretch(a):                              # 2-98 % linear stretch per band
    out = np.zeros(a.shape[1:] + (3,), np.uint8)
    for i in range(3):
        lo, hi = np.percentile(a[i], [2, 98]); out[..., i] = np.clip((a[i] - lo) / (hi - lo + 1e-9) * 255, 0, 255)
    return out
def paint(cls, colours):                     # class raster -> RGB
    out = np.full(cls.shape + (3,), 255, np.uint8)
    for v, c in colours.items(): out[cls == v] = hex2rgb(c)
    return out
def legend(name, handles, title=None, ncol=1):
    fig = plt.figure(figsize=(4, 0.3 + 0.42 * len(handles) / ncol))
    fig.legend(handles=handles, loc="center", frameon=False, title=title, ncol=ncol, fontsize=14,
               title_fontsize=14, handlelength=1.4, handleheight=1.1)
    fig.savefig(os.path.join(OUT, name), dpi=300, transparent=True, bbox_inches="tight"); plt.close(fig)
    print("  wrote", name)
def colourbar(name, cmap, vmin, vmax, label, ticks=None):
    fig, ax = plt.subplots(figsize=(4.5, 0.9)); fig.subplots_adjust(bottom=0.55, top=0.85)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)), cax=ax,
                      orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=13); cb.outline.set_visible(False)
    fig.savefig(os.path.join(OUT, name), dpi=300, transparent=True, bbox_inches="tight"); plt.close(fig)
    print("  wrote", name)

def maps():
    tc, box = read(os.path.join(P, f"{D}_truecolour.tif"))
    PANEL["tc"] = stretch(tc.astype(float)); save_rgb(PANEL["tc"], "01_true_colour.png", smooth=True)

    jrc = read(os.path.join(P, f"{D}_water_occurrence.tif"))[0][0].astype(float)
    cmap = LinearSegmentedColormap.from_list("occ", ["#f7f6f2", "#a9c8ee", BLUE, "#0d3b7a"])
    rgb = (cmap(np.clip(jrc, 0, 100) / 100)[..., :3] * 255).astype(np.uint8)
    rgb[(jrc == 0) | (jrc == 255)] = hex2rgb(LAND)
    PANEL["jrc"] = rgb; PANEL["jrc_cmap"] = cmap; save_rgb(rgb, "02_jrc_occurrence.png")
    colourbar("02_jrc_occurrence_legend.png", cmap, 0, 100,
              "JRC water occurrence 1984-2021 (% of observations)", [0, 25, 50, 75, 100])

    rf = read(os.path.join(P, "rf", f"{D}_rf_class.tif"))[0][0]
    PANEL["rf"] = paint(rf, {1: BLUE, 4: LAND, 9: UNSURE}); save_rgb(PANEL["rf"], "03_rf_water_mask.png")

    fdir = os.path.join(RM, D, "final")
    ftag = sorted(glob.glob(os.path.join(fdir, f"{D}_final_*.tif")))[-1]
    final = read(ftag)[0][0]
    prov = read(ftag.replace("_final_", "_provenance_"))[0][0]       # 1 hand label, 2 override, 3 model
    union = read(os.path.join(RM, D, f"{D}_union_s2_ndwi010_k1.tif"))[0][0]  # 2 = Sentinel-2 only
    save_rgb(paint(final, {1: BLUE, 4: LAND, 9: UNSURE}), "04_final_mask.png")
    src = np.where(final == 1, 1, np.where(final == 9, 9, 4))
    src[(final == 1) & (prov == 3) & (union == 2)] = 2                 # added by Sentinel-2
    src[(final == 1) & (prov == 2)] = 3                                # hand override -> water
    src[(final == 4) & (prov == 2)] = 5                                # hand override -> land
    PANEL["final"] = paint(src, {1: BLUE, 2: AQUA, 3: ORANGE, 5: RED, 4: LAND, 9: UNSURE})
    save_rgb(PANEL["final"], "04b_final_by_source.png")
    n = {k: int((src == k).sum()) for k in (1, 2, 3, 5)}
    legend("03_04_mask_legend.png", [Patch(fc=BLUE, label="water / river"), Patch(fc=LAND, label="land"),
           Patch(fc=UNSURE, label="unsure (no data)")])
    legend("04b_source_legend.png", [Patch(fc=BLUE, label="RF + hand labels"),
           Patch(fc=AQUA, label="added by Sentinel-2 (union)"), Patch(fc=ORANGE, label="hand fix: made water"),
           Patch(fc=RED, label="hand fix: made land"), Patch(fc=LAND, label="land")])

    sw = read(os.path.join(P, f"{D}_swir.tif"))[0]
    # SWIR 654 (SWIR1/NIR/red) shown as the file stores it, like figure 06b: water black, gravel bright
    PANEL["swir"] = np.moveaxis(sw, 0, -1); save_rgb(PANEL["swir"], "05_swir.png", smooth=True)
    # zoom panels: masks drawn over Landsat grey (as in figure 06b), land shows the grey
    grey = np.repeat((PANEL["tc"].mean(-1, keepdims=True) * 0.7 + 0.15 * 255).astype(np.uint8), 3, -1)
    def over(cls, colours):
        out = grey.copy()
        for v, c in colours.items(): out[cls == v] = hex2rgb(c)
        return out
    PANEL["s2"] = over(np.isin(union, (2, 3)).astype(np.uint8), {1: BLUE})  # >=1 of 3x3 S2 cells NDWI > 0.10
    PANEL["s2_plain"] = paint(np.where(np.isin(union, (2, 3)), 1, 4), {1: BLUE, 4: LAND})
    PANEL["rf_plain"] = paint(rf, {1: BLUE, 4: LAND, 9: UNSURE})
    PANEL["final_plain"] = paint(src, {1: BLUE, 2: AQUA, 3: ORANGE, 5: RED, 4: LAND, 9: UNSURE})
    PANEL["rf"] = over(rf, {1: BLUE})
    PANEL["final"] = over(src, {1: BLUE, 2: AQUA, 3: ORANGE, 5: RED})
    jz = grey.copy(); wet = (jrc > 0) & (jrc <= 100)               # JRC occurrence over grey
    jz[wet] = (PANEL["jrc_cmap"](jrc[wet] / 100)[:, :3] * 255).astype(np.uint8); PANEL["jrc_z"] = jz
    fw = final == 1                                                    # final water, outline only
    edge = fw & ~(np.roll(fw, 1, 0) & np.roll(fw, -1, 0) & np.roll(fw, 1, 1) & np.roll(fw, -1, 1))
    jf = jz.copy(); jf[edge] = hex2rgb(ORANGE); PANEL["jrc_final"] = jf
    return tc, box, rf, final, n

PANEL = {}
CELLS = {}

def zoom():
    """Square zoom on the braided reach (centred on wide channel 117, same window as the old
    figure 06b's orange box, no margin): 10a-e single panels and 11_zoom_combined
    (2 x 3: true colour, SWIR, JRC occurrence / Sentinel-2 NDWI mask, RF, final)."""
    import csv
    rows = [(int(r["row"]), int(r["col"])) for r in csv.DictReader(open(os.path.join(
        NR, "classify", "samples_2024-06-22.csv"), encoding="utf-8")) if r["poly_id"] == "117"]
    rc = int(np.median([r for r, _ in rows])); cc = int(np.median([c for _, c in rows])); hw = 160
    hr = int(hw * ASPECT)                                              # same ground distance N-S as E-W
    win = (slice(max(rc - hr, 0), rc + hr), slice(max(cc - hw, 0), cc + hw))
    km = 2 * hw * 0.00027027 * 111.32 * ASPECT
    PANEL["win"] = win
    names = [("tc", "10a_zoom_true_colour.png", "Landsat true colour"),
             ("swir", "10b_zoom_swir.png", "Landsat SWIR 654\n(SWIR1 / NIR / red)"),
             ("jrc_final", "10f_zoom_jrc_vs_final.png", "JRC occurrence (Valentina's mask)\n+ our final mask (outline)"),
             ("s2", "10c_zoom_s2_ndwi_mask.png", "Sentinel-2 water\n(10 m NDWI > 0.10)"),
             ("rf", "10d_zoom_rf_mask.png", "Our Random Forest\n(Landsat 30 m)"),
             ("final", "10e_zoom_final_mask.png", "Final mask\n(RF + Sentinel-2 + hand fixes)")]
    side = 1200
    sq = {}
    for k, fn, _ in names:
        im = Image.fromarray(PANEL[k][win]).resize((side, side), Image.LANCZOS if k in ("tc", "swir") else Image.NEAREST)
        im.save(os.path.join(OUT, fn), dpi=(300, 300)); sq[k] = np.asarray(im); print("  wrote", fn, im.size)
    # combined: 2 x 3 squares, a legend row under each row, ready to paste
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.2, 1, 0.08], wspace=0.04, hspace=0.1,
                          left=0.01, right=0.99, top=0.95, bottom=0.01)
    for i, (k, _, ttl) in enumerate(names):
        ax = fig.add_subplot(gs[2 * (i // 3), i % 3]); ax.imshow(sq[k]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#d6d5d0")
        ax.set_title(ttl, fontsize=13, fontweight="bold" if k == "final" else "normal")
        if i == 0:                                                      # 1 km scale bar, image pixel coords
            px = side / km; x0, y0 = 0.06 * side, 0.93 * side
            ax.plot([x0, x0 + px], [y0, y0], color="white", lw=3, solid_capstyle="butt")
            ax.text(x0 + px / 2, y0 - 25, "1 km", color="white", ha="center", va="bottom", fontsize=11, fontweight="bold")
    def leg(r, c, hs):
        lax = fig.add_subplot(gs[r, c]); lax.set_axis_off()
        lax.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False, fontsize=10,
                   handlelength=1.1, columnspacing=0.9)
    leg(1, 1, [Patch(fc="#111111", label="water"), Patch(fc="#e8e0d8", label="bare gravel")])
    host = fig.add_subplot(gs[1, 2]); host.set_axis_off(); cax = host.inset_axes([0.02, 0.72, 0.55, 0.16])
    host.legend(handles=[Line2D([], [], color=ORANGE, lw=2.5, label="final mask")], loc="upper right",
                bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=10)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=PANEL["jrc_cmap"], norm=plt.Normalize(0, 100)), cax=cax,
                      orientation="horizontal", ticks=[0, 50, 100]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9); cb.set_label("JRC: % of time wet, 1984-2021", fontsize=10)
    leg(3, 0, [Patch(fc=BLUE, label="water")])
    leg(3, 1, [Patch(fc=BLUE, label="river")])
    leg(3, 2, [Patch(fc=BLUE, label="RF"), Patch(fc=AQUA, label="added by Sentinel-2")] +
        ([Patch(fc=ORANGE, label="hand fix")] if (sq["final"] == hex2rgb(ORANGE)).all(-1).any() else []))
    fig.savefig(os.path.join(OUT, "11_zoom_combined.png"), dpi=300, facecolor="white"); plt.close(fig)
    print(f"  wrote 11_zoom_combined.png  (zoom {km:.1f} x {km:.1f} km)")

def sword(tc, box):
    import csv
    reaches = json.load(open(os.path.join(NR, "data", "sword_reaches_nenana.geojson")))
    nodes = list(csv.DictReader(open(os.path.join(NR, "data", "sword_nodes_nenana.csv"), encoding="utf-8")))
    inb = [r for r in nodes if box.left <= float(r["lon"]) <= box.right and box.bottom <= float(r["lat"]) <= box.top]
    h, w = tc.shape[1:]
    fig = plt.figure(figsize=(w * SCALE * ASPECT / 300, h * SCALE / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    bg = stretch(tc.astype(float)) / 255 * 0.55 + 0.45                 # faded true colour
    ax.imshow(bg, extent=(box.left, box.right, box.bottom, box.top), interpolation="nearest")
    for f in reaches["features"]:
        xy = np.array(f["geometry"]["coordinates"]); ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.8, alpha=0.6)
    wid = np.array([float(r["width_m"]) for r in inb])
    cmap = LinearSegmentedColormap.from_list("w", ["#a9c8ee", BLUE, "#0d3b7a"])
    ax.scatter([float(r["lon"]) for r in inb], [float(r["lat"]) for r in inb], c=wid, cmap=cmap,
               vmin=0, vmax=400, s=9, linewidths=0)
    ax.set_xlim(box.left, box.right); ax.set_ylim(box.bottom, box.top); ax.set_aspect("auto")
    fig.savefig(os.path.join(OUT, "06_sword_map.png"), dpi=300); plt.close(fig); print("  wrote 06_sword_map.png")
    colourbar("06_sword_legend.png", cmap, 0, 400, "SWORD node width (m); line = reach centreline", [0, 100, 200, 300, 400])
    allw = sorted(float(r["width_m"]) for r in nodes)
    return dict(n_reaches=len(reaches["features"]), n_nodes=len(nodes), n_box=len(inb),
                med_all=statistics.median(allw), med_box=statistics.median(wid.tolist()),
                narrow=sum(x < 375 for x in allw))

def unet():
    runs = sorted(glob.glob(os.path.join(NR, "unet", "runs", "*_2023-10-04")))
    day = os.path.basename(runs[-1])[:8]                               # newest batch of runs only
    runs = [r for r in runs if os.path.basename(r).startswith(day)]
    m = {k: [] for k in ("mean", "rf", "unet")}; seeds = {}
    for r in runs:
        j = json.load(open(os.path.join(r, "metrics.json"))); s = json.load(open(os.path.join(r, "config.json")))["seed"]
        seeds[s] = r
        for k in m: m[k].append(j[k]["test"]["mae_channel"])
    t = j["unet"]; CELLS["n"] = t["train"]["n_cells"] + t["test"]["n_cells"]
    CELLS["ch"] = t["train"]["n_channel_cells"] + t["test"]["n_channel_cells"]
    lab =["Naive guess\n(training mean)", "Per-cell random forest\non VIIRS", "U-Net\non VIIRS"]
    col = ["#c9c8c3", ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, k in enumerate(m):
        v = np.array(m[k]); ax.barh(i, v.mean(), color=col[i], height=0.6)
        ax.scatter(v, np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), color=INK, s=14, zorder=3)
        ax.text(v.mean() + 0.006, i + 0.33, f"{v.mean():.3f}", va="center", fontsize=13, fontweight="bold")
    ax.set_yticks(range(3), lab); ax.invert_yaxis(); ax.set_xlim(0, 0.24)
    ax.set_xlabel("test error in river + ice fraction per VIIRS cell (MAE, lower is better)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
    fig.text(0.01, -0.04, "Bars = mean of 5 random 75/25 spatial splits; dots = each split. "
             f"{CELLS['n']} hand-checked October cells ({CELLS['ch']} river/ice, {CELLS['n'] - CELLS['ch']} land).",
             fontsize=10, color=INK2)
    fig.savefig(os.path.join(OUT, "08_unet_error.png"), dpi=300, bbox_inches="tight", transparent=True); plt.close(fig)
    print("  wrote 08_unet_error.png")

    s = min(seeds, key=lambda k: abs(m["unet"][sorted(seeds).index(k)] - np.mean(m["unet"])))  # most typical split
    run = seeds[s]
    vi = read(os.path.join(NR, "patches", "nenana_20231004", "2023-10-04_viirs_375m.tif"))[0]
    with rasterio.open(sorted(glob.glob(os.path.join(RM, "2023-10-04", "final", "2023-10-04_fractions_375m_*.tif")))[-1]) as r:
        fr = {d: r.read(i + 1) for i, d in enumerate(r.descriptions)}
    split = read(os.path.join(run, "split_2023-10-04.tif"))[0][0]
    with rasterio.open(os.path.join(run, "pred_2023-10-04.tif")) as r:
        pred = r.read(list(r.descriptions).index("f_channel") + 1)
    cmap = LinearSegmentedColormap.from_list("f", ["#f7f6f2", "#a9c8ee", BLUE, "#0d3b7a"])
    ref = fr["f_river"] + fr["f_ice"]
    rr, cc = np.nonzero(split > 0); M = 3                              # zoom: checked cells + 3-cell margin
    win = (slice(max(rr.min() - M, 0), rr.max() + M + 1), slice(max(cc.min() - M, 0), cc.max() + M + 1))
    K = 60
    PANEL["u"] = dict(win=win, split=split, ref=ref, pred=pred, seed=s)
    v = stretch(vi[[2, 1, 0]][:, win[0], win[1]])
    save_rgb(np.kron(v, np.ones((K, K, 1), np.uint8)), "09a_unet_october_viirs_input.png", scale=1)   # I3/I2/I1
    def frac_png(a, mask, name):
        rgb = (cmap(np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)
        if mask is not None: rgb[~mask] = hex2rgb("#d9d8d3")
        rgb, sp = rgb[win], split[win]
        big = np.kron(rgb, np.ones((K, K, 1), np.uint8))
        for r_, c_ in zip(*np.nonzero(sp == 2)):                        # outline test cells
            big[r_ * K:(r_ + 1) * K, [c_ * K, c_ * K + 1, c_ * K + K - 2, c_ * K + K - 1]] = hex2rgb(ORANGE)
            big[[r_ * K, r_ * K + 1, r_ * K + K - 2, r_ * K + K - 1], c_ * K:(c_ + 1) * K] = hex2rgb(ORANGE)
        save_rgb(big, name, scale=1)
    frac_png(ref, split > 0, "09b_unet_october_reference.png")
    frac_png(pred, split > 0, "09c_unet_october_prediction.png")
    colourbar("09_fraction_legend.png", cmap, 0, 1, "river + ice fraction per 375 m cell", [0, 0.25, 0.5, 0.75, 1])
    legend("09_cells_legend.png", [Patch(fc="#d9d8d3", label="not hand-checked (no target)"),
           Patch(fc="none", ec=ORANGE, lw=2, label="test cells (held out)")])
    return m, s

def box_combined(box):
    """12: the whole study box, 5 strips side by side (same panels as the zoom), legends below."""
    names = [("tc", "Landsat\ntrue colour"), ("swir", "Landsat SWIR 654\n(SWIR1 / NIR / red)"),
             ("s2_plain", "Sentinel-2 water\n(NDWI > 0.10)"), ("rf_plain", "Our Random\nForest"),
             ("final_plain", "Final mask\n(RF + S2 + hand fixes)")]
    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.1], wspace=0.06, hspace=0.04,
                          left=0.01, right=0.99, top=0.92, bottom=0.01)
    for i, (k, ttl) in enumerate(names):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(PANEL[k], aspect=1 / ASPECT, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_color("#d6d5d0") for sp in ax.spines.values()]
        ax.set_title(ttl, fontsize=13, fontweight="bold" if k == "final_plain" else "normal")
        if i == 0:                                                      # 10 km scale bar
            px10 = 10 / (0.00027027 * 111.32 * ASPECT); x0, y0 = 40, PANEL[k].shape[0] - 60
            ax.plot([x0, x0 + px10], [y0, y0], color="white", lw=3, solid_capstyle="butt")
            ax.text(x0 + px10 / 2, y0 - 25, "10 km", color="white", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for col, hs in ((1, [Patch(fc="#111111", label="water")]), (2, [Patch(fc=BLUE, label="water")]),
                    (3, [Patch(fc=BLUE, label="river"), Patch(fc=UNSURE, label="unsure")]),
                    (4, [Patch(fc=BLUE, label="RF"), Patch(fc=AQUA, label="added by S2"), Patch(fc=ORANGE, label="hand fix")])):
        lax = fig.add_subplot(gs[1, col]); lax.set_axis_off()
        lax.legend(handles=hs, loc="upper center", ncol=1 if len(hs) > 2 else len(hs), frameon=False, fontsize=10,
                   handlelength=1.1, columnspacing=0.8)
    fig.text(0.01, 0.965, "Nenana study box, 2024-06-22 (13 × 49 km)", fontsize=15, fontweight="bold", va="bottom")
    fig.savefig(os.path.join(OUT, "12_study_box_combined.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 12_study_box_combined.png")

def unet_combined(seed, m):
    """13: October U-Net example, 3 panels + legends, one image."""
    ims = [np.asarray(Image.open(os.path.join(OUT, f))) for f in
           ("09a_unet_october_viirs_input.png", "09b_unet_october_reference.png", "09c_unet_october_prediction.png")]
    ttl = ["VIIRS input\n(I3 / I2 / I1, 375 m)", "Reference\n(from Landsat + hand check)", "U-Net prediction"]
    fig = plt.figure(figsize=(10, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], wspace=0.05, hspace=0.05, left=0.01, right=0.99, top=0.86, bottom=0.01)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(ims[i]); ax.set_xticks([]); ax.set_yticks([])
        [sp.set_color("#d6d5d0") for sp in ax.spines.values()]; ax.set_title(ttl[i], fontsize=13)
    cmap = LinearSegmentedColormap.from_list("f", ["#f7f6f2", "#a9c8ee", BLUE, "#0d3b7a"])
    host = fig.add_subplot(gs[1, 1]); host.set_axis_off(); cax = host.inset_axes([0.05, 0.6, 0.9, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)
    cb.set_label("river + ice fraction per cell", fontsize=10)
    lax = fig.add_subplot(gs[1, 2]); lax.set_axis_off()
    lax.legend(handles=[Patch(fc="none", ec=ORANGE, lw=2, label="test cells (held out)"),
                        Patch(fc="#d9d8d3", label="not hand-checked")], loc="upper center", frameon=False, fontsize=10)
    fig.text(0.01, 0.94, f"First U-Net on VIIRS, 2023-10-04 freeze-up (pipeline test, split {seed})\n"
             f"mean test error {np.mean(m['unet']):.3f}  vs  {np.mean(m['rf']):.3f} per-cell RF  vs  {np.mean(m['mean']):.3f} naive", fontsize=13,
             fontweight="bold", va="bottom")
    fig.savefig(os.path.join(OUT, "13_unet_october_combined.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 13_unet_october_combined.png")

def sword_combined(box, sw):
    """14: SWORD in one image: the Nenana's nodes coloured by width, study box outlined, stats beside it."""
    import csv
    nodes = list(csv.DictReader(open(os.path.join(NR, "data", "sword_nodes_nenana.csv"), encoding="utf-8")))
    reaches = json.load(open(os.path.join(NR, "data", "sword_reaches_nenana.geojson")))
    lon = np.array([float(r["lon"]) for r in nodes]); lat = np.array([float(r["lat"]) for r in nodes])
    wid = np.array([float(r["width_m"]) for r in nodes])
    cmap = LinearSegmentedColormap.from_list("w", ["#bcd6f2", BLUE, "#0d3b7a"]); norm = plt.Normalize(0, 400)
    fig = plt.figure(figsize=(13.33, 7.5))
    a = fig.add_axes([0.02, 0.05, 0.46, 0.82])
    for f in reaches["features"]:
        xy = np.array(f["geometry"]["coordinates"]); a.plot(xy[:, 0], xy[:, 1], color="#d6d5d0", lw=5, zorder=1,
                                                             solid_capstyle="round")
    a.scatter(lon, lat, c=wid, cmap=cmap, norm=norm, s=16, linewidths=0, zorder=2)
    a.add_patch(plt.Rectangle((box.left, box.bottom), box.right - box.left, box.top - box.bottom,
                              fill=False, ec=ORANGE, lw=2.2, zorder=3))
    a.text(box.right + 0.04, (box.top + box.bottom) / 2, "study box", color=ORANGE, fontsize=14,
           va="center", fontweight="bold")
    a.set_aspect(1 / np.cos(np.radians(lat.mean()))); a.tick_params(labelsize=10, colors=INK2)
    a.set_xticks([-149.0, -148.5, -148.0]); a.set_xticklabels(["149.0° W", "148.5° W", "148.0° W"])
    a.set_yticks([63.4, 63.8, 64.2]); a.set_yticklabels(["63.4° N", "63.8° N", "64.2° N"])
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(color="#efeeea", lw=0.8); a.set_axisbelow(True)
    stats = [("Whole river", f"{sw['n_reaches']} reaches, {sw['n_nodes']} nodes"),
             ("", "one node every ~200 m along the centreline,\neach with a width"),
             ("", f"median width {sw['med_all']:.0f} m"),
             ("", f"{sw['narrow']} of {sw['n_nodes']} nodes narrower than one\nVIIRS pixel (375 m)"),
             ("Study box (13 × 49 km)", f"{sw['n_box']} nodes, median width {sw['med_box']:.0f} m"),
             ("Source", "Google Earth Engine community catalogue\n(projects/sat-io/open-datasets/SWORD),\n"
                        "filtered to river_name = \"Nenana River\", type 1 (river)")]
    y = 0.84
    for head, body in stats:
        if head:
            y -= 0.02; fig.text(0.52, y, head, fontsize=14, fontweight="bold", va="top", color=ORANGE if "box" in head else INK)
            y -= 0.045
        fig.text(0.52, y, body, fontsize=13, va="top", color=INK, linespacing=1.25); y -= 0.038 * (body.count("\n") + 1) + 0.012
    cax = fig.add_axes([0.52, 0.1, 0.36, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal",
                      ticks=[0, 100, 200, 300, 400]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=10)
    cb.set_label("SWORD node width (m)", fontsize=12)
    fig.text(0.03, 0.93, "SWORD (SWOT River Database): the Nenana River", fontsize=18, fontweight="bold", va="bottom")
    fig.savefig(os.path.join(OUT, "14_sword_combined.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 14_sword_combined.png")

def _bars(ax, groups, series, colours, ylim=(0, 1.05), fmt="{:.2f}"):
    """grouped vertical bars with value labels; series = {name: [v per group]}"""
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=colours[i], label=name)
        ax.bar_label(b, [fmt.format(v) for v in vals], fontsize=10, padding=2, color=INK)
    ax.set_xticks(x, groups, fontsize=11)
    if ylim[1] >= 1: ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def rf_results():
    """15: Random Forest accuracy on one slide. Numbers from classify/qc/NOTES.md (v7, v8) and the
    reference-mask report; all polygon-held-out cross-validation or scoring on unseen dates."""
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.22, left=0.04, right=0.99, top=0.78, bottom=0.2)
    a = fig.add_subplot(gs[0, 0])
    _bars(a, ["river\nrecall", "river\nprecision", "land\nrecall"], {"v1 labels": [0.845, 0.931, 0.990]}, [BLUE])
    a.set_title("June 2024: cross-validation, v1 labels\n(whole polygons held out, 103 polygons)", fontsize=12)
    b = fig.add_subplot(gs[0, 1])
    _bars(b, ["2023-07-24", "2025-07-04\n(test date)"],
          {"main-channel recall": [0.972, 0.951], "land recall": [0.956, 0.958]}, [BLUE, "#b4b3ae"])
    b.set_title("Trained on June, applied to unseen summer dates\n(anchored features, no labels on these dates)", fontsize=12)
    b.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    c = fig.add_subplot(gs[0, 2])
    _bars(c, ["ice\nrecall", "ice\nprecision", "snow\nrecall", "snow\nprecision"],
          {"RF + channel rule": [0.859, 0.930, 0.887, 0.803]}, [ORANGE])
    c.set_title("October 2023 freeze-up: ice vs snow\n(cross-validation, polygons held out)", fontsize=12)
    fig.text(0.01, 0.93, "Random Forest on Landsat (30 m): accuracy so far", fontsize=17, fontweight="bold", va="bottom")
    fig.text(0.01, 0.875, "300 trees, 7 bands + NDWI / NDSI / NDVI.  Channel rule: frozen pixels inside the June "
             "channel = ice, outside = snow.", fontsize=11, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "15_rf_results.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 15_rf_results.png")

def unet_results():
    """16: U-Net results so far on one slide: error chart, per-class errors, setup."""
    runs = sorted(glob.glob(os.path.join(NR, "unet", "runs", "*_2023-10-04")))
    day = os.path.basename(runs[-1])[:8]; runs = [r for r in runs if os.path.basename(r).startswith(day)]
    J = [json.load(open(os.path.join(r, "metrics.json"))) for r in runs]
    keys = ("mean", "rf", "unet"); lab = ["Naive guess", "Per-cell RF on VIIRS", "U-Net on VIIRS"]
    col = ["#c9c8c3", ORANGE, BLUE]
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.8], wspace=0.25, left=0.1, right=0.99, top=0.8, bottom=0.14)
    a = fig.add_subplot(gs[0, 0])
    for i, k in enumerate(keys):
        v = np.array([j[k]["test"]["mae_channel"] for j in J]); a.barh(i, v.mean(), color=col[i], height=0.6)
        a.scatter(v, np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), color=INK, s=14, zorder=3)
        a.text(v.max() + 0.006, i, f"{v.mean():.3f}", va="center", fontsize=13, fontweight="bold")
    a.set_yticks(range(3), lab, fontsize=11); a.invert_yaxis(); a.set_xlim(0, 0.28)
    a.set_xlabel("test error, river + ice fraction per cell (MAE)", fontsize=10)
    a.grid(axis="x", color="#e4e3df"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_title("Error on held-out cells (lower is better)\nbars = mean of 5 splits, dots = each split", fontsize=12)
    b = fig.add_subplot(gs[0, 1])
    cls = [("mae_river", "river"), ("mae_ice", "ice"), ("mae_snow", "snow"), ("mae_land", "land")]
    _bars(b, [c for _, c in cls], {lab[i]: [np.mean([j[k]["test"][m] for j in J]) for m, _ in cls]
                                    for i, k in enumerate(keys)}, col, ylim=(0, 0.32), fmt="{:.2f}")
    b.set_yticks([0, 0.1, 0.2, 0.3]); b.set_ylim(0, 0.32); b.set_title("Error per class (test cells, mean of 5 splits)", fontsize=12)
    b.legend(frameon=False, fontsize=10, loc="upper left")
    t = J[0]["unet"]; n = t["train"]["n_cells"] + t["test"]["n_cells"]
    ch = t["train"]["n_channel_cells"] + t["test"]["n_channel_cells"]
    txt = (f"Setup\n"
           f"• Input: VIIRS I1–I5 (375 m), whole box\n"
           f"• Target: river / ice / snow / land fractions\n   from the Landsat reference mask\n"
           f"• {n} hand-checked October cells\n   ({ch} river/ice, {n - ch} land)\n"
           f"• 75 / 25 split in 4 × 4-cell blocks\n• 3-level U-Net, 118k parameters\n"
           f"\nTake-away\n"
           f"• Pipeline works end to end\n• Beats the naive guess clearly\n"
           f"• Ahead of per-cell RF on average,\n   but the splits overlap\n• Needs more checked cells")
    fig.text(0.735, 0.8, txt, fontsize=12, va="top", linespacing=1.45, color=INK)
    fig.text(0.01, 0.93, "First U-Net on VIIRS, 2023-10-04 freeze-up: results so far (pipeline test)",
             fontsize=17, fontweight="bold", va="bottom")
    fig.savefig(os.path.join(OUT, "16_unet_results.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 16_unet_results.png")

def viirs_square(date="2023-10-04", km=270):
    """km x km VIIRS (I1..I3) around the study box, cached in reference_masks/<date>/ (read-only on inputs)."""
    out = os.path.join(RM, date, f"{date}_viirs_square{km}km.tif")
    if os.path.exists(out): return out
    import sys
    from rasterio.transform import from_origin
    sys.path.insert(0, os.path.join(NR, "patches")); sys.path.insert(0, os.path.dirname(NR))
    import patch_config as C
    from add_viirs import PASSES, _granule, _resample
    from viirs_training_loader import load_viirs_training_pair
    lon_c, lat_c = (C.BOX[0] + C.BOX[2]) / 2, (C.BOX[1] + C.BOX[3]) / 2
    hl = km / 2 / 110.574; hw = km / 2 / (111.320 * np.cos(np.radians(lat_c)))
    box = (lon_c - hw, lat_c - hl, lon_c + hw, lat_c + hl); g = C.viirs_grid(box)
    sat, t_code, *_ = PASSES[date]; gitco, gimgo = _granule(sat, t_code)
    lat, lon, bands, _, _ = load_viirs_training_pair(gitco, gimgo, clip_bbox=(box[1] - 0.1, box[3] + 0.1,
                                                                              box[0] - 0.1, box[2] + 0.1))
    stack = np.stack([_resample(lat, lon, bands[b], g) for b in ("I1", "I2", "I3")]).astype(np.float32)
    with rasterio.open(out, "w", driver="GTiff", height=g["H"], width=g["W"], count=3, dtype="float32", crs=C.CRS,
                       transform=from_origin(g["x0"], g["y1"], g["res"], g["res"]), nodata=np.nan,
                       compress="deflate") as d:
        d.write(stack); [d.set_band_description(i + 1, b) for i, b in enumerate(("I1", "I2", "I3"))]
    print("  cached", os.path.basename(out)); return out

ZOOMC = "#ffd23f"                                                      # bright yellow, visible on VIIRS

def unet_221(m):
    """20: the October U-Net slide again, with VIIRS as an I2/I2/I1 composite and a wide locator."""
    Po = os.path.join(NR, "patches", "nenana_20231004")
    with rasterio.open(viirs_square("2023-10-04", 270)) as r:                  # 270 x 270 km around the box
        wide = r.read([2, 2, 1]).astype(float); wb = r.bounds
    with rasterio.open(os.path.join(Po, "2023-10-04_viirs_375m.tif")) as r:
        loc = r.read([2, 2, 1]).astype(float); lb = r.bounds; res = r.res[0]
    def rgb(a, ref):                                                   # 2-98 % stretch per band, from `ref`
        lohi = [np.nanpercentile(ref[i], [2, 98]) for i in range(3)]
        out = np.stack([np.clip((a[i] - lohi[i][0]) / (lohi[i][1] - lohi[i][0]), 0, 1) for i in range(3)], -1)
        return np.nan_to_num(out) ** 0.6                                # gamma: brighter mid-tones
    _, box = read(os.path.join(P, f"{D}_truecolour.tif"))
    u = PANEL["u"]; win = u["win"]
    zx0, zx1 = lb.left + win[1].start * res, lb.left + win[1].stop * res
    zy1, zy0 = lb.top - win[0].start * res, lb.top - win[0].stop * res
    zext = (zx0, zx1, zy0, zy1)
    cmap = LinearSegmentedColormap.from_list("f", ["#f7f6f2", "#a9c8ee", BLUE, "#0d3b7a"])
    def frac(a):
        out = cmap(np.clip(a, 0, 1))[..., :3]; out[~(u["split"] > 0)] = np.array(hex2rgb("#d9d8d3")) / 255
        return out[win]
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.5, 0.5, 0.5], height_ratios=[1, 0.09], wspace=0.06,
                          hspace=0.06, left=0.01, right=0.99, top=0.84, bottom=0.02)
    a = fig.add_subplot(gs[0, 0]); a.imshow(rgb(wide, wide), extent=(wb.left, wb.right, wb.bottom, wb.top),
                                            aspect=1 / ASPECT, interpolation="nearest")
    a.add_patch(plt.Rectangle((box.left, box.bottom), box.right - box.left, box.top - box.bottom, fill=False,
                              ec=ORANGE, lw=1.8))
    a.text(box.right + 0.05, box.top, "study box", color=ORANGE, fontsize=11, fontweight="bold", va="top")
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"VIIRS RGB, wider area (~{(wb.right - wb.left) * 111.32 * ASPECT:.0f} × "
                f"{(wb.top - wb.bottom) * 111.32:.0f} km)", fontsize=12)
    lw_ = loc[:, win[0], win[1]]
    panels = [(rgb(lw_, lw_),"VIIRS RGB\n(U-Net input)"),
              (frac(u["ref"]), "Reference\n(Landsat 30 m → per cell)"), (frac(u["pred"]), "U-Net prediction")]
    for i, (img, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i + 1]); ax.imshow(img, extent=zext, aspect=1 / ASPECT, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=12)
        if i == 0:
            for sp in ax.spines.values(): sp.set_color(ZOOMC); sp.set_linewidth(2.5)
        if i > 0:
            for (r_, c_) in zip(*np.nonzero(u["split"][win] == 2)):
                ax.add_patch(plt.Rectangle((zx0 + c_ * res, zy1 - (r_ + 1) * res), res, res, fill=False,
                                           ec=ORANGE, lw=1.2))
    a.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=ZOOMC, lw=1.8))
    host = fig.add_subplot(gs[1, 2]); host.set_axis_off(); cax = host.inset_axes([0.05, 0.55, 0.9, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)
    cb.set_label("river + ice fraction per VIIRS cell", fontsize=10)
    lax = fig.add_subplot(gs[1, 3]); lax.set_axis_off()
    lax.legend(handles=[Patch(fc="none", ec=ORANGE, lw=2, label="test cells (held out)"),
                        Patch(fc="#d9d8d3", label="not hand-checked")], loc="upper center", frameon=False, fontsize=10)
    lax0 = fig.add_subplot(gs[1, 0]); lax0.set_axis_off()
    lax0.legend(handles=[Patch(fc="none", ec=ORANGE, lw=2, label="study box (13 × 49 km)"),
                         Patch(fc="none", ec=ZOOMC, lw=2, label="area shown on the right")],
                loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.text(0.01, 0.94, f"First U-Net on VIIRS, 2023-10-04 freeze-up (pipeline test, split {u['seed']})",
             fontsize=16, fontweight="bold", va="bottom")
    fig.text(0.01, 0.895, f"Everything on the right is on the VIIRS grid: one value per cell. Mean test error "
             f"{np.mean(m['unet']):.3f} vs {np.mean(m['rf']):.3f} per-cell RF vs {np.mean(m['mean']):.3f} naive guess.",
             fontsize=11, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "20_unet_october_viirs221.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 20_unet_october_viirs221.png")

def pipeline():
    """17: method diagram, Landsat labels -> VIIRS U-Net, one image."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig = plt.figure(figsize=(16, 7.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 7.6)
    ax.set_axis_off()
    def box(x, y, w, h, head, body, fc="#ffffff", ec="#d6d5d0", hc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.15, y + h - 0.16, head, fontsize=12.5, fontweight="bold", va="top", color=hc)
        ax.text(x + 0.15, y + h - 0.55, body, fontsize=10.5, va="top", color=INK2, linespacing=1.35)
    def arrow(x0, y0, x1, y1, c=INK2, cs="arc3"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=c,
                                     connectionstyle=cs, shrinkA=2, shrinkB=2))
    ax.text(0.3, 7.25, "Method: from hand labels on Landsat to a U-Net on VIIRS", fontsize=17, fontweight="bold", va="top")
    # lane labels
    ax.text(0.3, 5.95, "LANDSAT\n30 m", fontsize=11, fontweight="bold", color=BLUE, va="center")
    ax.text(0.3, 2.5, "VIIRS\n375 m", fontsize=11, fontweight="bold", color=ORANGE, va="center")
    ax.plot([0.25, 15.75], [4.05, 4.05], color="#e4e3df", lw=1.2, ls="--")
    # Landsat lane
    box(1.4, 5.15, 2.9, 1.75, "1  Hand labels", "QGIS polygons on Landsat\nJune: river / land\nOctober: river / ice /\nsnow / land")
    box(4.9, 5.15, 3.0, 1.75, "2  Random Forest", "7 bands + NDWI / NDSI / NDVI\n300 trees\ncross-validation with whole\npolygons held out")
    box(8.5, 5.85, 3.4, 1.05, "3  June water mask", "RF  +  Sentinel-2 (NDWI > 0.10,\nsame day, 10 m)  +  hand fixes", fc="#eef4fc", ec=BLUE)
    box(8.5, 4.55, 3.4, 1.05, "4  October mask", "RF + channel rule: frozen inside\nJune channel = ice, outside = snow", fc="#fdf0ea", ec=ORANGE)
    box(12.5, 5.15, 3.1, 1.75, "Rejected on the way", "JRC occurrence as 2nd source\nlooser NDWI (< 0.10)\nQA cloud mask\nbalanced class weights",
        fc="#f7f6f2", ec="#e4e3df", hc=INK2)
    arrow(4.3, 6.0, 4.9, 6.0); arrow(7.9, 6.2, 8.5, 6.35); arrow(7.9, 5.8, 8.5, 5.1)
    arrow(10.2, 5.85, 10.2, 5.6)                                         # June channel feeds October rule
    # VIIRS lane
    box(1.4, 1.6, 3.2, 1.75, "5  375 m fractions", "both masks aggregated to the\nVIIRS grid: share of river /\nice / snow / land per cell")
    box(5.2, 1.6, 3.3, 1.75, "6  Trusted cells", "keep a cell if its river stays\ninside when VIIRS is shifted by\nits location error (±40-50 m)\n+ hand check in QGIS")
    box(9.1, 1.6, 3.1, 1.75, "7  U-Net", "input: VIIRS I1-I5, whole box\ntarget: the fractions\nloss only on checked cells\n75 / 25 split in cell blocks")
    box(12.8, 1.6, 2.8, 1.75, "Output", "river / ice / snow / land\nfraction for every\nVIIRS cell", fc="#eef4fc", ec=BLUE)
    arrow(10.2, 4.55, 3.0, 3.35, cs="arc3,rad=0.0"); arrow(4.6, 2.5, 5.2, 2.5); arrow(8.5, 2.5, 9.1, 2.5)
    arrow(12.2, 2.5, 12.8, 2.5)
    ax.text(0.3, 0.9, "Scenes: June 2024-06-22 (summer), October 2023-10-04 (freeze-up). Landsat and VIIRS from the same "
            "day; Sentinel-2 12 min after Landsat.", fontsize=10.5, color=INK2)
    fig.savefig(os.path.join(OUT, "17_pipeline.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 17_pipeline.png")

def viirs_cells():
    """18: which VIIRS cells are trusted, June and October + a zoom; cells over the water mask."""
    cols = {1: AQUA, 2: ORANGE, 3: "#9a9994"}
    def layer(d):
        st = read(os.path.join(RM, d, f"{d}_cell_status_375m.tif"))[0][0]
        with rasterio.open(sorted(glob.glob(os.path.join(RM, d, "final", f"{d}_fractions_375m_*.tif")))[-1]) as r:
            done = r.read(list(r.descriptions).index("done") + 1) > 0
            vb = r.bounds
        rgba = np.zeros(st.shape + (4,))
        for v, c in cols.items():
            rgba[st == v, :3] = np.array(hex2rgb(c)) / 255; rgba[st == v, 3] = 0.6
        return rgba, done, vb, st
    _, box = read(os.path.join(P, f"{D}_truecolour.tif"))
    lext = (box.left, box.right, box.bottom, box.top)
    fin = {}
    for d in ("2024-06-22", "2023-10-04"):
        f = read(sorted(glob.glob(os.path.join(RM, d, "final", f"{d}_final_*.tif")))[-1])[0][0]
        g = np.full(f.shape + (3,), 0.93); g[np.isin(f, (1, 2))] = np.array(hex2rgb(BLUE)) / 255
        fin[d] = g
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.42, 0.42, 1.25], height_ratios=[1, 0.08], wspace=0.06, hspace=0.05,
                          left=0.01, right=0.99, top=0.84, bottom=0.02)
    counts = {}
    for i, d in enumerate(("2024-06-22", "2023-10-04")):
        rgba, done, vb, st = layer(d); counts[d] = {k: int((st == k).sum()) for k in (1, 2, 3)}
        ax = fig.add_subplot(gs[0, i]); ax.imshow(fin[d], extent=lext, aspect=1 / ASPECT, interpolation="nearest")
        ax.imshow(rgba, extent=(vb.left, vb.right, vb.bottom, vb.top), aspect=1 / ASPECT, interpolation="nearest")
        ax.set_xlim(box.left, box.right); ax.set_ylim(box.bottom, box.top); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{'June' if d.startswith('2024') else 'October'}\n{counts[d][1]} trusted cells", fontsize=12)
        if i == 0:
            win = PANEL["win"]; T = rasterio.open(os.path.join(P, f"{D}_truecolour.tif")).transform
            (zx0, zy1), (zx1, zy0) = T * (win[1].start, win[0].start), T * (win[1].stop, win[0].stop)
            ax.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=INK, lw=1.2))
            zoomd = (rgba, vb, done)
    rgba, vb, done = zoomd
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(PANEL["final"], extent=lext, aspect=1 / ASPECT, interpolation="nearest")   # June final over grey
    res = (vb.right - vb.left) / rgba.shape[1]
    st = layer("2024-06-22")[3]
    for (r_, c_), v in np.ndenumerate(st):
        if v in cols:
            x0 = vb.left + c_ * res; y0 = vb.top - (r_ + 1) * res
            if zx0 - res < x0 < zx1 and zy0 - res < y0 < zy1:
                ax.add_patch(plt.Rectangle((x0, y0), res, res, fc=cols[v], alpha=0.18, ec="none"))
                ax.add_patch(plt.Rectangle((x0 + res * 0.04, y0 + res * 0.04), res * 0.92, res * 0.92,
                                           fill=False, ec=cols[v], lw=2.2))
    for x in np.arange(vb.left, vb.right + res / 2, res): ax.axvline(x, color="white", lw=0.4, alpha=0.7)
    for y in np.arange(vb.bottom, vb.top + res / 2, res): ax.axhline(y, color="white", lw=0.4, alpha=0.7)
    ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("June, zoom (4.2 × 4.2 km): each box is one VIIRS grid cell", fontsize=12)
    lax = fig.add_subplot(gs[1, :]); lax.set_axis_off()
    lax.legend(handles=[Patch(fc=BLUE, label="river (ice in October)"), Patch(fc=AQUA, alpha=0.6, label="trusted"),
                        Patch(fc=ORANGE, alpha=0.6, label="rejected: river would leave the cell (location error)"),
                        Patch(fc="#9a9994", alpha=0.6, label="rejected: too much unsure / no data")],
               loc="center", ncol=4, frameon=False, fontsize=11)
    fig.text(0.01, 0.955, "Which VIIRS cells can we trust?", fontsize=17, fontweight="bold", va="bottom")
    fig.text(0.01, 0.915, "A cell is trusted if it keeps at least half of its river when VIIRS is shifted by its "
             "location error (±40-50 m) in any of 8 directions.", fontsize=11, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "18_viirs_cells.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 18_viirs_cells.png", counts)

def october_mask():
    """19: October 2023-10-04 final mask (river / ice / snow / land), whole box + zoom with June channel outline."""
    Do = "2023-10-04"; Po = os.path.join(NR, "patches", "nenana_20231004")
    tc = stretch(read(os.path.join(Po, f"{Do}_truecolour.tif"))[0].astype(float))
    f = read(sorted(glob.glob(os.path.join(RM, Do, "final", f"{Do}_final_*.tif")))[-1])[0][0]
    SNOW, ICE = "#9fd8e8", "#7b3fa0"
    rgb = paint(f, {1: BLUE, 2: ICE, 3: SNOW, 4: LAND, 9: UNSURE})
    june = read(sorted(glob.glob(os.path.join(RM, D, "final", f"{D}_final_*.tif")))[-1])[0][0] == 1
    edge = june & ~(np.roll(june, 1, 0) & np.roll(june, -1, 0) & np.roll(june, 1, 1) & np.roll(june, -1, 1))
    win = PANEL["win"]
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.3, 0.3, 1, 1], height_ratios=[1, 0.09], wspace=0.05, hspace=0.05,
                          left=0.01, right=0.99, top=0.82, bottom=0.02)
    for i, (img, ttl) in enumerate(((tc, "Landsat\ntrue colour"), (rgb, "Final mask"))):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(img, aspect=1 / ASPECT, interpolation="nearest")
        ax.add_patch(plt.Rectangle((win[1].start, win[0].start), win[1].stop - win[1].start, win[0].stop - win[0].start,
                                   fill=False, ec=INK, lw=1)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=12)
    z = rgb[win].copy(); z[edge[win]] = hex2rgb("#222222")
    for i, (img, ttl) in enumerate(((tc[win], "Zoom: Landsat true colour"),
                                    (z, "Zoom: final mask  (black line = June channel)"))):
        ax = fig.add_subplot(gs[0, 2 + i]); ax.imshow(img, aspect=1 / ASPECT, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=12)
    lax = fig.add_subplot(gs[1, :]); lax.set_axis_off()
    lax.legend(handles=[Patch(fc=BLUE, label="open river"), Patch(fc=ICE, label="ice"), Patch(fc=SNOW, label="snow"),
                        Patch(fc=LAND, label="land"), Patch(fc=UNSURE, label="unsure"),
                        Line2D([], [], color="#222222", lw=2, label="June channel")],
               loc="center", ncol=6, frameon=False, fontsize=11)
    km2 = (0.00027027 * 111320) ** 2 * ASPECT / 1e6
    a = {k: int((f == v).sum()) * km2 for k, v in (("river", 1), ("ice", 2), ("snow", 3))}
    fig.text(0.01, 0.95, "October 2023-10-04 (freeze-up): river, ice and snow", fontsize=17, fontweight="bold", va="bottom")
    fig.text(0.01, 0.905, f"RF + channel rule: frozen pixels inside the June channel = ice, outside = snow.  "
             f"Open river {a['river']:.1f} km², ice {a['ice']:.1f} km², snow {a['snow']:.1f} km².",
             fontsize=11, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "19_october_mask.png"), dpi=300, facecolor="white"); plt.close(fig)
    print("  wrote 19_october_mask.png", {k: round(v, 1) for k, v in a.items()})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    OUT = a.out; os.makedirs(OUT, exist_ok=True)
    tc, box, rf, final, n = maps()
    zoom()
    sw = sword(tc, box)
    m, seed = unet()
    box_combined(box); unet_combined(seed, m); sword_combined(box, sw); rf_results(); unet_results()
    pipeline(); viirs_cells(); october_mask(); unet_221(m)
    import shutil; shutil.copy(os.path.join(NR, "classify", "qc", "figures", "07_feature_importance.png"),
                               os.path.join(OUT, "21_feature_importance.png"))
    km2 = (0.00027027 * 111320) ** 2 * ASPECT / 1e6                    # one grid cell on the ground
    with open(os.path.join(OUT, "SWORD_notes.txt"), "w", encoding="utf-8") as f:
        f.write(f"""SWORD (SWOT River Database) - slide notes
- What: global river centrelines built for the SWOT satellite mission (Altenau et al. 2021).
  Rivers are split into reaches (~10 km) and nodes (every ~200 m); each node carries a width.
- How we got it: Google Earth Engine community catalogue (projects/sat-io/open-datasets/SWORD),
  pulled with our script fetch_sword_river.py: filter river_name = "Nenana River", type 1 (river),
  no width filter (a 75-300 m filter would drop most of the Nenana).
- What it gave us: {sw['n_reaches']} reaches, {sw['n_nodes']} nodes; median node width {sw['med_all']:.0f} m;
  {sw['narrow']} of {sw['n_nodes']} nodes are narrower than one VIIRS pixel (375 m).
- How we used it: to pick the study box (13 x 49 km, {sw['n_box']} nodes, median width {sw['med_box']:.0f} m),
  to shortlist VIIRS overpasses that see the river, and for the geolocation check.
- Independent check of our maps: river width implied by the RF map ~246 m vs SWORD median in the
  box ~234 m (within 5 %), from data SWORD never saw.
""")
    with open(os.path.join(OUT, "ELEMENTS_notes.txt"), "w", encoding="utf-8") as f:
        f.write(f"""ELEMENTS - keyword notes (rebuild: make_presentation_elements.py; June = 2024-06-22, October = 2023-10-04)

PASTE-READY SLIDES (one image each, legends included)

17_pipeline - method in one picture
- Landsat lane: hand labels -> RF -> June water mask (RF + Sentinel-2 + hand fixes)
  -> October mask (RF + channel rule)
- VIIRS lane: 375 m fractions -> trusted cells -> U-Net -> fractions for every VIIRS cell
- grey box = what we tried and dropped (JRC, looser NDWI, QA cloud mask, balanced weights)

18_viirs_cells - which VIIRS cells we trust
- VIIRS location error ~ +-40-50 m; a narrow channel near a cell edge can fall in the neighbour
- shift the mask 8 ways; keep the cell if it keeps >= half its river
- June 547 trusted, October 509; orange = rejected because the river would leave the cell
- grid is in degrees: a cell is ~162 m E-W x 375 m N-S here, hence the tall boxes
- only trusted (and hand-checked) cells train the U-Net

19_october_mask - October freeze-up mask
- open river blue (10.3 km2), ice purple (7.3 km2), snow light blue (28.4 km2)
- legend colours: ice purple, snow light blue; black line = June channel
- frozen pixels inside the June channel = ice, outside = snow (channel rule)
- needed zero hand corrections in the checked cells

21_feature_importance - which inputs the Random Forest relies on (copy of qc/figures/07)
- importance = how much each input helps the trees separate the classes
  (mean decrease in impurity, "Gini importance"); the bars of one date add up to 100 %
- it is NOT accuracy: a high % means the trees split on that input often and usefully
- "June vs October" = two separate models, each trained on its own date's labels:
  June 2024-06-22 summer (river / land) vs October 2023-10-04 freeze-up (river / ice / snow / land)
  the comparison shows that the useful inputs change with the season
- June: NDSI 27 %, SWIR1 20 %, NDWI 13 % -> water is dark in SWIR, land is not
- October: NDWI 25 %, NDVI 19 %, green 14 % -> snow and ice are bright in the visible,
  so the model leans on NIR-based indices to tell ice / snow / water / vegetation apart
- bands vs indices about 50 / 50 on both dates (June 52 / 48, October 50 / 50)
- correlated inputs share credit (green is also inside NDWI and NDSI): read the ranking, not exact %
- the three indices are normalised differences of two bands: (A - B) / (A + B), from -1 to +1
  (the old chart said "green vs SWIR1" etc.; it now shows the formulas)
  NDWI = (green - NIR) / (green + NIR): water reflects green, absorbs NIR -> water positive
  NDSI = (green - SWIR1) / (green + SWIR1): water, snow and ice absorb SWIR1 -> positive;
         same formula as MNDWI; also separates water from bright gravel (gravel is bright in SWIR)
  NDVI = (NIR - red) / (NIR + red): vegetation reflects NIR, absorbs red -> vegetation positive
- dividing by the sum cancels overall brightness (haze, sun angle), so the index is about colour, not brightness

11_zoom_combined - zoom on braided reach, 4.2 x 4.2 km, June
- 2 x 3: true colour | SWIR 654 | JRC + final outline  /  Sentinel-2 water | RF | final; over Landsat grey
- JRC (Valentina's mask) shaded by % of time wet, our final water mask outlined in orange on top:
  JRC covers the whole braid plain (bars dry on the day too); the outline follows the channels wet that day
- SWIR 654: water black, gravel bright -> why SWIR separates gravel from water
- Sentinel-2: same day, +12 min, 10 m; Landsat cell wet if >= 1 of its 3x3 S2 cells NDWI > 0.10
- final = RF + Sentinel-2 + hand fixes; green = threads only Sentinel-2 caught

12_study_box_combined - whole study box, 13 x 49 km, June
- same five panels as 11
- RF {int((rf == 1).sum()) * km2:.1f} km2 river -> final {int((final == 1).sum()) * km2:.1f} km2 water
- added by Sentinel-2 {n[2] * km2:.1f} km2 ({n[2]} px); hand fixes {n[3]} px (66 polygons)
- JRC (Valentina's mask) tried first, cut-offs 25 / 50 / 75 %: rejected
  long-term average -> wet over dry bars, widens channel, misses narrow channels
- Sentinel-2 NDWI thresholds tried -0.15 ... +0.10: below 0 four- to five-fold more "water"
  (vegetation, damp ground); 0.0 / 0.05 still too much -> kept 0.10 + hand fixes

14_sword_combined - SWORD (see SWORD_notes.txt)
- 14 reaches, 993 nodes, ~200 m spacing, width per node
- most of the river narrower than one VIIRS pixel; study box = widest, braided part
- used for: siting the box, choosing VIIRS overpasses, geolocation check, width sanity check

15_rf_results - Random Forest accuracy
- cross-validation = whole polygons held out (pixels in a polygon are near-copies)
- June: v1 labels, river recall 0.84 / precision 0.93, land recall 0.99
  (current maps use v3: recall 0.86 / precision 0.90 on the same polygons)
- unseen dates: trained on June only, anchored features (haze correction)
  main-channel recall 0.97 / 0.95, land recall 0.96
- October: ice / snow with channel rule (frozen inside June channel = ice, outside = snow)
  ice 0.86 / 0.93, snow 0.89 / 0.80 (recall / precision)
- if asked: 2026-08-01 main channel 0.64 - silty water or channel moved, still open

16_unet_results - first U-Net on VIIRS (pipeline test, not a result)
- input VIIRS I1-I5, 375 m; target = river / ice / snow / land fractions from the Landsat reference
- {CELLS['n']} hand-checked October cells ({CELLS['ch']} river/ice, {CELLS['n'] - CELLS['ch']} land); 75/25 split in 4x4-cell blocks; 5 splits
- test error: naive {np.mean(m['mean']):.3f}, per-cell RF on VIIRS {np.mean(m['rf']):.3f}, U-Net {np.mean(m['unet']):.3f}
- splits overlap -> not a firm win; limited by checked cells, not the model
- next: check more cells (June too), smaller net + early stopping, April out-of-sample

13_unet_october_combined - one October split (split {seed}, the most typical of 5)
- VIIRS input | reference | U-Net prediction, zoomed to the checked corner; orange = test cells
- outside the checked corner the U-Net calls river/ice almost everywhere: never saw those areas

SINGLE PANELS (backups, same content as above)
01 true colour, 02 JRC occurrence (+legend), 03 RF mask, 04 final mask, 04b final by source,
05 SWIR 654, 06 SWORD over the study box (+legend), 08 U-Net error chart,
09a-c October U-Net panels (+legends), 10a-e zoom panels (1200 x 1200 px)
""")
    print("done ->", OUT)
