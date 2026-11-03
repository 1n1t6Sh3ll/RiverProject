"""As transfer_test.py, but every feature is anchored to the date's own dry land (JRC=0 median)."""
import sys, os, csv, numpy as np, rasterio
from sklearn.ensemble import RandomForestClassifier
CL = r"C:\Users\matus\dev\UG Research Assistant\RiverProject\Matus\narrow_rivers\classify"
sys.path.insert(0, CL); import train_rf as T; EX, C = T.EX, T.C
SRC = "2024-06-22"; TARGETS = ["2024-06-22", "2023-07-24", "2025-07-04", "2026-08-01"]; PX_M2 = 390.0
patch = lambda d: os.path.join(EX.PATCHES, f"nenana_{d.replace('-', '')}")
occ = rasterio.open(f"{patch(SRC)}/{SRC}_water_occurrence.tif").read(1).astype(float)
dry = occ == 0; occn = np.where(occ == 255, np.nan, occ)
with rasterio.open(f"{patch(SRC)}/{SRC}_sr.tif") as s: tf, H, W = s.transform, s.height, s.width
ylab, _, _ = EX.label_rasters(f"{patch(SRC)}/labels/labels_{SRC}.shp", (H, W), tf)
stable_river = (ylab == 1) & (occ >= 80) & (occ != 255); land_poly = ylab == 4

def load(d, keep):
    P = patch(d)
    with rasterio.open(f"{P}/{d}_sr.tif") as s: sr = s.read().astype(np.float32); prof = s.profile
    th = rasterio.open(f"{P}/{d}_thermal.tif").read(1).astype(np.float32)
    F = EX.features(sr, th)[keep]
    ok = np.isfinite(F).all(0)
    med = np.array([np.median(F[i][ok & dry]) for i in range(len(F))], np.float32)
    return F - med[:, None, None], ok, prof

rows = [r for r in csv.DictReader(open(f"{CL}/samples_{SRC}.csv")) if int(r["class_id"]) in (1, 4)]
R = np.array([int(r["row"]) for r in rows]); Cc = np.array([int(r["col"]) for r in rows])
y = np.array([int(r["class_id"]) for r in rows])
for drop_thermal in (False,):
    keep = [i for i, f in enumerate(EX.FEATURES) if not (drop_thermal and f == "thermal")]
    Fs, _, _ = load(SRC, keep)
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Fs[:, R, Cc].T, y)
    print(f"\n==== anchored, {'no thermal' if drop_thermal else 'all 10'}   top: " +
          ", ".join(f"{EX.FEATURES[keep[i]]} {v:.2f}" for i, v in sorted(enumerate(rf.feature_importances_), key=lambda t: -t[1])[:3]))
    print(f"{'date':11} {'river km2':>9} {'land rec':>8} {'stable river rec':>16} {'riv in JRC>0':>12} {'JRC>=90 hit':>11} {'conf med':>8} {'conf<0.8':>8}")
    for d in TARGETS:
        F, ok, prof = load(d, keep)
        pr = rf.predict_proba(F[:, ok].T)
        cls = np.full((H, W), C.NODATA_CLASS, np.uint8); conf = np.zeros((H, W), np.float32)
        cls[ok] = rf.classes_[pr.argmax(1)]; conf[ok] = pr.max(1); riv = cls == 1
        c = conf[ok]
        print(f"{d:11} {riv.sum()*PX_M2/1e6:9.2f} {(cls[land_poly]==4).mean():8.3f} {(cls[stable_river]==1).mean():16.3f} "
              f"{(np.nan_to_num(occn[riv])>0).mean():12.1%} {riv[(occ>=90)&(occ!=255)&ok].mean():11.1%} {np.median(c):8.3f} {(c<0.8).mean():8.1%}")
        if not drop_thermal and d != SRC:
            base = os.path.join(patch(d), "rf_transfer", f"{d}_rf_from_20240622_anchored")
            T.write(base + "_class.tif", cls, prof, "uint8", C.NODATA_CLASS, f"rf class, trained on {SRC}, anchored, all 10 features")
            T.write(base + "_confidence.tif", conf, prof, "float32", 0.0, "max probability")
            open(base + "_class.qml", "w", encoding="utf-8").write(T.class_qml(True))
            open(base + "_confidence.qml", "w", encoding="utf-8").write(T.conf_qml(0.8))
