"""Context-feature test, June 2024-06-22. Reads samples + rasters; writes nothing to the repo."""
import sys, os, csv, numpy as np, rasterio
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
CL = r"C:\Users\matus\dev\UG Research Assistant\RiverProject\Matus\narrow_rivers\classify"
sys.path.insert(0, CL); import extract_samples as EX
DATE = "2024-06-22"; P = os.path.join(EX.PATCHES, "nenana_20240622")
NARROW = {83, 101, 89, 91, 92}
PX_M2 = 390.0

sr = rasterio.open(f"{P}/{DATE}_sr.tif").read().astype(np.float32)
th = rasterio.open(f"{P}/{DATE}_thermal.tif").read(1).astype(np.float32)
base = EX.features(sr, th)                                   # 10 x H x W
valid = np.isfinite(base).all(0)
ndwi, nir = base[7], base[3]

def nanmean(a, k):
    m = np.isfinite(a); z = np.where(m, a, 0)
    s = ndi.uniform_filter(z, k); c = ndi.uniform_filter(m.astype(np.float32), k)
    return np.where(c > 0, s / np.maximum(c, 1e-6), np.nan)
nd_f = np.where(np.isfinite(ndwi), ndwi, -1)
ctx = {
    "ndwi_max3": ndi.maximum_filter(nd_f, 3),
    "ndwi_max5": ndi.maximum_filter(nd_f, 5),
    "ndwi_rel7": ndwi - nanmean(ndwi, 7),
    "nir_rel7":  nir - nanmean(nir, 7),
    "ndwi_std5": np.sqrt(np.maximum(nanmean(ndwi**2, 5) - nanmean(ndwi, 5)**2, 0)),
    "dist_water": np.minimum(ndi.distance_transform_edt(~(nd_f > 0.1)), 50).astype(np.float32),
}
occ = rasterio.open(f"{P}/{DATE}_water_occurrence.tif").read(1).astype(np.float32)
occ[occ == 255] = 0
jrc = {"jrc": occ, "jrc_max3": ndi.maximum_filter(occ, 3)}

rows = [r for r in csv.DictReader(open(f"{CL}/samples_{DATE}.csv")) if int(r["class_id"]) in (1, 4)]
R = np.array([int(r["row"]) for r in rows]); Cc = np.array([int(r["col"]) for r in rows])
y = np.array([int(r["class_id"]) for r in rows]); g = np.array([int(r["poly_id"]) for r in rows])
size = {p: (g == p).sum() for p in set(g)}
small_river = np.array([size[p] < 20 for p in g]) & (y == 1)
narrow = np.isin(g, list(NARROW)) & (y == 1)

def stack(names_ctx, names_jrc):
    layers = list(base) + [ctx[n] for n in names_ctx] + [jrc[n] for n in names_jrc]
    return np.stack(layers)

VARIANTS = {
    "A baseline":       ([], []),
    "B + image context": (list(ctx), []),
    "C + JRC":          ([], list(jrc)),
    "D context + JRC":  (list(ctx), list(jrc)),
}
qa = EX.qa_bad_mask(f"{P}/{DATE}_qa_pixel.tif")
pred_mask = valid & ~qa
for name, (nc, nj) in VARIANTS.items():
    F = stack(nc, nj); names = EX.FEATURES + nc + nj
    X = F[:, R, Cc].T
    ok = np.isfinite(X).all(1); X = np.where(np.isfinite(X), X, 0)
    oof = np.zeros_like(y)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    rv, ld = y == 1, y == 4
    rec = lambda m: (oof[m] == 1).mean()
    prec = ((oof == 1) & rv).sum() / (oof == 1).sum()
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X, y)
    Xs = F[:, pred_mask].T; Xs = np.where(np.isfinite(Xs), Xs, 0)
    area = (rf.predict(Xs) == 1).sum() * PX_M2 / 1e6
    imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])[:4]
    print(f"\n{name}   ({len(names)} features)")
    print(f"  river recall {rec(rv):.3f}  precision {prec:.3f}  | land->river {((oof==1)&ld).sum()}  land recall {(oof[ld]==4).mean():.3f}")
    print(f"  narrow polys 83/101/89/91/92: {rec(narrow):.3f}   river polys <20 px: {rec(small_river):.3f}")
    print("  per narrow poly: " + "  ".join(f"{p}:{(oof[(g==p)]==1).mean():.2f}" for p in sorted(NARROW)))
    print(f"  scene river area {area:.2f} km2   top features: " + ", ".join(f"{n} {v:.2f}" for n, v in imp))
