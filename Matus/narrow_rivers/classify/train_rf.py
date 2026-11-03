"""
Train a Random Forest on one date's hand labels and extend them to the rest of
that scene ("Mode A" -- within-date extension, agreed 2026-10-25).

Why one model per date: illumination and surface change between seasons, and a
pooled model would be tempted to recognise the date instead of the surface. A
summer scene therefore gets a 2-class model (river, land) -- which is physically
right, there is no ice or snow in it -- and a freeze-up scene gets all four.

Validation holds out WHOLE POLYGONS, never random pixels: pixels inside one
hand-drawn polygon are near-duplicates, so a random split would test the model
on copies of its own training data and report a number that means nothing.

Honesty of the result: this measures "does the model reproduce my labelling on
this scene", not "can it label a scene it has never seen". For that, ice and snow
would have to exist on more than one date.

Nothing is overwritten: outputs are new files named <date>_rf_*, and the hand
labels are only ever read.

Outputs (in patches/<patch>/rf/):
  <date>_rf_class.tif        1 river, 2 ice, 3 snow, 4 land, 9 unknown/masked
  <date>_rf_confidence.tif   max class probability, float32
  <date>_rf_class.qml        styled with the label-layer colours
  <date>_rf_confidence.qml
  <date>_rf_audit.geojson    random confident pixels, class hidden, for blind checking
  <date>_rf_audit_answers.csv  what the model said for those points — do not peek first
  <date>_rf_cv.json          cross-validation matrix + importances, for the accuracy figure

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\train_rf.py --date 2024-06-22
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\train_rf.py --date 2023-10-04 --min-conf 0.8
"""

import os, sys, csv, json, argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
sys.path.insert(0, HERE)
import patch_config as C
import extract_samples as EX

N_SPLITS = 5
AUDIT_N = 200
OUT_SUBDIR = "rf"      # everything this script writes lives in <patch>/rf/


# ── data ─────────────────────────────────────────────────────────────────────

def load_samples(date, drop_qa):
    path = os.path.join(HERE, f"samples_{date}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {path} missing — run extract_samples.py first")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    keep = [r for r in rows if int(r["class_id"]) in C.CLASSES]       # not class 9
    n_nodata = len(rows) - len(keep)
    n_qa = sum(1 for r in keep if int(r["qa_bad"]))
    if drop_qa and n_qa:
        keep = [r for r in keep if not int(r["qa_bad"])]

    X = np.array([[float(r[f]) for f in EX.FEATURES] for r in keep], dtype=np.float32)
    y = np.array([int(r["class_id"]) for r in keep])
    g = np.array([int(r["poly_id"]) for r in keep])
    print(f"  {len(keep):,} samples, {len(set(g))} polygons, "
          f"{len(set(y))} classes present")
    if n_nodata:
        print(f"    {n_nodata} class-9 (no data) rows excluded from training")
    if n_qa:
        print(f"    {n_qa} QA-flagged rows " +
              ("dropped" if drop_qa else "KEPT (pass --drop-qa to exclude)"))
    return X, y, g


# ── reporting ────────────────────────────────────────────────────────────────

def report(y_true, y_pred, classes):
    names = dict(C.CLASSES)
    print("\n  confusion matrix (rows = your label, columns = model)")
    head = "".join(f"{names[c][:5]:>7}" for c in classes)
    print(f"    {'':9}{head}{'recall':>9}")
    for c in classes:
        m = y_true == c
        row = "".join(f"{int(((y_pred == p) & m).sum()):7d}" for p in classes)
        rec = (y_pred[m] == c).mean() if m.sum() else float("nan")
        print(f"    {names[c]:9}{row}{rec:9.3f}")
    for c in classes:                                  # precision reads the other way
        p = y_pred == c
        prec = (y_true[p] == c).mean() if p.sum() else float("nan")
        print(f"      precision {names[c]:8} {prec:.3f}")
    print(f"    overall agreement {100 * (y_true == y_pred).mean():.2f} % "
          f"(land-heavy, so read the per-class numbers instead)")


def importances(rf):
    print("\n  feature importances")
    order = np.argsort(rf.feature_importances_)[::-1]
    for i in order:
        bar = "#" * int(round(40 * rf.feature_importances_[i] /
                              rf.feature_importances_.max()))
        print(f"    {EX.FEATURES[i]:8} {rf.feature_importances_[i]:.3f}  {bar}")


# ── styles ───────────────────────────────────────────────────────────────────

def class_qml(see_through=True):
    """Paletted style. see_through: every class keeps its colour but is drawn
    half transparent, so the imagery underneath stays readable and the result can
    be checked against it. Otherwise every class is solid."""
    alpha = {v: (140 if see_through else 255)
             for v in list(C.CLASSES) + [C.NODATA_CLASS]}
    items = "".join(
        f'     <paletteEntry value="{v}" alpha="{alpha[v]}" '
        f'color="{C.CLASS_COLOURS[v]}" '
        f'label="{v} {C.CLASSES.get(v, "unknown / masked")}"/>\n'
        for v in list(C.CLASSES) + [C.NODATA_CLASS])
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="paletted" band="1" opacity="1">\n'
            "   <colorPalette>\n" + items + "   </colorPalette>\n"
            "  </rasterrenderer></pipe></qgis>\n")


def conf_qml(min_conf):
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">\n'
            '   <rastershader><colorrampshader colorRampType="INTERPOLATED" clip="0">\n'
            '     <item value="0.25" color="#d73027" label="guessing"/>\n'
            f'     <item value="{min_conf - 0.001:.3f}" color="#fee08b" label="below the gate"/>\n'
            f'     <item value="{min_conf:.3f}" color="#ffffff" label="{min_conf:.2f} gate"/>\n'
            '     <item value="1" color="#1a9850" label="1 certain"/>\n'
            "   </colorrampshader></rastershader>\n"
            "  </rasterrenderer></pipe></qgis>\n")


def write(path, data, src_profile, dtype, nodata, desc):
    prof = dict(src_profile)
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)
        dst.set_band_description(1, desc)
    print(f"    {os.path.basename(path):34} {os.path.getsize(path)/1e6:6.1f} MB")


# ── main ─────────────────────────────────────────────────────────────────────

def main(date, min_conf, n_estimators, drop_qa, audit_n, seed, class_weight):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    out_dir = os.path.join(patch, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 66)
    print(f"Random Forest — {date}  (Mode A: train and predict within this date)")
    print("=" * 66)

    X, y, g = load_samples(date, drop_qa)
    classes = sorted(set(y))
    if min_conf is None:
        min_conf = 0.9 if len(classes) <= 2 else 0.8
        print(f"  confidence gate {min_conf} (default for a "
              f"{len(classes)}-class model)")

    def forest():
        # class_weight stays None by default. "balanced" inflates the rare classes
        # and pushes their boundary outwards: on 2023-10-04 it produced 37.9 km2 of
        # river against 11.9 km2 without it (June reference 16.0 km2). Good for
        # recall on a rare class, wrong for the unbiased AREA a fraction target needs.
        return RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=5, class_weight=class_weight,
            n_jobs=-1, random_state=seed)

    # ── validation: whole polygons held out ──────────────────────────────────
    n_groups = len(set(g))
    splits = min(N_SPLITS, n_groups)
    print(f"\n  cross-validation: {splits} folds, whole polygons held out")
    oof = np.zeros_like(y)
    for k, (tr, te) in enumerate(
            StratifiedGroupKFold(n_splits=splits, shuffle=True,
                                 random_state=seed).split(X, y, groups=g), 1):
        rf = forest().fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
        print(f"    fold {k}: trained on {len(tr):,} px / "
              f"{len(set(g[tr]))} polygons, tested on {len(te):,} px / "
              f"{len(set(g[te]))} polygons")
    report(y, oof, classes)
    cv = {"date": date, "classes": [int(c) for c in classes],
          "n_samples": int(len(y)), "n_polygons": int(n_groups),
          "folds": int(splits), "class_weight": class_weight,
          "matrix": [[int(((oof == pc) & (y == tc)).sum()) for pc in classes]
                     for tc in classes],
          "support": {int(c): int((y == c).sum()) for c in classes}}

    # ── final model on everything ───────────────────────────────────────────
    rf = forest().fit(X, y)
    importances(rf)
    cv["importances"] = {f: float(v) for f, v in
                         zip(EX.FEATURES, rf.feature_importances_)}
    with open(os.path.join(out_dir, f"{date}_rf_cv.json"), "w",
              encoding="utf-8") as f:
        json.dump(cv, f, indent=1)

    # ── predict the whole scene ─────────────────────────────────────────────
    print("\n  predicting the scene")
    with rasterio.open(os.path.join(patch, f"{date}_sr.tif")) as src:
        sr = src.read().astype(np.float32)
        profile = src.profile
        transform, H, W = src.transform, src.height, src.width
    with rasterio.open(os.path.join(patch, f"{date}_thermal.tif")) as src:
        thermal = src.read(1).astype(np.float32)

    feat = EX.features(sr, thermal)
    finite = np.isfinite(feat).all(axis=0)
    # QA_PIXEL is deliberately NOT used: CFMask flags silty summer water and river
    # ice as cloud (77.7 % of the main channel on 2023-07-24, at river temperature).
    # Scenes are checked for cloud by eye, and no box has cold cloud-top pixels.
    predictable = finite
    print(f"    {W} x {H} = {W*H:,} cells; {int(predictable.sum()):,} predictable "
          f"({int((~finite).sum()):,} no data)")

    cls = np.full((H, W), C.NODATA_CLASS, dtype=np.uint8)
    conf = np.zeros((H, W), dtype=np.float32)
    flat = feat[:, predictable].T
    proba = rf.predict_proba(flat)
    cls[predictable] = rf.classes_[proba.argmax(1)].astype(np.uint8)
    conf[predictable] = proba.max(1)

    low = predictable & (conf < min_conf)
    cls[low] = C.NODATA_CLASS
    print(f"    {int(low.sum()):,} cells below the {min_conf} gate -> class 9 "
          f"({100*low.sum()/max(1, predictable.sum()):.1f} % of predictable)")

    # hand labels always win over the model
    y_ras, _, _ = EX.label_rasters(
        os.path.join(patch, "labels", f"labels_{date}.shp"), (H, W), transform)
    hand = y_ras > 0
    cls[hand] = y_ras[hand]
    conf[hand] = 1.0
    print(f"    {int(hand.sum()):,} hand-labelled cells kept as drawn")

    names = dict(C.CLASSES); names[C.NODATA_CLASS] = "unknown"
    print("\n  resulting map")
    for c in list(C.CLASSES) + [C.NODATA_CLASS]:
        n = int((cls == c).sum())
        if n:
            print(f"    {names[c]:8} {n:9,} cells  {100*n/(W*H):5.1f} %")

    base = os.path.join(out_dir, f"{date}_rf")
    write(base + "_class.tif", cls, profile, "uint8", C.NODATA_CLASS, "rf class")
    write(base + "_confidence.tif", conf, profile, "float32", 0.0, "max probability")
    open(base + "_class.qml", "w", encoding="utf-8").write(class_qml(True))
    open(base + "_class_solid.qml", "w", encoding="utf-8").write(class_qml(False))
    open(base + "_confidence.qml", "w", encoding="utf-8").write(conf_qml(min_conf))

    # ── blind audit sample ──────────────────────────────────────────────────
    pool = np.flatnonzero(((cls != C.NODATA_CLASS) & ~hand).ravel())
    if pool.size and audit_n:
        rng = np.random.default_rng(seed)
        pick = rng.choice(pool, size=min(audit_n, pool.size), replace=False)
        feats = []
        answers = []
        for n, idx in enumerate(pick, 1):
            r, c = divmod(int(idx), W)
            lon, lat = transform * (c + 0.5, r + 0.5)
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"audit_id": n}})
            answers.append([n, r, c, f"{lat:.6f}", f"{lon:.6f}",
                            int(cls[r, c]), f"{conf[r, c]:.3f}"])
        with open(base + "_audit.geojson", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, indent=1)
        with open(os.path.join(out_dir, f"{date}_rf_audit_answers.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audit_id", "row", "col", "lat", "lon",
                        "model_class", "confidence"])
            w.writerows(answers)
        print(f"    {os.path.basename(base)}_audit.geojson     {len(feats)} points, "
              f"class hidden — classify them blind, then compare with "
              f"{OUT_SUBDIR}/{date}_rf_audit_answers.csv")

    print("\n  done. Nothing was written to the label shapefile.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random Forest within one date.")
    p.add_argument("--date", required=True)
    p.add_argument("--min-conf", type=float, default=None,
                   help="confidence gate; default 0.9 for 2 classes, 0.8 for more")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--drop-qa", action="store_true",
                   help="exclude labelled pixels that QA_PIXEL flags")
    p.add_argument("--audit-n", type=int, default=AUDIT_N)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class-weight", default=None, choices=[None, "balanced"],
                   help="default None; 'balanced' helps rare-class recall but "
                        "overstates that class's area")
    a = p.parse_args()
    main(a.date, a.min_conf, a.n_estimators, a.drop_qa, a.audit_n, a.seed,
         a.class_weight)
