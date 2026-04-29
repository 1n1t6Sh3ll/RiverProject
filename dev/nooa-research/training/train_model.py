"""
train_model.py  (v2)

Build and compare six river-ice classifiers:
  RF and XGBoost on three datasets (A, B, C).

Dataset A — Valentina confirmed only (6 enriched CSVs)
Dataset B — Valentina + Daksh v2 + May-2021 n50 pixels
Dataset C — Valentina + Daksh v2 (no n50)

Saves trained models, plots, and text report to training/model_outputs/.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
OUTPUT_DIR = TRAINING_DIR / "model_outputs"

DATASET_A_FILES = [
    TRAINING_DIR / "viirs_d20221005_t2219/outputs/training_candidates_2022-10-05_enriched.csv",
    TRAINING_DIR / "viirs_d20220311_t2218/outputs/training_candidates_2022-03-11_enriched.csv",
    TRAINING_DIR / "viirs_d20210503_t1947/outputs/training_candidates_2021-05-03_enriched.csv",
    TRAINING_DIR / "viirs_d20200304_t2056/outputs/training_candidates_2020-03-04_enriched.csv",
    TRAINING_DIR / "20230415/training_candidates_2023-04-15_enriched.csv",
    TRAINING_DIR / "20240314/training_candidates_2024-03-14_enriched.csv",
]
DAKSH_V2_FILE = TRAINING_DIR / "daksh_training_candidates_v2_cleaned_enriched.csv"
MAY_N50_FILE = TRAINING_DIR / "viirs_d20210503_t1947/outputs/training_candidates_2021-05-03_n50_enriched.csv"
MAY_ENRICHED_FILE = TRAINING_DIR / "viirs_d20210503_t1947/outputs/training_candidates_2021-05-03_enriched.csv"

FEATURE_COLUMNS = [
    "I1", "I2", "I3", "I4", "I5",
    "SZA", "SAA", "VZA", "VAA",
    "VIIRS_NDWI",
    "water_fraction",
    "modis_ndvi",
]

F1_CLASSES = [
    "ice_free_river_snow_free_land",
    "ice_free_river_snow_land",
    "ice_covered_river_snow_covered_land",
    "ice_covered_river_snow_free_land",
]


class Reporter:
    def __init__(self) -> None:
        self._buf = io.StringIO()

    def log(self, text: str = "") -> None:
        print(text)
        self._buf.write(text + "\n")

    def dump(self, path: Path) -> None:
        path.write_text(self._buf.getvalue(), encoding="utf-8")


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    X: pd.DataFrame
    y_encoded: np.ndarray


@dataclass
class ModelResult:
    key: str
    dataset_name: str
    model_name: str
    model_obj: object
    n_train: int
    n_test: int
    test_accuracy: float
    cv_mean: float
    cv_std: float
    report_text: str
    cm: np.ndarray
    feat_imp: pd.Series
    f1_map: Dict[str, float]


def _sanitize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace("", np.nan)
    return df


def _add_viirs_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIIRS NDWI and NDVI from I1/I2 bands in-place."""
    i1 = df["I1"].replace(0, np.nan)
    i2 = df["I2"].replace(0, np.nan)
    df["VIIRS_NDWI"] = (i1 - i2) / (i1 + i2)
    return df


def load_dataset_a(rep: Reporter) -> pd.DataFrame:
    frames = []
    for p in DATASET_A_FILES:
        df = pd.read_csv(p)
        df = _sanitize(df, ["ground_truth_class", "notes"])
        df["notes"] = df["notes"].fillna("").astype(str)
        df = df[df["ground_truth_class"].notna()]
        df = df[df["water_fraction"] < 0.90]
        df = df[~df["notes"].str.contains("Landsat null", case=False, na=False)]
        df["source"] = "valentina"
        frames.append(df)
    dataset_a = pd.concat(frames, ignore_index=True)
    rep.log("=== Dataset A (Valentina confirmed) ===")
    rep.log(f"Total rows: {len(dataset_a)}")
    for cls, cnt in dataset_a["ground_truth_class"].value_counts().sort_index().items():
        rep.log(f"  {cls}: {cnt}")
        if cnt < 5:
            rep.log(f"  WARNING: '{cls}' has < 5 samples")
    if len(dataset_a) < 30:
        rep.log("WARNING: Dataset A has < 30 samples — interpret cautiously.")
    rep.log("")
    return dataset_a


def load_daksh_v2(rep: Reporter) -> pd.DataFrame:
    df = pd.read_csv(DAKSH_V2_FILE)
    df = _sanitize(df, ["ground_truth_class", "notes"])
    df["notes"] = df["notes"].fillna("").astype(str)
    df = df[df["ground_truth_class"].notna()]
    df = df[~df["notes"].str.contains("CONFLICT", case=False, na=False)]
    df = df[~df["notes"].str.contains("EXCLUDED", case=False, na=False)]
    df = df[df["water_fraction"] < 0.90]
    if "water_fraction_sea" in df.columns:
        df = df[df["water_fraction_sea"] < 1.0]
    df = df[df["modis_ndvi"].notna()]
    for col in ["I3", "I4", "I5"]:
        if col not in df.columns:
            df[col] = np.nan
    rep.log(f"=== Daksh v2 after filters: {len(df)} rows ===")
    for cls, cnt in df["ground_truth_class"].value_counts().sort_index().items():
        rep.log(f"  {cls}: {cnt}")
    rep.log("")
    return df


def load_may_n50(dataset_a: pd.DataFrame, rep: Reporter) -> pd.DataFrame:
    n50 = pd.read_csv(MAY_N50_FILE)
    n50 = _sanitize(n50, ["ground_truth_class", "notes"])
    n50["notes"] = n50["notes"].fillna("").astype(str)
    n50 = n50[n50["ground_truth_class"].notna()]
    n50 = n50[n50["water_fraction"] < 0.90]
    n50 = n50[~n50["notes"].str.contains("Landsat null", case=False, na=False)]

    enriched = pd.read_csv(MAY_ENRICHED_FILE)
    enriched_coords = set(zip(enriched["lat"].round(5), enriched["lon"].round(5)))
    n50["_key"] = list(zip(n50["lat"].round(5), n50["lon"].round(5)))
    n50 = n50[~n50["_key"].isin(enriched_coords)].drop(columns=["_key"])
    n50 = n50[n50["modis_ndvi"].notna()]
    n50["source"] = "valentina_n50"
    rep.log(f"=== May 2021 n50 after filters: {len(n50)} rows ===")
    for cls, cnt in n50["ground_truth_class"].value_counts().sort_index().items():
        rep.log(f"  {cls}: {cnt}")
    rep.log("")
    return n50


def _impute_from_a(combined: pd.DataFrame, dataset_a: pd.DataFrame,
                   rep: Reporter, tag: str) -> pd.DataFrame:
    for col in ["I3", "I4", "I5", "modis_ndvi"]:
        med = dataset_a[col].median(skipna=True)
        missing = combined[col].isna().sum()
        combined[col] = combined[col].fillna(med)
        filled = missing - combined[col].isna().sum()
        if filled > 0:
            rep.log(f"  {tag} imputed {col}: {filled} values → median {med:.6f}")
    return combined


def _dedup_latlon(df: pd.DataFrame, rep: Reporter, tag: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").copy()
    removed = before - len(df)
    if removed:
        rep.log(f"  {tag}: removed {removed} duplicate (lat,lon) rows")
    return df


def build_datasets(dataset_a: pd.DataFrame, daksh: pd.DataFrame,
                   n50: pd.DataFrame, rep: Reporter) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep.log("=== Building Dataset B (A + Daksh v2 + n50) ===")
    b = pd.concat([dataset_a, daksh, n50], ignore_index=True, sort=False)
    b = _dedup_latlon(b, rep, "B")
    b = _impute_from_a(b, dataset_a, rep, "B")
    rep.log(f"  B total: {len(b)}  "
            f"(valentina={int((b['source']=='valentina').sum())}, "
            f"daksh={int((b['source']=='daksh').sum())}, "
            f"n50={int((b['source']=='valentina_n50').sum())})")
    for cls, cnt in b["ground_truth_class"].value_counts().sort_index().items():
        rep.log(f"    {cls}: {cnt}")
    rep.log("")

    rep.log("=== Building Dataset C (A + Daksh v2 only) ===")
    c = pd.concat([dataset_a, daksh], ignore_index=True, sort=False)
    c = _dedup_latlon(c, rep, "C")
    c = _impute_from_a(c, dataset_a, rep, "C")
    rep.log(f"  C total: {len(c)}  "
            f"(valentina={int((c['source']=='valentina').sum())}, "
            f"daksh={int((c['source']=='daksh').sum())})")
    for cls, cnt in c["ground_truth_class"].value_counts().sort_index().items():
        rep.log(f"    {cls}: {cnt}")
    rep.log("")

    return dataset_a, b, c


def _nan_check(df: pd.DataFrame, name: str, rep: Reporter) -> None:
    nan_counts = df[FEATURE_COLUMNS].isna().sum()
    rep.log(f"  NaN per feature in {name}:")
    for col, cnt in nan_counts.items():
        rep.log(f"    {col}: {int(cnt)}")
    for col, cnt in nan_counts.items():
        if cnt / len(df) > 0.30:
            raise RuntimeError(f"STOP: {name} feature '{col}' has >{30}% NaN")


def _verify_modis_enrichment(rep: Reporter) -> None:
    """
    Check that all input CSVs have modis_ndvi populated before training.
    Logs NaN rate per file. Raises RuntimeError if any file has > 10% NaN
    in modis_ndvi (indicates enrich_modis.py was not run on that file).
    """
    files_to_check = DATASET_A_FILES + [DAKSH_V2_FILE, MAY_N50_FILE]
    rep.log("=" * 60)
    rep.log("MODIS ENRICHMENT VERIFICATION")
    rep.log("=" * 60)
    for path in files_to_check:
        if not path.exists():
            rep.log(f"  SKIP (not found): {path.name}")
            continue
        df = pd.read_csv(path)
        if 'modis_ndvi' not in df.columns:
            raise RuntimeError(
                f"STOP: {path.name} has no modis_ndvi column — "
                f"run enrich_modis.py against this file first.")
        # Check NaN rate only among Landsat-confirmed rows — Landsat-null rows
        # are never sent to EE and will always have NaN modis_ndvi; they are
        # filtered out by the loaders before training anyway.
        # (Mirrors _is_confirmed() in enrich_modis.py without importing it.)
        if 'LS_SR_B2' in df.columns:
            notes_col = df.get('notes', pd.Series('', index=df.index))
            confirmed = (
                df['LS_SR_B2'].replace('', np.nan).notna()
                & ~notes_col.astype(str).str.contains('Landsat null', na=False)
            )
        else:
            confirmed = pd.Series(True, index=df.index)
        df_confirmed = df[confirmed]
        n_check  = len(df_confirmed) if len(df_confirmed) else len(df)
        nan_count = int(df_confirmed['modis_ndvi'].isna().sum()) \
                    if len(df_confirmed) else int(df['modis_ndvi'].isna().sum())
        nan_rate  = nan_count / n_check if n_check else 0
        n_total   = len(df)
        flag = "  ← WARNING: > 10% NaN — run enrich_modis.py" \
               if nan_rate > 0.10 else ""
        rep.log(f"  {path.name}: {nan_count}/{n_check} confirmed NaN "
                f"({nan_rate*100:.1f}%) [{n_total} rows total]{flag}")
        if nan_rate > 0.10:
            raise RuntimeError(
                f"STOP: {path.name} modis_ndvi is {nan_rate*100:.0f}% NaN "
                f"among confirmed rows. "
                f"Run: python training/enrich_modis.py --date <date> "
                f"or --file <path>")
    rep.log("")


def make_bundle(name: str, df: pd.DataFrame, le: LabelEncoder) -> DatasetBundle:
    X = df[FEATURE_COLUMNS].copy()
    y = le.transform(df["ground_truth_class"])
    return DatasetBundle(name=name, df=df, X=X, y_encoded=y)


def _xgb_feat_imp(model: XGBClassifier) -> pd.Series:
    gain = model.get_booster().get_score(importance_type="gain")
    vals = [gain.get(c, gain.get(f"f{i}", 0.0))
            for i, c in enumerate(FEATURE_COLUMNS)]
    return pd.Series(vals, index=FEATURE_COLUMNS).sort_values(ascending=False)


def _save_cm_png(cm: np.ndarray, names: List[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _eval(model, X, y, rep, tag):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_tr, idx_te = next(sss.split(X, y))
    te_counts = pd.Series(y[idx_te]).value_counts()
    use_loo = te_counts.min() < 2

    if use_loo:
        rep.log(f"  WARNING: {tag} falling back to LeaveOneOut (class <2 in test)")
        loo = LeaveOneOut()
        y_true = y
        y_pred = cross_val_predict(model, X, y, cv=loo)
        model.fit(X, y)
        return y_true, y_pred, len(X) - 1, 1, accuracy_score(y_true, y_pred)

    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return y_te, y_pred, len(X_tr), len(X_te), accuracy_score(y_te, y_pred)


def _cv(model, X, y, rep, tag):
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        sc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        return float(sc.mean()), float(sc.std())
    except ValueError:
        rep.log(f"  WARNING: {tag} 5-fold CV failed; using LOO")
        sc = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring="accuracy")
        return float(sc.mean()), float(sc.std())


def train_and_eval(key: str, mtype: str, bundle: DatasetBundle,
                   class_names: List[str], rep: Reporter) -> ModelResult:
    if mtype == "RF":
        mdl = RandomForestClassifier(n_estimators=200, max_depth=None,
                                     min_samples_leaf=2, class_weight="balanced",
                                     random_state=42)
    else:
        mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=42)

    y_true, y_pred, n_tr, n_te, acc = _eval(
        mdl, bundle.X, bundle.y_encoded, rep, f"{key}")
    cv_mean, cv_std = _cv(mdl, bundle.X, bundle.y_encoded, rep, key)

    labels = np.arange(len(class_names))
    rep_txt = classification_report(y_true, y_pred, labels=labels,
                                    target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rd = classification_report(y_true, y_pred, labels=labels,
                               target_names=class_names, output_dict=True,
                               zero_division=0)
    f1_map = {c: float(rd.get(c, {}).get("f1-score", 0.0)) for c in F1_CLASSES}

    if mtype == "RF":
        fi = pd.Series(mdl.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    else:
        fi = _xgb_feat_imp(mdl)

    rep.log(f"\n=== {key} ({mtype} on {bundle.name}) ===")
    rep.log(rep_txt)
    rep.log(f"Confusion matrix:\n{cm}")
    rep.log("Feature importances:")
    for f, v in fi.items():
        rep.log(f"  {f}: {v:.6f}")
    rep.log("")

    return ModelResult(key=key, dataset_name=bundle.name, model_name=mtype,
                       model_obj=mdl, n_train=n_tr, n_test=n_te,
                       test_accuracy=acc, cv_mean=cv_mean, cv_std=cv_std,
                       report_text=rep_txt, cm=cm, feat_imp=fi, f1_map=f1_map)


def print_pretraining(bundles: Dict[str, DatasetBundle], class_names: List[str],
                      le: LabelEncoder, rep: Reporter) -> None:
    rep.log("=" * 60)
    rep.log("VERIFICATION BEFORE TRAINING")
    rep.log("=" * 60)
    rep.log(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    rep.log("Label encoding:")
    for c in class_names:
        rep.log(f"  {c} -> {int(le.transform([c])[0])}")
    for name, b in bundles.items():
        rep.log(f"\n{name} shape: {b.X.shape}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        i_tr, i_te = next(sss.split(b.X, b.y_encoded))
        rep.log(f"  Train class counts: {pd.Series(b.y_encoded[i_tr]).value_counts().sort_index().to_dict()}")
        rep.log(f"  Test  class counts: {pd.Series(b.y_encoded[i_te]).value_counts().sort_index().to_dict()}")
        nan_tot = int(b.X.isna().sum().sum())
        rep.log(f"  Total NaN in features: {nan_tot}")
        if nan_tot > 0:
            for col, cnt in b.X.isna().sum().items():
                if cnt > 0:
                    rep.log(f"    {col}: {int(cnt)}")
        class_counts = b.df["ground_truth_class"].value_counts()
        for cls in F1_CLASSES:
            if class_counts.get(cls, 0) < 10:
                rep.log(f"  WARNING: '{cls}' has only {class_counts.get(cls, 0)} samples")
    rep.log("")


def comparison_table(results: Dict[str, ModelResult], rep: Reporter) -> None:
    order = ["RF-B", "RF-C", "XGB-B", "XGB-C"]
    hdr = f"{'Metric':<25}" + "".join(f"| {k:<8}" for k in order)
    rep.log("\n" + "=" * 60)
    rep.log("COMPARISON TABLE")
    rep.log("=" * 60)
    rep.log(hdr)
    rep.log("-" * len(hdr))

    def _row(label, vals):
        rep.log(f"{label:<25}" + "".join(f"| {v:<8}" for v in vals))

    _row("n_train", [str(results[k].n_train) for k in order])
    _row("n_test", [str(results[k].n_test) for k in order])
    _row("Test accuracy", [f"{results[k].test_accuracy:.4f}" for k in order])
    _row("CV mean accuracy", [f"{results[k].cv_mean:.4f}" for k in order])
    _row("CV std", [f"{results[k].cv_std:.4f}" for k in order])
    short = {"ice_free_river_snow_free_land": "F1 ice_free_snow_free",
             "ice_free_river_snow_land": "F1 ice_free_snow_land",
             "ice_covered_river_snow_covered_land": "F1 ice_cov_snow_cov",
             "ice_covered_river_snow_free_land": "F1 ice_cov_snow_free"}
    for full, label in short.items():
        _row(label, [f"{results[k].f1_map[full]:.4f}" for k in order])
    rep.log("")


def save_feat_imp_plot(results: Dict[str, ModelResult], path: Path) -> None:
    order = ["RF-B", "RF-C", "XGB-B", "XGB-C"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key in zip(axes.flatten(), order):
        s = results[key].feat_imp.sort_values(ascending=True)
        ax.barh(s.index, s.values)
        ax.set_title(key)
        ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = Reporter()
    _verify_modis_enrichment(rep)

    ds_a = load_dataset_a(rep)
    daksh = load_daksh_v2(rep)
    n50 = load_may_n50(ds_a, rep)

    ds_a_clean, ds_b, ds_c = build_datasets(ds_a, daksh, n50, rep)

    # Compute VIIRS derived indices on all datasets
    for df in [ds_a_clean, ds_b, ds_c]:
        _add_viirs_indices(df)

    for tag, df in [("A", ds_a_clean), ("B", ds_b), ("C", ds_c)]:
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        _nan_check(df, f"Dataset {tag}", rep)

    all_classes = sorted(
        set(ds_a_clean["ground_truth_class"].unique())
        | set(ds_b["ground_truth_class"].unique())
        | set(ds_c["ground_truth_class"].unique())
    )
    le = LabelEncoder().fit(all_classes)
    class_names = list(le.classes_)

    bb = make_bundle("Dataset B", ds_b, le)
    bc = make_bundle("Dataset C", ds_c, le)
    bundles = {"Dataset B": bb, "Dataset C": bc}

    print_pretraining(bundles, class_names, le, rep)

    results: Dict[str, ModelResult] = {}
    for key, mtype, bundle in [
        ("RF-B",  "RF",  bb), ("RF-C",  "RF",  bc),
        ("XGB-B", "XGB", bb), ("XGB-C", "XGB", bc),
    ]:
        results[key] = train_and_eval(key, mtype, bundle, class_names, rep)

    comparison_table(results, rep)

    for key, fname in [("RF-B","rf_B.pkl"),("RF-C","rf_C.pkl"),
                       ("XGB-B","xgb_B.pkl"),("XGB-C","xgb_C.pkl")]:
        joblib.dump(results[key].model_obj, OUTPUT_DIR / fname)
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")

    for key in results:
        tag = key.lower().replace("-", "_")
        _save_cm_png(results[key].cm, class_names,
                     OUTPUT_DIR / f"confusion_matrix_{tag}.png",
                     f"Confusion Matrix {key}")

    save_feat_imp_plot(results, OUTPUT_DIR / "feature_importance_comparison_v2.png")
    rep.dump(OUTPUT_DIR / "comparison_report_v2.txt")
    rep.log(f"\nAll artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
