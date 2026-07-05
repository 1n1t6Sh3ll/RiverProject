"""
compare_all_models.py

Runs CNN, Random Forest, and XGBoost on the same stratified 80/20 split
and produces a unified comparison report with accuracy, CV, F1, Dice, IoU.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train_cnn import (
    RiverIceCNN1D, prepare_datasets, to_tensors,
    FEATURE_COLUMNS, N_FEATURES, SEED,
    train_epoch, VALID_LABELS,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EPOCHS = 150

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def dice_iou(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    iou  =     tp / (     tp + fp + fn + 1e-9)
    return dice, iou


def per_class_f1(y_true, y_pred, n_classes):
    rd = {}
    for c in range(n_classes):
        mask = (y_true == c) | (y_pred == c)
        if mask.sum() == 0:
            rd[c] = 0.0
        else:
            tp = ((y_true == c) & (y_pred == c)).sum()
            fp = ((y_true != c) & (y_pred == c)).sum()
            fn = ((y_true == c) & (y_pred != c)).sum()
            p  = tp / (tp + fp + 1e-9)
            r  = tp / (tp + fn + 1e-9)
            rd[c] = 2 * p * r / (p + r + 1e-9)
    return rd


# ── CNN training (same logic as train_cnn.py) ─────────────────────────────────

def train_cnn(Xtr, ytr, Xva, yva, n_classes, device):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    Xtr_t, ytr_t = to_tensors(Xtr, ytr)
    Xva_t, _     = to_tensors(Xva, yva)

    model     = RiverIceCNN1D(N_FEATURES, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    counts    = np.bincount(ytr, minlength=n_classes).astype(float)
    weights   = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader    = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=16, shuffle=True)

    best_acc, best_state = -1.0, None
    for _ in range(EPOCHS):
        train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        with torch.no_grad():
            model.eval()
            va_pred = model(Xva_t.to(device)).cpu().argmax(1).numpy()
            va_acc  = accuracy_score(yva, va_pred)
        model.train()
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def cnn_cv(X, y, n_classes, device, epochs=80):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr, va in skf.split(X, y):
        sc   = StandardScaler().fit(X[tr])
        Xtr  = sc.transform(X[tr]); Xva = sc.transform(X[va])
        Xtr_t, ytr_t = to_tensors(Xtr, y[tr])
        Xva_t, _     = to_tensors(Xva, y[va])
        mdl  = RiverIceCNN1D(N_FEATURES, n_classes).to(device)
        opt  = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
        cnt  = np.bincount(y[tr], minlength=n_classes).astype(float)
        w    = torch.tensor(1.0 / (cnt + 1e-6), dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        ldr  = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=16, shuffle=True)
        for _ in range(epochs):
            train_epoch(mdl, ldr, opt, crit, device)
        with torch.no_grad():
            mdl.eval()
            preds = mdl(Xva_t.to(device)).cpu().argmax(1).numpy()
        accs.append(accuracy_score(y[va], preds))
    return float(np.mean(accs)), float(np.std(accs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    le = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    datasets = prepare_datasets()
    df = datasets["Combined"]
    print(f"Dataset: {len(df)} rows, {n_classes} classes\n")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = le.transform(df["ground_truth_class"])

    # Shared 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, y))

    scaler = StandardScaler().fit(X[tr_idx])
    Xtr_s  = scaler.transform(X[tr_idx])
    Xte_s  = scaler.transform(X[te_idx])
    ytr, yte = y[tr_idx], y[te_idx]

    # Val split for CNN early stopping (10% of train)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    sub_tr, sub_va = next(sss_val.split(Xtr_s, ytr))

    results = {}

    # ── 1. CNN ────────────────────────────────────────────────────────────────
    print("Training CNN...")
    cnn = train_cnn(Xtr_s[sub_tr], ytr[sub_tr], Xtr_s[sub_va], ytr[sub_va], n_classes, device)
    Xte_t, _ = to_tensors(Xte_s, yte)
    with torch.no_grad():
        cnn.eval()
        cnn_preds = cnn(Xte_t.to(device)).cpu().argmax(1).numpy()
    cnn_acc = accuracy_score(yte, cnn_preds)
    cnn_cm  = confusion_matrix(yte, cnn_preds, labels=list(range(n_classes)))
    cnn_cv_mean, cnn_cv_std = cnn_cv(X, y, n_classes, device, epochs=80)
    results["CNN"] = dict(acc=cnn_acc, preds=cnn_preds, cm=cnn_cm,
                          cv_mean=cnn_cv_mean, cv_std=cnn_cv_std,
                          report=classification_report(yte, cnn_preds,
                                  target_names=class_names, zero_division=0))
    print(f"  CNN test accuracy: {cnn_acc:.4f}  CV: {cnn_cv_mean:.4f}±{cnn_cv_std:.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def manual_cv(make_model_fn, X_cv, y_cv):
        accs = []
        for tr, va in skf.split(X_cv, y_cv):
            m = make_model_fn()
            m.fit(X_cv[tr], y_cv[tr])
            accs.append(accuracy_score(y_cv[va], m.predict(X_cv[va])))
        return float(np.mean(accs)), float(np.std(accs))

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=1)
    rf.fit(Xtr_s, ytr)
    rf_preds   = rf.predict(Xte_s)
    rf_acc     = accuracy_score(yte, rf_preds)
    rf_cm      = confusion_matrix(yte, rf_preds, labels=list(range(n_classes)))
    rf_cv_mean, rf_cv_std = manual_cv(
        lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=SEED, n_jobs=1),
        Xtr_s, ytr)
    results["RF"] = dict(acc=rf_acc, preds=rf_preds, cm=rf_cm,
                         cv_mean=rf_cv_mean, cv_std=rf_cv_std,
                         report=classification_report(yte, rf_preds,
                                 target_names=class_names, zero_division=0))
    print(f"  RF  test accuracy: {rf_acc:.4f}  CV: {rf_cv_mean:.4f}±{rf_cv_std:.4f}")

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        eval_metric="mlogloss", device="cpu",
                        random_state=SEED, n_jobs=1)
    xgb.fit(Xtr_s, ytr)
    xgb_preds   = xgb.predict(Xte_s)
    xgb_acc     = accuracy_score(yte, xgb_preds)
    xgb_cm      = confusion_matrix(yte, xgb_preds, labels=list(range(n_classes)))
    xgb_cv_mean, xgb_cv_std = manual_cv(
        lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              eval_metric="mlogloss", device="cpu",
                              random_state=SEED, n_jobs=1),
        Xtr_s, ytr)
    results["XGB"] = dict(acc=xgb_acc, preds=xgb_preds, cm=xgb_cm,
                          cv_mean=xgb_cv_mean, cv_std=xgb_cv_std,
                          report=classification_report(yte, xgb_preds,
                                  target_names=class_names, zero_division=0))
    print(f"  XGB test accuracy: {xgb_acc:.4f}  CV: {xgb_cv_mean:.4f}±{xgb_cv_std:.4f}")

    # ── Build report ──────────────────────────────────────────────────────────
    buf = io.StringIO()

    def w(line=""):
        print(line)
        buf.write(line + "\n")

    models = ["CNN", "RF", "XGB"]
    col = 28

    w("\n" + "=" * 70)
    w("MODEL COMPARISON REPORT — CNN vs Random Forest vs XGBoost")
    w(f"Dataset: Combined ({len(df)} rows)  |  Split: 80/20 stratified  |  Seed: {SEED}")
    w("=" * 70)

    # ── Overall metrics ───────────────────────────────────────────────────────
    w("\n--- OVERALL METRICS ---")
    hdr = f"{'Metric':<{col}}" + "".join(f"| {m:<12}" for m in models)
    w(hdr)
    w("-" * len(hdr))

    def row(label, vals):
        w(f"{label:<{col}}" + "".join(f"| {v:<12}" for v in vals))

    row("n_train",               [str(len(tr_idx))] * 3)
    row("n_test",                [str(len(te_idx))] * 3)
    row("Test accuracy",         [f"{results[m]['acc']:.4f}" for m in models])
    row("Balanced accuracy",     [f"{balanced_accuracy_score(yte, results[m]['preds']):.4f}" for m in models])
    row("CV mean (5-fold)",      [f"{results[m]['cv_mean']:.4f}" for m in models])
    row("CV std",                [f"{results[m]['cv_std']:.4f}" for m in models])
    row("Macro F1",              [f"{f1_score(yte, results[m]['preds'], average='macro', zero_division=0):.4f}" for m in models])
    row("Weighted F1",           [f"{f1_score(yte, results[m]['preds'], average='weighted', zero_division=0):.4f}" for m in models])
    row("Macro Precision",       [f"{precision_score(yte, results[m]['preds'], average='macro', zero_division=0):.4f}" for m in models])
    row("Macro Recall",          [f"{recall_score(yte, results[m]['preds'], average='macro', zero_division=0):.4f}" for m in models])
    row("Cohen's Kappa",         [f"{cohen_kappa_score(yte, results[m]['preds']):.4f}" for m in models])
    row("Matthews CC",           [f"{matthews_corrcoef(yte, results[m]['preds']):.4f}" for m in models])

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    w("\n--- PER-CLASS F1 ---")
    w(hdr)
    w("-" * len(hdr))
    for i, cls in enumerate(class_names):
        short = cls.replace("ice_", "").replace("_river_", " ").replace("_land", "")
        vals  = []
        for m in models:
            cm = results[m]["cm"]
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p  = tp / (tp + fp + 1e-9)
            r  = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            vals.append(f"{f1:.4f}")
        row(short, vals)

    # ── Dice & IoU ────────────────────────────────────────────────────────────
    w("\n--- DICE COEFFICIENT (per class) ---")
    w(hdr)
    w("-" * len(hdr))
    for i, cls in enumerate(class_names):
        short = cls.replace("ice_", "").replace("_river_", " ").replace("_land", "")
        vals  = []
        for m in models:
            d, _ = dice_iou(results[m]["cm"])
            vals.append(f"{d[i]:.4f}")
        row(short, vals)
    macro_dice = []
    for m in models:
        d, _ = dice_iou(results[m]["cm"])
        macro_dice.append(f"{d.mean():.4f}")
    row("Macro average", macro_dice)

    w("\n--- IoU / JACCARD (per class) ---")
    w(hdr)
    w("-" * len(hdr))
    for i, cls in enumerate(class_names):
        short = cls.replace("ice_", "").replace("_river_", " ").replace("_land", "")
        vals  = []
        for m in models:
            _, iou = dice_iou(results[m]["cm"])
            vals.append(f"{iou[i]:.4f}")
        row(short, vals)
    macro_iou = []
    for m in models:
        _, iou = dice_iou(results[m]["cm"])
        macro_iou.append(f"{iou.mean():.4f}")
    row("Macro average", macro_iou)

    # ── Confusion matrices ────────────────────────────────────────────────────
    for m in models:
        w(f"\n--- CONFUSION MATRIX: {m} ---")
        short_names = [c.split("_river_")[0].replace("ice_", "ice ") for c in class_names]
        w("True \\ Pred   " + "  ".join(f"{s[:8]:>8}" for s in short_names))
        cm = results[m]["cm"]
        for i, sn in enumerate(short_names):
            w(f"{sn[:13]:<14}" + "  ".join(f"{cm[i, j]:>8}" for j in range(n_classes)))

    # ── Full classification reports ───────────────────────────────────────────
    for m in models:
        w(f"\n--- CLASSIFICATION REPORT: {m} ---")
        w(results[m]["report"])

    w("\n" + "=" * 70)

    out_path = OUTPUT_DIR / "comparison_all_models.txt"
    out_path.write_text(buf.getvalue())
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
