"""
train_cnn.py

1D CNN classifier for river-ice classes using the same 12 features as train_model.py.
Input shape per sample: (12,) → reshaped to (1, 12) for Conv1d (channels-first).

Trains on Combined Dataset.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ── PATHS (mirrors train_model.py) ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

COMBINED_FILE = TRAINING_DIR / "combined_all.csv"

FEATURE_COLUMNS = [
    "I1", "I2", "I3", "I4", "I5",
    "SZA", "SAA", "VZA", "VAA",
    "VIIRS_NDWI",
    "water_fraction",
    "modis_ndvi",
]
N_FEATURES = len(FEATURE_COLUMNS)

VALID_LABELS = {
    "ice_covered_river_snow_covered_land",
    "ice_covered_river_snow_free_land",
    "ice_free_river_snow_free_land",
    "ice_free_river_snow_land",
}

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── MODEL ─────────────────────────────────────────────────────────────────────

class RiverIceCNN1D(nn.Module):
    """
    1D CNN for tabular feature classification.
    Input:  (batch, 1, n_features)   — 1 channel, n_features "time steps"
    Output: (batch, n_classes)        — logits
    """
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        pool_out = max(1, n_features // 3)   # derived from actual input size
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(pool_out),   # → (batch, 64, pool_out)
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # → (batch, 64 * pool_out)
            nn.Linear(64 * pool_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


# ── DATA LOADING (same filters as train_model.py) ─────────────────────────────

def _sanitize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace("", np.nan)
    return df


def _add_viirs_ndwi(df: pd.DataFrame) -> pd.DataFrame:
    i1 = df["I1"].replace(0, np.nan)
    i2 = df["I2"].replace(0, np.nan)
    df["VIIRS_NDWI"] = (i1 - i2) / (i1 + i2)
    return df



def load_combined_csv() -> pd.DataFrame:
    df = pd.read_csv(COMBINED_FILE)
    df = _sanitize(df, ["ground_truth_class", "notes"])
    df["notes"] = df["notes"].fillna("").astype(str)
    df = df[df["ground_truth_class"].notna()]
    df = df[df["water_fraction"].replace("", np.nan).astype(float) < 0.90]
    df = df[~df["notes"].str.contains("Landsat null", case=False, na=False)]
    df = df[~df["notes"].str.contains("CONFLICT", case=False, na=False)]
    df = df[~df["notes"].str.contains("EXCLUDED", case=False, na=False)]
    df = df[df["modis_ndvi"].replace("", np.nan).astype(float).notna()]
    df = df[df["ground_truth_class"].isin(VALID_LABELS)]
    df["source"] = "combined"
    return df



def _nan_guard(df: pd.DataFrame, name: str) -> None:
    nan_counts = df[FEATURE_COLUMNS].isna().sum()
    total_nan = int(nan_counts.sum())
    if total_nan > 0:
        bad = nan_counts[nan_counts > 0].to_dict()
        raise RuntimeError(f"NaN in {name} features before training: {bad}")


def prepare_datasets() -> Dict[str, pd.DataFrame]:
    comb = load_combined_csv()
    # impute any missing I3/I4/I5/modis_ndvi using column medians
    for col in ["I3", "I4", "I5", "modis_ndvi"]:
        if col in comb.columns:
            comb[col] = comb[col].fillna(comb[col].median())
    _add_viirs_ndwi(comb)
    for col in FEATURE_COLUMNS:
        comb[col] = pd.to_numeric(comb[col], errors="coerce")
    _nan_guard(comb, "Combined")
    return {"Combined": comb}


# ── TRAINING HELPERS ──────────────────────────────────────────────────────────

def to_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # X shape: (N, 12) → (N, 1, 12) for Conv1d
    xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y, dtype=torch.long)
    return xt, yt


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X_t: torch.Tensor, y_np: np.ndarray, device, class_names):
    model.eval()
    logits = model(X_t.to(device)).cpu()
    preds  = logits.argmax(dim=1).numpy()
    acc    = accuracy_score(y_np, preds)
    report = classification_report(y_np, preds, target_names=class_names,
                                   zero_division=0)
    rd     = classification_report(y_np, preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_np, preds)
    f1_map = {c: float(rd.get(c, {}).get("f1-score", 0.0)) for c in class_names}
    return acc, preds, report, cm, f1_map


def cv_accuracy(dataset_name, X: np.ndarray, y: np.ndarray,
                n_classes: int, device, epochs=60) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, va_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr = sc.transform(X[tr_idx]); Xva = sc.transform(X[va_idx])
        Xtr_t, ytr_t = to_tensors(Xtr, y[tr_idx])
        Xva_t = to_tensors(Xva, y[va_idx])[0]
        mdl = RiverIceCNN1D(N_FEATURES, n_classes).to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
        # use same weighted loss as main training
        counts = np.bincount(y[tr_idx], minlength=n_classes).astype(float)
        w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=16, shuffle=True)
        for _ in range(epochs):
            train_epoch(mdl, loader, opt, crit, device)
        with torch.no_grad():
            mdl.eval()
            preds = mdl(Xva_t.to(device)).cpu().argmax(1).numpy()
        accs.append(accuracy_score(y[va_idx], preds))
    return float(np.mean(accs)), float(np.std(accs))


def save_cm_png(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title); fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    datasets = prepare_datasets()

    all_classes = sorted(VALID_LABELS)
    le = LabelEncoder().fit(all_classes)
    n_classes  = len(all_classes)
    class_names = list(le.classes_)

    print(f"Classes ({n_classes}): {class_names}\n")
    print("=" * 60)

    buf = io.StringIO()

    results = {}
    EPOCHS = 150

    for ds_name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset {ds_name}  —  {len(df)} rows")
        print(f"{'='*60}")

        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = le.transform(df["ground_truth_class"])

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(sss.split(X, y))

        scaler = StandardScaler().fit(X[tr_idx])
        Xtr = scaler.transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])

        Xtr_t, ytr_t = to_tensors(Xtr, y[tr_idx])
        Xte_t        = to_tensors(Xte, y[te_idx])[0]

        model    = RiverIceCNN1D(N_FEATURES, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Class weights for imbalanced data
        counts = np.bincount(y[tr_idx], minlength=n_classes).astype(float)
        weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Hold out 10% of train for val-based early stopping (no leakage from test)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        sub_tr, sub_va = next(sss_val.split(Xtr, y[tr_idx]))
        Xtr_t2, ytr_t2 = to_tensors(Xtr[sub_tr], y[tr_idx][sub_tr])
        Xva_t2 = to_tensors(Xtr[sub_va], y[tr_idx][sub_va])[0]
        yva_np2 = y[tr_idx][sub_va]

        loader = DataLoader(TensorDataset(Xtr_t2, ytr_t2), batch_size=16, shuffle=True)

        best_val_acc = -1.0
        best_state = None
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            with torch.no_grad():
                model.eval()
                val_preds = model(Xva_t2.to(device)).cpu().argmax(1).numpy()
                val_acc = accuracy_score(yva_np2, val_preds)
            model.train()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 30 == 0:
                print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={loss:.4f}  val_acc={val_acc:.3f}")

        model.load_state_dict(best_state)
        acc, preds, report, cm, f1_map = evaluate(model, Xte_t, y[te_idx], device, class_names)

        print(f"\n  Test accuracy: {acc:.4f}")
        print(f"\n{report}")
        print(f"  Confusion matrix:\n{cm}\n")

        cv_mean, cv_std = cv_accuracy(ds_name, X, y, n_classes, device, epochs=80)
        print(f"  5-fold CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        results[ds_name] = {
            "acc": acc, "cv_mean": cv_mean, "cv_std": cv_std,
            "report": report, "cm": cm, "f1_map": f1_map,
            "n_train": len(tr_idx), "n_test": len(te_idx),
        }

        tag = ds_name.lower().replace(" ", "_")
        torch.save(model.state_dict(), OUTPUT_DIR / f"cnn1d_{tag}.pt")
        joblib.dump(scaler, OUTPUT_DIR / f"scaler_{tag}.pkl")
        save_cm_png(cm, class_names,
                    OUTPUT_DIR / f"confusion_matrix_cnn1d_{tag}.png",
                    f"1D CNN — Dataset {ds_name}")

        line = (f"CNN1D-{ds_name}: test={acc:.4f}  "
                f"CV={cv_mean:.4f}±{cv_std:.4f}  "
                f"n_train={len(tr_idx)}  n_test={len(te_idx)}")
        buf.write(line + "\n")

    F1_SHORT = {
        "ice_free_river_snow_free_land":          "F1 ice_free_snow_free",
        "ice_free_river_snow_land":               "F1 ice_free_snow_land",
        "ice_covered_river_snow_covered_land":    "F1 ice_cov_snow_cov",
        "ice_covered_river_snow_free_land":       "F1 ice_cov_snow_free",
    }

    print("\n" + "=" * 60)
    print("COMPARISON TABLE — 1D CNN")
    print("=" * 60)
    order = ["Combined"]
    hdr = f"{'Metric':<25}" + "".join(f"| {'CNN-'+k:<10}" for k in order)
    print(hdr)
    print("-" * len(hdr))

    def _row(label, vals):
        print(f"{label:<25}" + "".join(f"| {v:<10}" for v in vals))

    _row("n_train",          [str(results[k]["n_train"])          for k in order])
    _row("n_test",           [str(results[k]["n_test"])           for k in order])
    _row("Test accuracy",    [f"{results[k]['acc']:.4f}"          for k in order])
    _row("CV mean accuracy", [f"{results[k]['cv_mean']:.4f}"      for k in order])
    _row("CV std",           [f"{results[k]['cv_std']:.4f}"       for k in order])
    for full, label in F1_SHORT.items():
        _row(label, [f"{results[k]['f1_map'].get(full, 0.0):.4f}" for k in order])

    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
    (OUTPUT_DIR / "cnn1d_report.txt").write_text(buf.getvalue())
    print(f"\nAll CNN artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
