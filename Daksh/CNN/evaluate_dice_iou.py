"""
evaluate_dice_iou.py

Loads the trained 1D CNN and computes per-class and macro-averaged
Dice coefficient and IoU (Jaccard index) on the held-out test split.

Dice  = 2*TP / (2*TP + FP + FN)
IoU   =   TP / (TP + FP + FN)
"""

from __future__ import annotations

import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from train_cnn import (
    RiverIceCNN1D,
    prepare_datasets,
    to_tensors,
    FEATURE_COLUMNS,
    N_FEATURES,
    SEED,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def dice_iou_from_cm(cm: np.ndarray):
    """Return per-class Dice and IoU from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    iou  = tp / (tp + fp + fn + 1e-9)
    return dice, iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le     = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    scaler = joblib.load(OUTPUT_DIR / "scaler_combined.pkl")
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    datasets = prepare_datasets()
    df = datasets["Combined"]

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = le.transform(df["ground_truth_class"])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, y))

    Xte = scaler.transform(X[te_idx])
    Xte_t, _ = to_tensors(Xte, y[te_idx])
    y_true = y[te_idx]

    model = RiverIceCNN1D(N_FEATURES, n_classes)
    model.load_state_dict(torch.load(OUTPUT_DIR / "cnn1d_combined.pt",
                                     map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        preds = model(Xte_t.to(device)).cpu().argmax(1).numpy()

    cm = confusion_matrix(y_true, preds, labels=list(range(n_classes)))
    dice, iou = dice_iou_from_cm(cm)

    col_w = max(len(c) for c in class_names) + 2
    header = f"{'Class':<{col_w}} {'Dice':>8}  {'IoU':>8}"
    print("\n" + "=" * len(header))
    print("Dice & IoU — 1D CNN (Combined, test split)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        print(f"{name:<{col_w}} {dice[i]:>8.4f}  {iou[i]:>8.4f}")
    print("-" * len(header))
    print(f"{'Macro average':<{col_w}} {dice.mean():>8.4f}  {iou.mean():>8.4f}")
    print("=" * len(header))

    report_path = OUTPUT_DIR / "dice_iou_report.txt"
    lines = [
        "Dice & IoU — 1D CNN (Combined, test split)\n",
        f"{'Class':<{col_w}} {'Dice':>8}  {'IoU':>8}\n",
        "-" * len(header) + "\n",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"{name:<{col_w}} {dice[i]:>8.4f}  {iou[i]:>8.4f}\n")
    lines.append("-" * len(header) + "\n")
    lines.append(f"{'Macro average':<{col_w}} {dice.mean():>8.4f}  {iou.mean():>8.4f}\n")
    report_path.write_text("".join(lines))
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
