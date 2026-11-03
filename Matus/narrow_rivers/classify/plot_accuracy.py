"""
Draw the accuracy of each date's Random Forest from the saved cross-validation.

Reads <patch>/rf/<date>_rf_cv.json (written by train_rf.py) so nothing is
retrained here. One row per date:

  left   confusion matrix, each row normalised to the class you labelled, so the
         diagonal reads as recall and the off-diagonal shows where it goes wrong
  middle recall and precision per class, with the number of labelled pixels
  right  feature importances

Every number comes from folds in which WHOLE POLYGONS were held out, so it is not
inflated by near-duplicate pixels inside one polygon. It measures how well the
model reproduces the hand labelling on the same scene; it says nothing about a
date the model has not seen.

Outputs:
  narrow_rivers/classify/accuracy_<dates>.png

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\plot_accuracy.py --dates 2024-06-22 2023-10-04
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
import patch_config as C

OUT_SUBDIR = "rf"


def load(date):
    p = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}", OUT_SUBDIR,
                     f"{date}_rf_cv.json")
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: {p} missing — run train_rf.py for {date} first")
    return json.load(open(p, encoding="utf-8"))


def main(dates, out):
    cvs = [load(d) for d in dates]
    fig, axes = plt.subplots(len(cvs), 3, figsize=(15, 4.6 * len(cvs)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]})

    for row, cv in enumerate(cvs):
        classes = cv["classes"]
        names = [C.CLASSES[c] for c in classes]
        m = np.array(cv["matrix"], dtype=float)
        support = {int(k): v for k, v in cv["support"].items()}

        # ── confusion matrix, normalised per true class ──────────────────────
        ax = axes[row][0]
        norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i, j]:.2f}\n{int(m[i, j]):,}",
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "#333")
        ax.set_xticks(range(len(classes)), names, fontsize=9)
        ax.set_yticks(range(len(classes)), names, fontsize=9)
        ax.set_xlabel("model said"); ax.set_ylabel("you labelled")
        ax.set_title(f"{cv['date']}  —  confusion\n"
                     f"{cv['n_samples']:,} px, {cv['n_polygons']} polygons, "
                     f"{cv['folds']} folds", fontsize=10)

        # ── recall / precision ──────────────────────────────────────────────
        ax = axes[row][1]
        recall = np.diag(m) / m.sum(axis=1)
        prec = np.divide(np.diag(m), m.sum(axis=0),
                         out=np.zeros(len(classes)), where=m.sum(axis=0) > 0)
        x = np.arange(len(classes))
        ax.bar(x - 0.2, recall, 0.4, label="recall", color="#2c7fb8")
        ax.bar(x + 0.2, prec, 0.4, label="precision", color="#7fcdbb")
        for i in x:
            ax.text(i - 0.2, recall[i] + 0.02, f"{recall[i]:.2f}",
                    ha="center", fontsize=8)
            ax.text(i + 0.2, prec[i] + 0.02, f"{prec[i]:.2f}",
                    ha="center", fontsize=8)
        ax.axhline(1.0, color="#bbb", lw=0.8, zorder=0)
        ax.set_xticks(x, [f"{n}\n{support[c]:,} px" for n, c in zip(names, classes)],
                      fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("share")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("per class  (whole polygons held out)", fontsize=10)

        # ── importances ─────────────────────────────────────────────────────
        ax = axes[row][2]
        imp = cv.get("importances", {})
        if imp:
            k = sorted(imp, key=imp.get)
            ax.barh(range(len(k)), [imp[i] for i in k], color="#969696")
            ax.set_yticks(range(len(k)), k, fontsize=8)
            ax.set_xlabel("importance")
        ax.set_title("what the model used", fontsize=10)

    fig.suptitle("Random Forest accuracy — reproducing the hand labels within each "
                 "date (Mode A)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out or os.path.join(HERE, "accuracy_" + "_".join(dates) + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"{path}  {os.path.getsize(path)/1e6:.2f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Accuracy figure from saved CV results.")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    main(a.dates, a.out)
