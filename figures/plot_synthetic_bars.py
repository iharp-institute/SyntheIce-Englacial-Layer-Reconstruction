#!/usr/bin/env python
"""Paper figure: TraceLink scores on SyntheIce by category (grouped bars).

    python figures/plot_synthetic_bars.py --results results/synthetic --iou 0.50

Reads iou_threshold_sensitivity.csv written by
tracelink_exp_synthetic/run_batch.py and plots macro precision, recall,
F1 and mean IoU for the four categories at one IoU matching threshold.
Saves combined_macro_metrics_by_category_iou0pXX.(pdf|png).
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["close_To_synthetic0", "mid_level0", "close_To_real0", "noisybucket0"]
NAMES = {"close_To_synthetic0": "Close-to-Synthetic", "mid_level0": "Mid-Level",
         "close_To_real0": "Close-to-Real", "noisybucket0": "Noisy-Level"}
METRICS = [("Precision", "macro_precision"), ("Recall", "macro_recall"),
           ("F1", "macro_f1"), ("Mean IoU", "macro_mean_iou")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/synthetic")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--out", default=None, help="output folder (default <results>/figures)")
    args = ap.parse_args()

    with open(os.path.join(args.results, "iou_threshold_sensitivity.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if abs(float(r["iou_threshold"]) - args.iou) < 1e-9]
    by_cat = {r["category"].replace("synthetic_radargram_output_", ""): r for r in rows}
    cats = [c for c in ORDER if c in by_cat]
    if not cats:
        raise SystemExit(f"No rows for IoU {args.iou} in the sensitivity CSV")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    x = np.arange(len(METRICS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, c in enumerate(cats):
        vals = [float(by_cat[c][col]) for _, col in METRICS]
        bars = ax.bar(x + (i - (len(cats) - 1) / 2) * width, vals, width, label=NAMES[c])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in METRICS], fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score", fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.05), frameon=False, fontsize=6)
    plt.tight_layout()

    out = args.out or os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"combined_macro_metrics_by_category_iou{args.iou:.2f}".replace(".", "p"))
    fig.savefig(base + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    print(f"Saved {base}.pdf / .png")


if __name__ == "__main__":
    main()
