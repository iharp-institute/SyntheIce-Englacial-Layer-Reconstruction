#!/usr/bin/env python
"""Figures for the XCheck batch run (Cell 13).

    python xcheck_synthetic/plot_results.py --results results/xcheck

Reads xcheck_per_radargram.csv and xcheck_per_category.csv and saves to
<results>/figures/:
  xcheck_metrics_by_category   grouped bars: precision / recall / F1 / Jaccard
  xcheck_tp_fp_fn              stacked TP / FP / FN totals per category
  xcheck_f1_scatter            one dot per radargram (bar = mean)
  xcheck_f1_box                F1 distribution per category
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["close_To_synthetic0", "mid_level0", "close_To_real0", "noisybucket0"]
NAMES = {"close_To_synthetic0": "Close-to-Synthetic", "mid_level0": "Mid-Level",
         "close_To_real0": "Close-to-Real", "noisybucket0": "Noisy-Level"}
PAL = ["#1C7293", "#4A9875", "#E8A838", "#E24B4A"]


def save(fig, out, name):
    base = os.path.join(out, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {base}.png")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/xcheck")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results, "xcheck_per_radargram.csv"))
    summ = pd.read_csv(os.path.join(args.results, "xcheck_per_category.csv")).set_index("category")
    bks = [b for b in ORDER if b in summ.index] + [b for b in summ.index if b not in ORDER]
    labels = [NAMES.get(b, b) for b in bks]
    color = {b: PAL[i % len(PAL)] for i, b in enumerate(bks)}
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)

    # 1. grouped bars
    cols = ["precision_mean", "recall_mean", "f1_mean", "jaccard_mean"]
    x = np.arange(len(cols))
    w = 0.18
    off = np.linspace(-(len(bks) - 1) / 2 * w, (len(bks) - 1) / 2 * w, len(bks))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, b in enumerate(bks):
        vals = [float(summ.loc[b, c]) for c in cols]
        bars = ax.bar(x + off[i], vals, w, label=NAMES.get(b, b), color=color[b],
                      alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "Jaccard (IoU)"], fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(f"XCheck metrics per category (mean over {len(df)} wide radargrams)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    clean(ax)
    save(fig, out, "xcheck_metrics_by_category")

    # 2. stacked TP / FP / FN
    tp = [int(summ.loc[b, "TP"]) for b in bks]
    fp = [int(summ.loc[b, "FP"]) for b in bks]
    fn = [int(summ.loc[b, "FN"]) for b in bks]
    xp = np.arange(len(bks))
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(xp, tp, color="#2E7D32", label="TP", edgecolor="white")
    ax.bar(xp, fp, bottom=tp, color="#E24B4A", label="FP", edgecolor="white")
    ax.bar(xp, fn, bottom=[a + b for a, b in zip(tp, fp)], color="#5B8DD9", label="FN",
           edgecolor="white")
    for bar, v in zip(b1, tp):
        if v:
            ax.text(bar.get_x() + bar.get_width() / 2, v / 2, str(v), ha="center",
                    va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count (summed over radargrams)")
    ax.set_title("Total TP / FP / FN per category", fontweight="bold")
    ax.legend()
    clean(ax)
    save(fig, out, "xcheck_tp_fp_fn")

    # 3. F1 scatter
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, b in enumerate(bks):
        v = df[df["category"] == b]["f1"].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, color=color[b],
                   alpha=0.75, s=55, edgecolors="white", linewidths=0.5, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, colors=color[b], linewidths=2.2, zorder=4)
    ax.set_xticks(range(len(bks)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Radargram-level F1 per category (bar = mean)", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    clean(ax)
    save(fig, out, "xcheck_f1_scatter")

    # 4. box plot
    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot([df[df["category"] == b]["f1"].values for b in bks], patch_artist=True,
                    widths=0.45, medianprops=dict(color="white", linewidth=2.0))
    for patch, b in zip(bp["boxes"], bks):
        patch.set_facecolor(color[b])
        patch.set_alpha(0.85)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("F1 distribution per category", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    clean(ax)
    save(fig, out, "xcheck_f1_box")


if __name__ == "__main__":
    main()
