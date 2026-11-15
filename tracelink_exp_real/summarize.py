#!/usr/bin/env python
"""Summary table and bar charts for the real-data sweep.

    python tracelink_exp_real/summarize.py --results results/real
    python tracelink_exp_real/summarize.py --results results/real --min-mean-iou 0.50

Reads <results>/iou_threshold_sensitivity/iou_0.XX/batch_real_results.csv
written by run_batch.py.

By default every processed radargram is averaged. With --min-mean-iou X,
only radargrams whose mean IoU is >= X are kept. The same rule is applied
separately at each IoU matching threshold, so the kept set can differ
between thresholds. The number kept out of the total is always printed,
written to the CSV and shown on the figures.

Outputs in <results>/summary/:
  real_summary.csv                       one row per threshold
  real_metrics_by_threshold.(pdf|png)    grouped bars, all thresholds
  real_metrics_iou0pXX.(pdf|png)         Precision / Recall / F1 / Mean IoU at --headline
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS = [("Precision", "precision_global"), ("Recall", "recovery_rate"),
           ("F1", "f1"), ("Mean IoU", "mean_iou")]
COLORS = ["#1f8cff", "#26c281", "#f5a623", "#9b59b6"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(rows, min_mean_iou):
    iou = np.array([float(r["mean_iou"]) for r in rows])
    keep = np.ones(len(rows), bool) if min_mean_iou is None else iou >= min_mean_iou
    kept = [r for r, k in zip(rows, keep) if k]
    out = {"N": len(rows), "kept": len(kept)}
    for name, col in METRICS:
        out[name] = float(np.mean([float(r[col]) for r in kept])) if kept else float("nan")
    out["pre_merge"] = int(sum(float(r["pre_merge"]) for r in kept))
    out["post_merge"] = int(sum(float(r["post_merge"]) for r in kept))
    out["merges"] = out["pre_merge"] - out["post_merge"]
    out["merge_reduction_pct"] = 100.0 * out["merges"] / out["pre_merge"] if out["pre_merge"] else float("nan")
    return out


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def save(fig, base):
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {base}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/real")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--min-mean-iou", type=float, default=None,
                    help="optional: keep only radargrams with mean IoU >= this value")
    ap.add_argument("--headline", type=float, default=0.50,
                    help="IoU threshold for the single-threshold bar chart")
    args = ap.parse_args()

    base = os.path.join(args.results, "iou_threshold_sensitivity")
    out_dir = os.path.join(args.results, "summary")
    os.makedirs(out_dir, exist_ok=True)
    rule = "all radargrams" if args.min_mean_iou is None else f"mean IoU >= {args.min_mean_iou:.2f}"

    res = {}
    for t in args.thresholds:
        p = os.path.join(base, f"iou_{t:.2f}", "batch_real_results.csv")
        if not os.path.exists(p):
            print(f"WARNING: missing {p}")
            continue
        res[t] = summarise(load(p), args.min_mean_iou)

    if not res:
        raise SystemExit("No result CSVs found. Run tracelink_exp_real/run_batch.py first.")

    print("\n" + "=" * 104)
    print(f"REAL DATA SUMMARY  (files averaged: {rule})")
    print("=" * 104)
    print(f"{'IoU':>5} {'Files':>11} {'Precision':>10} {'Recall':>8} {'F1':>7} {'Mean IoU':>9} "
          f"{'Pre-merge':>10} {'Post-merge':>11} {'Merges':>8} {'Reduction':>10}")
    print("-" * 104)
    for t, r in res.items():
        print(f"{t:>5.2f} {r['kept']:>5}/{r['N']:<5} {r['Precision']:>10.3f} {r['Recall']:>8.3f} "
              f"{r['F1']:>7.3f} {r['Mean IoU']:>9.3f} {r['pre_merge']:>10} {r['post_merge']:>11} "
              f"{r['merges']:>8} {r['merge_reduction_pct']:>9.2f}%")
    print("=" * 104)

    with open(os.path.join(out_dir, "real_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iou_threshold", "filter", "files_kept", "files_total", "precision", "recall",
                    "f1", "mean_iou", "pre_merge", "post_merge", "merges", "merge_reduction_pct"])
        for t, r in res.items():
            w.writerow([t, rule, r["kept"], r["N"], r["Precision"], r["Recall"], r["F1"],
                        r["Mean IoU"], r["pre_merge"], r["post_merge"], r["merges"],
                        r["merge_reduction_pct"]])

    style()
    names = [m for m, _ in METRICS]
    x = np.arange(len(names))
    width = 0.8 / len(res)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (t, r) in enumerate(res.items()):
        vals = [r[m] for m in names]
        bars = ax.bar(x + (i - (len(res) - 1) / 2) * width, vals, width,
                      color=COLORS[i % len(COLORS)], edgecolor="black",
                      label=f"IoU matching = {t:.2f}  ({r['kept']}/{r['N']} files)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(f"Real data: sensitivity to the IoU matching threshold ({rule})")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    save(fig, os.path.join(out_dir, "real_metrics_by_threshold"))

    if args.headline in res:
        r = res[args.headline]
        vals = [r[m] for m in names]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bars = ax.bar(x, vals, color=COLORS, width=0.6, edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_title(f"{r['kept']} of {r['N']} real radargrams, IoU matching = {args.headline:.2f}",
                     fontsize=12)
        save(fig, os.path.join(out_dir, f"real_metrics_iou{args.headline:.2f}".replace(".", "p")))


if __name__ == "__main__":
    main()
