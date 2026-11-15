#!/usr/bin/env python
"""Part 3: TraceLink on SyntheIce, all four categories, IoU threshold sweep.

    python tracelink_exp_synthetic/run_batch.py \
        --data ../SyntheIce-Dataset/data/full --out results/synthetic

For every normal (non-wide) radargram in each category folder:
  1. run TraceLink on data/mask (synthetic auto_params preset),
  2. score the predicted layer map against data/layer_id_map
     (pixel IoU + Hungarian matching) at every threshold in --thresholds.

TraceLink runs once per file; all thresholds are scored from the same
prediction. Wide radargrams are skipped, as in the original notebook.

Outputs in --out:
  batch_results_<category>_iou0pXX.csv   one row per file
  batch_summary_<category>_iou0pXX.json  macro / micro summary
  iou_threshold_sensitivity.csv / .json  all categories x thresholds
  per_file/<category>/<stem>_traced.{h5,json}   (with --save-per-file)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import DEFAULT_THRESHOLDS, aggregate, score_layer_maps  # noqa: E402
from tracelink import (SYNTHETIC_PRESET, atomic_save_json, run_tracelink,  # noqa: E402
                       save_h5_layer_map)

CATEGORIES = ["close_To_synthetic0", "mid_level0", "close_To_real0", "noisybucket0"]
PAPER_NAMES = {"close_To_synthetic0": "Close-to-Synthetic", "mid_level0": "Mid-Level",
               "close_To_real0": "Close-to-Real", "noisybucket0": "Noisy-Level"}


def tag(t):
    return f"iou{float(t):.2f}".replace(".", "p")


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def evaluate_file(path, thresholds, save_dir=None):
    with h5py.File(path, "r") as f:
        mask = f["data/mask"][:]
        gt = f["data/layer_id_map"][:]
        meta = dict(f["metadata"].attrs)
    if mask.sum() == 0 or gt.max() == 0:
        return None

    t0 = time.perf_counter()
    layer_map, lookup, info = run_tracelink(mask, preset=SYNTHETIC_PRESET, return_info=True)
    elapsed = time.perf_counter() - t0
    scores = score_layer_maps(gt, layer_map, thresholds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        save_h5_layer_map(os.path.join(save_dir, stem + "_traced.h5"), layer_map, lookup,
                          {"gap_cols": info["gap_cols"], "row_tol": info["row_tol"]},
                          os.path.basename(path))
        atomic_save_json(os.path.join(save_dir, stem + "_traced.json"), {
            "source_file": os.path.basename(path),
            "params": {"gap_cols": info["gap_cols"], "row_tol": info["row_tol"],
                       "merge_pass": {"n_merges": info["n_merges"],
                                      "n_iters": info["n_iters"], "strict": True}},
            "metadata": {k: _jsonable(v) for k, v in meta.items()},
            "layer_lookup": {f"L{k}": v for k, v in lookup.items()},
        })

    rows = {}
    for t, s in scores.items():
        rows[t] = {"file": os.path.basename(path), **s,
                   "pred_pre_merge": info["pred_pre_merge"],
                   "n_merges": info["n_merges"], "n_iters": info["n_iters"],
                   "gap_cols": info["gap_cols"], "row_tol": info["row_tol"],
                   "elapsed_sec": elapsed, "iou_threshold": t}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="folder holding synthetic_radargram_output_<category>/ folders")
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--save-per-file", action="store_true",
                    help="also save each traced layer map (.h5) and lookup (.json)")
    ap.add_argument("--limit", type=int, default=None, help="first N files per category (quick test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sensitivity = []
    full = {f"{t:.2f}": {} for t in args.thresholds}

    for cat in args.categories:
        folder = os.path.join(args.data, f"synthetic_radargram_output_{cat}")
        files = [p for p in sorted(glob.glob(os.path.join(folder, "*.h5")))
                 if "wide" not in os.path.basename(p).lower()]
        if args.limit:
            files = files[:args.limit]
        if not files:
            sys.exit(f"No normal .h5 files in {folder}")
        print(f"\n=== {cat}: {len(files)} radargrams (wide skipped)")

        per_t = {t: [] for t in args.thresholds}
        save_dir = os.path.join(args.out, "per_file", cat) if args.save_per_file else None
        for i, p in enumerate(files, 1):
            rows = evaluate_file(p, args.thresholds, save_dir)
            if rows is None:
                print(f"  skip {os.path.basename(p)} (empty)")
                continue
            for t in args.thresholds:
                per_t[t].append(rows[float(t)])
            r = rows[0.5] if 0.5 in rows else next(iter(rows.values()))
            print(f"  [{i:>3}/{len(files)}] {r['file'][:52]:<52} GT={r['gt_n']:>3} "
                  f"pred={r['pred_n']:>3} F1@{r['iou_threshold']:.2f}={r['f1']:.3f}")

        for t in args.thresholds:
            rows = per_t[t]
            summ = aggregate(rows)
            summ.update({"category": f"synthetic_radargram_output_{cat}", "iou_threshold": float(t)})
            summ["totals"]["pred_pre_merge"] = sum(r["pred_pre_merge"] for r in rows)
            summ["totals"]["n_merges"] = sum(r["n_merges"] for r in rows)
            summ["elapsed_sec_total"] = float(sum(r["elapsed_sec"] for r in rows))
            full[f"{t:.2f}"][cat] = summ

            stem = f"synthetic_radargram_output_{cat}_{tag(t)}"
            with open(os.path.join(args.out, f"batch_results_{stem}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            atomic_save_json(os.path.join(args.out, f"batch_summary_{stem}.json"), summ)

            m, tot = summ["macro"], summ["totals"]
            sensitivity.append({
                "iou_threshold": float(t), "category": cat, "paper_name": PAPER_NAMES.get(cat, cat),
                "files_processed": summ["files"],
                "macro_precision": m["precision"], "macro_recall": m["recall"],
                "macro_f1": m["f1"], "macro_mean_iou": m["mean_iou"],
                "macro_median_iou": m["median_iou"], "macro_median_f1": m["median_f1"],
                "macro_over_seg": m["over_seg"], "macro_under_seg": m["under_seg"],
                "micro_precision": summ["micro"]["precision"],
                "micro_recall": summ["micro"]["recall"], "micro_f1": summ["micro"]["f1"],
                "total_TP": tot["TP"], "total_FP": tot["FP"], "total_FN": tot["FN"],
                "GT_layers": tot["GT_layers"], "pred_layers": tot["pred_layers"],
                "pred_pre_merge": tot["pred_pre_merge"], "n_merges": tot["n_merges"],
            })
        t0 = args.thresholds[0]
        tot = full[f"{t0:.2f}"][cat]["totals"]
        print(f"  merge pass: {tot['pred_pre_merge']} -> {tot['pred_layers']} "
              f"({tot['n_merges']} merges)")

    with open(os.path.join(args.out, "iou_threshold_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        w.writeheader()
        w.writerows(sensitivity)
    atomic_save_json(os.path.join(args.out, "iou_threshold_sensitivity.json"),
                     {"analysis": "IoU threshold sensitivity",
                      "iou_thresholds": [float(t) for t in args.thresholds],
                      "categories": args.categories, "results": full})

    print("\n" + "=" * 92)
    print(f"{'IoU':>5}  {'Category':<20}{'Files':>6}{'Prec':>9}{'Recall':>9}{'F1':>9}{'mIoU':>9}")
    print("-" * 92)
    for r in sensitivity:
        print(f"{r['iou_threshold']:>5.2f}  {r['paper_name']:<20}{r['files_processed']:>6}"
              f"{r['macro_precision']:>9.4f}{r['macro_recall']:>9.4f}"
              f"{r['macro_f1']:>9.4f}{r['macro_mean_iou']:>9.4f}")
    print("=" * 92)
    print(f"Saved results to {args.out}/")


if __name__ == "__main__":
    main()
