#!/usr/bin/env python
"""Part 1: XCheck on every wide SyntheIce radargram (4 tiles each).

    python xcheck_synthetic/run_batch.py \
        --tiles ../SyntheIce-Dataset/data/xcheck_set/tiles --out results/xcheck

For each wide radargram (its tiles are grouped by the parent_radargram
attribute):
  1. run XCheck on the 4 tile centerlines (one-to-one boundary matches)
  2. build the ground-truth boundary matches from the tiles'
     skeleton_layer_id_map
  3. score with +/-1 px row tolerance (fuzzy_metrics)

Outputs in --out:
  xcheck_per_radargram.csv   one row per wide radargram
  xcheck_per_category.csv    per-category means and totals
  xcheck_global.json         TP / FP / FN summed over all radargrams
  xcheck_predictions.json    predicted matches per radargram (for stitch_one.py)
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xcheck_synthetic.xcheck import (DEPTH_TOL, DOT_THRESH, WINDOW, build_gt,  # noqa: E402
                                     fuzzy_metrics, load_crops, run_xcheck)

PAPER_NAMES = {"close_To_synthetic0": "Close-to-Synthetic", "mid_level0": "Mid-Level",
               "close_To_real0": "Close-to-Real", "noisybucket0": "Noisy-Level"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True, help="folder of *_cropped_XX.h5 tiles")
    ap.add_argument("--out", default="results/xcheck")
    ap.add_argument("--tol", type=int, default=1, help="row tolerance for matching to GT (px)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tiles, "*.h5")))
    if not files:
        sys.exit(f"No tile files in {args.tiles}. Run SyntheIce-Dataset/scripts/crop_wide_to_tiles.py first.")

    groups = defaultdict(lambda: defaultdict(list))
    for f in files:
        with h5py.File(f, "r") as h:
            parent = str(h["metadata"].attrs["parent_radargram"])
            bucket = str(h["metadata"].attrs.get("bucket", "unknown"))
        groups[bucket][parent].append(f)

    total = sum(len(v) for v in groups.values())
    print(f"XCheck params: WINDOW={WINDOW}  DEPTH_TOL={DEPTH_TOL}  DOT_THRESH={DOT_THRESH}")
    print(f"Running XCheck on {total} wide radargrams\n")

    rows, preds, done = [], {}, 0
    for bucket in sorted(groups):
        for parent, crops in sorted(groups[bucket].items()):
            centerlines, skel_maps, metas = load_crops(crops)
            pred = run_xcheck(centerlines)
            pred_set, gt_set = set(pred), build_gt(skel_maps)
            m = fuzzy_metrics(pred_set, gt_set, tol=args.tol)
            tilt = float(metas[0].get("tilt_angle", 0.0))
            m.update({"category": bucket, "parent": parent, "tilt_angle": tilt,
                      "n_gt": len(gt_set), "n_pred": len(pred_set)})
            rows.append(m)
            preds[parent] = sorted(pred)
            done += 1
            print(f"  [{done:>3}/{total}] {bucket:<22} GT={m['n_gt']:>3} Pred={m['n_pred']:>3} "
                  f"TP={m['TP']:>3} FP={m['FP']:>2} FN={m['FN']:>2} F1={m['f1']:.3f} tilt={tilt:.1f}")

    df = pd.DataFrame(rows)
    summary = (df.groupby("category")
                 .agg(n_radargrams=("parent", "count"),
                      TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
                      precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                      f1_mean=("f1", "mean"), jaccard_mean=("jaccard", "mean"),
                      f1_std=("f1", "std"), n_gt_total=("n_gt", "sum"),
                      n_pred_total=("n_pred", "sum"))
                 .round(4).reset_index())
    summary.insert(1, "paper_name", summary["category"].map(PAPER_NAMES).fillna(summary["category"]))

    tp, fp, fn = int(df["TP"].sum()), int(df["FP"].sum()), int(df["FN"].sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    j = tp / (tp + fp + fn + 1e-9)
    glob_res = {"radargrams": len(df), "gt_boundary_crossings": int(df["n_gt"].sum()),
                "predictions": int(df["n_pred"].sum()), "TP": tp, "FP": fp, "FN": fn,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "jaccard": round(j, 4),
                "params": {"window": WINDOW, "depth_tol": DEPTH_TOL,
                           "dot_thresh": DOT_THRESH, "gt_row_tolerance": args.tol}}

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "xcheck_per_radargram.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "xcheck_per_category.csv"), index=False)
    with open(os.path.join(args.out, "xcheck_global.json"), "w") as fh:
        json.dump(glob_res, fh, indent=2)
    with open(os.path.join(args.out, "xcheck_predictions.json"), "w") as fh:
        json.dump(preds, fh)

    print("\n" + "=" * 70)
    print("PER-CATEGORY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\n" + "=" * 70)
    print("GLOBAL TOTALS (all categories)")
    print("=" * 70)
    print(f"  Radargrams evaluated : {glob_res['radargrams']}")
    print(f"  GT boundary crossings: {glob_res['gt_boundary_crossings']}")
    print(f"  Predictions          : {glob_res['predictions']}")
    print(f"  TP / FP / FN         : {tp} / {fp} / {fn}")
    print(f"  Precision {p:.4f}  Recall {r:.4f}  F1 {f:.4f}  Jaccard {j:.4f}")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
