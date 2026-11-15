#!/usr/bin/env python
"""Run the real-data protocol on one radargram and save its artifacts.

    python tracelink_exp_real/run_single.py Data_20120516_01_084.mat \
        --layer-dir ./Nick-layer-data-mat/Layer_ --radar-dir ./Nick-raw-radargram-mat/ \
        --iou 0.50 --out results/real_single

Saves <stem>_gapped.npy, <stem>_gt_band.npy, <stem>_traced.h5,
<stem>_gaps.json and <stem>_overlay.png. Requires NDH_Tools.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from figures.plot_overlay import plot_overlay  # noqa: E402
from tracelink import atomic_save_json, save_h5_layer_map  # noqa: E402
from tracelink_exp_real.real_pipeline import (load_expert_layer_id_map,  # noqa: E402
                                              prepare_and_trace, score_prepared)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--radar-dir", required=True)
    ap.add_argument("--coverage", type=float, default=0.70)
    ap.add_argument("--gap-seed", type=int, default=128)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--out", default="results/real_single")
    args = ap.parse_args()

    gt_raw, (H, W), coll = load_expert_layer_id_map(args.file, args.layer_dir, args.radar_dir)
    prep = prepare_and_trace(gt_raw, args.coverage, args.gap_seed, args.band)
    if prep is None:
        raise SystemExit(f"No layer spans >= {args.coverage:.0%} of the width; nothing to test.")
    row, per_gap, iou_res = score_prepared(prep, args.iou, args.band)

    stem = os.path.splitext(args.file)[0]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, f"{stem}_gapped.npy"), prep["gapped_bin"])
    np.save(os.path.join(args.out, f"{stem}_gt_band.npy"), prep["gt_band"])
    save_h5_layer_map(os.path.join(args.out, f"{stem}_traced.h5"), prep["layer_map"], prep["lookup"],
                      {"gap_cols": prep["gap_cols"], "row_tol": prep["row_tol"]}, args.file)
    atomic_save_json(os.path.join(args.out, f"{stem}_gaps.json"),
                     {"file": args.file, "seed": args.gap_seed, "coverage_threshold": args.coverage,
                      "qualifying_layers": prep["qual"], "gaps": per_gap, "scores": row,
                      "clean_report": prep["clean_report"]})
    matched = [d for d in iou_res["per_gapped_layer"] if d["recovered"]]
    png = plot_overlay(prep["gt_band"], prep["layer_map"],
                       {d["gt_layer_id"] for d in matched}, {d["matched_pred_id"] for d in matched},
                       os.path.join(args.out, f"{stem}_overlay.png"),
                       title=f"{stem}  recall={row['recovery_rate']:.2f} F1={row['f1']:.2f} "
                             f"mIoU={row['mean_iou']:.2f}")

    print(f"{args.file}: {H}x{W}, {len(prep['qual'])} qualifying layers, {row['n_gaps']} gaps")
    print(f"  merge {row['pre_merge']} -> {row['post_merge']}")
    print(f"  IoU >= {args.iou:.2f}: precision {row['precision_global']:.3f}  recall "
          f"{row['recovery_rate']:.3f}  F1 {row['f1']:.3f}  mean IoU {row['mean_iou']:.3f}")
    print(f"  gap bridging tol 0/1/2: {row['bridge_tol0']:.1%} / {row['bridge_tol1']:.1%} / "
          f"{row['bridge_tol2']:.1%}")
    print(f"  saved {png}")


if __name__ == "__main__":
    main()
