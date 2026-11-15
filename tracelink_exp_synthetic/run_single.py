#!/usr/bin/env python
"""Run TraceLink on one SyntheIce radargram, print scores, save outputs and a plot.

    python tracelink_exp_synthetic/run_single.py \
        ../SyntheIce-Dataset/data/full/synthetic_radargram_output_close_To_real0/synthetic_radargram_output_close_To_real067.h5 \
        --iou 0.50 --out results/single
"""

import argparse
import os
import sys

import h5py
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import iou_matrix, score_layer_maps  # noqa: E402
from figures.plot_overlay import plot_overlay  # noqa: E402
from tracelink import (SYNTHETIC_PRESET, atomic_save_json, run_tracelink,  # noqa: E402
                       save_h5_layer_map)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--out", default="results/single")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        mask = f["data/mask"][:]
        gt = f["data/layer_id_map"][:]
    layer_map, lookup, info = run_tracelink(mask, preset=SYNTHETIC_PRESET, return_info=True)
    s = score_layer_maps(gt, layer_map, [args.iou])[args.iou]

    iou, gt_ids, pred_ids = iou_matrix(gt, layer_map)
    r, c = linear_sum_assignment(-iou) if iou.size else ([], [])
    pairs = [(gt_ids[i], pred_ids[j]) for i, j in zip(r, c) if iou[i, j] >= args.iou]

    stem = os.path.splitext(os.path.basename(args.h5))[0]
    os.makedirs(args.out, exist_ok=True)
    save_h5_layer_map(os.path.join(args.out, stem + "_traced.h5"), layer_map, lookup,
                      {"gap_cols": info["gap_cols"], "row_tol": info["row_tol"]},
                      os.path.basename(args.h5))
    atomic_save_json(os.path.join(args.out, stem + "_traced.json"),
                     {"source_file": os.path.basename(args.h5), "params": info,
                      "scores": s, "layer_lookup": {f"L{k}": v for k, v in lookup.items()}})
    png = plot_overlay(gt, layer_map, {g for g, _ in pairs}, {p for _, p in pairs},
                       os.path.join(args.out, stem + "_overlay.png"),
                       title=f"{stem}  P={s['precision']:.2f} R={s['recall']:.2f} "
                             f"F1={s['f1']:.2f} mIoU={s['mean_iou']:.2f}")

    print(f"{stem}")
    print(f"  gap_cols={info['gap_cols']}  row_tol={info['row_tol']}  "
          f"merge {info['pred_pre_merge']} -> {info['pred_post_merge']}")
    print(f"  GT layers {s['gt_n']}  predicted {s['pred_n']}  TP {s['TP']}  FP {s['FP']}  FN {s['FN']}")
    print(f"  IoU >= {args.iou:.2f}: precision {s['precision']:.3f}  recall {s['recall']:.3f}  "
          f"F1 {s['f1']:.3f}  mean IoU {s['mean_iou']:.3f}")
    print(f"  saved {png}")


if __name__ == "__main__":
    main()
