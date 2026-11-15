#!/usr/bin/env python
"""Part 4: TraceLink on real expert-annotated radargrams, Multiple IoU threshold.

    python tracelink_exp_real/run_batch.py \
        --layer-dir ./Nick-layer-data-mat/Layer_ \
        --radar-dir ./Nick-raw-radargram-mat/ \
        --out results/real

Protocol per radargram (settings used for the paper):
  1. rasterize the expert picks, keeping each layer's identity
  2. clean: per-column de-dup, despeckle, +/-1 px jitter removal
  3. keep layers spanning >= --coverage of the width (0.70)
  4. inject 2-4 seeded gaps (10-30 columns each) into those layers (seed 128)
  5. run TraceLink on the gapped mask (real auto_params preset)
  6. score with +/-2 px thick IoU + Hungarian matching against the
     gapped GT band (injected gaps excluded) at every --thresholds value

Steps 1-5 run once per file; every threshold is scored from the same
prediction, which gives the same numbers as one run per threshold.

Requires NDH_Tools (for reading the CReSIS .mat files).

Outputs, matching the notebook layout:
  <out>/iou_threshold_sensitivity/iou_0.XX/batch_real_results.csv
  <out>/iou_threshold_sensitivity/iou_0.XX/batch_real_gaps.csv
  <out>/iou_threshold_sensitivity/iou_0.XX/batch_real_summary.json
  <out>/iou_threshold_sensitivity/iou_threshold_sensitivity_table.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracelink import atomic_save_json  # noqa: E402
from tracelink_exp_real.real_pipeline import (load_expert_layer_id_map,  # noqa: E402
                                              prepare_and_trace, score_prepared)


def summarize(rows, skipped):
    macro = lambda k: float(np.mean([r[k] for r in rows]))
    tg = sum(r["n_gaps"] for r in rows)
    micro = {t: (sum(round(r[f"bridge_tol{t}"] * r["n_gaps"]) for r in rows) / tg if tg else 0.0)
             for t in (0, 1, 2)}
    return {"files_processed": len(rows), "files_skipped": len(skipped),
            "skipped": skipped, "total_gaps": tg,
            "total_collisions": sum(r["collisions"] for r in rows),
            "pre_merge_total": sum(r["pre_merge"] for r in rows),
            "post_merge_total": sum(r["post_merge"] for r in rows),
            "macro": {"recovery_rate": macro("recovery_rate"), "mean_iou": macro("mean_iou"),
                      "precision_global": macro("precision_global"), "f1": macro("f1"),
                      "bridge_tol0": macro("bridge_tol0"), "bridge_tol1": macro("bridge_tol1"),
                      "bridge_tol2": macro("bridge_tol2")},
            "micro_bridge": micro}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer-dir", required=True,
                    help="prefix of the expert layer files, e.g. ./Nick-layer-data-mat/Layer_")
    ap.add_argument("--radar-dir", required=True, help="folder of raw radargram .mat files")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--coverage", type=float, default=0.70)
    ap.add_argument("--gap-seed", type=int, default=128)
    ap.add_argument("--band", type=int, default=2, help="half-width of the thick-IoU band, px")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--file-glob", default="*.mat")
    ap.add_argument("--limit", type=int, default=None, help="first N files (quick test)")
    args = ap.parse_args()

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.radar_dir, args.file_glob)))
    if not files:
        d = os.path.dirname(args.layer_dir)
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, args.file_glob)))
    if args.limit:
        files = files[:args.limit]
    print(f"Found {len(files)} .mat files")

    sens_dir = os.path.join(args.out, "iou_threshold_sensitivity")
    rows = {t: [] for t in args.thresholds}
    gaps = {t: [] for t in args.thresholds}
    skipped = []

    for i, fn in enumerate(files, 1):
        try:
            gt_raw, _, coll = load_expert_layer_id_map(fn, args.layer_dir, args.radar_dir)
            prep = prepare_and_trace(gt_raw, args.coverage, args.gap_seed, args.band)
        except Exception as e:  # missing or malformed files are skipped, as in the notebook
            print(f"  SKIP {fn[:40]}: {type(e).__name__}: {e}")
            skipped.append((fn, f"{type(e).__name__}: {e}"))
            continue
        if prep is None:
            skipped.append((fn, "no qualifying layers"))
            continue
        for t in args.thresholds:
            row, per_gap, _ = score_prepared(prep, t, args.band)
            rows[t].append({"file": fn, "collisions": coll, **row})
            gaps[t].extend({"file": fn, **g} for g in per_gap)
        r = rows[args.thresholds[0]][-1]
        print(f"  [{i:>4}/{len(files)}] {fn[:40]:<40} layers={r['n_qualifying']:>3} "
              f"gaps={r['n_gaps']:>3} pre->post merge={r['pre_merge']}->{r['post_merge']}")

    if not any(rows.values()):
        sys.exit(f"No results (skipped {len(skipped)})")

    table = []
    for t in args.thresholds:
        d = os.path.join(sens_dir, f"iou_{t:.2f}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "batch_real_results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[t][0].keys()))
            w.writeheader()
            w.writerows(rows[t])
        if gaps[t]:
            with open(os.path.join(d, "batch_real_gaps.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(gaps[t][0].keys()))
                w.writeheader()
                w.writerows(gaps[t])
        s = summarize(rows[t], skipped)
        atomic_save_json(os.path.join(d, "batch_real_summary.json"), s)
        table.append({"iou_threshold": float(t), "files_processed": s["files_processed"],
                      "files_skipped": s["files_skipped"], "total_gaps": s["total_gaps"],
                      "macro_recovery_rate": s["macro"]["recovery_rate"],
                      "macro_mean_iou": s["macro"]["mean_iou"],
                      "macro_precision": s["macro"]["precision_global"],
                      "macro_f1": s["macro"]["f1"],
                      "macro_bridge_tol0": s["macro"]["bridge_tol0"],
                      "macro_bridge_tol1": s["macro"]["bridge_tol1"],
                      "macro_bridge_tol2": s["macro"]["bridge_tol2"],
                      "pre_merge_total": s["pre_merge_total"],
                      "post_merge_total": s["post_merge_total"]})

    with open(os.path.join(sens_dir, "iou_threshold_sensitivity_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    atomic_save_json(os.path.join(sens_dir, "settings.json"),
                     {"layer_dir": args.layer_dir, "radar_dir": args.radar_dir,
                      "coverage_threshold": args.coverage, "gap_seed": args.gap_seed,
                      "band_half_width": args.band, "iou_thresholds": args.thresholds})

    print("\n" + "=" * 78)
    print(f"  REAL DATA: {table[0]['files_processed']} radargrams processed, "
          f"{table[0]['files_skipped']} skipped (macro averages over ALL processed files)")
    print("=" * 78)
    print(f"{'IoU':>6} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Mean IoU':>10} {'Merges':>16}")
    for r in table:
        print(f"{r['iou_threshold']:>6.2f} {r['macro_precision']:>10.3f} "
              f"{r['macro_recovery_rate']:>10.3f} {r['macro_f1']:>8.3f} {r['macro_mean_iou']:>10.3f} "
              f"{r['pre_merge_total']:>7}->{r['post_merge_total']}")
    print("=" * 78)
    print(f"Saved to {sens_dir}/  (next: python tracelink_exp_real/summarize.py --results {args.out})")


if __name__ == "__main__":
    main()
