#!/usr/bin/env python
"""Chain XCheck matches across all 4 tiles of one wide radargram (Cell 9).

    python xcheck_synthetic/stitch_one.py synthetic_radargram_output_close_To_real001_wide \
        --tiles ../SyntheIce-Dataset/data/xcheck_set/tiles \
        --data ../SyntheIce-Dataset/data/full --out results/xcheck/stitch

Runs XCheck on the 4 tiles, joins matched layers into chains with
union-find, and checks every chain against the parent radargram:
a chain is TP if all its pieces belong to one parent layer, FP if they
belong to two or more. FN = parent layers crossing a tile boundary that
no TP chain covers. Saves a two-panel figure and a JSON summary.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xcheck_synthetic.xcheck import (WINDOW, _skel_from_layer_ids, load_crops,  # noqa: E402
                                     run_xcheck, stitch_chains)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent", help="parent radargram name, without .h5")
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--data", required=True, help="folder holding the category folders")
    ap.add_argument("--out", default="results/xcheck/stitch")
    args = ap.parse_args()

    crops = sorted(glob.glob(os.path.join(args.tiles, f"{args.parent}_cropped_*.h5")))
    if not crops:
        sys.exit(f"No tiles for {args.parent} in {args.tiles}")
    parent_file = next((p for p in glob.glob(os.path.join(args.data, "synthetic_radargram_output_*",
                                                          args.parent + ".h5"))), None)
    if parent_file is None:
        sys.exit(f"Parent file {args.parent}.h5 not found under {args.data}")

    centerlines, skel_maps, _ = load_crops(crops)
    with h5py.File(parent_file, "r") as f:
        parent_mask = f["data/mask"][:]
        parent_skel = (f["derived/skeleton_layer_id_map"][:] if "derived/skeleton_layer_id_map" in f
                       else _skel_from_layer_ids(f["data/layer_id_map"][:]))

    preds = run_xcheck(centerlines)
    m, results, _, gmap, extra = stitch_chains(preds, skel_maps, parent_skel, window=WINDOW)

    print(f"{args.parent}: {len(crops)} tiles, {len(preds)} boundary matches")
    print(f"  links applied {m['links_applied']}, failed {m['links_failed']}")
    print(f"  GT multi-tile layers {m['gt_multi_crop_layers']}, predicted chains {m['predicted_chains']}")
    print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}")
    print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
          f"F1 {m['f1']:.3f}  Jaccard {m['jaccard']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.parent}_stitch.json"), "w") as fh:
        json.dump(m, fh, indent=2)

    H, W = gmap.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for root, gid in extra["gid_of"].items():
        overlay[gmap == gid] = (0.0, 0.9, 0.3, 0.85) if results.get(root) == "TP" else (0.95, 0.2, 0.2, 0.85)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    axes[0].imshow(parent_mask[:, :W], cmap="gray", aspect="auto", alpha=0.7)
    axes[0].imshow(overlay, aspect="auto")
    axes[1].imshow(parent_skel[:, :W] > 0, cmap="inferno", aspect="auto", alpha=0.8)
    for ax, c in zip(axes, ("yellow", "cyan")):
        for k in extra["offsets"][1:]:
            ax.axvline(k, color=c, linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Depth (pixels)")
    axes[0].set_title(f"Stitched chains: green = TP ({m['TP']}), red = FP ({m['FP']})   "
                      f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    axes[0].legend(handles=[mpatches.Patch(color=(0.0, 0.9, 0.3), label=f"TP chains ({m['TP']})"),
                            mpatches.Patch(color=(0.95, 0.2, 0.2), label=f"FP chains ({m['FP']})")],
                   loc="lower right")
    axes[1].set_title(f"Ground truth (multi-tile layers: {m['gt_multi_crop_layers']}, missed: {m['FN']})")
    plt.suptitle(f"Full-width stitching: {args.parent}", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(args.out, f"{args.parent}_stitch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
