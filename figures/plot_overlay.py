"""Overlay plot: traced layers vs ground truth (used by the run_single scripts).

Green line = predicted layer matched to a GT layer (TP), red = unmatched
prediction (FP), yellow = missed GT layer (FN), white = GT.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

GREEN = (0.15, 0.95, 0.30, 1.0)
RED = (1.00, 0.30, 0.30, 1.0)
YELLOW = (1.00, 0.92, 0.20, 1.0)


def _polylines(layer_map, pid):
    ys, xs = np.nonzero(layer_map == pid)
    if xs.size < 2:
        return []
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux, start = np.unique(xs, return_index=True)
    end = np.r_[start[1:], xs.size]
    my = np.array([np.median(ys[s:e]) for s, e in zip(start, end)])
    if ux.size < 2:
        return []
    gaps = np.diff(ux)
    split = np.where(gaps > max(3.0 * float(np.median(gaps)), 3.0))[0]
    starts, ends = np.r_[0, split + 1], np.r_[split + 1, ux.size]
    return [np.column_stack([ux[s:e], my[s:e]]) for s, e in zip(starts, ends) if e - s >= 2]


def plot_overlay(gt_layer_map, pred_layer_map, matched_gt, matched_pred, path,
                 title=None, figsize=(10, 6)):
    H, W = pred_layer_map.shape
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("black")

    gt_disp = np.zeros((H, W, 4), dtype=np.float32)
    gt_disp[gt_layer_map > 0] = [1.0, 1.0, 1.0, 0.6]
    fn = [g for g in np.unique(gt_layer_map) if g > 0 and int(g) not in matched_gt]
    for g in fn:
        gt_disp[gt_layer_map == g] = YELLOW
    ax.imshow(gt_disp, aspect="auto", interpolation="nearest")

    segs, cols = [], []
    for pid in [int(p) for p in np.unique(pred_layer_map) if p > 0]:
        color = GREEN if pid in matched_pred else RED
        for s in _polylines(pred_layer_map, pid):
            segs.append(s)
            cols.append(color)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.0, alpha=0.95))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("Distance (columns)")
    ax.set_ylabel("Depth (rows)")
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
