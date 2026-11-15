"""
Layer-level scoring, as used for TraceLink on SyntheIce.

Taken from the batch evaluator in Conse_Layer_Tracing_ConC_F.ipynb:

1. Build the IoU matrix between every ground-truth layer and every
   predicted layer (pixel IoU on the layer-ID maps).
2. Match GT to predicted layers one-to-one with the Hungarian algorithm
   (scipy.optimize.linear_sum_assignment on -IoU).
3. A matched pair counts as a true positive when IoU >= threshold.
4. precision = TP / predicted layers, recall = TP / GT layers.
5. Mean / median IoU are over the true-positive pairs only.

The IoU matrix and the Hungarian assignment do not depend on the
threshold, so ``score_layer_maps`` computes them once and scores every
threshold from the same assignment. This gives the same numbers as
re-running the notebook once per threshold, only faster.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_THRESHOLDS = (0.20, 0.30, 0.50, 0.70)


def iou_matrix(gt_layer_map, pred_layer_map):
    """Return (iou, gt_ids, pred_ids). iou has shape (n_gt, n_pred)."""
    gt_ids = sorted(int(i) for i in np.unique(gt_layer_map) if i > 0)
    pred_ids = sorted(int(i) for i in np.unique(pred_layer_map) if i > 0)
    gt_n, pred_n = len(gt_ids), len(pred_ids)
    if gt_n == 0 or pred_n == 0:
        return np.zeros((gt_n, pred_n)), gt_ids, pred_ids

    gt_idx = {g: i for i, g in enumerate(gt_ids)}
    pred_idx = {p: i for i, p in enumerate(pred_ids)}
    inter = np.zeros((gt_n, pred_n), dtype=np.int64)
    gt_flat, pred_flat = gt_layer_map.ravel(), pred_layer_map.ravel()
    both = (gt_flat > 0) & (pred_flat > 0)
    if both.any():
        gi = np.fromiter((gt_idx[int(g)] for g in gt_flat[both]), dtype=np.int64)
        pi = np.fromiter((pred_idx[int(p)] for p in pred_flat[both]), dtype=np.int64)
        np.add.at(inter, (gi, pi), 1)
    gt_count = np.array([int((gt_layer_map == g).sum()) for g in gt_ids])
    pred_count = np.array([int((pred_layer_map == p).sum()) for p in pred_ids])
    union = gt_count[:, None] + pred_count[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, gt_ids, pred_ids


def score_layer_maps(gt_layer_map, pred_layer_map,
                     thresholds=DEFAULT_THRESHOLDS, iou_frag_th=0.05):
    """Score one prediction at several IoU thresholds.

    Returns {threshold: metrics dict}. Each metrics dict has gt_n, pred_n,
    TP, FP, FN, precision, recall, f1, mean_iou, median_iou, over_seg,
    under_seg.

    over_seg  = GT layers overlapping > 1 predicted layer with IoU > iou_frag_th
    under_seg = predicted layers overlapping > 1 GT layer with IoU > iou_frag_th
    """
    iou, gt_ids, pred_ids = iou_matrix(gt_layer_map, pred_layer_map)
    gt_n, pred_n = len(gt_ids), len(pred_ids)
    out = {}

    if gt_n == 0 or pred_n == 0:
        for t in thresholds:
            out[float(t)] = {"gt_n": gt_n, "pred_n": pred_n, "TP": 0,
                             "FP": pred_n, "FN": gt_n, "precision": 0.0,
                             "recall": 0.0, "f1": 0.0, "mean_iou": 0.0,
                             "median_iou": 0.0, "over_seg": 0, "under_seg": 0}
        return out

    row_ind, col_ind = linear_sum_assignment(-iou)
    over_seg = int(sum(1 for i in range(gt_n) if (iou[i, :] > iou_frag_th).sum() > 1))
    under_seg = int(sum(1 for j in range(pred_n) if (iou[:, j] > iou_frag_th).sum() > 1))

    for t in thresholds:
        matched = [float(iou[r, c]) for r, c in zip(row_ind, col_ind) if iou[r, c] >= t]
        TP = len(matched)
        FP, FN = pred_n - TP, gt_n - TP
        precision = TP / pred_n if pred_n else 0.0
        recall = TP / gt_n if gt_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[float(t)] = {
            "gt_n": gt_n, "pred_n": pred_n, "TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_iou": float(np.mean(matched)) if matched else 0.0,
            "median_iou": float(np.median(matched)) if matched else 0.0,
            "over_seg": over_seg, "under_seg": under_seg,
        }
    return out


def aggregate(rows):
    """Macro (per-file mean) and micro (summed counts) summary of per-file rows."""
    if not rows:
        return {}
    macro = lambda k: float(np.mean([r[k] for r in rows]))
    tp = sum(r["TP"] for r in rows)
    fp = sum(r["FP"] for r in rows)
    fn = sum(r["FN"] for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "files": len(rows),
        "macro": {"precision": macro("precision"), "recall": macro("recall"),
                  "f1": macro("f1"), "mean_iou": macro("mean_iou"),
                  "median_iou": macro("median_iou"),
                  "median_f1": float(np.median([x["f1"] for x in rows])),
                  "over_seg": macro("over_seg"), "under_seg": macro("under_seg")},
        "micro": {"precision": p, "recall": r, "f1": f},
        "totals": {"TP": tp, "FP": fp, "FN": fn,
                   "GT_layers": sum(x["gt_n"] for x in rows),
                   "pred_layers": sum(x["pred_n"] for x in rows)},
    }
