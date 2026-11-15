"""Shared scoring code: per-pixel IoU, Hungarian matching, P/R/F1, threshold sweep."""

from .metrics import DEFAULT_THRESHOLDS, aggregate, iou_matrix, score_layer_maps

__all__ = ["DEFAULT_THRESHOLDS", "aggregate", "iou_matrix", "score_layer_maps"]
