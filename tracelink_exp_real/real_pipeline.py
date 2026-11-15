"""
Real-data protocol for TraceLink (Part 4).

Functions extracted unchanged from Real_Annotated_Layer_Tracing (2).ipynb,
the run used for the IEEE BigData 2026 paper:

  clean_layer_id_map      per-column de-dup, despeckle, +/-1 px jitter removal
  layer_column_coverage   fraction of columns a layer spans
  inject_gaps             seeded random gaps in qualifying layers
  dilate_gt_bands         +/-2 px vertical bands per layer (thick IoU)
  derive_merge_tolerances merge-pass slope tolerances from the mask's own slopes (MAD)
  iou_level_score         thick IoU + Hungarian: recovery, precision, F1, mean IoU
  link_level_score        per-gap bridging check at row tolerance 0 / 1 / 2

``load_expert_layer_id_map`` and ``evaluate_layer_id_map`` are the
notebook's ``_load`` and ``_eval`` with their settings passed in as
arguments instead of being read from the enclosing function.
"""

import os

import numpy as np
from skimage.morphology import skeletonize

from tracelink import (REAL_PRESET, auto_params, compute_slope_confidence,
                       merge_layers_strict, renumber_layers_by_row, trace_layers)


def clean_layer_id_map(gt_layer_id_map, min_area=None, median_window=3,
                           max_jitter_disp=2, verbose=False):
        """Conservative noise removal on a rasterized layer-ID map.

        Stage 1: per-column de-duplication (multiple rows of same layer in one
                  column -> keep median row). Fixes rasterization fattening.
        Stage 2: connected-component area despeckle (remove tiny stray blobs).
                  min_area derived from component-size histogram if None.
        Stage 4: per-layer +/-1px jitter removal via short 1-D median on row(col).
                  Window kept short (3-5) so true slope/curvature is preserved.

        NO isotropic morphology, NO closing (which could merge ~17px-spaced layers).
        Returns (cleaned_id_map, report).
        """
        from skimage.measure import label, regionprops
        from scipy.ndimage import median_filter

        out = gt_layer_id_map.copy()
        report = {}
        layer_ids = [int(x) for x in np.unique(out) if x > 0]

        def _build_coord_map(arr):
            _ay, _ax = np.nonzero(arr)
            if not len(_ay):
                return {}
            _al = arr[_ay, _ax]
            _srt = np.argsort(_al, kind='stable')
            _ays, _axs, _als = _ay[_srt], _ax[_srt], _al[_srt]
            _u, _us = np.unique(_als, return_index=True)
            _ue = np.r_[_us[1:], len(_als)]
            return {int(u): (_ays[s:e], _axs[s:e])
                    for u, s, e in zip(_u, _us, _ue) if u > 0}

        _EMPTY = np.array([], dtype=np.intp)

        # Stage 1: per-column de-dup (keep median row per (layer,column))
        dedup = 0
        _cmap1 = _build_coord_map(out)
        for lid in layer_ids:
            ys, xs = _cmap1.get(lid, (_EMPTY, _EMPTY))
            if len(xs) == 0:
                continue
            order = np.argsort(xs); xs_s, ys_s = xs[order], ys[order]
            ucols, starts = np.unique(xs_s, return_index=True)
            ends = np.r_[starts[1:], len(xs_s)]
            for c, s, e in zip(ucols, starts, ends):
                if e - s > 1:
                    rows = ys_s[s:e]; keep = int(np.median(rows))
                    for r in rows:
                        out[r, c] = 0
                    out[keep, c] = lid
                    dedup += (e - s) - 1
        report["dedup_removed"] = int(dedup)

        # Stage 2: despeckle by connected-component area
        binary = (out > 0).astype(np.uint8)
        lbl = label(binary, connectivity=2)
        areas = np.array([p.area for p in regionprops(lbl)])
        if min_area is None and len(areas):
            hist = np.bincount(areas.astype(int))
            # speckle = area-1 spike; cut just above only if it dominates. cap 5.
            derived = 2 if (len(hist) > 2 and hist[1] > np.median(areas)) else 1
            min_area = int(min(max(derived, 1), 5))
        elif min_area is None:
            min_area = 1
        report["min_area"] = int(min_area)
        despeck = 0
        if min_area > 1:
            n_lab = int(lbl.max())
            if n_lab > 0:
                area_map = np.zeros(n_lab + 1, dtype=np.int64)
                for p in regionprops(lbl):
                    area_map[p.label] = p.area
                remove_mask = (area_map[lbl] > 0) & (area_map[lbl] < min_area)
                despeck = int(remove_mask.sum())
                out[remove_mask] = 0
        report["despeckle_removed"] = int(despeck)

        # Stage 4: per-layer +/-1px jitter removal (short 1-D median on row(col))
        max_disp = 0; jl = 0
        if median_window and median_window >= 3:
            _cmap4 = _build_coord_map(out)
            for lid in layer_ids:
                ys, xs = _cmap4.get(lid, (_EMPTY, _EMPTY))
                if len(xs) < median_window:
                    continue
                order = np.argsort(xs); xs_s, ys_s = xs[order], ys[order]
                ucols, idx = np.unique(xs_s, return_index=True)
                rows_arr = ys_s[idx].astype(float)
                sm = median_filter(rows_arr, size=median_window, mode="nearest")
                changed = np.where(sm.astype(int) != rows_arr.astype(int))[0]
                if len(changed):
                    layer_changed = False
                    for k in changed:
                        c = int(ucols[k]); old_r = int(rows_arr[k]); new_r = int(sm[k])
                        # GUARDRAIL: only accept a smoothing move of <= max_jitter_disp
                        # px. Larger "moves" are not jitter (they come from gaps/
                        # outliers) and must not alter true geometry.
                        if abs(new_r - old_r) > max_jitter_disp:
                            continue
                        max_disp = max(max_disp, abs(new_r - old_r))
                        out[old_r, c] = 0; out[new_r, c] = lid
                        layer_changed = True
                    if layer_changed:
                        jl += 1
        report["jitter_layers"] = int(jl)
        report["jitter_max_disp"] = int(max_disp)

        report["layers_in"]  = len(layer_ids)
        report["layers_out"] = len([x for x in np.unique(out) if x > 0])
        if verbose:
            print(f"  [clean] dedup={report['dedup_removed']} "
                  f"min_area={report['min_area']} despeckle={report['despeckle_removed']} "
                  f"jitter_layers={report['jitter_layers']}(max {report['jitter_max_disp']}px) "
                  f"layers {report['layers_in']}->{report['layers_out']}")
        return out, report


def dilate_gt_bands(gt_layer_id_map, half_width=2):
        """Dilate each rasterized GT layer VERTICALLY into a band of half_width px,
        per-layer, so the real GT matches the synthetic GT's thick representation.

        Vertical-only dilation: a pick at (r,c) fills rows [r-half_width, r+half_width]
        in that column. Per-layer stamping means bands never merge across layers
        (each keeps its own ID). half_width=2 -> 5px band, well under the ~17px
        inter-layer spacing, so adjacent layers cannot touch.

        Returns gt_band_id_map (int32).
        """
        H, W = gt_layer_id_map.shape
        out = np.zeros_like(gt_layer_id_map, dtype=np.int32)
        # Vectorized: shift the whole map by each row offset and take element-wise
        # maximum. Higher lid wins on overlap (matches ascending-order stamp behavior).
        for dr in range(-half_width, half_width + 1):
            r0, r1 = max(0, dr), min(H, H + dr)
            s0, s1 = max(0, -dr), min(H, H - dr)
            np.maximum(out[r0:r1, :], gt_layer_id_map[s0:s1, :], out=out[r0:r1, :])
        return out


def dilate_band_map(id_map, half_width=2):
        """Generic per-ID vertical band dilation (used for BOTH gt and prediction
        so IoU is thick-vs-thick: high and tolerant of +/-1-2px skeleton shift)."""
        return dilate_gt_bands(id_map, half_width=half_width)


def layer_column_coverage(gt_layer_id_map, lid):
        """Fraction of total columns this layer spans (measured on thin/original)."""
        W = gt_layer_id_map.shape[1]
        cols = np.unique(np.where(gt_layer_id_map == lid)[1])
        return len(cols) / W, cols


def inject_gaps(gt_layer_id_map, qualifying_ids, seed=128,
                    n_gaps_range=(2, 4), gap_width_range=(10, 30),
                    endpoint_margin=20, min_gap_sep=40):
        """Punch random column gaps into qualifying layers. Operates on whatever
        representation is passed in (we pass the THIN id map so gaps are clean)."""
        rng = np.random.default_rng(seed)
        gapped = gt_layer_id_map.copy()
        gaps = []
        W = gapped.shape[1]
        col_idx = np.arange(W)[None, :]
        _ys_inj, _xs_inj = np.nonzero(gt_layer_id_map)
        if len(_ys_inj):
            _lids_inj = gt_layer_id_map[_ys_inj, _xs_inj]
            _srt_inj = np.lexsort((_xs_inj, _lids_inj))
            _xs_inj_s, _lids_inj_s = _xs_inj[_srt_inj], _lids_inj[_srt_inj]
            _ulids_inj, _uidx_inj = np.unique(_lids_inj_s, return_index=True)
            _uend_inj = np.r_[_uidx_inj[1:], len(_lids_inj_s)]
            _lcmap_inj = {int(u): np.unique(_xs_inj_s[s:e])
                         for u, s, e in zip(_ulids_inj, _uidx_inj, _uend_inj) if u > 0}
        else:
            _lcmap_inj = {}
        for lid in qualifying_ids:
            cols = _lcmap_inj.get(lid, np.array([]))
            if len(cols) == 0:
                continue
            c_lo, c_hi = int(cols.min()), int(cols.max())
            span_lo, span_hi = c_lo + endpoint_margin, c_hi - endpoint_margin
            if span_hi - span_lo < gap_width_range[0]:
                continue
            n = int(rng.integers(n_gaps_range[0], n_gaps_range[1] + 1))
            placed = []; attempts = 0
            while len(placed) < n and attempts < n * 25:
                attempts += 1
                w = int(rng.integers(gap_width_range[0], gap_width_range[1] + 1))
                if span_hi - w <= span_lo:
                    continue
                start = int(rng.integers(span_lo, span_hi - w + 1)); end = start + w
                if all(end + min_gap_sep <= s or start - min_gap_sep >= e
                       for (s, e) in placed):
                    placed.append((start, end))
            for (start, end) in sorted(placed):
                region = (gapped == lid) & (col_idx >= start) & (col_idx < end)
                gapped[region] = 0
                gaps.append({"layer_id": int(lid), "col_start": int(start),
                             "col_end": int(end), "width": int(end - start)})
        return gapped, (gapped > 0).astype(np.uint8), gaps


def derive_row_tolerance(gt_layer_id_map, cap=2):
        """Link-sampling tolerance = min(cap, min_inter_layer_gap//2 - 1), >=0."""
        H, W = gt_layer_id_map.shape
        # Vectorized: sort all non-zero pixels by (col, row), diff adjacent same-col pairs
        _ys, _xs = np.nonzero(gt_layer_id_map)
        if len(_ys) == 0:
            return cap
        _srt = np.lexsort((_ys, _xs))
        _ys_s, _xs_s = _ys[_srt], _xs[_srt]
        _ls_s = gt_layer_id_map[_ys_s, _xs_s]
        _same_col = _xs_s[1:] == _xs_s[:-1]
        _diff_lab = _ls_s[1:] != _ls_s[:-1]
        _row_gaps = _ys_s[1:] - _ys_s[:-1]
        _valid_gaps = _same_col & _diff_lab
        if not _valid_gaps.any():
            return cap
        min_sep = int(_row_gaps[_valid_gaps].min())
        return max(0, int(min(cap, (min_sep // 2) - 1)))


def derive_merge_tolerances(lookup):
        """Derive (slope_tol_floor, slope_tol_ratio) from observed slope spread of
        multi-fragment layers. Real layers wobble more than synthetic; gate must
        reflect THIS mask. Falls back to research-backed defaults if too little data."""
        slopes = [v["slope"] for v in lookup.values() if v.get("n_fragments", 1) >= 2]
        if len(slopes) < 3:
            return 0.08, 0.50
        s = np.array(slopes)
        mad = np.median(np.abs(s - np.median(s))) * 1.4826
        return float(max(0.05, 2.0 * mad)), 0.50


def _build_iou(pred_layer_map, gt_id_map):
        """IoU matrix + id lists for predicted vs GT (thick). Returns
        (iou, gt_ids, pred_ids, gt_idx, pred_idx)."""
        gt_ids   = sorted(int(i) for i in np.unique(gt_id_map) if i > 0)
        pred_ids = sorted(int(i) for i in np.unique(pred_layer_map) if i > 0)
        gt_n, pred_n = len(gt_ids), len(pred_ids)
        if gt_n == 0 or pred_n == 0:
            return None, gt_ids, pred_ids, {}, {}
        gt_idx   = {g: i for i, g in enumerate(gt_ids)}
        pred_idx = {p: i for i, p in enumerate(pred_ids)}
        gf, pf = gt_id_map.ravel().astype(np.intp), pred_layer_map.ravel().astype(np.intp)
        # Pixel counts via bincount: single O(H*W) pass instead of O(N_ids * H*W)
        _max_id = max(int(gt_ids[-1]), int(pred_ids[-1])) + 1
        _gt_bc = np.bincount(gf, minlength=_max_id)
        _pr_bc = np.bincount(pf, minlength=_max_id)
        gt_c = _gt_bc[np.array(gt_ids, dtype=np.intp)]
        pr_c = _pr_bc[np.array(pred_ids, dtype=np.intp)]
        # Intersection via LUT + bincount: no per-pixel Python loop
        inter = np.zeros((gt_n, pred_n), dtype=np.int64)
        both = (gf > 0) & (pf > 0)
        if both.any():
            _mg = int(gf[both].max()) + 1; _mp = int(pf[both].max()) + 1
            _gl = np.zeros(_mg, dtype=np.intp); _pl = np.zeros(_mp, dtype=np.intp)
            for _i, _g in enumerate(gt_ids):
                if _g < _mg: _gl[_g] = _i
            for _i, _p in enumerate(pred_ids):
                if _p < _mp: _pl[_p] = _i
            _gi = _gl[gf[both]]; _pi = _pl[pf[both]]
            _flat = np.bincount(_gi * pred_n + _pi, minlength=gt_n * pred_n)
            inter = _flat.reshape(gt_n, pred_n)
        union = gt_c[:, None] + pr_c[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        return iou, gt_ids, pred_ids, gt_idx, pred_idx


def iou_level_score(pred_layer_map, gt_band_id_map, gapped_layer_ids,
                        iou_thresh=0.20, band_half_width=2):
        """Thick-GT IoU + Hungarian against FULL GT (matches synthetic metric).
        Reports gapped-layer recovery (recall) + global spurious precision.
        Also returns pred_to_gt mapping for Fix-2 link scoring."""
        from scipy.optimize import linear_sum_assignment
        gapped_set = set(int(i) for i in gapped_layer_ids)
        # THICK-vs-THICK: band the (thin, skeleton) prediction to match the GT band
        # thickness. Thin-vs-thick would cap IoU at ~1/thickness; banding both sides
        # makes IoU high AND tolerant of +/-1-2px skeleton shift. gt_band_id_map is
        # already a band (from dilate_gt_bands).
        pred_band = dilate_band_map(pred_layer_map, half_width=band_half_width)
        iou, gt_ids, pred_ids, gt_idx, pred_idx = _build_iou(pred_band, gt_band_id_map)
        if iou is None:
            return ({"n_gapped": len(gapped_set), "n_pred": len(pred_ids),
                     "recovered": 0, "recovery_rate": 0.0, "mean_iou_gapped": 0.0,
                     "spurious": len(pred_ids), "precision_global": 0.0,
                     "recall": 0.0, "f1": 0.0, "per_gapped_layer": []}, {})
        r_ind, c_ind = linear_sum_assignment(-iou)
        matched = {gt_ids[r]: (pred_ids[c], float(iou[r, c]))
                   for r, c in zip(r_ind, c_ind) if iou[r, c] >= iou_thresh}
        matched_pred = {p for (p, _) in matched.values()}
        pred_to_gt = {p: g for g, (p, _) in matched.items()}

        per_layer, iou_vals, recovered = [], [], 0
        for gid in sorted(gapped_set):
            if gid in matched:
                pid, v = matched[gid]; rec = v >= iou_thresh
            else:
                pid, v, rec = 0, 0.0, False
            recovered += int(rec); iou_vals.append(v)
            per_layer.append({"gt_layer_id": gid, "matched_pred_id": int(pid),
                              "iou": float(v), "recovered": bool(rec)})
        n_gapped = len(gapped_set)
        recovery_rate = recovered / n_gapped if n_gapped else 0.0
        mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0
        spurious = sum(1 for p in pred_ids if p not in matched_pred)
        precision = (len(pred_ids) - spurious) / len(pred_ids) if pred_ids else 0.0
        f1 = (2*precision*recovery_rate/(precision+recovery_rate)
              if (precision+recovery_rate) else 0.0)
        return ({"n_gapped": n_gapped, "n_pred": len(pred_ids),
                 "recovered": recovered, "recovery_rate": recovery_rate,
                 "mean_iou_gapped": mean_iou, "spurious": spurious,
                 "precision_global": precision, "recall": recovery_rate, "f1": f1,
                 "per_gapped_layer": per_layer}, pred_to_gt)


def link_level_score(pred_layer_map, gt_band_id_map, gaps, tol, pred_to_gt):
        """Fix 2: gap 'bridged' if predicted IDs on both flanks map to the SAME GT
        layer (via pred_to_gt) OR are literally the same predicted ID. Robust to
        tracer splits elsewhere on the layer. Samples within +/-tol rows of the GT
        band at the columns flanking the gap."""
        H, W = pred_layer_map.shape

        def pred_at(gid, col):
            if col < 0 or col >= W:
                return 0
            gt_rows = np.where(gt_band_id_map[:, col] == gid)[0]
            if len(gt_rows) == 0:
                return 0
            r0 = max(0, int(gt_rows.min()) - tol)
            r1 = min(H, int(gt_rows.max()) + tol + 1)
            band = pred_layer_map[r0:r1, col]
            nz = band[band > 0]
            if len(nz) == 0:
                return 0
            vals, counts = np.unique(nz, return_counts=True)
            return int(vals[np.argmax(counts)])

        per_gap = []
        for g in gaps:
            gid, cs, ce = g["layer_id"], g["col_start"], g["col_end"]
            left  = [p for p in (pred_at(gid, cs-1), pred_at(gid, cs-2)) if p > 0]
            right = [p for p in (pred_at(gid, ce),   pred_at(gid, ce+1)) if p > 0]
            left_gt  = {pred_to_gt.get(p, -p) for p in left}
            right_gt = {pred_to_gt.get(p, -p) for p in right}
            bridged = bool(left and right and (
                           (set(left) & set(right)) or
                           (gid in (left_gt & right_gt))))
            per_gap.append({**g, "left_pred": left[0] if left else 0,
                            "right_pred": right[0] if right else 0,
                            "bridged": bridged})
        n = len(per_gap)
        return per_gap, (sum(p["bridged"] for p in per_gap) / n if n else 0.0)


# ----------------------------------------------------------------------
# Loader (notebook: _load). Needs NDH_Tools.
# ----------------------------------------------------------------------

def load_expert_layer_id_map(file_name, layer_dir, radar_dir):
    """Rasterize the expert picks of one radargram into a layer-ID map.

    layer_dir is a path prefix such as './Nick-layer-data-mat/Layer_';
    the layer file is layer_dir + file_name. The radar file is
    radar_dir + file_name and is only used for its depth and distance axes.

    Returns (id_map int32 HxW, (H, W), collisions) where collisions counts
    picks of one layer that landed on a pixel already taken by another.
    """
    from NDH_Tools.loadmat import loadmat
    from NDH_Tools.radar_load import radar_load

    gt = loadmat(layer_dir + file_name)
    gd = gt['layer_info'][()][2]
    gl = gt['layer_info'][()][5].T
    rd, dd = radar_load(radar_dir + file_name, 0, 1)
    H, W = dd['new_data'].shape
    idm = np.zeros((H, W), dtype=np.int32)
    coll = 0
    _depth_ax = np.asarray(dd['depth_axis']).ravel()
    _dist_ax = np.asarray(rd['distance']).ravel()
    _gd_flat = np.asarray(gd).ravel()
    _d_sort = np.argsort(_depth_ax, kind='stable')
    _d_sorted = _depth_ax[_d_sort]
    _c_sort = np.argsort(_dist_ax, kind='stable')
    _c_sorted = _dist_ax[_c_sort]
    for i, layer in enumerate(gl):
        lid = i + 1
        _lyr = np.asarray(layer).ravel()
        _valid = ~np.isnan(_lyr)
        if not _valid.any():
            continue
        _d_vals = _lyr[_valid]
        _g_vals = _gd_flat[np.where(_valid)[0]]
        _si = np.searchsorted(_d_sorted, _d_vals).clip(0, len(_d_sorted) - 1)
        _si0 = np.maximum(_si - 1, 0)
        _d_idx = _d_sort[np.where(np.abs(_d_sorted[_si0] - _d_vals) < np.abs(_d_sorted[_si] - _d_vals), _si0, _si)]
        _ti = np.searchsorted(_c_sorted, _g_vals).clip(0, len(_c_sorted) - 1)
        _ti0 = np.maximum(_ti - 1, 0)
        _c_idx = _c_sort[np.where(np.abs(_c_sorted[_ti0] - _g_vals) < np.abs(_c_sorted[_ti] - _g_vals), _ti0, _ti)]
        _in_b = (_d_idx < H) & (_c_idx < W)
        _dv, _cv = _d_idx[_in_b], _c_idx[_in_b]
        _cur = idm[_dv, _cv]
        _new_mask = _cur == 0
        _coll_mask = (~_new_mask) & (_cur != lid)
        coll += int(_coll_mask.sum())
        idm[_dv[_new_mask], _cv[_new_mask]] = lid
    return idm, (H, W), coll


# ----------------------------------------------------------------------
# One radargram: clean -> select -> gap -> trace (notebook: _eval)
# ----------------------------------------------------------------------

def prepare_and_trace(gt_raw, coverage_th=0.70, gap_seed=128, band_half_width=2):
    """Everything in the notebook's _eval that does not depend on the IoU
    threshold. Returns None if no layer reaches coverage_th, else a dict
    with the traced layer map, the gapped GT band map and bookkeeping.
    """
    gt_clean, crep = clean_layer_id_map(gt_raw)
    ids = [int(x) for x in np.unique(gt_clean) if x > 0]
    W_cl = gt_clean.shape[1]
    ys_cl, xs_cl = np.nonzero(gt_clean)
    if len(ys_cl):
        lids_cl = gt_clean[ys_cl, xs_cl]
        pairs_cl = np.unique(np.stack([lids_cl, xs_cl], axis=1), axis=0)
        ulids_cl, ucnts_cl = np.unique(pairs_cl[:, 0], return_counts=True)
        cov_cl = {int(u): c / W_cl for u, c in zip(ulids_cl, ucnts_cl)}
    else:
        cov_cl = {}
    qual = [l for l in ids if cov_cl.get(l, 0) >= coverage_th]
    if not qual:
        return None

    gapped_id, gapped_bin, gaps = inject_gaps(gt_clean, qual, seed=gap_seed)
    # GT band from the GAPPED map: IoU measures fragment labelling, not gap
    # reconstruction (unfilled gaps are excluded, only mislabels/splits count).
    gt_band = dilate_gt_bands(gapped_id, half_width=band_half_width)
    skel = skeletonize(np.isin(gapped_id, qual)).astype(np.uint8)
    gc, rt = auto_params(skel, verbose=False, **REAL_PRESET)
    lmap, lk = trace_layers(gapped_bin, gc, rt, min_width=1, min_pixels=2)
    n_pre = len(lk)
    stf, str_ = derive_merge_tolerances(lk)
    lmap, lk, nm, _ = merge_layers_strict(lmap, lk, gap_cols_hint=gc,
                                          slope_tol_floor=stf, slope_tol_ratio=str_)
    lmap, lk = renumber_layers_by_row(lmap, lk)
    compute_slope_confidence(lk)
    return {"layer_map": lmap, "lookup": lk, "gt_band": gt_band, "qual": qual,
            "gaps": gaps, "gapped_bin": gapped_bin, "clean_report": crep,
            "gap_cols": int(gc), "row_tol": int(rt), "pre_merge": int(n_pre),
            "post_merge": int(len(lk)), "n_merges": int(nm)}


def score_prepared(prep, iou_thresh, band_half_width=2):
    """Score a prepare_and_trace result at one IoU threshold.
    Returns (row, per_gap_rows) with the notebook's CSV columns."""
    lmap, gt_band, qual, gaps = prep["layer_map"], prep["gt_band"], prep["qual"], prep["gaps"]
    iou_res, p2g = iou_level_score(lmap, gt_band, qual, iou_thresh=iou_thresh,
                                   band_half_width=band_half_width)
    tol = derive_row_tolerance(gt_band, cap=2)
    rates = {}
    for t in (0, 1, 2):
        _, r = link_level_score(lmap, gt_band, gaps, t, p2g)
        rates[t] = r
    per_gap, _ = link_level_score(lmap, gt_band, gaps, tol, p2g)
    row = {"n_qualifying": len(qual), "n_gaps": len(gaps), "derived_tol": tol,
           "bridge_tol0": rates[0], "bridge_tol1": rates[1], "bridge_tol2": rates[2],
           "recovery_rate": iou_res["recovery_rate"], "mean_iou": iou_res["mean_iou_gapped"],
           "spurious": iou_res["spurious"], "n_pred": iou_res["n_pred"],
           "precision_global": iou_res["precision_global"], "f1": iou_res["f1"],
           "pre_merge": prep["pre_merge"], "post_merge": prep["post_merge"],
           "n_merges": prep["n_merges"]}
    return row, per_gap, iou_res
