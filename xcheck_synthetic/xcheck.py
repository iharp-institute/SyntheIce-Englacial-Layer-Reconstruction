"""
XCheck across 4 adjacent tiles (Part 1).

Helpers from the bulk-evaluation notebook cells (Cell 11 and Cell 9),
unchanged except for being importable:

  run_xcheck     XCheck boundary matching on the tile centerlines, with a
                 Hungarian step that makes the matching one-to-one
  build_gt       ground-truth boundary matches from the tiles'
                 skeleton_layer_id_map (tiles keep the parent's layer IDs)
  fuzzy_metrics  TP / FP / FN with a +/-1 px row tolerance (used for the table)
  metrics        exact-match TP / FP / FN
  load_crops     read the 4 tiles of one wide radargram
  stitch_chains  chain the matched layers across all 4 tiles (union-find)
                 and check each chain against the parent radargram

XCheck parameters (Hassan et al., ICDMW 2025):
  WINDOW = 10 columns on each side of the boundary
  DEPTH_TOL = 3 rows
  DOT_THRESH = 6 (minimum dot product of the two binary row vectors)
"""

from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

WINDOW = 10
DEPTH_TOL = 3
DOT_THRESH = 6


def _extract_vector(mask, row, cols):
    """Binary presence vector for one row inside the boundary window."""
    return np.array([float(mask[row, c]) if 0 <= c < mask.shape[1] else 0.0
                     for c in cols], dtype=np.float32)


def _lid_from_window(layer_map, row, cols):
    """Most common non-zero layer ID in one row of the boundary window."""
    vals = [int(layer_map[row, c]) for c in cols
            if 0 <= c < layer_map.shape[1] and layer_map[row, c] > 0]
    return Counter(vals).most_common(1)[0][0] if vals else None


def build_gt(skel_id_maps, window=WINDOW):
    """Ground-truth matches: for every layer present on both sides of a
    boundary, one tuple (r, r+1, centroid_row_left, centroid_row_right)."""
    gt = set()
    for r in range(len(skel_id_maps) - 1):
        g1, g2 = skel_id_maps[r], skel_id_maps[r + 1]
        c1 = list(range(g1.shape[1] - window, g1.shape[1]))
        c2 = list(range(0, window))
        lr1, lr2 = {}, {}
        for row in np.where(np.any(g1[:, c1] > 0, axis=1))[0]:
            lid = _lid_from_window(g1, row, c1)
            if lid:
                lr1.setdefault(lid, []).append(row)
        for row in np.where(np.any(g2[:, c2] > 0, axis=1))[0]:
            lid = _lid_from_window(g2, row, c2)
            if lid:
                lr2.setdefault(lid, []).append(row)
        cen1 = {lid: int(round(np.mean(rows))) for lid, rows in lr1.items()}
        cen2 = {lid: int(round(np.mean(rows))) for lid, rows in lr2.items()}
        for lid in set(cen1) & set(cen2):
            gt.add((r, r + 1, cen1[lid], cen2[lid]))
    return gt


def run_xcheck(centerlines, window=WINDOW, depth_tol=DEPTH_TOL, dot_thresh=DOT_THRESH):
    """XCheck on consecutive tiles. Returns [(r, r+1, row_left, row_right), ...].

    For each pair of boundary rows within depth_tol of each other, the
    binary row vectors of the two windows are compared by dot product;
    pairs with dot >= dot_thresh are candidates. A Hungarian assignment
    (cost = row distance) then keeps at most one match per row.
    """
    all_matches = []
    for r in range(len(centerlines) - 1):
        m1, m2 = centerlines[r], centerlines[r + 1]
        c1 = list(range(m1.shape[1] - window, m1.shape[1]))
        c2 = list(range(0, window))
        rows1 = np.where(np.any(m1[:, c1], axis=1))[0]
        rows2 = np.where(np.any(m2[:, c2], axis=1))[0]

        raw = []
        for i in rows1:
            v1 = _extract_vector(m1, i, c1)
            for j in rows2:
                if abs(int(i) - int(j)) > depth_tol:
                    continue
                dp = float(np.dot(v1, _extract_vector(m2, j, c2)))
                if dp >= dot_thresh:
                    raw.append((int(i), int(j), dp))
        if not raw:
            continue

        ru = sorted(set(x[0] for x in raw))
        cu = sorted(set(x[1] for x in raw))
        LARGE = 1e9
        cost = np.full((len(ru), len(cu)), LARGE)
        dp_lk = {(x[0], x[1]): x[2] for x in raw}
        for ii, ri in enumerate(ru):
            for jj, rj in enumerate(cu):
                if (ri, rj) in dp_lk:
                    cost[ii, jj] = abs(ri - rj)
        ri_idx, ci_idx = linear_sum_assignment(cost)
        for ii, jj in zip(ri_idx, ci_idx):
            if cost[ii, jj] < LARGE:
                all_matches.append((r, r + 1, ru[ii], cu[jj]))
    return all_matches


def metrics(pred_set, gt_set):
    """Exact-match TP / FP / FN and precision, recall, F1, Jaccard."""
    TP = len(pred_set & gt_set)
    FP = len(pred_set - gt_set)
    FN = len(gt_set - pred_set)
    P = TP / (TP + FP + 1e-9)
    R = TP / (TP + FN + 1e-9)
    F = 2 * P * R / (P + R + 1e-9)
    J = TP / (TP + FP + FN + 1e-9)
    return dict(TP=TP, FP=FP, FN=FN, precision=round(P, 4), recall=round(R, 4),
                f1=round(F, 4), jaccard=round(J, 4))


def fuzzy_metrics(pred_set, gt_set, tol=1):
    """Like metrics(), but a prediction matches a GT tuple on the same
    boundary when both rows are within +/-tol px (each GT used once)."""
    pl = list(pred_set)
    gl = list(gt_set)
    mg, mp = set(), set()
    for pi, p in enumerate(pl):
        pr, pr1, pir, pjr = p
        bg, bd = None, 999
        for gi, g in enumerate(gl):
            if gi in mg:
                continue
            gr, gr1, gir, gjr = g
            if gr != pr or gr1 != pr1:
                continue
            if abs(pir - gir) <= tol and abs(pjr - gjr) <= tol:
                d = abs(pir - gir) + abs(pjr - gjr)
                if d < bd:
                    bd, bg = d, gi
        if bg is not None:
            mg.add(bg)
            mp.add(pi)
    TP = len(mp)
    FP = len(pl) - TP
    FN = len(gl) - len(mg)
    P = TP / (TP + FP + 1e-9)
    R = TP / (TP + FN + 1e-9)
    F = 2 * P * R / (P + R + 1e-9)
    J = TP / (TP + FP + FN + 1e-9)
    return dict(TP=TP, FP=FP, FN=FN, precision=round(P, 4), recall=round(R, 4),
                f1=round(F, 4), jaccard=round(J, 4))


def _skel_from_layer_ids(lid):
    skel = np.zeros_like(lid, dtype=np.int32)
    for lv in np.unique(lid):
        if lv == 0:
            continue
        skel[skeletonize(lid == lv)] = lv
    return skel


def load_crops(crop_files):
    """Return (centerlines, skel_maps, metas) for the tiles, sorted by file name."""
    centerlines, skel_maps, metas = [], [], []
    for f in sorted(crop_files):
        with h5py.File(f, "r") as h:
            centerlines.append(h["derived/centerline"][:])
            if "derived/skeleton_layer_id_map" in h:
                skel_maps.append(h["derived/skeleton_layer_id_map"][:])
            else:
                skel_maps.append(_skel_from_layer_ids(h["data/layer_id_map"][:]))
            metas.append(dict(h["metadata"].attrs))
    return centerlines, skel_maps, metas


# ----------------------------------------------------------------------
# Chain stitching across all tiles (Cell 9)
# ----------------------------------------------------------------------

def _lid_at_boundary_row(skel_map, row, side, window, row_expand=2):
    w = skel_map.shape[1]
    cols = range(w - window, w) if side == "right" else range(0, window)
    values = []
    for rr in range(max(0, row - row_expand), min(skel_map.shape[0], row + row_expand + 1)):
        for c in cols:
            if 0 <= c < w and skel_map[rr, c] > 0:
                values.append(int(skel_map[rr, c]))
    return Counter(values).most_common(1)[0][0] if values else None


def stitch_chains(predictions, skel_maps, parent_skel, window=WINDOW):
    """Chain matched layers across tiles and score the chains.

    Union-find joins (tile k, layer) with (tile k+1, layer) for every
    XCheck match. A chain spanning >= 2 tiles is a TP if all its pieces
    fall on one parent layer, otherwise FP. FN = parent layers spanning
    >= 2 tiles that no TP chain covers.

    ``window`` is the boundary window used to read layer IDs at each
    match. Use the same window that produced the predictions (10 for
    run_xcheck; the notebook's Cell 9 used 7 to match its own Cell 6).

    Returns (metrics dict, chain_results, chain_detail, global_id_map, extra).
    """
    n = len(skel_maps)
    widths = [m.shape[1] for m in skel_maps]
    offsets = [sum(widths[:k]) for k in range(n)]
    H = skel_maps[0].shape[0]
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for k, skel in enumerate(skel_maps):
        for lid in np.unique(skel):
            if lid:
                find((k, int(lid)))

    ok = failed = 0
    for (r, r1, row_r, row_r1) in predictions:
        a = _lid_at_boundary_row(skel_maps[r], row_r, "right", window)
        b = _lid_at_boundary_row(skel_maps[r1], row_r1, "left", window)
        if a is not None and b is not None:
            union((r, int(a)), (r1, int(b)))
            ok += 1
        else:
            failed += 1

    groups = defaultdict(list)
    for node in list(parent):
        groups[find(node)].append(node)
    multi = {root: nodes for root, nodes in groups.items() if len({x[0] for x in nodes}) >= 2}

    def parent_lid(k, lid):
        rows, cols = np.where(skel_maps[k] == lid)
        if rows.size == 0:
            return None
        vals = parent_skel[rows, cols + offsets[k]]
        vals = vals[vals > 0]
        return int(Counter(vals.tolist()).most_common(1)[0][0]) if vals.size else None

    chain_results, chain_detail = {}, {}
    for root, nodes in multi.items():
        plids = {p for p in (parent_lid(k, lid) for k, lid in nodes) if p is not None}
        chain_results[root] = "TP" if len(plids) == 1 else "FP"
        chain_detail[root] = {"parent_lids": plids,
                              "crops_spanned": sorted({x[0] for x in nodes}), "nodes": nodes}

    gt_multi = set()
    for plid in np.unique(parent_skel):
        if plid == 0:
            continue
        cols = np.where(np.any(parent_skel == plid, axis=0))[0]
        touched = {k for k in range(n)
                   if np.any((cols >= offsets[k]) & (cols < offsets[k] + widths[k]))}
        if len(touched) >= 2:
            gt_multi.add(int(plid))

    covered = set()
    for root, res in chain_results.items():
        if res == "TP":
            covered |= chain_detail[root]["parent_lids"]

    TP = sum(v == "TP" for v in chain_results.values())
    FP = sum(v == "FP" for v in chain_results.values())
    FN = len(gt_multi - covered)
    P = TP / (TP + FP + 1e-9)
    R = TP / (TP + FN + 1e-9)
    F = 2 * P * R / (P + R + 1e-9)
    J = TP / (TP + FP + FN + 1e-9)

    gid_of = {root: i + 1 for i, root in enumerate(multi)}
    global_id_map = np.zeros((H, sum(widths)), dtype=np.int32)
    for k, skel in enumerate(skel_maps):
        for lid in np.unique(skel):
            if lid == 0:
                continue
            gid = gid_of.get(find((k, int(lid))), 0)
            if gid:
                rows, cols = np.where(skel == lid)
                global_id_map[rows, cols + offsets[k]] = gid

    m = {"TP": TP, "FP": FP, "FN": FN, "precision": P, "recall": R, "f1": F, "jaccard": J,
         "gt_multi_crop_layers": len(gt_multi), "predicted_chains": len(multi),
         "links_applied": ok, "links_failed": failed}
    extra = {"offsets": offsets, "widths": widths, "gid_of": gid_of}
    return m, chain_results, chain_detail, global_id_map, extra
