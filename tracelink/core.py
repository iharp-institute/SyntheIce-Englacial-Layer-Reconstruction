"""
TraceLink: slope-aware greedy fragment linking for englacial layers.

Extracted unchanged from Conse_Layer_Tracing_ConC_F.ipynb (the same code
is used for the synthetic and the real-data experiments). The only
change is that ``auto_params`` exposes the row-tolerance factor and floor
as arguments, because the real-data notebook raised them from
(0.40, 5) to (0.60, 8).

Pipeline (see ``run_tracelink``):
    skeletonize -> auto_params -> trace_layers -> merge_layers_strict
    -> renumber_layers_by_row -> compute_slope_confidence
"""

import bisect
import hashlib
import json
import os

import h5py
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

# auto_params settings used in the paper
SYNTHETIC_PRESET = {"row_tol_factor": 0.40, "row_tol_floor": 5}
REAL_PRESET = {"row_tol_factor": 0.60, "row_tol_floor": 8}


def _layer_hash(blob_props_list):
    """A4: content-addressed ID. SHA1 of sorted (col_min, col_max, row_min, row_max)
    tuples, truncated to 10 hex chars. Same content -> same ID across runs."""
    sig = tuple(sorted(
        (int(p.bbox[1]), int(p.bbox[3]-1), int(p.bbox[0]), int(p.bbox[2]-1))
        for p in blob_props_list
    ))
    h = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()
    return h[:10]


def _vectorized_relabel(lbl, mapping_arr):
    """Efficiency: O(H*W) single pass replaces O(N_blobs * H * W) per-blob masking.
    mapping_arr[old_label] = new_lid (0 for dropped blobs)."""
    return mapping_arr[lbl]


def trace_layers(mask, gap_cols, row_tol, min_width=1, min_pixels=2,
                 slope_factor=2.5, return_skel=False):
    """
    Greedy sequential layer assignment with slope-aware row gating.

    Fixes applied vs original:
      A1: Priority key reversed to (-row_diff, right): row-similarity wins
           over column proximity. This was the core misassignment bug.
      A2: Track recent slope (drow/dcol) per layer; reject candidates whose
           implied row jump exceeds slope_factor x predicted drift.
      A3: min_width default = 1 (was 2). Thin valid fragments no longer dropped.
           Separate min_pixels filter for true noise.
      A4: Each layer gets a content-hash ID alongside the sequential lid.
      Eff: Single vectorized relabel via lookup_arr[lbl] instead of per-blob
            boolean masking (~10-100x faster on dense masks).
      Eff: Sorted layer_right list + bisect for candidate window (O(log L)
            lookup instead of O(L) scan per blob).
    """
    skel_mask = skeletonize(mask > 0).astype(np.uint8)
    lbl = label(skel_mask, connectivity=2)

    props = {}
    for p in regionprops(lbl):
        width = p.bbox[3] - p.bbox[1]
        if width >= min_width and p.area >= min_pixels:
            props[p.label] = p

    # Sort by left edge; ties broken by centroid row
    sorted_labs = sorted(props, key=lambda x: (props[x].bbox[1], props[x].centroid[0]))

    layer_blobs = {}   # lid -> [blob_labels]
    layer_right = {}   # lid -> right edge of last blob
    layer_row   = {}   # lid -> running-mean centroid row
    layer_slope = {}   # lid -> recent drow/dcol (per column)
    layer_last_col_row = {}  # lid -> (right_col, row_at_that_col) of last blob
    next_lid = 1

    # Sorted index of (right_edge, lid) for fast candidate window lookup
    right_index = []  # sorted list of (right_edge, lid)

    for la in sorted_labs:
        a = props[la]
        a_left, a_right = a.bbox[1], a.bbox[3]
        a_row = a.centroid[0]

        # Candidate window: layers whose right edge in [a_left - gap_cols, a_left]
        lo = bisect.bisect_left(right_index, (a_left - gap_cols, -1))
        hi = bisect.bisect_right(right_index, (a_left, float('inf')))

        best_lid = None
        # A1: priority key = (-row_diff, right): row similarity FIRST
        best_key = (-float('inf'), -1)

        for idx in range(lo, hi):
            right, lid = right_index[idx]
            gap = a_left - right
            if gap < 0:
                continue  # structurally shouldn't happen, defensive

            # A2: slope-aware predicted row at a_left
            last_col, last_row = layer_last_col_row[lid]
            col_dist = max(a_left - last_col, 1)
            predicted_row = last_row + layer_slope[lid] * col_dist
            row_diff = abs(predicted_row - a_row)

            # Tolerance scales with predicted drift magnitude
            expected_drift = abs(layer_slope[lid]) * col_dist
            adaptive_tol = row_tol + slope_factor * expected_drift

            if row_diff <= adaptive_tol:
                key = (-row_diff, right)  # max row similarity, then max right
                if key > best_key:
                    best_key = key
                    best_lid = lid

        if best_lid is not None:
            layer_blobs[best_lid].append(la)
            old_right = layer_right[best_lid]
            old_last_col, old_last_row = layer_last_col_row[best_lid]

            # Remove old entry from right_index
            old_pos = bisect.bisect_left(right_index, (old_right, best_lid))
            if old_pos < len(right_index) and right_index[old_pos] == (old_right, best_lid):
                right_index.pop(old_pos)

            # Update slope: simple EWMA with new (col_dist, row_dist) sample
            new_col_dist = max(a_left - old_last_col, 1)
            instant_slope = (a_row - old_last_row) / new_col_dist
            # EWMA with alpha=0.5: adapts but stays stable
            layer_slope[best_lid] = 0.5 * layer_slope[best_lid] + 0.5 * instant_slope

            layer_right[best_lid] = a_right
            layer_last_col_row[best_lid] = (a_right - 1, a_row)
            # Running-mean centroid row (kept for diagnostics)
            n = len(layer_blobs[best_lid])
            layer_row[best_lid] += (a_row - layer_row[best_lid]) / n

            # Re-insert into sorted right_index
            bisect.insort(right_index, (a_right, best_lid))
        else:
            # New layer
            layer_blobs[next_lid] = [la]
            layer_right[next_lid] = a_right
            layer_row[next_lid]   = a_row
            layer_slope[next_lid] = 0.0  # unknown until 2nd blob arrives
            layer_last_col_row[next_lid] = (a_right - 1, a_row)
            bisect.insort(right_index, (a_right, next_lid))
            next_lid += 1

    # Efficient vectorized relabel
    max_lab = int(lbl.max())
    mapping_arr = np.zeros(max_lab + 1, dtype=np.int32)
    for lid, blob_list in layer_blobs.items():
        for lab in blob_list:
            mapping_arr[lab] = lid
    layer_map = _vectorized_relabel(lbl, mapping_arr)

    # Build lookup with content-hash IDs (A4)
    lookup = {}
    for lid, blob_list in layer_blobs.items():
        ordered = sorted(blob_list, key=lambda x: props[x].bbox[1])
        blob_objs = [props[lab] for lab in ordered]
        layer_hash = _layer_hash(blob_objs)
        frags = []
        for fid, p in enumerate(blob_objs, 1):
            frags.append({
                "fragment_id": fid,
                "col_range": [int(p.bbox[1]), int(p.bbox[3] - 1)],
                "row_range": [int(p.bbox[0]), int(p.bbox[2] - 1)],
                "n_pixels":  int(p.area),
            })
        lookup[lid] = {
            "hash_id": layer_hash,
            "mean_row": float(layer_row[lid]),
            "slope":   float(layer_slope[lid]),
            "n_fragments": len(frags),
            "fragments": frags,
        }

    if return_skel:
        return layer_map, lookup, skel_mask
    return layer_map, lookup


def auto_params(skel_mask, min_width=1, verbose=False,
                row_tol_factor=0.40, row_tol_floor=5):
    """Estimate (gap_cols, row_tol) from the mask's own geometry.

    row_tol = max(row_tol_floor, int(inter_spacing * row_tol_factor)).
    The defaults (0.40, 5) are the synthetic-data setting. The real-data
    experiment uses (0.60, 8), see REAL_PRESET below.
    """
    H, W = skel_mask.shape
    lbl = label(skel_mask, connectivity=2)
    blobs = [p for p in regionprops(lbl) if p.bbox[3] - p.bbox[1] >= min_width]

    overlap_row_diffs = []
    for i, a in enumerate(blobs):
        for j in range(i + 1, len(blobs)):
            b = blobs[j]
            col_overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
            if col_overlap >= 10:
                rd = abs(a.centroid[0] - b.centroid[0])
                if rd > 2:
                    overlap_row_diffs.append(rd)

    if len(overlap_row_diffs) >= 5:
        inter_spacing = float(np.percentile(overlap_row_diffs, 10))
    else:
        rows = np.where(skel_mask.sum(axis=1) > 0)[0]
        n_est = max(len(blobs) // 3, 5)
        inter_spacing = (int(rows[-1]) - int(rows[0])) / n_est if len(rows) > 1 else 30

    inter_spacing = max(8.0, inter_spacing)
    row_tol = max(row_tol_floor, int(inter_spacing * row_tol_factor))

    sorted_blobs = sorted(blobs, key=lambda p: p.centroid[0])
    band_merge = max(3, row_tol // 2)
    bands = [[sorted_blobs[0]]] if sorted_blobs else []
    for p in sorted_blobs[1:]:
        if p.centroid[0] - bands[-1][-1].centroid[0] <= band_merge:
            bands[-1].append(p)
        else:
            bands.append([p])

    intra_gaps = []
    for band in bands:
        ordered = sorted(band, key=lambda p: p.bbox[1])
        for i in range(len(ordered) - 1):
            g = ordered[i + 1].bbox[1] - ordered[i].bbox[3]
            if 0 < g < W // 2:
                intra_gaps.append(g)

    gap_cols = max(20, int(np.percentile(intra_gaps, 95))) if intra_gaps else max(30, W // 8)
    if verbose:
        print(f"  inter_spacing={inter_spacing:.1f}  row_tol={row_tol}  gap_cols={gap_cols}")
    return gap_cols, row_tol


def atomic_save_json(path, obj):
    """S3-style atomic write: write to .tmp then replace."""
    import os, json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_h5_layer_map(h5_path, layer_map, lookup, params, source_file):
    """S2: save full per-pixel layer_map to H5 alongside JSON.
    JSON loses pixel-level info; this preserves it for downstream IoU re-checks."""
    import h5py, os
    tmp = h5_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("layer_map", data=layer_map.astype(np.int32),
                         compression="gzip", compression_opts=4)
        f.attrs["source_file"] = source_file
        f.attrs["gap_cols"]    = int(params.get("gap_cols", -1))
        f.attrs["row_tol"]     = int(params.get("row_tol", -1))
        f.attrs["n_layers"]    = int(len(lookup))
        # Store hash IDs as a parallel array for downstream consumers
        hash_grp = f.create_group("layer_hashes")
        for lid, info in lookup.items():
            hash_grp.attrs[f"L{lid}"] = info["hash_id"]
    os.replace(tmp, h5_path)


def _frag_centroid(frag):
    """Centroid (col, row) of a fragment from its bbox."""
    cc = (frag["col_range"][0] + frag["col_range"][1]) / 2.0
    rc = (frag["row_range"][0] + frag["row_range"][1]) / 2.0
    return cc, rc


def _layer_extents(lookup, lid):
    frags = lookup[lid]["fragments"]
    col_min = min(f["col_range"][0] for f in frags)
    col_max = max(f["col_range"][1] for f in frags)
    return col_min, col_max


def _median_intra_gap(lookup, lid):
    """Median column gap between consecutive fragments inside this layer."""
    frags = sorted(lookup[lid]["fragments"], key=lambda f: f["col_range"][0])
    if len(frags) < 2:
        return None
    gaps = [max(frags[i+1]["col_range"][0] - frags[i]["col_range"][1], 0)
            for i in range(len(frags)-1)]
    return float(np.median(gaps))


def _refit_layer_slope(combined_frags):
    """Re-fit slope via linear regression on fragment centroids."""
    if len(combined_frags) < 2:
        return 0.0
    cs = np.array([_frag_centroid(f)[0] for f in combined_frags])
    rs = np.array([_frag_centroid(f)[1] for f in combined_frags])
    if cs.std() < 1e-6:
        return 0.0
    return float(np.polyfit(cs, rs, 1)[0])


def _layer_polyfit(fragments):
    """Linear regression (slope, intercept) on fragment centroids.

    Returns (None, None) if < 2 fragments or all centroids share a column.
    The tracer's EWMA-tracked slope is reactive to local noise and unsafe
    for merge decisions; a polyfit on the full fragment list is stable.
    """
    if len(fragments) < 2:
        return None, None
    cs = np.array([_frag_centroid(f)[0] for f in fragments])
    rs = np.array([_frag_centroid(f)[1] for f in fragments])
    if cs.std() < 1e-6:
        return 0.0, float(rs.mean())
    m, b = np.polyfit(cs, rs, 1)
    return float(m), float(b)


def merge_layers_strict(layer_map, lookup, gap_cols_hint=None,
                        min_intra_gap_floor=5, gap_ratio=2.0,
                        row_tol_base=3.0, slope_factor=1.5,
                        slope_tol_floor=0.03, slope_tol_ratio=0.30,
                        max_iters=10, verbose=False):
    """
    Item 1: STRICT post-trace merge pass.

    Bookkeeping operation: pixels with predicted ID B are retagged to A.
    Skeleton pixels, mask, and the gap between A and B are NOT modified.

    Merge layer pair (A, B) where A is column-left of B if ALL hold:
      (1) A.col_max < B.col_min                        (column-disjoint)
      (2) gap <= gap_ratio * max(median_intra_gap_A,
                                 median_intra_gap_B,
                                 gap_cols_hint,
                                 min_intra_gap_floor)  (relative gap)
      (3) |fitted_A(B.first_col) - B.first centroid row|
              <= row_tol_base + slope_factor * |fitted_A.slope| * gap
      (4) |fitted_A.slope - fitted_B.slope| <= max(slope_tol_floor,
                              slope_tol_ratio * max(|slope_A|, |slope_B|))

    *_fitted* slopes are recomputed via np.polyfit on fragment centroids each
    iteration: the tracer's EWMA `slope` field is too noisy to gate merges.

    Iterates until no merges happen (fixed point) or max_iters reached.
    Each merge re-fits slope/mean_row/hash_id of the absorbing layer.

    Returns: (new_layer_map, new_lookup, n_merges_total, n_iters)
    """
    layer_map = layer_map.copy()
    lookup = {k: {**v, "fragments": list(v["fragments"])} for k, v in lookup.items()}

    total_merges = 0
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        merged_this_pass = 0
        consumed = set()

        # Pre-compute fits, intra gaps, extents for this iteration
        fits    = {lid: _layer_polyfit(v["fragments"]) for lid, v in lookup.items()}
        intras  = {lid: _median_intra_gap(lookup, lid) for lid in lookup}
        extents = {}
        for lid, v in lookup.items():
            sorted_f = sorted(v["fragments"], key=lambda f: f["col_range"][0])
            extents[lid] = (sorted_f[0]["col_range"][0],
                            sorted_f[-1]["col_range"][1])

        for a_id in sorted(lookup.keys()):
            if a_id in consumed or a_id not in lookup:
                continue

            a_col_min, a_col_max = extents[a_id]
            a_slope, a_intercept = fits[a_id]
            # Fallback for single-fragment layers
            if a_slope is None:
                a_slope = 0.0
                a_intercept = _frag_centroid(lookup[a_id]["fragments"][0])[1]
            a_intra = intras[a_id]

            best_b, best_gap = None, float("inf")

            for b_id in sorted(lookup.keys()):
                if b_id == a_id or b_id in consumed or b_id not in lookup:
                    continue
                b_col_min, b_col_max = extents[b_id]

                # (1) column-disjoint
                if a_col_max >= b_col_min:
                    continue
                gap = b_col_min - a_col_max
                b_intra = intras[b_id]

                # (2) relative gap: reference scale is whichever is LARGEST of:
                #   - the first-pass gap_cols (gap the tracer already failed to bridge)
                #   - each layer's median intra-fragment gap
                #   - a small floor (prevents trivial merges on dense layers)
                # Conditions (3)+(4) below prevent bad merges across this allowed range.
                ref = max(a_intra if a_intra is not None else min_intra_gap_floor,
                          b_intra if b_intra is not None else min_intra_gap_floor,
                          gap_cols_hint if gap_cols_hint is not None else 0,
                          min_intra_gap_floor)
                if gap > gap_ratio * ref:
                    continue

                # (3) extrapolated row via fitted line (not EWMA)
                b_first_frag = sorted(lookup[b_id]["fragments"],
                                      key=lambda f: f["col_range"][0])[0]
                b_first_col, b_first_row = _frag_centroid(b_first_frag)
                predicted = a_slope * b_first_col + a_intercept
                row_diff = abs(predicted - b_first_row)
                adaptive_tol = row_tol_base + slope_factor * abs(a_slope) * gap
                if row_diff > adaptive_tol:
                    continue

                # (4) slope similarity using fitted slopes
                b_slope, _ = fits[b_id]
                if b_slope is None:
                    b_slope = 0.0
                slope_diff = abs(a_slope - b_slope)
                slope_tol = max(slope_tol_floor,
                                slope_tol_ratio * max(abs(a_slope), abs(b_slope)))
                if slope_diff > slope_tol:
                    continue

                if gap < best_gap:
                    best_b, best_gap = b_id, gap

            if best_b is not None:
                # Bookkeeping: retag pixels, append fragments
                layer_map[layer_map == best_b] = a_id

                combined = sorted(lookup[a_id]["fragments"] + lookup[best_b]["fragments"],
                                  key=lambda f: f["col_range"][0])
                for new_fid, frag in enumerate(combined, 1):
                    frag["fragment_id"] = new_fid
                lookup[a_id]["fragments"]   = combined
                lookup[a_id]["n_fragments"] = len(combined)

                total_pix = sum(f["n_pixels"] for f in combined)
                if total_pix > 0:
                    lookup[a_id]["mean_row"] = float(
                        sum(_frag_centroid(f)[1] * f["n_pixels"] for f in combined)
                        / total_pix
                    )

                # Re-fit slope on combined fragments (stored back as the new EWMA seed)
                m_new, _ = _layer_polyfit(combined)
                if m_new is not None:
                    lookup[a_id]["slope"] = float(m_new)

                sig = tuple(sorted(
                    (int(f["col_range"][0]), int(f["col_range"][1]),
                     int(f["row_range"][0]), int(f["row_range"][1]))
                    for f in combined
                ))
                lookup[a_id]["hash_id"] = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:10]

                del lookup[best_b]
                consumed.add(best_b)
                merged_this_pass += 1

        total_merges += merged_this_pass
        if verbose:
            print(f"  merge pass iteration {iteration}: {merged_this_pass} merged")
        if merged_this_pass == 0:
            break

    return layer_map, lookup, total_merges, iteration


def compute_slope_confidence(lookup, k=0.1):
    """
    Item 3: annotate each layer with a confidence in [0, 1] based on slope
    consistency between consecutive fragments.

    confidence = exp(-std(consecutive_slopes) / k)
      - 1-2 fragments: confidence = 1.0 (insufficient data to measure variance)
      - 3+ fragments: based on per-pair slope variance

    A low confidence means the layer's fragments don't lie on a smooth curve -
    candidate for a wrong merge. A high confidence means the fragments are
    consistently sloped, regardless of slope magnitude (steep tilted layers
    still score high if they tilt uniformly).

    Mutates lookup in-place; also returns it.
    """
    for lid, info in lookup.items():
        frags = sorted(info["fragments"], key=lambda f: f["col_range"][0])
        if len(frags) < 3:
            info["confidence"] = 1.0
            continue
        slopes = []
        for i in range(len(frags) - 1):
            c1, r1 = _frag_centroid(frags[i])
            c2, r2 = _frag_centroid(frags[i+1])
            col_dist = max(c2 - c1, 1.0)
            slopes.append((r2 - r1) / col_dist)
        slope_std = float(np.std(slopes))
        info["confidence"] = float(np.exp(-slope_std / k))
    return lookup


def renumber_layers_by_row(layer_map, lookup):
    """Reassign layer IDs in ascending order of mean_row (top of mask = ID 1).

    Fixes two issues:
      (a) trace_layers assigns IDs in left-to-right column order, not top-to-
          bottom row order. A layer starting at col 0 with mean_row 300 gets
          ID 1 while a layer starting at col 5 with mean_row 20 gets ID 2.
      (b) merge_layers_strict deletes the absorbed layer's ID, leaving gaps
          in the sequence (e.g. 1,2,3,4,5,6,8,9: missing 7).

    After this call: IDs are contiguous starting at 1, sorted top-to-bottom by
    each layer's mean_row. layer_map pixels are remapped consistently.
    Returns (new_layer_map, new_lookup).
    """
    if not lookup:
        return layer_map, lookup

    # Sort by mean_row; tie-break by column-min so the result is deterministic.
    def sort_key(item):
        lid, info = item
        col_min = min(f["col_range"][0] for f in info["fragments"])
        return (info["mean_row"], col_min)

    ordered = sorted(lookup.items(), key=sort_key)

    # Build remap: old_id -> new_id (1-indexed)
    remap = {old_lid: new_lid for new_lid, (old_lid, _) in enumerate(ordered, 1)}

    # Vectorized layer_map remap via lookup array
    max_old = max(remap.keys())
    remap_arr = np.zeros(max_old + 1, dtype=layer_map.dtype)
    for old_lid, new_lid in remap.items():
        remap_arr[old_lid] = new_lid
    # Pixels with value 0 (background) stay 0; others are remapped
    new_layer_map = np.where(layer_map > 0, remap_arr[layer_map], 0)

    new_lookup = {remap[old_lid]: info for old_lid, info in ordered}
    return new_layer_map, new_lookup


def run_tracelink(mask, preset=None, min_width=1, min_pixels=2,
                  confidence_k=0.1, return_info=False):
    """Run the full TraceLink pipeline on a binary layer mask.

    Parameters
    ----------
    mask : (H, W) array, nonzero = layer pixel
    preset : dict, optional
        auto_params settings, e.g. SYNTHETIC_PRESET (default) or REAL_PRESET.

    Returns
    -------
    layer_map : (H, W) int32, predicted layer ID per skeleton pixel (0 = bg)
    lookup : dict, per-layer info (hash_id, mean_row, slope, fragments, confidence)
    info : dict (only if return_info), gap_cols, row_tol, pre/post-merge counts
    """
    preset = SYNTHETIC_PRESET if preset is None else preset
    skel = skeletonize(mask > 0).astype(np.uint8)
    gap_cols, row_tol = auto_params(skel, verbose=False, **preset)
    layer_map, lookup = trace_layers(mask, gap_cols, row_tol,
                                     min_width=min_width, min_pixels=min_pixels)
    n_pre = len(lookup)
    layer_map, lookup, n_merges, n_iters = merge_layers_strict(
        layer_map, lookup, gap_cols_hint=gap_cols, verbose=False)
    layer_map, lookup = renumber_layers_by_row(layer_map, lookup)
    compute_slope_confidence(lookup, k=confidence_k)
    if not return_info:
        return layer_map, lookup
    info = {"gap_cols": int(gap_cols), "row_tol": int(row_tol),
            "pred_pre_merge": int(n_pre), "pred_post_merge": int(len(lookup)),
            "n_merges": int(n_merges), "n_iters": int(n_iters)}
    return layer_map, lookup, info
