# SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction [Paper](https://arxiv.org/pdf/2611.14728)

## Abstract

Englacial layers in radio-echo-sounding radargrams (subsurface ice images) encode the past climate and flow history of ice sheets and glaciers, but extracting them at scale is bottlenecked by time-consuming manual annotation and by semi-automated detectors that return broken, disconnected fragments rather than continuous horizons. We address two coupled problems in layer annotation products: repairing partial annotations by connecting their fragments into coherent full-layer reconstructions across an entire profile, and validating layer continuity across the boundaries of adjacent tiles. We introduce a slope-aware greedy fragment-linking algorithm that reconstructs full-width layers from partial detections, assigning each fragment to its parent layer through row-similarity gating and an exponentially weighted slope predictor (EWMA) and attaching a content-based hash ID to each reconstruction for stable cross-radargram identification. For development and rigorous quantitative evaluation, we generate a synthetic dataset of 600 radargrams across two sizes (normal and wide) and multiple complexity levels, including realistic noise. The dataset includes 150 wide radargrams divided into four adjacent tiles, each with exact ground-truth annotations. We test this dataset with XCheck, a consistency-based validation framework: We extend XCheck from two to four consecutive tiles, checking consistency across every boundary, the synthetic data is realiable enough to drive real-world algorithms. Our major finding generalizes beyond ice: when linear features in spatial data are broken into fragments but keep a consistent local geometry, linking them by predicted trajectory is more reliable than by appearance. Evaluated against expert-labeled real radargrams, the method achieves high precision, recall, F1, and mean IoU, establishing transfer of our findings to a real-world dataset.

## SyntheIce Experiments: XCheck, TraceLink, and Evaluation

Code for the experiments in the paper *SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction*.

## Paper

The paper has been accepted in the proceedings of the 2026 IEEE International Conference on Big Data (BigData), Phoenix, AZ, USA, Dec. 2026.

> The synthetic dataset and its generator are in the companion repository: **[SyntheIce-Dataset](https://github.com/iharp-institute/SyntheIce-Dataset)**.

---

## What is in this repository

| Part | What it does | Folder |
|---|---|---|
| **1. XCheck on SyntheIce** | Runs the XCheck consistency check across 4 adjacent tiles cut from each wide synthetic radargram | `xcheck_synthetic/` |
| **2. TraceLink** | Slope-aware algorithm that groups broken layer fragments back into full-width layers | `tracelink/` |
| **3. TraceLink on SyntheIce** | Evaluates TraceLink on 400 synthetic radargrams with exact ground truth | `tracelink_exp_synthetic/` |
| **4. TraceLink on real data** | Evaluates TraceLink on real expert-annotated Greenland radargrams using a controlled gap-injection protocol | `tracelink_exp_real/` |
| **5. Evaluation** | Shared scoring code: IoU on thickened bands, Hungarian matching, precision, recall, F1, mean IoU, multiple IoU threshold settings | `evaluation/` |

The pipeline runs in this order:

```
SyntheIce data ──► Wide radargrams ──► cropped into adjacent 4 tiles and saved ──► XCheck (cross-tile consistency)
              └──► TraceLink ──► Evaluation (synthetic)
Real expert annotations ──► select + inject gaps ──► TraceLink ──► Evaluation (real)
```

---

## 1. XCheck on synthetic data

[XCheck](https://doi.org/10.1109/ICDMW69685.2025.00015) is our published method for checking whether layer annotations agree across neighbouring radargrams, to check the consistency of englacial layers. The original method compares **two** consecutive or non-consecutive images. Here we extend it to **four** consecutive tiles.

**Protocol**

1. Take 150 wide SyntheIce radargrams from all four categories.
2. Crop each into 4 adjacent tiles (600 tiles in total) and save each tile's mask and metadata.
3. Run XCheck at each shared tile boundary to check the consistency of layers.
4. Score XCheck's matches against the ground truth of the original uncropped radargram.

**Why it matters:** this shows two things. First, layer continuity holds across tile boundaries. Second, the synthetic data works with a published, real-world algorithm.

**Results** (148 radargrams evaluated, 37 per category)

| Category | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Close-to-Synthetic | 1.000 | 0.774 | 0.861 | 0.774 |
| Mid-Level | 1.000 | 0.720 | 0.825 | 0.720 |
| Close-to-Real | 0.999 | 0.723 | 0.827 | 0.723 |
| Noisy-Level | 1.000 | 0.787 | 0.872 | 0.787 |
| **All** | **1.000** | **0.754** | **0.860** | **0.754** |

XCheck almost never makes a wrong match (precision near 1.0). Its errors are missed boundary matches, which is why IoU equals recall.

**Settings** (from the XCheck paper): a 10-column window on each side of a boundary, a depth tolerance of 3 rows and a dot-product threshold of 6. A Hungarian step keeps the matching one-to-one. Ground truth is one row per layer per boundary (the layer's centroid row in the window), taken from the tiles' `skeleton_layer_id_map`. A prediction counts as correct when both of its rows are within ±1 px of a ground-truth match on the same boundary. The category rows are per-radargram means. The **All** row sums TP, FP and FN over all 148 radargrams.

`xcheck_synthetic/stitch_one.py` goes one step further for a single wide radargram: it chains the matched layers across all four tiles and checks each chain against the parent radargram.

---

## 2. TraceLink: slope-aware fragment linking

Automatic tracers break layers into fragments. **TraceLink repairs these annotations** by deciding which fragments belong to the same layer. It only records the fragment-to-layer mapping. It does **not** fill in the gaps or invent pixels.

**How it works**

1. Skeletonize the mask and extract fragments (connected components).
2. Sort fragments left to right, then top to bottom.
3. For each fragment, find candidate layers that end just before it, within a column window.
4. Predict where each candidate layer should be at this column using its slope, tracked with an exponentially weighted moving average (EWMA, weight α = 0.5).
5. Accept a candidate only if the fragment's row lies within a tolerance that widens with slope and distance. Among accepted candidates, pick the closest row (**row similarity first**).
6. If no candidate fits, start a new layer.
7. Run a **strict merge pass** that joins layers that were over-split, gated by a line fit.
8. Renumber layers top to bottom and give each one a content-based hash ID, so it keeps a stable identity.

**Self-tuning:** tolerances are derived from each radargram (for example, the slope tolerance comes from the median absolute deviation of observed slopes). Nothing is hand-tuned per image.

**Two settings.** The row tolerance is `max(floor, factor × inter-layer spacing)`. Synthetic data uses factor 0.40 with floor 5 (`SYNTHETIC_PRESET`). Real data uses 0.60 with floor 8 (`REAL_PRESET`), because real masks are denser and wobblier.

**Use it on your own mask:**

```python
from tracelink import run_tracelink, SYNTHETIC_PRESET   # or REAL_PRESET

layer_map, lookup = run_tracelink(mask, preset=SYNTHETIC_PRESET)
# layer_map: predicted layer ID per skeleton pixel
# lookup[layer_id]: hash_id, mean_row, slope, fragments, confidence
```

---

## 3. TraceLink on synthetic data

- **Data:** 400 SyntheIce radargrams, 100 per category.
- **Ground truth:** the SyntheIce layer-identity map, so every fragment's true parent layer is known.
- **Results (IoU matching threshold θ = 0.50, macro-averaged):** on the two cleaner categories, TraceLink is almost perfect (recall 1.00, F1 0.97 to 0.98). As the data gets denser and more broken, scores drop gradually, down to F1 0.85 on Noisy-Level. On Close-to-Real, precision falls more than recall (0.84 vs 0.95), so the main error there is extra splits rather than missed layers. On Noisy-Level, both precision and recall drop.

| Category | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| Close-to-Synthetic | 0.96 | 1.00 | 0.98 | 0.99 |
| Mid-Level | 0.95 | 1.00 | 0.97 | 0.99 |
| Close-to-Real | 0.84 | 0.95 | 0.89 | 0.93 |
| Noisy-Level | 0.82 | 0.89 | 0.85 | 0.90 |

**Effect of the strict merge pass** (predicted layer count before and after merging)

| Category | Before | After | Merges |
|---|---|---|---|
| Close-to-Synthetic | 2075 | 2032 | 43 |
| Mid-Level | 2392 | 2362 | 30 |
| Close-to-Real | 3246 | 3228 | 18 |
| Noisy-Level | 5147 | 5085 | 62 |

---

## 4. TraceLink on real expert annotations

Real expert labels are clean, but they have no controlled fragmentation, so there is nothing to score linking against. We build a controlled test from them.

**Protocol**

1. **Select:** keep expert layers that span at least **70%** of the radargram width. These are continuous enough to serve as ground truth.
2. **Clean:** remove speckle and small jitter. This step changes no layer geometry beyond ±1 to 2 pixels:
   - per-column de-duplication
   - connected-component despeckle, with the threshold taken from the data
   - short median filter for ±1 px jitter
3. **Break:** inject 2 to 4 random gaps (10 to 30 columns each) into every selected layer, with a fixed seed (128), to imitate detector breakage.
4. **Link:** run TraceLink on the broken layers.
5. **Score:** compare the result against the original unbroken layer.

**Why this test is fair**

- **The answer is known,** because the gaps are cut from layers that were continuous to begin with.
- **It doesn't depend on a second noisy label,** because the reference is the original expert layer.
- **It is reproducible,** because gaps are seeded and TraceLink's parameters are derived from each radargram.
- **The method isn't penalized for gaps,** because injected gaps are excluded from the reference. The score measures fragment grouping, not gap filling.

**Data:** CReSIS Greenland radargrams with expert layer annotations. Of 1136 `.mat` files, **770** were processed. The other 366 were skipped, either because no layer spans 70% of the width or because the file has no `layer_info`.

**Results over all 770 radargrams** (macro averages, as printed by `run_batch.py`)

| IoU threshold θ | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| 0.20 | 0.787 | 0.980 | 0.864 | 0.690 |
| 0.30 | 0.760 | 0.938 | 0.829 | 0.679 |
| **0.50** | **0.635** | **0.714** | **0.647** | **0.589** |
| 0.70 | 0.494 | 0.494 | 0.455 | 0.459 |

**Results reported in the paper.** The paper reports the averages over a subset of radargrams: those with mean IoU ≥ 0.65 **and** F1 ≥ 0.65. At θ = 0.50 that is **333 of the 770** radargrams (43%).

| IoU threshold θ | Radargrams kept | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|
| 0.20 | 464 / 770 | 0.828 | 0.995 | 0.900 | 0.808 |
| 0.30 | 452 / 770 | 0.817 | 0.985 | 0.889 | 0.809 |
| **0.50** | **333 / 770** | **0.776** | **0.929** | **0.842** | **0.821** |
| 0.70 | 188 / 770 | 0.730 | 0.884 | 0.795 | 0.838 |

- **Read the paper's subset with care.** The kept files are chosen by the same metrics that are reported. The subset shrinks as θ rises, which is why its mean IoU goes *up* (0.808 to 0.838) while the mean IoU over all files goes *down* (0.690 to 0.459).
- **Merge pass:** the paper's merge figure (14024 → 13677, 347 merges, 2.47%) is over the 439 radargrams with mean IoU ≥ 0.50 and F1 ≥ 0.65 at θ = 0.50. `summarize.py` prints the merge totals for whichever files you average over.

`tracelink_exp_real/summarize.py` averages over all processed radargrams by default. `--min-mean-iou X` keeps only radargrams with mean IoU ≥ X, and the script always reports how many files were kept.

---

## 5. Evaluation

The same scoring code is used for synthetic and real data, so the results can be compared directly.

- **Thick IoU:** both predicted and true layers are widened into equal ±2 px bands before IoU is computed. This keeps thin layers from being unfairly penalized by one-pixel offsets.
- **One-to-one matching:** predicted layers are matched to true layers with the Hungarian algorithm on the IoU matrix.
- **A match counts** when IoU ≥ θ (headline θ = 0.50).
- **Metrics:** precision, recall, F1, mean IoU and median IoU, plus merge counts and over- or under-segmentation counts.
- **Sensitivity:** every result is repeated for θ from 0.20 to 0.70.

---

## Repository layout

```
SyntheIce-Experiments/
├── xcheck_synthetic/          # Part 1: 4-tile XCheck on SyntheIce
│   ├── xcheck.py              #   run_xcheck, build_gt, fuzzy_metrics, stitch_chains
│   ├── run_batch.py           #   all wide radargrams -> per-radargram / per-category / global tables
│   ├── plot_results.py        #   bar, TP/FP/FN, scatter and box plots
│   └── stitch_one.py          #   chain layers across all 4 tiles of one radargram
├── tracelink/                 # Part 2: TraceLink algorithm
│   └── core.py                #   trace_layers, auto_params, merge pass, run_tracelink()
├── tracelink_exp_synthetic/   # Part 3: TraceLink on synthetic radargrams
│   ├── run_batch.py           #   all categories, IoU threshold sweep
│   └── run_single.py          #   one radargram + overlay plot
├── tracelink_exp_real/        # Part 4: selection, cleaning, gap injection, TraceLink on real data
│   ├── real_pipeline.py       #   cleaning, gap injection, thick-IoU and gap-bridging scores
│   ├── run_batch.py           #   all radargrams, IoU threshold sweep
│   ├── run_single.py          #   one radargram + overlay plot
│   └── summarize.py           #   summary table + bar charts
├── evaluation/                # Part 5: IoU, Hungarian matching, P/R/F1, threshold sweep
│   └── metrics.py
├── figures/                   # Scripts that regenerate the paper figures
│   ├── plot_synthetic_bars.py
│   └── plot_overlay.py
├── results/                   # Saved metric tables (created when you run the scripts)
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/iharp-institute/SyntheIce-Englacial-Layer-Reconstruction.git
cd SyntheIce-Englacial-Layer-Reconstruction
pip install -r requirements.txt
```

1. Download the dataset from [SyntheIce-Dataset](https://github.com/USER/SyntheIce-Dataset). The commands below assume it sits next to this repository.
2. Run the experiments in order:

```bash
# Part 1: XCheck across 4 tiles (tiles come from SyntheIce-Dataset/scripts/crop_wide_to_tiles.py)
python xcheck_synthetic/run_batch.py --tiles ../SyntheIce-Dataset/data/xcheck_set/tiles --out results/xcheck
python xcheck_synthetic/plot_results.py --results results/xcheck

# Part 3: TraceLink on SyntheIce, thresholds 0.20 / 0.30 / 0.50 / 0.70
python tracelink_exp_synthetic/run_batch.py --data ../SyntheIce-Dataset/data/full --out results/synthetic
python figures/plot_synthetic_bars.py --results results/synthetic --iou 0.50

# Part 4: TraceLink on real expert annotations (needs NDH_Tools and the .mat files)
python tracelink_exp_real/run_batch.py \
    --layer-dir ./Nick-layer-data-mat/Layer_ \
    --radar-dir ./Nick-raw-radargram-mat/ \
    --out results/real
python tracelink_exp_real/summarize.py --results results/real
```

One radargram at a time:

```bash
python xcheck_synthetic/stitch_one.py synthetic_radargram_output_close_To_real001_wide \
    --tiles ../SyntheIce-Dataset/data/xcheck_set/tiles --data ../SyntheIce-Dataset/data/full
python tracelink_exp_synthetic/run_single.py <path/to/radargram.h5> --iou 0.50
python tracelink_exp_real/run_single.py Data_20120516_01_084.mat \
    --layer-dir ./Nick-layer-data-mat/Layer_ --radar-dir ./Nick-raw-radargram-mat/
```

Every script accepts `--help`.

**Real data:** the raw radargrams come from the publicly available [CReSIS](https://data.cresis.ku.edu/) archive. Reading the `.mat` files needs `NDH_Tools`.

<!-- TODO: say where the expert layer files can be obtained, or that they are available on request -->

---

## Citation

M. B. Hassan, B. A. Tama, S. Purushotham, and V. P. Janeja, "SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction," in Proc. 2026 IEEE International Conference on Big Data (BigData), Phoenix, AZ, USA, Dec. 2026, to appear.

```bibtex
@inproceedings{hassan2026syntheice,
  title     = {{SyntheIce}: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction},
  author    = {Hassan, Muhammad Behroze and Tama, Bayu Adhi and Purushotham, Sanjay and Janeja, Vandana P.},
  booktitle = {2026 IEEE International Conference on Big Data (BigData)},
  address   = {Phoenix, AZ, USA},
  month     = dec,
  year      = {2026}
}

@inproceedings{hassan2025xcheck,
  title     = {{XCheck}: A Consistency-Based Validation Framework for Englacial Layers Annotations},
  author    = {Hassan, Muhammad Behroze and Tama, Bayu Adhi and Purushotham, Sanjay and Janeja, Vandana P.},
  booktitle = {IEEE International Conference on Data Mining Workshops (ICDMW)},
  year      = {2025},
  doi       = {10.1109/ICDMW69685.2025.00015}
}
```

## License

[MIT]

## Acknowledgments

This work has been funded by the NSF HDR Institute for Harnessing Data and Model Revolution in the Polar Regions (iHARP), Award #2118285.
We acknowledge Dr. Nicholas D. Holschuh for providing the expert-labeled ice layer dataset.
