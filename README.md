# SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction [Paper]

## Abstract

Englacial layers in radio-echo-sounding radargrams (subsurface ice images) encode the past climate and flow history of ice sheets and glaciers, but extracting them at scale is bottlenecked by time-consuming manual annotation and by semi-automated detectors that return broken, disconnected fragments rather than continuous horizons. We address two coupled problems in layer annotation products: repairing partial
annotations by connecting their fragments into coherent full-layer reconstructions across an entire profile, and validating layer continuity across the boundaries of adjacent tiles. We introduce a slope-aware greedy fragment-linking algorithm that reconstructs full-width layers from partial detections, assigning each fragment to its parent layer through row-similarity gating and an exponentially weighted slope predictor (EWMA), and attaching a content-based hash ID to each reconstruction for stable cross-radargram identification. For development and rigorous quantitative evaluation, we generate a synthetic dataset of 600 radargrams spanning multiple complexity levels, including noise close to realistic conditions, 150 are wide radargrams cropped into four adjacent tiles. All come with exact ground-truth annotations, giving full control for validation. We test this dataset with XCheck, a consistency-based validation framework: whereas XCheck previously operated on only two consecutive images, we apply it across four consecutive tiles, enforcing continuity at every boundary, which also indicates that the
synthetic data is realistic enough to drive real-world algorithms. Our major finding generalizes beyond ice: when linear features in spatial data are broken into fragments but keep a consistent local geometry, linking them by predicted trajectory is more reliable than by appearance. Evaluated against expert-labeled real radargrams, the method achieves high precision, recall, F1, and mean IoU, establishing transfer of our findings to a real-world dataset.

## SyntheIce Experiments: XCheck, TraceLink, and Evaluation

Code for the experiments in the paper *SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction*

## Paper
The paper has been accepted in the proceedings of 2026 IEEE International Conference on Big Data (BigData), Phoenix, AZ, USA, Dec. 2026


> The synthetic dataset and its generator are in the companion repository: **[SyntheIce-Dataset](https://github.com/USER/SyntheIce-Dataset)**.

---

## What is in this repository

| Part | What it does |
|---|---|
| **1. XCheck on SyntheIce** | Runs the XCheck consistency check across 4 adjacent tiles cut from each wide synthetic radargram |
| **2. TraceLink** | Slope-aware algorithm that groups broken layer fragments back into full-width layers |
| **3. TraceLink on SyntheIce** | Evaluates TraceLink on 400 synthetic radargrams with exact ground truth |
| **4. TraceLink on real data** | Evaluates TraceLink on real expert-annotated Greenland radargrams using a controlled gap-injection protocol |
| **5. Evaluation** | Shared scoring code: IoU on thickened bands, Hungarian matching, precision, recall, F1, mean IoU, multiple IoU threshold settings|

The pipeline runs in this order:

```
SyntheIce data ──► Wide radargrams ──► cropped into adjacent 4 tiles and saved ──► XCheck (cross-tile consistency)
              └──► TraceLink ──► Evaluation (synthetic)
Real expert annotations ──► select + inject gaps ──► TraceLink ──► Evaluation (real)
```

---

## 1. XCheck on synthetic data

[XCheck](https://doi.org/10.1109/ICDMW69685.2025.00015) is our published method for checking whether layer annotations agree across neighbouring radargrams to check consistency of englacial layers. The original method compares **two** consecutive/non-consecutive images. Here we extend it to **four** consecutive tiles.

**Protocol**

1. Take 150 wide SyntheIce radargrams from all four categories.
2. Crop each into 4 adjacent tiles (600 tiles in total) and save each tile's mask and meta data.
3. Run XCheck at each shared tile boundary to check consistency of layers.
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
3. **Break:** inject random gaps with a fixed seed to imitate detector breakage.
4. **Link:** run TraceLink on the broken layers.
5. **Score:** compare the result against the original unbroken layer.

**Why this test is fair**

- **The answer is known,** because the gaps are cut from layers that were continuous to begin with.
- **It doesn't depend on a second noisy label,** because the reference is the original expert layer.
- **It is reproducible,** because gaps are seeded and TraceLink's parameters are derived from each radargram.
- **The method isn't penalized for gaps,** because injected gaps are excluded from the reference. The score measures fragment grouping, not gap filling.

**Data:** CReSIS 2012 Greenland P3 radargrams with expert annotations, [N] radargrams.
<!-- TODO: state the exact real-data sample count; reviewers flagged it as missing -->

**Results (θ = 0.50)**

| Precision | Recall | F1 | Mean IoU |
|---|---|---|---|
| 0.78 | 0.93 | 0.84 | 0.82 |

- The merge pass cuts predicted layers from 14024 to 13677 (347 merges, 2.47%).
- **Threshold sweep from 0.20 to 0.70:** F1 falls only from 0.900 to 0.795, while the mean IoU of matched layers *rises* from 0.808 to 0.838. The layers TraceLink keeps are high-overlap traces, not loose matches that only pass at a lenient threshold.

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
├── tracelink/                 # Part 2: TraceLink algorithm
├── tracelink_exp_synthetic/   # Part 3: TraceLink on 400 synthetic radargrams
├── tracelink_exp_real/        # Part 4: selection, cleaning, gap injection, TraceLink on real data
├── evaluation/                # Part 5: thick IoU, Hungarian matching, IoU threshold settings
├── figures/                   # Scripts that regenerate the paper figures
└── results/                   # Saved metric tables
```

## Quick start

```bash
git clone https://github.com/USER/SyntheIce-Experiments.git
cd SyntheIce-Experiments
pip install -r requirements.txt
```

1. Download the dataset from [SyntheIce-Dataset](https://github.com/USER/SyntheIce-Dataset).
2. Run the experiments in order: `xcheck_synthetic/`, `exp_synthetic/`, `exp_real/`.

Real Data: The Raw radargrams come ferom the publicly available archive: [CReSIS](https://data.cresis.ku.edu/).

---

## Citation 

M. B. Hassan, B. A. Tama, S. Purushotham, and V. P. Janeja, "SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction," in Proc. 2026 IEEE International Conference on Big Data (BigData), Phoenix, AZ, USA, Dec. 2026, to appear

```bibtex
@inproceedings{hassan2026syntheice,
  title     = {SyntheIce: A Synthetic Radargram Annotation Benchmark for Englacial Layer Reconstruction},
  author    = {Hassan, Muhammad Behroze and Tama, Bayu Adhi and Purushotham, Sanjay and Janeja, Vandana P.},
  booktitle = {2026 IEEE International Conference on Big Data (BigData)},
  address   = {Phoenix, AZ, USA},
  month     = dec,
  year      = {2026}
}

@inproceedings{hassan2025xcheck,
  title     = {XCheck: A Consistency-Based Validation Framework for Englacial Layers Annotations},
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
We acknowledge Dr. Nicholas D. Holschuh for providing ice layer expert-labeled dataset.
