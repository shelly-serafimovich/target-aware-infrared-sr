# Target-Aware Infrared Super-Resolution for Small Drone Preservation

This repository contains a research project on **infrared image super-resolution for scenes containing very small thermal drone targets**.

The central question is not only whether a super-resolution method produces a visually better image, but whether it **preserves weak targets that matter for downstream detection**.

We use [DifIISR](https://github.com/zirui0625/DifIISR) as the diffusion-based infrared super-resolution backbone and develop a **zero-shot target-aware guidance mechanism** designed to increase the local prominence of small thermal targets while limiting unnecessary changes to the background.

---

## Motivation

Conventional super-resolution metrics such as PSNR or SSIM measure global reconstruction quality. In thermal drone imagery, however, the operationally important information may occupy only a few pixels.

A super-resolution result can therefore look globally plausible while suppressing, blurring, or altering a tiny target enough to hurt detection.

This project evaluates super-resolution from two complementary perspectives:

1. **Image quality** — PSNR, SSIM, LPIPS, CLIPIQA, MUSIQ, and NIQE.
2. **Target preservation and detection** — GT-matched target recall, small-target recall, detector precision/recall, AP, and background false alarms using a frozen YOLO detector.

---

## Method Overview

Our final method, **V3-B**, adds target-aware latent guidance on top of a frozen DifIISR inference pipeline. No DifIISR weights are fine-tuned.

The guidance pipeline is:

1. Bicubic-upsample the low-resolution thermal input.
2. Propose candidate local maxima using local contrast.
3. Measure each candidate with a pooled **signal-to-clutter ratio (SCR)** computed from a target core and surrounding ring.
4. Estimate **neighbour density (ND)** to prefer spatially supported target-like responses over isolated noise peaks.
5. Convert these measurements into frozen candidate weights.
6. Optimize the predicted clean latent for a small number of steps using a target-prominence objective, while regularizing global fidelity and background changes.
7. Decode the guided latent using the frozen DifIISR decoder.

The selected V3-B configuration uses:

| Parameter | Value |
|---|---:|
| Core size | 24 × 24 |
| Outer window | 48 × 48 |
| Top-k pixels for pooled SCR | 15 |
| SCR gate | 1.8 |
| Neighbour-density gate | 0.40 |
| Guidance steps | 3 |
| Guidance learning rate | 0.02 |
| Fidelity weight | 10.0 |
| Background-fidelity weight | 20.0 |

The final configuration was selected in a pilot ablation and then frozen before full validation.

---

## Dataset and Annotation Refinement

The project dataset contains **7,032 thermal images** using the original train/validation/test split:

| Split | Images |
|---|---:|
| Train | 2,809 |
| Validation | 2,115 |
| Test | 2,108 |
| **Total** | **7,032** |

Early EDA revealed problematic bounding boxes in part of the dataset, particularly annotations that were much larger than the actual thermal target. A corrected YOLO annotation set was therefore created while preserving the original image split and class information.

Corrected labels are stored under:

```text
annotations/corrected_yolo/
├── train/labels/
├── valid/labels/
└── test/labels/
```

All drone classes are treated as the same downstream detection target during the frozen-detector evaluation.

The project also contains exploratory background analysis based on OpenCLIP embeddings and K-Means clustering. Background metadata is stored in:

```text
metadata/background_metadata.csv
```

Representative examples are available under:

```text
examples/background_clusters/
```

---

## Experimental Pipeline

The main experimental stages are implemented as reproducible notebooks:

| Notebook | Purpose |
|---|---|
| `00_environment_check.ipynb` | Environment and dependency checks |
| `01_difiisr_setup.ipynb` | DifIISR setup |
| `01_thermal_drone_dataset_eda.ipynb` | Dataset EDA and annotation analysis |
| `02_difiisr_baseline_evaluation.ipynb` | Original DifIISR baseline evaluation |
| `03_thermal_drone_detector.ipynb` | Frozen thermal-drone YOLO detector |
| `04_target_size_analysis.ipynb` | Target-size analysis |
| `05_difiisr_target_dynamics.ipynb` | Study of small-target behaviour through DifIISR |
| `06_target_aware_difiisr.ipynb` | Initial target-aware guidance experiments |
| `06_target_aware_difiisr_v2.ipynb` | Refined target-aware formulation |
| `07_difiisr_bicubic_input_dataset.ipynb` | Controlled bicubic-input dataset preparation |
| `08_recall_guidance_ablation_v3.ipynb` | V3 guidance ablation and final configuration selection |
| `09_full_validation_standalone.ipynb` | Frozen full-validation experiment and final comparison |

The final notebook is intentionally standalone and reproduces the frozen evaluation once the dataset, DifIISR weights, and detector weights are available.

---

## Full Validation

The frozen final experiment uses a common validation set of **2,114 images**, containing:

- **1,817 annotated targets**
- **928 small targets**
- **297 background-only images**

A small target is defined as a target whose **upsampled pre-encoder bounding-box width and height are both ≤ 24 pixels**.

The primary selection/evaluation criterion for the target-aware method is **GT-matched recall at IoU ≥ 0.25**, with special attention to the small-target subset.

Three outputs are compared:

- **Original DifIISR** — previously generated frozen DifIISR validation output.
- **BASE** — unguided DifIISR output generated inside the final controlled runner.
- **V3-B selected** — the same controlled pipeline with pooled-SCR + neighbour-density target-aware latent guidance.

---

## Final Results

### Detection-oriented results

| Method | Target Recall @ IoU≥0.25 | Small-Target Recall | Precision | Detector Recall | AP50 | mAP50-95 | Background FAs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original DifIISR | 50.14% | 17.89% | 68.62% | 35.44% | 36.83% | 18.70% | 52 |
| BASE | 51.07% | 18.43% | **77.82%** | 35.50% | **38.94%** | **19.19%** | 59 |
| **V3-B selected** | **51.35%** | **18.86%** | 74.04% | **36.11%** | 38.63% | 18.60% | 64 |

Compared with BASE, V3-B improves:

- overall GT-matched target recall by **+0.28 percentage points**;
- small-target recall by **+0.43 percentage points**.

The paired target analysis is more informative than the aggregate change alone:

- **10 targets were saved and 5 were lost** versus BASE;
- among small targets, **5 were saved and only 1 was lost**.

The gain is accompanied by a trade-off: V3-B produces more background false alarms and slightly lower precision and mAP than BASE.

### Image-quality results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIPIQA ↑ | MUSIQ ↑ | NIQE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original DifIISR | **28.18** | **0.7283** | 0.3729 | **0.5771** | **44.43** | 9.0833 |
| BASE | 28.03 | 0.7111 | **0.3503** | 0.5448 | 41.69 | **9.0657** |
| V3-B selected | 27.55 | 0.7030 | 0.3696 | 0.5481 | 41.02 | 9.0833 |

V3-B is therefore **not a universal image-quality improvement**. Its value is task-oriented: it shifts the SR output toward preserving a subset of weak and small targets that the unguided baseline misses, at the cost of additional false alarms and some degradation in conventional quality metrics.

This trade-off is the main empirical result of the project.

Full compact results are stored in:

```text
results/full_validation_v3b/
├── FINAL_COMPARISON.csv
├── quality_summary.csv
├── detection_summary.csv
├── yolo_dataset_summary.csv
├── paired_saved_lost.csv
├── config.json
├── excluded_images.csv
└── README.md
```

Large generated SR images, per-image metric tables, detector caches, and JSONL detection outputs are intentionally kept outside the repository because they are reproducible and substantially larger.

---

## Core Scripts

The reusable final-stage code is under `scripts/`:

```text
scripts/
├── v3_recall_guidance_runner.py   # BASE and target-aware SR generation
├── v3_recall_yolo_eval.py         # recall-oriented pilot evaluation
├── v3_full_quality_eval.py        # full image-quality evaluation
└── v3_full_yolo_eval.py           # frozen-YOLO full evaluation + saved/lost analysis
```

The final guidance runner freezes DifIISR and performs optimization only on the predicted latent representation.

---

## Reproducing the Final Experiment

The recommended entry point is:

```text
notebooks/09_full_validation_standalone.ipynb
```

The notebook:

1. mounts Google Drive;
2. clones this repository;
3. clones DifIISR and checks out the frozen commit;
4. creates the required Python environment;
5. builds the common validation manifest;
6. runs BASE and V3-B inference;
7. computes full image-quality metrics;
8. evaluates all methods with the same frozen YOLO detector;
9. exports the final comparison tables and saved/lost analysis.

The DifIISR version used in the frozen experiment is pinned to:

```text
09ca97ea48d481656dd8090e84099c963059ac41
```

Dataset paths and frozen YOLO weights are expected to be available in Google Drive as configured near the top of Notebook 09.

---

## Repository Structure

```text
target-aware-infrared-sr/
├── annotations/                 # corrected YOLO annotations
├── examples/                    # representative background examples
├── figures/                     # report / experiment figures
├── metadata/                    # dataset and background metadata
├── notebooks/                   # complete experimental workflow
├── results/
│   ├── detector/                # frozen-detector training/validation summaries
│   └── full_validation_v3b/     # final compact experiment results
├── scripts/                     # reusable V3 guidance and evaluation code
├── .gitignore
└── README.md
```

---

## Takeaway

For tiny thermal targets, **better super-resolution should not be defined only by global perceptual or reconstruction quality**.

Our experiments show that targeted zero-shot guidance can recover additional detections—especially among very small targets—but that this benefit must be evaluated together with its cost in false alarms and image-quality degradation.

The repository is structured to make that trade-off explicit and reproducible rather than reporting only a single aggregate SR metric.
