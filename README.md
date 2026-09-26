# Target-Aware Infrared Super-Resolution

Target-aware infrared image super-resolution for preserving small thermal drone targets.

## Project Goal

This project investigates super-resolution for thermal imagery containing very small drone targets.

The main motivation is that standard super-resolution methods may improve overall image quality while degrading or suppressing small targets that are important for downstream detection.

The project uses DifIISR as the baseline diffusion-based infrared super-resolution method and explores target-aware guidance for better preservation of small thermal targets.

## Current Stage

The project is currently focused on:

1. Dataset exploration and validation
2. Bounding-box annotation refinement
3. Background characterization
4. Preparation of the corrected dataset for DifIISR baseline experiments

The target-aware diffusion guidance method will be developed and evaluated in later stages.

## Dataset

The dataset contains **7,032 thermal images** divided into the original train, validation, and test splits:

| Split | Images |
|---|---:|
| Train | 2,809 |
| Validation | 2,115 |
| Test | 2,108 |
| **Total** | **7,032** |

The dataset contains thermal drone targets from several annotation classes. For the main super-resolution task, all drone classes are treated as thermal targets while the original class information is preserved.

## Annotation Refinement

Initial EDA revealed that some annotations, particularly FPV-drone annotations, contained bounding boxes that were substantially larger than the actual thermal target.

A refinement procedure was therefore applied to suspicious annotations while preserving the original dataset split and class information.

The corrected YOLO annotations are stored in:

```text
annotations/
└── corrected_yolo/
    ├── train/
    │   └── labels/
    ├── valid/
    │   └── labels/
    └── test/
        └── labels/
```

The annotation files follow standard YOLO format:

```text
class_id x_center y_center width height
```

Images without targets are represented by empty label files.

## Background Analysis

The dataset contains **927 background-only images**.

To better characterize the different thermal backgrounds, background-only images were embedded using **OpenCLIP ViT-B/32** and explored using **K-Means clustering**.

Ten clusters were retained for exploratory background characterization. The clusters should not be interpreted as ground-truth semantic classes; they provide a practical grouping of visually similar background conditions.

Following manual inspection, the clusters were characterized as:

| Cluster | Manual interpretation |
|---:|---|
| 0 | Half sky / half ground |
| 1 | Pole, no sky |
| 2 | Sky with moon |
| 3 | Half sky / half ground |
| 4 | Sky with light pole |
| 5 | Cloudy sky with moon |
| 6 | Car / ground background |
| 7 | Half sky / half ground |
| 8 | Mostly sky with some ground |
| 9 | Wall with object |

The complete mapping between background images, dataset splits, cluster IDs, and manual background descriptions is stored in:

```text
metadata/background_metadata.csv
```

## Representative Background Examples

Five representative images from each background cluster are included for visual interpretation:

```text
examples/
└── background_clusters/
    ├── cluster_00_half_sky_half_ground/
    ├── cluster_01_pole_no_sky/
    ├── cluster_02_sky_with_moon/
    ├── cluster_03_half_sky_half_ground/
    ├── cluster_04_sky_with_light_pole/
    ├── cluster_05_cloudy_sky_with_moon/
    ├── cluster_06_car_ground_background/
    ├── cluster_07_half_sky_half_ground/
    ├── cluster_08_mostly_sky_with_some_ground/
    └── cluster_09_wall_with_object/
```

These examples are intended to document the visual interpretation of the clusters rather than serve as additional training data.

## Repository Structure

```text
target-aware-infrared-sr/
├── annotations/
│   └── corrected_yolo/
├── examples/
│   └── background_clusters/
├── metadata/
│   └── background_metadata.csv
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_difiisr_setup.ipynb
│   └── 01_thermal_drone_dataset_eda.ipynb
├── .gitignore
└── README.md
```

## Research Direction

The next stage of the project is to establish the original DifIISR model as a baseline on the corrected dataset.

The baseline will then be compared with a target-aware approach designed to preserve small thermal targets during the diffusion-based super-resolution process.

Evaluation will focus on target preservation and downstream detection behavior, with conventional image-quality metrics used as complementary measures.

## Status

Work in progress.
