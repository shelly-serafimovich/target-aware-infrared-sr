# METRICS_EDA

Exploratory analysis of candidate **target-preservation metrics** for guiding DifIISR / ResShift super-resolution on thermal drone images.
The question: can a cheap, differentiable score computed on a small patch tell "a drone is here" from "background only", on the low-resolution (LR) encoder input?

Everything is produced by one notebook, [EDA.ipynb](EDA.ipynb). All outputs are in [RESULTS/](RESULTS/).

## Data

| Item | Used for | In repo |
|---|---|---|
| `thermal-drone.v1i.yolov8.zip` (Roboflow HR images) | image sizes, examples | no (`*.zip` is git-ignored) |
| `new_dataset/exact_padded_float32-*.zip` | DifIISR encoder-input NPYs (float32, CxHxW, padded), **valid split only** | no |
| [new_dataset/manifest.csv](new_dataset/manifest.csv) | maps image stems to NPYs | yes |
| `../annotations/corrected_yolo/` | corrected YOLO labels (train / valid / test) | yes |

The zips are extracted on first run into `extracted/` (git-ignored cache).

### Patch groups

Each patch is 48x48 px: a 24x24 **core** surrounded by a 12 px **ring**. Drone patches are centred on drone boxes of at most 24x24 px.

| Group | Patches | Definition |
|---|---|---|
| `drone` | 922 | drone box centred in the core |
| `non_drone` | 1188 | 4 random crops from each of the 297 pure-background images |
| `new_non_drone` | 922 | background patch in the same image as a drone patch, at least 8 px away from any box |

## Metrics

| Metric | Short description |
|---|---|
| Pool SCR | signal-to-clutter ratio of the top 2.5% core pixels (k = 15) against ring statistics; operating threshold tau = 1.8 |
| `soft_count` | sigmoid-soft count of core pixels above k = 1.5 sigma |
| `neighbour_density` | density of those pixels' neighbours (k = 1.5) |
| `compactness` | compactness of the thresholded blob (k = 2) |
| Hoyer | Hoyer sparsity of the core minus the ring mean |

Seed 42.

## Results layout

```
RESULTS/
  cell*.png                    dataset and example figures from the first cells
  eda_exports/                 per-patch records, metric JSON/NPZ, SCR/Hoyer/blob comparison figures
  chosen_graphs_and_tables/    curated subset selected for the report
  EDA_executed.ipynb           executed copy of the notebook with outputs
```

## Reproducing

Tested with Python 3.9.5, numpy 2.0.2, scipy 1.13.1, opencv 5.0.0, plus matplotlib and jupyter.

1. Place `thermal-drone.v1i.yolov8.zip` and the `new_dataset/exact_padded_float32-*.zip` parts in this folder (the notebook extracts them on first run).
2. Run from this folder:

```powershell
py -3.9 -m jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA_executed --output-dir RESULTS --ExecutePreprocessor.timeout=-1
```

Notes:
- Paths are relative to this folder, so run from `METRICS_EDA/`.
- On Windows, if a write fails with `Errno 22` or `Permission denied`, delete the old `RESULTS/*.png` and re-run.
- The Hoyer, blob and SCR sections also write JSON/NPZ exports to `RESULTS/eda_exports/`.
