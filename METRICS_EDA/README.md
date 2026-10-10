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

The full funnel is in [table_data_funnel.csv](RESULTS/appendix/table_data_funnel.csv).

## Metrics

| Metric | Short description |
|---|---|
| Pool SCR | signal-to-clutter ratio of the top 2.5% core pixels (k = 15) against ring statistics; operating threshold tau = 1.8 |
| `mean_nb` | mean 8-neighbour adjacency among the top-K (K = 15) core pixels, a spatial-compactness score |
| `soft_count` | sigmoid-soft count of core pixels above k = 1.5 sigma |
| `neighbour_density` | density of those pixels' neighbours (k = 1.5) |
| `compactness` | compactness of the thresholded blob (k = 2) |
| Hoyer | Hoyer sparsity of the core minus the ring mean |

All parameters are listed in [table_parameters.csv](RESULTS/appendix/table_parameters.csv).

Statistics: Mann-Whitney U, Cliff's delta, AUROC with a 95% bootstrap CI resampled **by source image** (200 resamples), Spearman rho between metrics. Seed 42.

## Main findings

AUROC of drone vs negatives (full table: [table_metric_summary.csv](RESULTS/appendix/table_metric_summary.csv)):

| Metric | AUROC (all negatives) | 95% CI | TPR @ FPR 5% | rho with SCR |
|---|---|---|---|---|
| Pool SCR | 0.714 | [0.692, 0.734] | 0.321 | 1.00 |
| `mean_nb` | 0.653 | [0.633, 0.676] | 0.000 (ties) | 0.57 |
| `soft_count` | 0.714 | [0.693, 0.733] | 0.287 | 0.95 |
| `neighbour_density` | 0.717 | [0.698, 0.738] | 0.315 | 0.99 |
| `compactness` | 0.657 | [0.636, 0.676] | 0.000 | 0.80 |
| Hoyer | 0.435 | [0.416, 0.455] | 0.012 | -0.22 |

- All metrics except Hoyer rank drones above both negative groups (Mann-Whitney p < 1e-29).
- Blob-style metrics (`soft_count`, `neighbour_density`, `compactness`) are almost collinear with SCR and add nothing over it.
- Hoyer is **reversed** (AUROC < 0.5): background patches are sparser than drones. Negative result.
- Combining SCR with `mean_nb` gives no gain at matched FPR ([table_scr_plus_mean_nb_rules.csv](RESULTS/appendix/table_scr_plus_mean_nb_rules.csv)).
- `mean_nb` with K = 15 is a compromise between ties and AUROC ([table_mean_nb_variants.csv](RESULTS/appendix/table_mean_nb_variants.csv)); simpler adjacency variants are near chance.

### Causal tests

Erasing a drone (linear interpolation of the surrounding pixels, blended by alpha) and injecting a synthetic peak into background patches:

| Test | Result |
|---|---|
| Erasure, alpha 1.0 to 0.0 | median SCR 2.03 to 1.67, still 44% detected at SCR >= 1.8; `mean_nb` detection falls from 29% to 5% |
| Injection, peak 0 to 8 x ring std | SCR detection 27% to 99.9%; `mean_nb` detection 8% to 89% |

Interpretation: SCR responds to the injected peak but does not drop to the background level when the target is erased, while `mean_nb` tracks the erasure more closely.
See [table_target_erasure.csv](RESULTS/appendix/table_target_erasure.csv) and [table_target_injection.csv](RESULTS/appendix/table_target_injection.csv).

## Results layout

```
RESULTS/
  cell*.png                    dataset and example figures from the first cells
  eda_exports/                 per-patch records, metric JSON/NPZ, SCR/Hoyer/blob comparison figures
  appendix/                    tables (table_*.csv) and figures (fig_*.png) from the statistics cells
  chosen_graphs_and_tables/    curated subset selected for the report appendices
  EDA_executed.ipynb           executed copy of the notebook with outputs
```

Appendix files:

| Topic | Files in `RESULTS/appendix/` |
|---|---|
| Data and parameters | `table_data_funnel.csv`, `table_parameters.csv` |
| Separation statistics | `table_metric_summary.csv`, `table_vs_negatives_rank_tests.csv`, `table_operating_points.csv` |
| ROC / ECDF | `fig_roc_all_metrics_and_youden_j.png`, `fig_ecdf_and_operating_points.png` |
| SCR and `mean_nb` | `fig_scr_vs_mean_nb_scatter.png`, `table_scr_plus_mean_nb_rules.csv`, `table_mean_nb_auroc_by_scr_bin.csv` |
| `mean_nb` design | `fig_mean_nb_definition_examples.png`, `fig_mean_nb_distribution_roc_k.png`, `table_mean_nb_variants.csv` |
| Dependence on target size | `fig_tpr_by_drone_size.png`, `table_tpr_by_drone_size.csv` |
| Erasure / injection | `fig_target_erasure.png`, `fig_target_injection.png`, `table_target_erasure.csv`, `table_target_injection.csv` |
| Qualitative | `fig_drone_patches_by_scr_quantile.png` |

## Reproducing

Tested with Python 3.9.5, numpy 2.0.2, scipy 1.13.1, opencv 5.0.0, plus pandas, matplotlib and jupyter.

1. Place `thermal-drone.v1i.yolov8.zip` and the `new_dataset/exact_padded_float32-*.zip` parts in this folder (the notebook extracts them on first run).
2. Run from this folder:

```powershell
py -3.9 -m jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA_executed --output-dir RESULTS --ExecutePreprocessor.timeout=-1
```

Notes:
- Paths are relative to this folder, so run from `METRICS_EDA/`.
- On Windows, if a write fails with `Errno 22` or `Permission denied`, delete the old `RESULTS/*.png` and `RESULTS/appendix/*.csv` and re-run.
- The Hoyer, blob and SCR sections also write JSON/NPZ exports to `RESULTS/eda_exports/`.
