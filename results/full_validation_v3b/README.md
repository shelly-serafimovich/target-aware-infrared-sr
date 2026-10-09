# Notebook 09 — Full validation results

This directory contains the compact, reproducible outputs from `notebooks/09_full_validation_standalone.ipynb` for the frozen final configuration **V3-B, 3 guidance steps, LR=0.02**.

## Included

- `FINAL_COMPARISON.csv` — combined detection + image-quality comparison across Original DifIISR, BASE, and V3-B selected.
- `quality_summary.csv` — aggregate PSNR, SSIM, LPIPS, CLIPIQA, MUSIQ, and NIQE statistics.
- `detection_summary.csv` — target-level recall, small-target recall, background false alarms, precision, recall, AP50, and mAP50-95.
- `yolo_dataset_summary.csv` — compact frozen-YOLO summary.
- `paired_saved_lost.csv` — paired target analysis versus BASE / Original DifIISR.
- `config.json` — frozen final guidance configuration and DifIISR commit.
- `excluded_images.csv` — images excluded from the common validation set due to unsupported resolution.

## Main observation

V3-B improves GT-matched target recall over BASE from **51.07% to 51.35%** and small-target recall from **18.43% to 18.86%**. In the paired analysis versus BASE, V3-B saves **10** targets and loses **5** overall; for small targets it saves **5** and loses **1**. The gain comes with a trade-off: precision decreases from **77.82% to 74.04%**, background false alarms increase, and reconstruction / perceptual quality metrics do not uniformly improve.

## Intentionally not tracked here

Large per-image outputs, generated SR images, YOLO caches, JSONL detections, and full per-image quality tables remain external artifacts because they are substantially larger and are reproducible from Notebook 09. Qualitative saved/lost examples can be added separately under `figures/` for the paper/report.
