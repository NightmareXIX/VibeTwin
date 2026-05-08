# CWRU Load-Shift Report

## Protocol
- Leave-one-load-out evaluation across the four motor loads: `0`, `1`, `2`, and `3`.
- Healthy train windows come only from non-held-out loads using the existing `train` split.
- Healthy validation windows come only from non-held-out loads using the existing `val` split.
- Test healthy windows come from the held-out load using the existing `test` split.
- Test fault windows come from the held-out load using the existing fault-test windows.
- Fold-specific z-score normalization is refit on healthy train windows only after reconstructing pre-z-score values from the saved preprocessing stats.
- Existing `window_manifest.csv` was sufficient for load indexing; no additional preprocessing metadata file was needed.

## Dataset Defaults Reused
- Window size: `2048`
- Window stride: `1024`
- Original preprocessing normalization: `zscore_global` fit on `healthy_train_only`

## Held-Out Load 0
- Train healthy windows: `981`
- Val healthy windows: `207`
- Test healthy windows: `34`
- Test fault windows: `353`
| Model | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 0.174591 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | 1.059918 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| Isolation Forest | 0.066208 | 1.000000 | 1.000000 | 0.956640 | 0.916883 | 1.000000 | 0.941176 |

## Held-Out Load 1
- Train healthy windows: `817`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `353`
| Model | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 0.213272 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | 0.961994 | 1.000000 | 1.000000 | 0.979196 | 0.959239 | 1.000000 | 0.217391 |
| Isolation Forest | 0.108204 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |

## Held-Out Load 2
- Train healthy windows: `816`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `352`
| Model | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 0.255343 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | 1.050482 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | 0.089633 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## Held-Out Load 3
- Train healthy windows: `815`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `354`
| Model | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 0.204598 | 1.000000 | 1.000000 | 0.980609 | 0.961957 | 1.000000 | 0.202899 |
| OC-SVM | 0.895319 | 1.000000 | 1.000000 | 0.987448 | 0.975207 | 1.000000 | 0.130435 |
| Isolation Forest | 0.085488 | 1.000000 | 1.000000 | 0.990210 | 0.980609 | 1.000000 | 0.101449 |

## Mean/Std Across Folds
| Model | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- |
| AE | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.983666 +/- 0.018840 | 0.968525 +/- 0.036067 | 1.000000 +/- 0.000000 | 0.300725 +/- 0.412137 |
| OC-SVM | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.980174 +/- 0.016802 | 0.961648 +/- 0.032059 | 1.000000 +/- 0.000000 | 0.336957 +/- 0.390549 |
| Isolation Forest | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.986006 +/- 0.017325 | 0.972965 +/- 0.033138 | 1.000000 +/- 0.000000 | 0.267903 +/- 0.390467 |

## Model Comparison
- AE does not hold a clean overall advantage under load shift; the shallow baselines are at least competitive on both mean F1 and false alarm rate.

## Saved Artifacts
- Metrics JSON: `artifacts/metrics/cwru_load_shift_metrics.json`
- Report: `artifacts/metrics/cwru_load_shift_report.md`
- Summary plot: `artifacts/plots/cwru_load_shift_summary.png`
