# CWRU Threshold Calibration Report

## Protocol
- Same leave-one-load-out folds as the load-shift study.
- Same healthy-train-only model fitting.
- Same healthy-val-only threshold fitting.
- Same held-out healthy plus held-out fault testing.
- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.
- MAD uses the raw median absolute deviation over healthy validation scores.
- Load-shift base: artifacts/metrics/cwru_load_shift_metrics.json

## Dataset Defaults Reused
- Window size: `2048`
- Window stride: `1024`
- Original preprocessing normalization: `zscore_global` fit on `healthy_train_only`

## Held-Out Load 0
- Train healthy windows: `981`
- Val healthy windows: `207`
- Test healthy windows: `34`
- Test fault windows: `353`
| Model | Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AE | mean_plus_3std | 0.174591 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| AE | percentile_99 | 0.155439 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| AE | percentile_99_5 | 0.161583 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| AE | median_plus_3mad | 0.153417 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| AE | median_plus_4mad | 0.168552 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | mean_plus_3std | 1.059918 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | percentile_99 | 1.018246 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | percentile_99_5 | 1.159359 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | median_plus_3mad | 0.026439 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| OC-SVM | median_plus_4mad | 0.387596 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| Isolation Forest | mean_plus_3std | 0.066208 | 1.000000 | 1.000000 | 0.956640 | 0.916883 | 1.000000 | 0.941176 |
| Isolation Forest | percentile_99 | 0.073747 | 1.000000 | 1.000000 | 0.956640 | 0.916883 | 1.000000 | 0.941176 |
| Isolation Forest | percentile_99_5 | 0.079568 | 1.000000 | 1.000000 | 0.959239 | 0.921671 | 1.000000 | 0.882353 |
| Isolation Forest | median_plus_3mad | 0.005567 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| Isolation Forest | median_plus_4mad | 0.029110 | 1.000000 | 1.000000 | 0.955345 | 0.914508 | 1.000000 | 0.970588 |

## Held-Out Load 1
- Train healthy windows: `817`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `353`
| Model | Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AE | mean_plus_3std | 0.213272 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | percentile_99 | 0.190833 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | percentile_99_5 | 0.193018 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | median_plus_3mad | 0.205404 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | median_plus_4mad | 0.221785 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | mean_plus_3std | 0.961994 | 1.000000 | 1.000000 | 0.979196 | 0.959239 | 1.000000 | 0.217391 |
| OC-SVM | percentile_99 | 0.785606 | 1.000000 | 1.000000 | 0.967123 | 0.936340 | 1.000000 | 0.347826 |
| OC-SVM | percentile_99_5 | 0.869058 | 1.000000 | 1.000000 | 0.973793 | 0.948925 | 1.000000 | 0.275362 |
| OC-SVM | median_plus_3mad | 0.340872 | 1.000000 | 1.000000 | 0.943850 | 0.893671 | 1.000000 | 0.608696 |
| OC-SVM | median_plus_4mad | 0.770061 | 1.000000 | 1.000000 | 0.964481 | 0.931398 | 1.000000 | 0.376812 |
| Isolation Forest | mean_plus_3std | 0.108204 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| Isolation Forest | percentile_99 | 0.083467 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| Isolation Forest | percentile_99_5 | 0.088904 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| Isolation Forest | median_plus_3mad | 0.027290 | 1.000000 | 1.000000 | 0.983287 | 0.967123 | 1.000000 | 0.173913 |
| Isolation Forest | median_plus_4mad | 0.060445 | 1.000000 | 1.000000 | 0.994366 | 0.988796 | 1.000000 | 0.057971 |

## Held-Out Load 2
- Train healthy windows: `816`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `352`
| Model | Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AE | mean_plus_3std | 0.255343 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | percentile_99 | 0.205230 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | percentile_99_5 | 0.206230 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | median_plus_3mad | 0.221638 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| AE | median_plus_4mad | 0.253419 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | mean_plus_3std | 1.050482 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | percentile_99 | 0.916563 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | percentile_99_5 | 1.082605 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | median_plus_3mad | 0.321871 | 1.000000 | 1.000000 | 0.998582 | 0.997167 | 1.000000 | 0.014493 |
| OC-SVM | median_plus_4mad | 0.747108 | 1.000000 | 1.000000 | 0.998582 | 0.997167 | 1.000000 | 0.014493 |
| Isolation Forest | mean_plus_3std | 0.089633 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | percentile_99 | 0.062482 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | percentile_99_5 | 0.073794 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | median_plus_3mad | 0.044736 | 1.000000 | 1.000000 | 0.998582 | 0.997167 | 1.000000 | 0.014493 |
| Isolation Forest | median_plus_4mad | 0.075998 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## Held-Out Load 3
- Train healthy windows: `815`
- Val healthy windows: `172`
- Test healthy windows: `69`
- Test fault windows: `354`
| Model | Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AE | mean_plus_3std | 0.204598 | 1.000000 | 1.000000 | 0.980609 | 0.961957 | 1.000000 | 0.202899 |
| AE | percentile_99 | 0.176050 | 1.000000 | 1.000000 | 0.913548 | 0.840855 | 1.000000 | 0.971014 |
| AE | percentile_99_5 | 0.179941 | 1.000000 | 1.000000 | 0.917098 | 0.846890 | 1.000000 | 0.927536 |
| AE | median_plus_3mad | 0.198852 | 1.000000 | 1.000000 | 0.964578 | 0.931579 | 1.000000 | 0.376812 |
| AE | median_plus_4mad | 0.222281 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | mean_plus_3std | 0.895319 | 1.000000 | 1.000000 | 0.987448 | 0.975207 | 1.000000 | 0.130435 |
| OC-SVM | percentile_99 | 0.700214 | 1.000000 | 1.000000 | 0.981969 | 0.964578 | 1.000000 | 0.188406 |
| OC-SVM | percentile_99_5 | 0.869936 | 1.000000 | 1.000000 | 0.986072 | 0.972527 | 1.000000 | 0.144928 |
| OC-SVM | median_plus_3mad | 0.195631 | 1.000000 | 1.000000 | 0.954178 | 0.912371 | 1.000000 | 0.492754 |
| OC-SVM | median_plus_4mad | 0.552741 | 1.000000 | 1.000000 | 0.976552 | 0.954178 | 1.000000 | 0.246377 |
| Isolation Forest | mean_plus_3std | 0.085488 | 1.000000 | 1.000000 | 0.990210 | 0.980609 | 1.000000 | 0.101449 |
| Isolation Forest | percentile_99 | 0.079992 | 1.000000 | 1.000000 | 0.988827 | 0.977901 | 1.000000 | 0.115942 |
| Isolation Forest | percentile_99_5 | 0.081566 | 1.000000 | 1.000000 | 0.990210 | 0.980609 | 1.000000 | 0.101449 |
| Isolation Forest | median_plus_3mad | 0.017606 | 1.000000 | 1.000000 | 0.981969 | 0.964578 | 1.000000 | 0.188406 |
| Isolation Forest | median_plus_4mad | 0.044873 | 1.000000 | 1.000000 | 0.987448 | 0.975207 | 1.000000 | 0.130435 |

## Mean/Std Across Loads
### AE
| Threshold Rule | Threshold mean+/-std | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.211951 +/- 0.028872 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.983666 +/- 0.018840 | 0.968525 +/- 0.036067 | 1.000000 +/- 0.000000 | 0.300725 +/- 0.412137 |
| percentile_99 | 0.181888 +/- 0.018429 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.966901 +/- 0.036065 | 0.938250 +/- 0.066696 | 1.000000 +/- 0.000000 | 0.492754 +/- 0.492860 |
| percentile_99_5 | 0.185193 +/- 0.016499 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.967788 +/- 0.034761 | 0.939759 +/- 0.064508 | 1.000000 +/- 0.000000 | 0.481884 +/- 0.482565 |
| median_plus_3mad | 0.194828 +/- 0.025307 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.979658 +/- 0.020680 | 0.960931 +/- 0.039669 | 1.000000 +/- 0.000000 | 0.344203 +/- 0.408682 |
| median_plus_4mad | 0.216509 +/- 0.030510 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.988514 +/- 0.019895 | 0.978036 +/- 0.038042 | 1.000000 +/- 0.000000 | 0.250000 +/- 0.433013 |
- Best false-alarm rule: `median_plus_4mad`
- Mean false alarm rate improves from `0.300725` to `0.250000` with mean F1 changing from `0.983666` to `0.988514`.

### OC-SVM
| Threshold Rule | Threshold mean+/-std | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.991928 +/- 0.067603 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.980174 +/- 0.016802 | 0.961648 +/- 0.032059 | 1.000000 +/- 0.000000 | 0.336957 +/- 0.390549 |
| percentile_99 | 0.855157 +/- 0.121669 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.975787 +/- 0.017116 | 0.953265 +/- 0.032747 | 1.000000 +/- 0.000000 | 0.384058 +/- 0.376324 |
| percentile_99_5 | 0.995240 +/- 0.128638 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.978480 +/- 0.016877 | 0.958399 +/- 0.032247 | 1.000000 +/- 0.000000 | 0.355072 +/- 0.384877 |
| median_plus_3mad | 0.221203 +/- 0.125540 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.962666 +/- 0.021155 | 0.928838 +/- 0.040173 | 1.000000 +/- 0.000000 | 0.528986 +/- 0.351505 |
| median_plus_4mad | 0.614377 +/- 0.155791 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.973417 +/- 0.016567 | 0.948722 +/- 0.031681 | 1.000000 +/- 0.000000 | 0.409420 +/- 0.364828 |
- Best false-alarm rule: `mean_plus_3std`
- Mean false alarm rate improves from `0.336957` to `0.336957` with mean F1 changing from `0.980174` to `0.980174`.

### Isolation Forest
| Threshold Rule | Threshold mean+/-std | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.087383 +/- 0.014921 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.986006 +/- 0.017325 | 0.972965 +/- 0.033138 | 1.000000 +/- 0.000000 | 0.267903 +/- 0.390467 |
| percentile_99 | 0.074922 +/- 0.007982 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.985660 +/- 0.017251 | 0.972287 +/- 0.033002 | 1.000000 +/- 0.000000 | 0.271526 +/- 0.388970 |
| percentile_99_5 | 0.080958 +/- 0.005403 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.986656 +/- 0.016225 | 0.974162 +/- 0.031115 | 1.000000 +/- 0.000000 | 0.253197 +/- 0.365118 |
| median_plus_3mad | 0.023800 +/- 0.014329 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.979473 +/- 0.016063 | 0.960253 +/- 0.030590 | 1.000000 +/- 0.000000 | 0.344203 +/- 0.384724 |
| median_plus_4mad | 0.052607 +/- 0.017468 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.984290 +/- 0.017292 | 0.969627 +/- 0.033012 | 1.000000 +/- 0.000000 | 0.289749 +/- 0.395790 |
- Best false-alarm rule: `percentile_99_5`
- Mean false alarm rate improves from `0.267903` to `0.253197` with mean F1 changing from `0.986006` to `0.986656`.

## Calibration Interpretation
- AE benefits the most from improved calibration in terms of mean false-alarm reduction (0.050725), with mean F1 change +0.004848.
- The score ranking remains robust under load shift, but absolute score calibration drifts enough that a single validation-derived threshold can over-trigger on some unseen loads. For VibeTwin, this points to a calibration problem rather than a representation problem: the models can separate healthy and faulty windows, yet deployment needs thresholding that is robust to operating-condition shift. Under this study, AE's best calibration rule is `median_plus_4mad`, yielding mean false alarm rate `0.250000` and mean F1 `0.988514` across loads.

## Saved Artifacts
- Metrics JSON: `artifacts/metrics/cwru_threshold_calibration_metrics.json`
- Report: `artifacts/metrics/cwru_threshold_calibration_report.md`
- Summary plot: `artifacts/plots/cwru_threshold_calibration_summary.png`
