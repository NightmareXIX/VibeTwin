# CWRU ResDilatedAE Threshold Calibration Report

## Protocol
- Inference free from saved score arrays only.
- No retraining, no new model inference, no preprocessing changes, and no raw-data edits.
- Seed evaluated: `42`.
- Held-out loads evaluated: `0, 1, 2, 3`.
- Score splits reused per fold: `val_healthy`, `test_healthy`, `test_fault`.
- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.
- MAD uses the raw median absolute deviation over healthy validation scores.
- Saved run root: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42`

## Held-Out Load 0
- Fold directory: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_0`
- Saved scores: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_0/val_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_0/test_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_0/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.014289 | 1.000000 | 1.000000 | 0.955345 | 0.914508 | 1.000000 | 0.970588 |
| percentile_99 | 0.014813 | 1.000000 | 1.000000 | 0.955345 | 0.914508 | 1.000000 | 0.970588 |
| percentile_99_5 | 0.015481 | 1.000000 | 1.000000 | 0.955345 | 0.914508 | 1.000000 | 0.970588 |
| median_plus_3mad | 0.011354 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |
| median_plus_4mad | 0.012780 | 1.000000 | 1.000000 | 0.954054 | 0.912145 | 1.000000 | 1.000000 |

## Held-Out Load 1
- Fold directory: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_1`
- Saved scores: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_1/val_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_1/test_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_1/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.018456 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| percentile_99 | 0.022255 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| percentile_99_5 | 0.022683 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| median_plus_3mad | 0.011770 | 1.000000 | 1.000000 | 0.979196 | 0.959239 | 1.000000 | 0.217391 |
| median_plus_4mad | 0.013421 | 1.000000 | 1.000000 | 0.988796 | 0.977839 | 1.000000 | 0.115942 |

## Held-Out Load 2
- Fold directory: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_2`
- Saved scores: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_2/val_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_2/test_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_2/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.017998 | 1.000000 | 1.000000 | 0.997167 | 0.994350 | 1.000000 | 0.028986 |
| percentile_99 | 0.020758 | 1.000000 | 1.000000 | 0.998582 | 0.997167 | 1.000000 | 0.014493 |
| percentile_99_5 | 0.021781 | 1.000000 | 1.000000 | 0.998582 | 0.997167 | 1.000000 | 0.014493 |
| median_plus_3mad | 0.012321 | 1.000000 | 1.000000 | 0.985994 | 0.972376 | 1.000000 | 0.144928 |
| median_plus_4mad | 0.013904 | 1.000000 | 1.000000 | 0.992948 | 0.985994 | 1.000000 | 0.072464 |

## Held-Out Load 3
- Fold directory: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_3`
- Saved scores: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_3/val_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_3/test_healthy_scores.npy`, `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/load_3/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.020324 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| percentile_99 | 0.022849 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| percentile_99_5 | 0.023972 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| median_plus_3mad | 0.015239 | 1.000000 | 1.000000 | 0.990210 | 0.980609 | 1.000000 | 0.101449 |
| median_plus_4mad | 0.017372 | 1.000000 | 1.000000 | 0.998590 | 0.997183 | 1.000000 | 0.014493 |

## Mean/Std Across Held-Out Loads
| Threshold Rule | Threshold mean+/-std | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.017767 +/- 0.002189 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.987422 +/- 0.018556 | 0.975806 +/- 0.035465 | 1.000000 +/- 0.000000 | 0.257140 +/- 0.412080 |
| percentile_99 | 0.020169 +/- 0.003184 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.987775 +/- 0.018750 | 0.976510 +/- 0.035853 | 1.000000 +/- 0.000000 | 0.253517 +/- 0.414128 |
| percentile_99_5 | 0.020979 +/- 0.003269 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.987775 +/- 0.018750 | 0.976510 +/- 0.035853 | 1.000000 +/- 0.000000 | 0.253517 +/- 0.414128 |
| median_plus_3mad | 0.012671 +/- 0.001522 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.977363 +/- 0.014020 | 0.956092 +/- 0.026493 | 1.000000 +/- 0.000000 | 0.365942 +/- 0.368409 |
| median_plus_4mad | 0.014369 +/- 0.001779 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.983597 +/- 0.017407 | 0.968290 +/- 0.033135 | 1.000000 +/- 0.000000 | 0.300725 +/- 0.405328 |

## Practical Takeaway
- Best practical rule: `percentile_99`
- Selection rule: Selected as the lowest-mean-false-alarm rule among threshold rules that keep mean F1 within 0.01 and mean recall fault within 0.02 of the current mean_plus_3std baseline.
- Mean false alarm rate moves from `0.257140` to `0.253517` (`-0.003623`), mean F1 moves from `0.987422` to `0.987775` (`+0.000354`), and mean precision moves from `0.975806` to `0.976510` (`+0.000704`).
- Load-0 check: Held-out load 0 does not show a clear threshold effect. Under `percentile_99`, its FAR is `0.970588` versus `0.970588` for `mean_plus_3std`. The best load-0 rule is `mean_plus_3std`, which lowers FAR from `0.970588` to `0.970588`, but the residual FAR is still too high to call the issue mostly solved by threshold transfer alone.

## Comparison vs Earlier Saved CWRU Load-Shift References
- Threshold values are not directly comparable across models because the score scales differ.
| Baseline | Metric | Best Calibrated ResDilatedAE | Saved Baseline Mean | Delta |
| --- | --- | --- | --- | --- |
| AE | AUROC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| AE | AUPRC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| AE | F1 | 0.987775 +/- 0.018750 | 0.983666 | +0.004110 |
| AE | Precision | 0.976510 +/- 0.035853 | 0.968525 | +0.007985 |
| AE | Recall Fault | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| AE | False Alarm Rate | 0.253517 +/- 0.414128 | 0.300725 | -0.047208 |
| OC-SVM | AUROC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| OC-SVM | AUPRC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| OC-SVM | F1 | 0.987775 +/- 0.018750 | 0.980174 | +0.007601 |
| OC-SVM | Precision | 0.976510 +/- 0.035853 | 0.961648 | +0.014863 |
| OC-SVM | Recall Fault | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| OC-SVM | False Alarm Rate | 0.253517 +/- 0.414128 | 0.336957 | -0.083440 |
| Isolation Forest | AUROC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| Isolation Forest | AUPRC | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| Isolation Forest | F1 | 0.987775 +/- 0.018750 | 0.986006 | +0.001769 |
| Isolation Forest | Precision | 0.976510 +/- 0.035853 | 0.972965 | +0.003546 |
| Isolation Forest | Recall Fault | 1.000000 +/- 0.000000 | 1.000000 | +0.000000 |
| Isolation Forest | False Alarm Rate | 0.253517 +/- 0.414128 | 0.267903 | -0.014386 |

## Saved Artifacts
- Metrics JSON: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json`
- Report: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_report.md`
