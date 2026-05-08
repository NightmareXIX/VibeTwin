# Paderborn ResDilatedAE Threshold Calibration Report

## Protocol
- Inference only from saved `best.pt` checkpoints.
- No retraining, no preprocessing changes, and no raw-data edits.
- Seeds evaluated: `42`, `7`, `123`.
- Splits scored: `val_healthy`, `test_healthy`, `test_fault`.
- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.
- MAD uses the raw median absolute deviation over healthy validation scores.

## Dataset Defaults Reused
- Device used: `cuda`
- Effective batch size: `256`
- Window size: `2048`
- Stride: `1024`

## Seed 42
- Run directory: `artifacts/generative_upgrades/resdilated_ae/seed_42`
- Checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_42/checkpoints/best.pt`
- Saved scores: `artifacts/generative_upgrades/resdilated_ae/seed_42/val_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_42/test_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_42/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.000489 | 0.865734 | 0.751198 | 0.999200 | 0.601824 | 0.014805 |
| percentile_99 | 0.000532 | 0.865734 | 0.747941 | 0.999307 | 0.597617 | 0.012732 |
| percentile_99_5 | 0.000676 | 0.865734 | 0.742861 | 0.999645 | 0.591038 | 0.006455 |
| median_plus_3mad | 0.000277 | 0.865734 | 0.809662 | 0.996660 | 0.681750 | 0.070236 |
| median_plus_4mad | 0.000326 | 0.865734 | 0.787826 | 0.997797 | 0.650862 | 0.044179 |

## Seed 7
- Run directory: `artifacts/generative_upgrades/resdilated_ae/seed_7`
- Checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_7/checkpoints/best.pt`
- Saved scores: `artifacts/generative_upgrades/resdilated_ae/seed_7/val_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_7/test_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_7/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.000441 | 0.843718 | 0.754338 | 0.999253 | 0.605846 | 0.013917 |
| percentile_99 | 0.000469 | 0.843718 | 0.752109 | 0.999333 | 0.602947 | 0.012377 |
| percentile_99_5 | 0.000574 | 0.843718 | 0.746792 | 0.999606 | 0.596045 | 0.007225 |
| median_plus_3mad | 0.000240 | 0.843718 | 0.798698 | 0.996637 | 0.666355 | 0.069111 |
| median_plus_4mad | 0.000281 | 0.843718 | 0.782266 | 0.997845 | 0.643288 | 0.042698 |

## Seed 123
- Run directory: `artifacts/generative_upgrades/resdilated_ae/seed_123`
- Checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_123/checkpoints/best.pt`
- Saved scores: `artifacts/generative_upgrades/resdilated_ae/seed_123/val_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_123/test_healthy_scores.npy`, `artifacts/generative_upgrades/resdilated_ae/seed_123/test_fault_scores.npy`
| Threshold Rule | Threshold | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.000346 | 0.865940 | 0.803987 | 0.999156 | 0.672605 | 0.017470 |
| percentile_99 | 0.000382 | 0.865940 | 0.798428 | 0.999340 | 0.664777 | 0.013502 |
| percentile_99_5 | 0.000492 | 0.865940 | 0.782121 | 0.999613 | 0.642359 | 0.007639 |
| median_plus_3mad | 0.000194 | 0.865940 | 0.831244 | 0.996505 | 0.712999 | 0.076868 |
| median_plus_4mad | 0.000230 | 0.865940 | 0.822853 | 0.997719 | 0.700142 | 0.049212 |

## Mean/Std Across Seeds
| Threshold Rule | Threshold mean+/-std | AUROC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- |
| mean_plus_3std | 0.000425 +/- 0.000073 | 0.858464 +/- 0.012771 | 0.769841 +/- 0.029613 | 0.999203 +/- 0.000049 | 0.626758 +/- 0.039755 | 0.015397 +/- 0.001849 |
| percentile_99 | 0.000461 +/- 0.000075 | 0.858464 +/- 0.012771 | 0.766159 +/- 0.028023 | 0.999327 +/- 0.000017 | 0.621780 +/- 0.037332 | 0.012871 +/- 0.000575 |
| percentile_99_5 | 0.000581 +/- 0.000092 | 0.858464 +/- 0.012771 | 0.757258 +/- 0.021622 | 0.999621 +/- 0.000021 | 0.609814 +/- 0.028296 | 0.007106 +/- 0.000601 |
| median_plus_3mad | 0.000237 +/- 0.000042 | 0.858464 +/- 0.012771 | 0.813201 +/- 0.016559 | 0.996601 +/- 0.000084 | 0.687035 +/- 0.023767 | 0.072072 +/- 0.004192 |
| median_plus_4mad | 0.000279 +/- 0.000048 | 0.858464 +/- 0.012771 | 0.797648 +/- 0.022004 | 0.997787 +/- 0.000064 | 0.664764 +/- 0.030871 | 0.045363 +/- 0.003415 |

## Practical Tradeoff
- Best practical rule: `percentile_99_5`
- Selection rule: Selected as the lowest-mean-false-alarm rule among threshold rules that keep mean F1 within 0.015 and mean recall fault within 0.02 of the current mean_plus_3std baseline.
- Mean false alarm rate moves from `0.015397` to `0.007106` (`-0.008291`), mean F1 moves from `0.769841` to `0.757258` (`-0.012583`), and mean recall fault moves from `0.626758` to `0.609814` (`-0.016945`).

## Comparison vs Isolation Forest Baseline
- Threshold values are not directly comparable across models because the score scales differ.
| Metric | Best Calibrated ResDilatedAE | Isolation Forest | Delta |
| --- | --- | --- | --- |
| AUROC | 0.858464 +/- 0.012771 | 0.914274 | -0.055810 |
| F1 | 0.757258 +/- 0.021622 | 0.633790 | +0.123468 |
| Precision | 0.999621 +/- 0.000021 | 0.999419 | +0.000202 |
| Recall Fault | 0.609814 +/- 0.028296 | 0.464029 | +0.145785 |
| False Alarm Rate | 0.007106 +/- 0.000601 | 0.008291 | -0.001184 |

## Saved Artifacts
- Metrics JSON: `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`
- Report: `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_report.md`
