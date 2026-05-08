# Paderborn Generative Upgrade Report

## Label Provenance
- Total bearings: `32`
- Verified bearings: `0`
- Inferred bearings: `32`
- All current Paderborn evaluation labels remain bearing-family inferences; local support PDFs exist but were not parsed automatically in this pass.

## Setup
- Device used: `cuda`
- Selected signal channel: `vibration_1`
- Window size: `2048`
- Stride: `1024`
- Threshold rule: `mean_plus_3std`
- Frequency-loss weight: `0.100`
- Effective batch size: `256`
- CUDA available at runtime: `True`

## Overall Comparison
| Model | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CompactAE | 0.546229 | 0.976361 | 0.260799 | 0.998116 | 0.149996 | 0.008705 |
| IsolationForest | 0.914274 | 0.996756 | 0.633790 | 0.999419 | 0.464029 | 0.008291 |
| ResDilatedAE | 0.865940 | 0.995162 | 0.803987 | 0.999156 | 0.672605 | 0.017470 |

## Interpretation
- Best upgraded generative model: `ResDilatedAE`
- Competitive with Isolation Forest: `False`
- Capacity/loss read: The Paderborn gap looked like both under-capacity and loss-design weakness: deeper residual context and spectral guidance materially improved ranking and detection.
- Summary note: ResDilatedAE improves over the compact AE but still trails Isolation Forest enough that the generative path is promising rather than fully competitive.

## ResDilatedAE
- Threshold: `0.000346`
- AUROC: `0.865940`
- AUPRC: `0.995162`
- F1: `0.803987`
- Precision: `0.999156`
- Recall fault: `0.672605`
- False alarm rate: `0.017470`
- Final train loss: `0.000313`
- Final val loss: `0.000151`
- Final train time loss: `0.000222`
- Final val time loss: `0.000103`
- Final train freq loss: `0.000906`
- Final val freq loss: `0.000482`
- Final train KL loss: `0.000000`
- Final val KL loss: `0.000000`
- Parameter count: `222657`
- Model size on disk: `0.882` MB
- Training time: `1884.14` seconds

### By Damage Group
| Damage Group | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KA | 0.804980 | 0.984146 | 0.716532 | 0.997800 | 0.558967 | 0.017470 | 239390 |
| KB | 0.994012 | 0.998419 | 0.979001 | 0.994917 | 0.963587 | 0.017470 | 59923 |
| KI | 0.897423 | 0.991674 | 0.834558 | 0.998131 | 0.717048 | 0.017470 | 219769 |

### By Operating Condition
| Condition | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N09_M07_F10 | 0.809705 | 0.992838 | 0.723131 | 0.998954 | 0.566667 | 0.018268 | 129757 |
| N15_M01_F10 | 0.889046 | 0.996039 | 0.834071 | 0.998719 | 0.716027 | 0.028179 | 129572 |
| N15_M07_F04 | 0.886479 | 0.995974 | 0.821735 | 0.999559 | 0.697625 | 0.009474 | 129938 |
| N15_M07_F10 | 0.877903 | 0.995655 | 0.830264 | 0.999360 | 0.710111 | 0.013961 | 129815 |

## Saved Artifacts
- Run directory: `artifacts/generative_upgrades/resdilated_ae/seed_123`
- Best checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_123/checkpoints/best.pt`
- Latest checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_123/checkpoints/latest.pt`
- Training history JSON: `artifacts/generative_upgrades/resdilated_ae/seed_123/history.json`
- Run status JSON: `artifacts/generative_upgrades/resdilated_ae/seed_123/status.json`
- Training log: `artifacts/generative_upgrades/resdilated_ae/seed_123/train.log`
- Metrics JSON: `artifacts/generative_upgrades/resdilated_ae/seed_123/metrics.json`
- Markdown report: `artifacts/generative_upgrades/resdilated_ae/seed_123/report.md`
- Summary plot: `artifacts/generative_upgrades/resdilated_ae/seed_123/summary.png`
