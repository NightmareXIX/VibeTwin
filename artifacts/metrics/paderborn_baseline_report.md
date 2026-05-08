# Paderborn Baseline Report

## Label Provenance
- Total bearings: `32`
- Verified bearings: `0`
- Inferred bearings: `32`
- Damage-group counts: `{"KA": 12, "KB": 3, "KI": 11}`
- All current evaluation labels remain family-rule inferences; no support PDFs were parsed automatically in this pass.

## Setup
- Processed root: `data/processed/paderborn`
- Selected signal channel: `vibration_1`
- Window size: `2048`
- Stride: `1024`
- Threshold rule: `mean_plus_3std`
- AE epochs: `8`
- AE batch size: `128`
- OC-SVM variant: `SGDOneClassSVM_linear_full_train`

## Overall Comparison
| Model | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AE | 0.546229 | 0.976361 | 0.260799 | 0.998116 | 0.149996 | 0.008705 |
| OC-SVM | 0.577091 | 0.976227 | 0.000320 | 1.000000 | 0.000160 | 0.000000 |
| Isolation Forest | 0.914274 | 0.996756 | 0.633790 | 0.999419 | 0.464029 | 0.008291 |

## AE
- Threshold: `0.164825`
- Threshold rule: `mean_plus_3std`
- Precision: `0.998116`
- Recall on fault windows: `0.149996`
- False alarm rate on healthy test windows: `0.008705`

### By Damage Group
| Damage Group | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KA | 0.522550 | 0.944721 | 0.194000 | 0.994319 | 0.107486 | 0.008705 | 239390 |
| KB | 0.730347 | 0.922592 | 0.706280 | 0.995537 | 0.547269 | 0.008705 | 59923 |
| KI | 0.521821 | 0.939178 | 0.161629 | 0.992455 | 0.087979 | 0.008705 | 219769 |

### By Operating Condition
| Condition | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N09_M07_F10 | 0.480475 | 0.968231 | 0.064713 | 0.985695 | 0.033455 | 0.014947 | 129757 |
| N15_M01_F10 | 0.577325 | 0.978022 | 0.332729 | 0.996918 | 0.199688 | 0.018944 | 129572 |
| N15_M07_F04 | 0.591759 | 0.980035 | 0.319799 | 0.999919 | 0.190337 | 0.000474 | 129938 |
| N15_M07_F10 | 0.574642 | 0.978004 | 0.300046 | 0.999913 | 0.176505 | 0.000473 | 129815 |

- Final train loss: `0.054781`
- Final val loss: `0.052571`
- Parameter count: `41409`

## OC-SVM
- Threshold: `0.000425`
- Threshold rule: `mean_plus_3std`
- Precision: `1.000000`
- Recall on fault windows: `0.000160`
- False alarm rate on healthy test windows: `0.000000`

### By Damage Group
| Damage Group | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KA | 0.592380 | 0.947917 | 0.000526 | 1.000000 | 0.000263 | 0.000000 | 239390 |
| KB | 0.344747 | 0.738886 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 59923 |
| KI | 0.623788 | 0.954039 | 0.000182 | 1.000000 | 0.000091 | 0.000000 | 219769 |

### By Operating Condition
| Condition | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N09_M07_F10 | 0.699704 | 0.984854 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 129757 |
| N15_M01_F10 | 0.511282 | 0.973227 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 129572 |
| N15_M07_F04 | 0.595576 | 0.977030 | 0.001077 | 1.000000 | 0.000539 | 0.000000 | 129938 |
| N15_M07_F10 | 0.510741 | 0.972575 | 0.000200 | 1.000000 | 0.000100 | 0.000000 | 129815 |

## Isolation Forest
- Threshold: `0.068030`
- Threshold rule: `mean_plus_3std`
- Precision: `0.999419`
- Recall on fault windows: `0.464029`
- False alarm rate on healthy test windows: `0.008291`

### By Damage Group
| Damage Group | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KA | 0.889347 | 0.990537 | 0.544214 | 0.998439 | 0.374047 | 0.008291 | 239390 |
| KB | 0.964585 | 0.990531 | 0.841299 | 0.996800 | 0.727767 | 0.008291 | 59923 |
| KI | 0.927708 | 0.993729 | 0.657557 | 0.998702 | 0.490133 | 0.008291 | 219769 |

### By Operating Condition
| Condition | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N09_M07_F10 | 0.769902 | 0.989100 | 0.268793 | 0.995065 | 0.155383 | 0.023725 | 129757 |
| N15_M01_F10 | 0.949294 | 0.998165 | 0.754840 | 0.999771 | 0.606304 | 0.004262 | 129572 |
| N15_M07_F04 | 0.946978 | 0.998094 | 0.672249 | 0.999863 | 0.506341 | 0.002132 | 129938 |
| N15_M07_F10 | 0.951445 | 0.998253 | 0.740647 | 0.999830 | 0.588175 | 0.003076 | 129815 |

## CWRU Comparison Note
- Compared with the current CWRU baselines, Paderborn looks materially harder in a more conservative way: mean F1 dropped from 0.9989 to 0.2983 while mean false alarm changed from 0.0124 to 0.0057, so the bigger issue is missed faults rather than excess alarms.

## Saved Artifacts
- AE model: `artifacts/models/paderborn_ae_baseline.pt`
- AE metrics: `artifacts/metrics/paderborn_ae_metrics.json`
- OC-SVM metrics: `artifacts/metrics/paderborn_ocsvm_metrics.json`
- Isolation Forest metrics: `artifacts/metrics/paderborn_iforest_metrics.json`
- AE scores: `artifacts/metrics/paderborn_ae_scores.csv`
- OC-SVM scores: `artifacts/metrics/paderborn_ocsvm_scores.csv`
- Isolation Forest scores: `artifacts/metrics/paderborn_iforest_scores.csv`
- Summary plot: `artifacts/plots/paderborn_baseline_summary.png`
