# Paderborn Table IV Calibration Consistency Update

Date: 2026-04-30

## Executive Summary

The Table IV calibration comparison has been corrected to keep the selected architecture fixed to `ResDilatedAE-T`.

The updated comparison now uses the saved Paderborn ablation artifacts for internal variant `resdilated_time` across seeds `42`, `7`, and `123`, and exposes the paper-facing alias row:

- `variant = resdilated_ae_time`
- `threshold_rule = train_mean_plus_3std`

No retraining was needed. The result was recomputed from saved score arrays.

## Artifact Audit

For each seed under `artifacts/paderborn_ablation/resdilated_time/seed_<seed>/`, the following files were present:

- `best.pt`
- `train_healthy_scores.npy`
- `val_healthy_scores.npy`
- `test_healthy_scores.npy`
- `test_fault_scores.npy`

The ablation run directories do not store combined `test_scores.npy` or `test_labels.npy`. That is expected for this artifact family and did not block recomputation.

## Updated Outputs

The following existing artifacts were updated in place:

- `artifacts/paderborn_ablation/evaluation/per_seed_metrics.csv`
- `artifacts/paderborn_ablation/evaluation/ablation_summary_val_p99_5.csv`
- `artifacts/paderborn_ablation/evaluation/calibration_comparison_resdilated_full.csv`
- `artifacts/paderborn_ablation/evaluation/evaluation_report.md`
- `artifacts/paderborn_ablation/evaluation/metrics_summary.json`

Behavioral changes:

- The calibration comparison is no longer hard-wired to `resdilated_full`.
- The calibration comparison CSV now reports the threshold sweep for the selected model family `resdilated_time`, exposed as `resdilated_ae_time`.
- The ablation summary CSV now includes a `threshold_rule` column while keeping `calibration` for compatibility.
- The summary CSV now includes a paper-facing `resdilated_ae_time` row for the training-only threshold comparator.

## Recomputed ResDilatedAE-T Train-Threshold Result

Aggregate over seeds `42`, `7`, and `123`:

| Variant | Threshold Rule | AUROC | AUPRC | F1 | Precision | Recall Fault | FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `resdilated_ae_time` | `train_mean_plus_3std` | 0.859644 +/- 0.018251 | 0.994890 +/- 0.000747 | 0.787098 +/- 0.022723 | 0.999313 +/- 0.000121 | 0.649613 +/- 0.030990 | 0.013660 +/- 0.001774 |

Per-seed recomputed rows in `per_seed_metrics.csv`:

| Variant | Seed | Threshold Rule | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | FAR |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resdilated_time` | 42 | `train_mean_plus_3std` | 0.000391059 | 0.865973 | 0.995132 | 0.784261 | 0.999338 | 0.645366 | 0.013147 |
| `resdilated_time` | 7 | `train_mean_plus_3std` | 0.000372392 | 0.873888 | 0.995485 | 0.811106 | 0.999419 | 0.682507 | 0.012199 |
| `resdilated_time` | 123 | `train_mean_plus_3std` | 0.000426711 | 0.839072 | 0.994052 | 0.765927 | 0.999182 | 0.620965 | 0.015634 |

## Updated Calibration Comparison

The retargeted calibration comparison for the selected architecture now reads:

| Variant | Threshold Rule | Threshold Mean | AUROC | AUPRC | F1 | Precision | Recall Fault | FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resdilated_ae_time` | `train_mean_plus_3std` | 0.000396721 | 0.859644 | 0.994890 | 0.787098 | 0.999313 | 0.649613 | 0.013660 |
| `resdilated_ae_time` | `val_mean_plus_3std` | 0.000387182 | 0.859644 | 0.994890 | 0.788103 | 0.999285 | 0.651008 | 0.014252 |
| `resdilated_ae_time` | `val_p99_5` | 0.000516632 | 0.859644 | 0.994890 | 0.776496 | 0.999647 | 0.635335 | 0.006889 |

## Table IV LaTeX Row

Paper-facing row for the training-only threshold comparator:

```tex
ResDilatedAE-T & train-thr & 0.860 $\pm$ 0.018 & 0.995 $\pm$ 0.001 & 0.787 $\pm$ 0.023 & 0.999 $\pm$ 0.000 & 0.650 $\pm$ 0.031 & 0.0137 $\pm$ 0.0018 \\
```

## Reproducibility Note

- Retraining was not needed.
- Fresh train-score inference was not needed.
- The updated numbers were recomputed directly from the saved `train_healthy_scores.npy`, `val_healthy_scores.npy`, `test_healthy_scores.npy`, and `test_fault_scores.npy` arrays already present in the ablation artifact directories.
