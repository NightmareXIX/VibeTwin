# Paderborn Deep SVDD Smoke Validation

## Scope

This report validates the existing one-epoch Deep SVDD smoke-test artifacts only. No full training was run, no code was edited, and no paper files were edited.

## Artifacts Inspected

- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/deep_svdd.pt`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/val_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/test_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/test_fault_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/percentile_99_5/val_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/percentile_99_5/test_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/percentile_99_5/test_labels.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/percentile_99_5/metrics.json`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_42/percentile_99_5/run_config.json`

## Validation Summary

Overall status: PASS.

No obvious implementation artifact bug was found in the one-epoch smoke output.

## Required Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Checkpoint exists | PASS | `deep_svdd.pt` exists, size `76496` bytes |
| Center `c` saved | PASS | checkpoint contains key `center` |
| Center shape is `[embedding_dim]` | PASS | center shape `[64]`, embedding dim `64` |
| Validation scores finite/nonnegative | PASS | all finite, all `>= 0` |
| Combined test scores finite/nonnegative | PASS | all finite, all `>= 0` |
| Test labels contain both classes | PASS | labels `[0, 1]` |
| Larger score means more abnormal | PASS | run config records `larger_is_more_abnormal`; fault mean is greater than healthy mean; AUROC `0.837016` |
| Threshold equals validation p99.5 | PASS | absolute delta `7.22e-11` |
| FAR recomputation matches metrics | PASS | absolute delta `0.0` |
| Healthy-only training recorded | PASS | `training = healthy_train_windows_only` |
| Healthy-validation-only thresholding recorded | PASS | `threshold_calibration = healthy_validation_scores_only`; threshold fit split `val_healthy` |
| Fault data excluded from training/threshold | PASS | `fault_data_used_for_training_or_threshold = false` |

## Checkpoint Details

Checkpoint keys:

- `best_epoch`
- `best_validation_mean_score`
- `center`
- `data_protocol`
- `embedding_dim`
- `history`
- `model_name`
- `model_settings`
- `model_state_dict`
- `parameter_count`
- `seed`
- `training_settings`

Center summary:

| Field | Value |
| --- | ---: |
| shape | `[64]` |
| embedding_dim | `64` |
| mean | `-0.0192886` |
| std | `0.307054` |
| min | `-0.966075` |
| max | `0.681120` |

## Score Distribution Summary

| Split | Count | Mean | Std | Min | Max | Finite | Nonnegative | Approx Unique |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| validation healthy | 16884 | 0.00113662 | 0.000651025 | 0.000241065 | 0.0132355 | yes | yes | 16875 |
| test healthy | 16886 | 0.00113088 | 0.000692864 | 0.000264208 | 0.0171157 | yes | yes | 16880 |
| test fault | 519082 | 0.00599123 | 0.00572690 | 0.000167423 | 0.0677370 | yes | yes | 514332 |
| combined test | 535968 | 0.00583811 | 0.00570088 | 0.000167423 | 0.0677370 | yes | yes | 530986 |

## Metric Cross-Checks

Metrics from `metrics.json`:

- AUROC: `0.8370155084865989`
- AUPRC: `0.9940023601895684`
- F1: `0.6475914599201842`
- Precision: `0.999654182382603`
- Recall fault: `0.4789224053232437`
- FAR: `0.0050929764301788465`
- Threshold: `0.004823317357804625`
- Score source: `trained_deep_svdd`

Recomputed checks:

- Validation p99.5 threshold delta: `7.22e-11`
- FAR recomputation delta: `0.0`
- Validation score count: `16884`
- Combined test score count: `535968`
- `threshold_meta.fit_split`: `val_healthy`
- `threshold_meta.fit_count`: `16884`

## Warnings

None.

Specific warning checks:

- Scores are not all identical: PASS.
- Scores are finite: PASS.
- Scores are nonnegative: PASS.
- Scores are not extremely close to zero: PASS.
- Fault score distribution is separated upward from healthy score distribution in the one-epoch smoke run: PASS.

## Notes

This is only a one-epoch smoke run. The metrics are useful for sanity checking the implementation and artifact protocol, but they should not be treated as the final Deep SVDD baseline result for the paper.
