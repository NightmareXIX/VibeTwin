# Paderborn Ablation Evaluation Report

## Files Evaluated
- Artifacts root: `artifacts/v3/paderborn_ablation`
- Evaluation root: `artifacts/v3/paderborn_ablation/evaluation`
- Requested variants: `resdilated_time`
- Requested seeds: `42, 7, 123, 2024, 2026`
- Evaluated run count: `5`
- `artifacts/v3/paderborn_ablation/resdilated_time/seed_42`
- `artifacts/v3/paderborn_ablation/resdilated_time/seed_7`
- `artifacts/v3/paderborn_ablation/resdilated_time/seed_123`
- `artifacts/v3/paderborn_ablation/resdilated_time/seed_2024`
- `artifacts/v3/paderborn_ablation/resdilated_time/seed_2026`

## Missing Runs
- none

## Threshold Rules
- `val_p99_5`: 99.5th percentile of healthy validation scores.
- `val_mean_plus_3std`: healthy validation mean plus three standard deviations; validation-calibrated.
- `train_mean_plus_3std`: healthy training-score mean plus three standard deviations; no-validation-calibration comparator.
- Rules evaluated in this run: `val_p99_5, val_mean_plus_3std, train_mean_plus_3std`
- Main ablation threshold: `val_p99_5`

## Main Ablation Summary
| variant | residual | dilation | frequency_loss | threshold_rule | calibration | auroc_mean | auprc_mean | f1_mean | precision_mean | recall_fault_mean | far_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resdilated_time | true | true | false | val_p99_5 | val_p99_5 | 0.868620 | 0.995249 | 0.789601 | 0.999652 | 0.653102 | 0.007000 |
| resdilated_ae_time | true | true | false | train_mean_plus_3std | train_mean_plus_3std | 0.868620 | 0.995249 | 0.798464 | 0.999343 | 0.665283 | 0.013372 |

## Calibration Comparison Summary
| threshold_rule | threshold_value_mean | auroc_mean | auprc_mean | f1_mean | precision_mean | recall_fault_mean | far_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_mean_plus_3std | 0.000357 | 0.868620 | 0.995249 | 0.798464 | 0.999343 | 0.665283 | 0.013372 |
| val_mean_plus_3std | 0.000348 | 0.868620 | 0.995249 | 0.799309 | 0.999307 | 0.666480 | 0.014154 |
| val_p99_5 | 0.000459 | 0.868620 | 0.995249 | 0.789601 | 0.999652 | 0.653102 | 0.007000 |

## Selected Architecture Consistency
- Table IV calibration comparison is fixed to `ResDilatedAE-T` using internal ablation variant `resdilated_time`.
- Paper-facing alias row: `resdilated_ae_time` with `threshold_rule = train_mean_plus_3std`.

## Score Availability Audit
- Present for each `resdilated_time` seed (`42`, `7`, `123`): `train_healthy_scores.npy`, `val_healthy_scores.npy`, `test_healthy_scores.npy`, `test_fault_scores.npy`, `best.pt`.
- Absent by design in the ablation run directories: combined `test_scores.npy` and `test_labels.npy`.

## Group-Wise Recall Analysis
- Detailed breakdown report: `artifacts/v3/paderborn_ablation/evaluation/paderborn_group_breakdown_report.md`
- Operating-condition misses are highest for `N09_M07_F10` at `44.70% +/- 1.40%` missed fault windows, versus `N15_M07_F10` at `32.90% +/- 5.61%`.
- Damage-family misses are highest for `KA` at `47.26% +/- 2.58%`, while `KB` is lowest at `7.71% +/- 2.10%`.
- Metadata alignment was count-verified for every selected seed, but row-level attachment still relies on manifest write order matching array write order.
- Assumption: No explicit saved window IDs accompany the score arrays, so row-level attachment uses preprocessing order.
- Assumption: That attachment order was source-verified from `scripts/preprocess_paderborn.py`, which writes test arrays and manifest rows in the same sequential loop with shared split offsets.
- Assumption: Operating-condition values come from the manifest `condition_code` field and are exported as `operating_condition`.
- Assumption: Damage groups remain bearing-family inferences (`KA`, `KB`, `KI`) from preprocessing and are not PDF-verified labels.

| seed | healthy_metadata_count | healthy_score_count | healthy_aligned | fault_metadata_count | fault_score_count | fault_aligned |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 16886 | 16886 | true | 519082 | 519082 | true |
| 42 | 16886 | 16886 | true | 519082 | 519082 | true |
| 123 | 16886 | 16886 | true | 519082 | 519082 | true |

## Recompute Status
- Fresh train-score inference was used because at least one `train_healthy_scores.npy` file was missing.

## Train Score Inference
- Computed `artifacts/v3/paderborn_ablation/resdilated_time/seed_42/train_healthy_scores.npy` for `resdilated_time` seed `42` with `82231` scores on `cuda`.
- Computed `artifacts/v3/paderborn_ablation/resdilated_time/seed_7/train_healthy_scores.npy` for `resdilated_time` seed `7` with `82231` scores on `cuda`.
- Computed `artifacts/v3/paderborn_ablation/resdilated_time/seed_123/train_healthy_scores.npy` for `resdilated_time` seed `123` with `82231` scores on `cuda`.
- Computed `artifacts/v3/paderborn_ablation/resdilated_time/seed_2024/train_healthy_scores.npy` for `resdilated_time` seed `2024` with `82231` scores on `cuda`.
- Computed `artifacts/v3/paderborn_ablation/resdilated_time/seed_2026/train_healthy_scores.npy` for `resdilated_time` seed `2026` with `82231` scores on `cuda`.

## Warnings Or Reproducibility Issues
- none
