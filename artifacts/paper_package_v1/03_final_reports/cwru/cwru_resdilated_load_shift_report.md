# CWRU ResDilatedAE Load-Shift Report

## Protocol
- Leave-one-load-out evaluation across the four motor loads: `0`, `1`, `2`, and `3`.
- Healthy train windows come only from non-held-out loads using the existing `train` split.
- Healthy validation windows come only from non-held-out loads using the existing `val` split.
- Test healthy windows come from the held-out load using the existing `test` split.
- Test fault windows come from the held-out load using the existing fault-test windows.
- Fold-specific z-score normalization is refit on healthy train windows only after reconstructing pre-z-score values from the saved preprocessing stats.
- This run keeps the harder CWRU load-shift protocol intact and adds only the final chosen generative model family.

## ResDilatedAE Settings
- Device used: `cuda`
- Effective batch size: `128`
- Epoch budget: `50` with patience `8`
- Learning rate: `0.0003`
- Weight decay: `0.0001`
- Dropout: `0.05`
- Frequency-loss weight: `0.1`
- Threshold rule: `mean_plus_3std`
- Window size: `2048`
- Window stride: `1024`

## Per-Load Results
| Held-Out Load | Train H | Val H | Test H | Test F | Threshold | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 981 | 207 | 34 | 353 | 0.014289 | 1.000000 | 1.000000 | 0.955345 | 0.914508 | 1.000000 | 0.970588 |
| 1 | 817 | 172 | 69 | 353 | 0.018456 | 1.000000 | 1.000000 | 0.997175 | 0.994366 | 1.000000 | 0.028986 |
| 2 | 816 | 172 | 69 | 352 | 0.017998 | 1.000000 | 1.000000 | 0.997167 | 0.994350 | 1.000000 | 0.028986 |
| 3 | 815 | 172 | 69 | 354 | 0.020324 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## Mean/Std Comparison
| Model | AUROC mean+/-std | AUPRC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall Fault mean+/-std | False Alarm Rate mean+/-std |
| --- | --- | --- | --- | --- | --- | --- |
| ResDilatedAE | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.987422 +/- 0.018556 | 0.975806 +/- 0.035465 | 1.000000 +/- 0.000000 | 0.257140 +/- 0.412080 |
| AE | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.983666 +/- 0.018840 | 0.968525 +/- 0.036067 | 1.000000 +/- 0.000000 | 0.300725 +/- 0.412137 |
| OC-SVM | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.980174 +/- 0.016802 | 0.961648 +/- 0.032059 | 1.000000 +/- 0.000000 | 0.336957 +/- 0.390549 |
| Isolation Forest | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.986006 +/- 0.017325 | 0.972965 +/- 0.033138 | 1.000000 +/- 0.000000 | 0.267903 +/- 0.390467 |

## Practical Comparison
- Against the earlier CompactAE load-shift result, ResDilatedAE improves mean F1 (0.987 vs 0.984) and reduces mean false alarm rate (0.257 vs 0.301). The strongest earlier shallow/baseline reference on mean F1 was `Isolation Forest` at `0.986`; ResDilatedAE lands at `0.987` under the same leakage-safe load-shift protocol.

## Manual Commands
- All loads: `python scripts/eval_cwru_resdilated_load_shift.py --held-out-loads 0 1 2 3 --seed 42 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --patience 8 --freq-loss-weight 0.1 --dropout 0.05 --save-every-epochs 1 --threshold-rule mean_plus_3std`
- Resume all: `python scripts/eval_cwru_resdilated_load_shift.py --held-out-loads 0 1 2 3 --seed 42 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --patience 8 --freq-loss-weight 0.1 --dropout 0.05 --save-every-epochs 1 --threshold-rule mean_plus_3std --resume`

## Saved Artifacts
- Metrics JSON: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_metrics.json`
- Report: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_report.md`
- Summary plot: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_summary.png`
- Root run directory: `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42`
