# Paderborn ResDilatedAE MC-Dropout Uncertainty Report

## Protocol
- Inference only from saved `best.pt` checkpoints.
- Current backbone reused directly with inference-time dropout activation only.
- Seeds evaluated: `42, 7, 123`.
- MC passes per window: `10`.
- Score threshold rule: `percentile_99_5` fit on validation healthy MC-mean scores.
- Uncertainty threshold rule: `variance_percentile_99_5` using validation healthy score variance at `99.5` percentile.
- Device used: `cuda` with effective batch size `256`.

## Baseline Calibrated Metrics
| Seed | Threshold | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.000676 | 0.865734 | 0.742861 | 0.999645 | 0.591038 | 0.006455 |
| 7 | 0.000574 | 0.843718 | 0.746792 | 0.999606 | 0.596045 | 0.007225 |
| 123 | 0.000492 | 0.865940 | 0.782121 | 0.999613 | 0.642359 | 0.007639 |

## Uncertainty-Aware Metrics
| Seed | MC Score Thr | Uncertainty Thr | AUROC | F1 | Precision | Recall Fault | False Alarm Rate | Deferred | FAR Delta vs MC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.001313 | 0.00000001 | 0.737356 | 0.597068 | 0.999667 | 0.425646 | 0.003157 | 145062 (27.0654%) | -0.003239 |
| 7 | 0.001170 | 0.00000003 | 0.741776 | 0.594558 | 0.999669 | 0.423099 | 0.003159 | 140365 (26.1891%) | -0.002704 |
| 123 | 0.000961 | 0.00000001 | 0.589889 | 0.150032 | 0.996090 | 0.081125 | 0.004168 | 299355 (55.8531%) | -0.002820 |

## Uncertainty Concentration
| Seed | Miscls Higher? | Miscls/Correct Unc | Hardest Condition | Cond F1 | Cond Unc Gap | Hardest Damage | Damage F1 | Damage Unc Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | no | 0.000 | N09_M07_F10 | 0.666653 | -0.00000500 | KA | 0.648923 | -0.00000480 |
| 7 | no | 0.000 | N09_M07_F10 | 0.669646 | -0.00001255 | KA | 0.640198 | -0.00001089 |
| 123 | no | 0.000 | N09_M07_F10 | 0.687055 | -0.00000179 | KA | 0.654676 | -0.00000153 |

## Mean/Std Across Seeds
| Setting | AUROC mean+/-std | F1 mean+/-std | Precision mean+/-std | Recall mean+/-std | FAR mean+/-std |
| --- | --- | --- | --- | --- | --- |
| Deterministic Baseline | 0.858464 +/- 0.012771 | 0.757258 +/- 0.021622 | 0.999621 +/- 0.000021 | 0.609814 +/- 0.028296 | 0.007106 +/- 0.000601 |
| MC No Defer | 0.814763 +/- 0.008950 | 0.741399 +/- 0.011132 | 0.999646 +/- 0.000023 | 0.589272 +/- 0.014114 | 0.006416 +/- 0.000563 |
| Uncertainty Aware | 0.689674 +/- 0.086444 | 0.447219 +/- 0.257375 | 0.998476 +/- 0.002066 | 0.309957 +/- 0.198178 | 0.003495 +/- 0.000583 |

| Metric | Value |
| --- | --- |
| Uncertainty threshold | 0.00000002 +/- 0.00000001 |
| Total deferred rate | 36.3692% +/- 16.8793% |
| Healthy deferred rate | 0.5922% +/- 0.0485% |
| Fault deferred rate | 37.5331% +/- 17.4296% |
| FAR delta vs MC no defer | -0.002921 +/- 0.000281 |

## Practical Take
- Deferring high-uncertainty windows moved mean false alarm rate from `0.006416` to `0.003495` (`-0.002921`), mean F1 from `0.741399` to `0.447219` (`-0.294179`), and mean recall fault from `0.589272` to `0.309957` (`-0.279315`).
- Misclassified windows had higher uncertainty in `0`/`3` seeds.
- The hardest operating condition had above-overall uncertainty in `0`/`3` seeds.
- The hardest damage group had above-overall fault uncertainty in `0`/`3` seeds.

## Saved Artifacts
- Metrics JSON: `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json`
- Report: `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_report.md`
