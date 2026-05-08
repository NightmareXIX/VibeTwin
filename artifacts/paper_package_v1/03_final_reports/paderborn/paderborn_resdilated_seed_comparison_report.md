# Paderborn ResDilatedAE Seed Comparison

Compared completed `ResDilatedAE` Paderborn runs for seeds `42`, `7`, and `123`.
Summary statistics below use the sample standard deviation across the 3 saved runs.

## Located Runs

| Seed | Run Folder | Metrics File |
| --- | --- | --- |
| 42 | `artifacts/generative_upgrades/resdilated_ae/seed_42/` | `artifacts/generative_upgrades/resdilated_ae/seed_42/metrics.json` |
| 7 | `artifacts/generative_upgrades/resdilated_ae/seed_7/` | `artifacts/generative_upgrades/resdilated_ae/seed_7/metrics.json` |
| 123 | `artifacts/generative_upgrades/resdilated_ae/seed_123/` | `artifacts/generative_upgrades/resdilated_ae/seed_123/metrics.json` |

## Per-Seed Metrics

| Seed | AUROC | F1 | Recall Fault | False Alarm Rate | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.865734 | 0.751198 | 0.601824 | 0.014805 | 0.000489 |
| 7 | 0.843718 | 0.754338 | 0.605846 | 0.013917 | 0.000441 |
| 123 | 0.865940 | 0.803987 | 0.672605 | 0.017470 | 0.000346 |

## Three-Seed Summary

| Metric | Mean | Std |
| --- | ---: | ---: |
| AUROC | 0.858464 | 0.012771 |
| F1 | 0.769841 | 0.029613 |
| Recall Fault | 0.626758 | 0.039755 |
| False Alarm Rate | 0.015397 | 0.001849 |
| Threshold | 0.000425 | 0.000073 |

## Stability Take

`ResDilatedAE` looks stable enough across these 3 seeds to keep as the main generative model.
There is no failed or collapsed seed, AUROC is tightly grouped, and thresholded detection is consistently strong.
The main variability is the usual recall vs. false-alarm tradeoff: seed `123` gives the best `F1` and `recall_fault`, but it also has the highest `false_alarm_rate`.

In short, the seed sensitivity is noticeable but not alarming:

- `AUROC` varies by about `0.013` std.
- `F1` varies by about `0.030` std.
- `recall_fault` varies by about `0.040` std.
- `false_alarm_rate` stays in a narrow band from `0.013917` to `0.017470`.

That is stable enough for a main generative model, but it is still worth reporting mean and std rather than a single-seed point estimate.

## Brief Comparison vs. Paderborn Isolation Forest Baseline

Saved baseline: `artifacts/metrics/paderborn_iforest_metrics.json`

| Metric | ResDilatedAE 3-Seed Mean | Isolation Forest | Mean Delta |
| --- | ---: | ---: | ---: |
| AUROC | 0.858464 | 0.914274 | -0.055810 |
| F1 | 0.769841 | 0.633790 | +0.136051 |
| Recall Fault | 0.626758 | 0.464029 | +0.162730 |
| False Alarm Rate | 0.015397 | 0.008291 | +0.007106 |

Briefly: the 3-seed `ResDilatedAE` summary beats the saved Isolation Forest baseline on `F1` and `recall_fault`, but still trails it on `AUROC` and produces a higher `false_alarm_rate`.
So it looks like a strong main generative model, but not yet a full replacement for Isolation Forest as the overall Paderborn baseline.
