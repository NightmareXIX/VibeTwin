# Paderborn Deployment Metrics Report

## Setup
- Final chosen model benchmarked: `ResDilatedAE` using saved seed `123` and calibrated threshold rule `percentile_99_5`.
- Threshold value for the saved final run: `0.000492`
- CPU benchmarking used `1` Torch thread(s) with representative saved windows only.
- Benchmark pool: `512` windows of length `2048` drawn from `val_healthy`, `test_healthy`, and `test_fault`.

## CPU Benchmarks
| Model | Params | Weights MB | Checkpoint MB | Single ms | Batch64 ms | Batch64 win/s | Peak RSS Delta MB | Saved F1 | Saved AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResDilatedAE | 222657 | 0.877 | 0.882 | 8.832 | 403.243 | 158.7 | 162.074 | 0.782 | 0.866 |
| CompactAE | 41409 | 0.164 | 0.166 | 1.552 | 84.688 | 755.7 | 2.746 | 0.261 | 0.546 |

## Practical Take
- `ResDilatedAE` keeps a tiny on-disk footprint, but the measured CPU runtime RSS bump is much larger in PyTorch. That still fits an industrial PC or gateway-style edge deployment, but it is not a microcontroller-class model.
- The tradeoff versus `CompactAE` is straightforward: `ResDilatedAE` is larger and slower on CPU, but it buys a large jump in saved Paderborn detection quality under the chosen calibrated setup. CompactAE peak RSS delta was about `2.7` MB.
- Isolation Forest comparison is partially blocked: no serialized Paderborn Isolation Forest estimator or scaler was saved under `artifacts/models/`; only metrics and score CSVs exist, so a real inference benchmark would require refitting the model, which would violate the no-retraining constraint.
- Saved Isolation Forest reference remains useful for context: F1 `0.634`, AUROC `0.914`, FAR `0.0083`.

## Saved Artifacts
- Metrics JSON: `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json`
- Report: `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_report.md`
