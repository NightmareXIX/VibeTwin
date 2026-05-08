# AE Baseline Report

## Training Summary
- Device: `cpu`
- Model parameter count: `41409`
- Final train loss: `0.126561`
- Final val loss: `0.126697`

## Threshold
- Rule: `mean_plus_3std`
- Threshold: `0.232328`
- Val error mean: `0.126697`
- Val error std: `0.035211`

## Evaluation
- AUROC: `1.000000`
- AUPRC: `1.000000`
- F1: `1.000000`
- Precision: `1.000000`
- Recall on fault windows: `1.000000`
- False alarm rate on healthy test windows: `0.000000`

## Shape Check
- Input batch shape: `(64, 1, 2048)`
- Output batch shape: `(64, 1, 2048)`

## Saved Artifacts
- Model: `artifacts/models/cwru_ae_baseline.pt`
- History JSON: `artifacts/metrics/cwru_ae_history.json`
- Threshold JSON: `artifacts/metrics/cwru_ae_threshold.json`
- Metrics JSON: `artifacts/metrics/cwru_ae_metrics.json`
- Scores CSV: `artifacts/metrics/cwru_ae_scores.csv`
- Loss curve: `artifacts/plots/cwru_ae_loss_curve.png`
- Val histogram: `artifacts/plots/cwru_ae_val_error_hist.png`
- Test histogram: `artifacts/plots/cwru_ae_test_error_hist.png`
