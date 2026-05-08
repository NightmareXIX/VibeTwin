# Candidate Table: CWRU Final Model vs Baselines

- Protocol: harder leave-one-load-out CWRU load shift.
- ResDilatedAE rows use the saved final model family; the last row is the best practical calibrated rule.
- Baseline rows come from the earlier saved CWRU baseline load-shift summary.

| Model | Threshold Setup | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| AE | saved mean_plus_3std | 1.000 +/- 0.000 | 0.984 +/- 0.019 | 0.969 +/- 0.036 | 1.000 +/- 0.000 | 0.301 +/- 0.412 |
| OC-SVM | saved mean_plus_3std | 1.000 +/- 0.000 | 0.980 +/- 0.017 | 0.962 +/- 0.032 | 1.000 +/- 0.000 | 0.337 +/- 0.391 |
| Isolation Forest | saved mean_plus_3std | 1.000 +/- 0.000 | 0.986 +/- 0.017 | 0.973 +/- 0.033 | 1.000 +/- 0.000 | 0.268 +/- 0.390 |
| ResDilatedAE | saved mean_plus_3std | 1.000 +/- 0.000 | 0.987 +/- 0.019 | 0.976 +/- 0.035 | 1.000 +/- 0.000 | 0.257 +/- 0.412 |
| ResDilatedAE | percentile_99 | 1.000 +/- 0.000 | 0.988 +/- 0.019 | 0.977 +/- 0.036 | 1.000 +/- 0.000 | 0.254 +/- 0.414 |
