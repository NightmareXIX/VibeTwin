# Candidate Table: Paderborn Final Model vs Baselines

- Primary dataset comparison table.
- ResDilatedAE rows are 3-seed mean +/- std.
- Baseline rows are the saved single-run references from the current codebase.

| Model | Threshold Setup | AUROC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| CompactAE | mean_plus_3std | 0.546 | 0.261 | 0.998 | 0.150 | 0.0087 |
| OC-SVM | mean_plus_3std | 0.577 | 0.000 | 1.000 | 0.000 | 0.0000 |
| Isolation Forest | mean_plus_3std | 0.914 | 0.634 | 0.999 | 0.464 | 0.0083 |
| ResDilatedAE | mean_plus_3std | 0.858 +/- 0.013 | 0.770 +/- 0.030 | 0.999 +/- 0.000 | 0.627 +/- 0.040 | 0.0154 +/- 0.0018 |
| ResDilatedAE | percentile_99_5 | 0.858 +/- 0.013 | 0.757 +/- 0.022 | 1.000 +/- 0.000 | 0.610 +/- 0.028 | 0.0071 +/- 0.0006 |
