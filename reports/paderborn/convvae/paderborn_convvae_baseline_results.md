# Paderborn ConvVAE Baseline Results

## Aggregation

Regenerated the unified Paderborn baseline summaries from existing artifacts only:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models compact_ae ocsvm isolation_forest conv_vae resdilated_ae --threshold-rule percentile_99_5 --seeds 42 7 123 --device cpu --skip-train-if-artifacts-exist
```

No training was run. The runner reused 15 complete unified artifacts.

Updated outputs:

- `artifacts/paderborn_unified_baselines/summary.csv`
- `artifacts/paderborn_unified_baselines/summary_by_model.csv`
- `artifacts/paderborn_unified_baselines/latex_table_by_model.tex`

## ConvVAE Seed-Wise Results

| Seed | AUROC | AUPRC | F1 | Precision | Recall Fault | FAR | Threshold | Score Source |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.529831 | 0.973019 | 0.248265 | 0.998656 | 0.141752 | 0.005863 | 2.510491 | existing checkpoint inference |
| 7 | 0.530559 | 0.972961 | 0.246133 | 0.998588 | 0.140365 | 0.006100 | 2.518982 | existing score arrays |
| 123 | 0.529357 | 0.972975 | 0.246603 | 0.998578 | 0.140671 | 0.006159 | 2.505224 | existing score arrays |

## ConvVAE Mean Plus Std

| Metric | Mean | Std |
| --- | ---: | ---: |
| AUROC | 0.529916 | 0.000605 |
| AUPRC | 0.972985 | 0.000030 |
| F1 | 0.247000 | 0.001120 |
| Precision | 0.998607 | 0.000043 |
| Recall Fault | 0.140930 | 0.000729 |
| FAR | 0.006041 | 0.000157 |

ConvVAE is very stable across the three seeds, but the stable operating point is low recall and low F1.

## Comparison Against CompactAE

| Metric | CompactAE Mean | ConvVAE Mean | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.522099 | 0.529916 | +0.007817 |
| AUPRC | 0.973836 | 0.972985 | -0.000851 |
| F1 | 0.202889 | 0.247000 | +0.044112 |
| Recall Fault | 0.113282 | 0.140930 | +0.027648 |
| FAR | 0.007758 | 0.006041 | -0.001717 |

ConvVAE is a modest improvement over CompactAE on AUROC, F1, recall, and FAR, while AUPRC is essentially tied and slightly lower. This supports treating ConvVAE as a stronger autoencoding baseline than CompactAE, but not as a competitive final method.

## Comparison Against Isolation Forest

| Metric | Isolation Forest Mean | ConvVAE Mean | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.912755 | 0.529916 | -0.382839 |
| AUPRC | 0.996668 | 0.972985 | -0.023683 |
| F1 | 0.575152 | 0.247000 | -0.328152 |
| Recall Fault | 0.404172 | 0.140930 | -0.263242 |
| FAR | 0.004580 | 0.006041 | +0.001461 |

Isolation Forest remains much stronger than ConvVAE under the unified Paderborn p99.5 protocol. The main gap is recall and F1, not precision.

## Comparison Against ResDilatedAE

| Metric | ResDilatedAE Mean | ConvVAE Mean | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.858464 | 0.529916 | -0.328548 |
| AUPRC | 0.994847 | 0.972985 | -0.021862 |
| F1 | 0.757258 | 0.247000 | -0.510258 |
| Recall Fault | 0.609814 | 0.140930 | -0.468884 |
| FAR | 0.007106 | 0.006041 | -0.001066 |

ConvVAE has slightly lower FAR than ResDilatedAE, but this comes with much lower recall and F1. ResDilatedAE is still the stronger generative baseline and the better paper method.

## Paper Table Recommendation

ConvVAE should be included in the paper baseline table if space allows, especially now that it has three seeds under the same unified protocol. It is useful evidence because it shows that a higher-capacity VAE-style reconstruction model improves over CompactAE only modestly and does not close the gap to Isolation Forest or ResDilatedAE.

If the main table must be compact, ConvVAE can go in an extended baseline or supplementary table. It should not be presented as a headline competitor.

## Suggested Discussion Interpretation

The ConvVAE result is best interpreted as a negative-but-informative stronger baseline. It is stable across seeds and slightly better than CompactAE on F1 and recall, but it remains far behind Isolation Forest and ResDilatedAE. This suggests that simply increasing reconstruction-model capacity or adding a latent variational bottleneck is not enough for Paderborn under healthy-only training and validation-only p99.5 calibration.

The discussion should emphasize that ResDilatedAE's advantage is not just because it is a neural reconstruction model. Its architecture and scoring behavior produce much higher fault recall and F1, whereas ConvVAE remains conservative and under-detects faults at the calibrated threshold.

## Validation

Validation checks passed for all 15 regenerated rows:

- `summary.csv` includes `conv_vae` rows for seeds `42`, `7`, and `123`.
- `summary_by_model.csv` includes `conv_vae`.
- `latex_table_by_model.tex` includes `conv_vae`.
- FAR recomputation from `test_scores.npy`, `test_labels.npy`, and `metrics.json` matches.
- Thresholds equal the 99.5th percentile of each run's `val_healthy_scores.npy`.
- Each run's `run_config.json` records healthy-validation-only threshold calibration and no fault use for training or thresholding.
- Test labels contain both healthy (`0`) and fault (`1`) samples.
