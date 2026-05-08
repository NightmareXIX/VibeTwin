# Candidate Table: Deployment Summary

- CPU benchmark summary for the final saved deployment study.

| Model | Params | Weights MB | Checkpoint MB | Single ms | Batch64 ms | Batch64 win/s | Peak RSS Delta MB | Saved F1 | Saved AUROC | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResDilatedAE | 222657 | 0.877 | 0.882 | 8.832 | 403.243 | 158.7 | 162.074 | 0.782 | 0.866 |  |
| CompactAE | 41409 | 0.164 | 0.166 | 1.552 | 84.688 | 755.7 | 2.746 | 0.261 | 0.546 |  |
| Isolation Forest | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.634 | 0.914 | benchmark blocked by missing serialized estimator |
