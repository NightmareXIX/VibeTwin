# Candidate Table: Uncertainty Summary

- Negative-result summary for MC-dropout uncertainty.

| Setting | AUROC | F1 | Precision | Recall Fault | False Alarm Rate | Deferred Rate |
| --- | --- | --- | --- | --- | --- | --- |
| Deterministic Baseline | 0.858 +/- 0.013 | 0.757 +/- 0.022 | 1.000 +/- 0.000 | 0.610 +/- 0.028 | 0.0071 +/- 0.0006 | n/a |
| MC No Defer | 0.815 +/- 0.009 | 0.741 +/- 0.011 | 1.000 +/- 0.000 | 0.589 +/- 0.014 | 0.0064 +/- 0.0006 | n/a |
| Uncertainty Aware | 0.690 +/- 0.086 | 0.447 +/- 0.257 | 0.998 +/- 0.002 | 0.310 +/- 0.198 | 0.0035 +/- 0.0006 | 36.3692% |
