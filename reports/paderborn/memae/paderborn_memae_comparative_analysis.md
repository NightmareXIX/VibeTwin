# Paderborn MemAE Comparative Analysis

Workstream note for the MemAE comparator's diagnostic and comparative analysis against
ResDilatedAE-T. The generated tables and the full written note live under
`artifacts/generative_upgrades/memae/analysis/` (local run output, not tracked); this file
records the decisions and the findings so they survive outside that directory.

## What Was Run

```powershell
.\.venv-cuda\Scripts\python.exe scripts/eval_paderborn_deployment_metrics.py --include-memae --memae-seed 42 `
    --output-dir artifacts/generative_upgrades/memae/analysis --output-stem memae_deployment
.\.venv-cuda\Scripts\python.exe scripts/compare_memae_resdilated.py
```

Neither script retrains anything. `compare_memae_resdilated.py` reads the saved per-window score
arrays written by the unified Paderborn evaluation; the deployment script loads saved checkpoints
and profiles them on CPU. The `--output-dir` / `--output-stem` redirect keeps the comparison run
out of `artifacts/generative_upgrades/resdilated_ae/deployment/`, so the tracked ResDilatedAE-T
deployment numbers are untouched.

## Gate Decision

The analysis plan gates the per-condition breakdown and the miss-overlap comparison on the
comparator being a working detector, at AUROC ≥ 0.75.

MemAE's headline three-seed AUROC under `percentile_99_5` is **0.6972 ± 0.0567**, below the gate.
Seed 42 alone reaches 0.7527, but the headline is the mean, and selecting the branch on the best
seed would be exactly the kind of after-the-fact choice the gate exists to prevent.

**Branch taken: FAR-matched recall and the deployment profile only.** The per-condition miss
breakdown and the miss-overlap analysis were not produced. At its calibrated operating point
MemAE flags 22.3% of fault windows against 61.0% for ResDilatedAE-T, and its seed-to-seed AUROC
spread (0.057) is of the same order as the differences between operating conditions a breakdown
would report. A per-condition miss rate drawn from those scores would measure the scorer's noise,
and a miss-overlap figure between a working detector and a barely-working one is arithmetic rather
than complementarity. Withholding both is the honest result: the threshold-transfer
generalization and the complementarity claim cannot be tested against this comparator.

## Findings

**MemAE is mid-field, not an outlier failure.** Three-seed AUROC across the whole comparison:
Isolation Forest 0.913, ResDilatedAE-T 0.858, MemAE 0.697, Deep SVDD 0.676, ConvVAE 0.530,
CompactAE 0.522, OC-SVM 0.345. MemAE places third of seven and first among the deep autoencoders,
ahead of CompactAE by 0.175 and ConvVAE by 0.167 AUROC, with roughly double CompactAE's recall.
Every generic reconstruction autoencoder in the comparison falls in a 0.52-0.70 band; the result
to explain is that ResDilatedAE-T escapes that band, not that MemAE sits inside it. The manuscript
must not describe the MemAE row as a failure - the table itself would contradict it.

**The comparator trains without the frequency-domain loss, and this needs disclosing.**
`train_generative_upgrades.py` forces `freq_loss_weight = 0.0` for `model_kind == "memae"`, while
ResDilatedAE-T and ConvVAE train at 0.1. The spectral term is VibeTwin's contribution rather than
part of MemAE as published, so adding it would report a hybrid neither paper proposes - but a
reviewer will still ask whether the comparator was handicapped. The empirical answer is in the
same table: ConvVAE trains *with* the frequency loss and reaches 0.530 AUROC, well below MemAE's
0.697 without it, so the term is not on its own what separates the models. State this proactively
in the Experiment section.

**FAR-matched recall does not reverse the ordering.** Sweeping every model's threshold to a common
false-alarm target leaves the ranking unchanged at FAR 0.005 / 0.0069 / 0.01. ResDilatedAE-T
recalls 0.610 → 0.622 across those targets, Isolation Forest 0.404 → 0.464, MemAE 0.223 → 0.243.
MemAE's loss is not a threshold artifact.

Two threshold bases are reported. `val_fitted` fits the (1 − FAR) percentile on validation healthy
scores only, so the realized test FAR is a measured outcome and the repo's no-fitting-on-test rule
holds; these are the numbers that may enter the paper. `test_oracle` places the threshold to hit
the target FAR exactly on test healthy windows — it reads test data, exists only to rule out
validation-to-test threshold drift as an explanation, and must not back a headline claim. The two
bases agree on the ordering. As a correctness check, `val_fitted` at FAR 0.005 reproduces the
tracked `percentile_99_5` metrics exactly for all seven models across all three seeds.

**The mechanism-disabled control softens the Phase 3 probe result.** At full protocol across three
seeds, memory-live scores AUROC 0.6972 ± 0.0567 against 0.7249 ± 0.0449 for the λ = 0, α = 0
control — Δ = −0.028, smaller than either arm's seed standard deviation. The reduced-scale probe
had put the control 0.073 ahead. The defensible reading at paper scale is that the memory
mechanism neither helps nor demonstrably hurts on this benchmark, **not** that it is what costs
the detector its performance. The narrow claim still holds unchanged: MemAE, as specified, under
matched calibration and capacity on the Paderborn benchmark, did not improve over its own
memory-free control.

**The λ sweep supports the chosen setting.** At the paper protocol on seed 42, λ = 1/N = 0.002
gives the lowest validation reconstruction loss (0.00973 vs 0.01648 at 2/N and 0.02471 at 3/N) and
also the highest AUROC. Selection was made on validation loss, so no test information entered it.

**The memory is neither inert nor collapsed.** 415–445 of 500 slots are utilized across seeds, the
top slot takes 1.3–1.6% of the mass, and addressing entropy sits at 4.81–4.91 against a uniform
6.21. The result is not a degenerate-memory artifact.

**Deployment cost does not favour VibeTwin.** MemAE carries 226,969 parameters against
ResDilatedAE-T's 222,657 (+1.94%, inside the ±15% capacity-parity target) at an essentially
identical 0.881 MB checkpoint, and scores a single window on CPU in 2.75 ms against 7.66 ms — it is
the *cheaper* model. Addressing a 500-slot bank costs less than the dilated residual stack it
replaces. The deployment axis is worth reporting as part of the fairness argument, since it shows
the comparator was not starved of capacity, but not as a consolation for the accuracy result.

## What This Selects For The Manuscript

The negative-result branch, together with the workflow-generality and deployment-cost framings.
Confirmed with the user on 2026-08-30; the plan's §2 status table records the settled position and
the stop-and-review gate before the paper-writing phases is cleared.

- Report the loss plainly and keep the tables as they are.
- State the claim narrowly, every time: *MemAE, as specified, under matched calibration and
  capacity on the Paderborn benchmark, did not improve over its own memory-free control.* Nothing
  measured here speaks to CMAE or the other 2024–2026 members of the memory-augmented family.
- The workflow-generality argument strengthens: a structurally different backbone dropped into the
  same healthy-only training, validation-fitted calibration and leakage-safe evaluation with no
  protocol change, and the harness reported it as worse. That is evidence of an unbiased
  evaluation, not of a weak workflow.
- The FAR-matched table is the protocol's standard deployment view computed for every model. It
  must be introduced that way, not as MemAE's rescue after the headline row goes against it.
- The threshold-transfer generalization and the complementarity analysis are out of reach on this
  comparator, and the manuscript should say so rather than extract them from noise. The existing
  single-model threshold-transfer limitation stays as written; the fusion item drops out of future
  work, since no analysis in the paper motivates it.
- The mechanism-disabled control is exculpatory, not attributive. It rules out a defective encoder
  and licenses nothing about where the loss comes from.
- A plausible explanation, and it must be labelled as one: the memory addresses a failure mode
  this benchmark does not exhibit. Addressing entropy of 4.81-4.91 nats over 500 slots means
  roughly 120-135 slots are blended per latent position, so the intended bottleneck does not bind,
  and the λ sweep showed that tightening it degrades reconstruction and AUROC together rather than
  improving separation.
