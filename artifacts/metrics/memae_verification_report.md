# MemAE correctness cross-check

Device: `cuda` | seed 42 | addressing `dot`, N = 500, lambda = 0.002, alpha = 0.0002

**11/11 implementation checks passed.**

Contrary findings — results about the method under this protocol, not implementation faults: check 12 (Memory attribution: live mechanism vs disabled control).

### Implementation checks

| # | Check | Group | Status | Observation |
|---|---|---|---|---|
| - | Capacity matched to ResDilatedAE | mechanism | pass | MemAE 226,969 vs ResDilatedAE 222,657 (+1.94%) |
| 1 | Addressing weights sum to one per position | mechanism | pass | max deviation 2.384e-07 (init path 2.384e-07, sharp path 2.384e-07) |
| 2 | Shrinkage is the identity at lambda = 0 | mechanism | pass | max \|w_hat - w\| = 5.364e-07 |
| 3 | Raising lambda shrinks the active slot count | mechanism | pass | 500.0 -> 25.7 -> 18.0 -> 14.4 slots for lambda in {0, 1/N, 2/N, 3/N} (N=500) |
| 4 | Gradient reaches the memory bank | mechanism | pass | \|\|dL/dM\|\| = 4.219e-02 over 500/500 slots with non-zero gradient |
| 5 | Cosine similarity stays in [-1, 1] | mechanism | pass | similarity range [-0.482439, 0.463772] |
| 6 | Addressing is per latent position | mechanism | pass | attention (4, 128, 500), reconstruction (4, 1, 2048) |
| 7 | Overfits a small healthy subset | behavioural | pass | recon MSE 1.6037 -> 0.0498 over 200 epochs (96.8% of window variance explained) |
| 8 | Memory utilization after training | behavioural | pass | 55.5/500 slots survive shrinkage per position; 66/500 slots utilized (13.2%); top slot holds 3.265% of the addressing mass |
| 9 | Addressing entropy sharpens during training | behavioural | pass | train_mem_loss 5.3742 -> 3.9509 (uniform reference log(N) = 6.2146) |
| 10 | Degenerate control (alpha = 0, lambda = 0) | behavioural | pass | AUROC full 0.6941 vs control 0.7669 (delta -0.0728); recall 0.2041 vs 0.2119; surviving slots 55.5 vs 500.0 |

### Findings

| # | Check | Group | Status | Observation |
|---|---|---|---|---|
| 11 | Score distribution sanity | behavioural | as expected | val healthy modes = 1; median fault 0.0119 vs median healthy 0.0074 |
| 12 | Memory attribution: live mechanism vs disabled control | behavioural | CONTRARY | AUROC 0.6941 (memory live) vs 0.7669 (control), delta -0.0728; recon 0.0087 vs 0.0046 |

## Expectations

- **capacity — Capacity matched to ResDilatedAE** (assertion): |delta| <= 15% of 222657
- **check 1 — Addressing weights sum to one per position** (assertion): max |sum - 1| < 1e-5
- **check 2 — Shrinkage is the identity at lambda = 0** (assertion): max |w_hat - w| < 1e-6
- **check 3 — Raising lambda shrinks the active slot count** (assertion): mean non-zero slots per position is non-increasing in lambda, and strictly lower at 3/N than at 0
- **check 4 — Gradient reaches the memory bank** (assertion): M.grad is not None and has non-zero norm
- **check 5 — Cosine similarity stays in [-1, 1]** (assertion): -1 - 1e-6 <= d <= 1 + 1e-6 for the cosine addressing path, whichever mode is the default
- **check 6 — Addressing is per latent position** (assertion): attention shape (4, 128, 500), reconstruction shape (4, 1, 2048)
- **check 7 — Overfits a small healthy subset** (assertion): final recon MSE < 0.3 and < half of the first epoch
- **check 8 — Memory utilization after training** (assertion): shrinkage active (mean surviving slots < 0.9N), >10% of slots carry mean weight above 1/(10N), and no slot holds >50% of the mass
- **check 9 — Addressing entropy sharpens during training** (assertion): final train_mem_loss below 0.95 log(N) and not materially above the first epoch
- **check 10 — Degenerate control (alpha = 0, lambda = 0)** (assertion): the control must differ mechanically (all slots survive at lambda = 0, far fewer with shrinkage live)
- **check 11 — Score distribution sanity** (finding): validation healthy scores unimodal and fault median above healthy median
- **check 12 — Memory attribution: live mechanism vs disabled control** (finding): the memory-enabled model outperforms the control; otherwise any advantage MemAE shows is architectural and the paper must attribute it that way

## Probe run

32768 healthy train windows, 4096 validation, 4096 healthy test, 4096 fault test, 40 epochs, batch 256.

| Variant | recon MSE (val) | addressing entropy | AUROC | recall | FAR |
|---|---|---|---|---|---|
| MemAE (memory live) | 0.0087 | 3.9487 | 0.6941 | 0.2041 | 0.0107 |
| Control (lambda = 0, alpha = 0) | 0.0046 | 6.1790 | 0.7669 | 0.2119 | 0.0110 |

## Notes

- The probe run is a reduced-scale smoke check, not a headline result. Paper numbers come from the full three-seed run and `eval_paderborn_baselines_unified.py`.
- Check 12 is only readable at the default probe size. Reduced runs cannot separate the two variants from their own run-to-run spread.
- Deviations from the reference release (`donggong1/memae-anomaly-detection`) are recorded in `implementation_docs/memae_phase3_notes.md`.
