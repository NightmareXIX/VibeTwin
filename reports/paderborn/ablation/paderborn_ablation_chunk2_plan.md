# Paderborn Ablation Chunk 2 Plan

## Summary

Chunk 2 creates only the controlled Paderborn ablation training harness and this written plan artifact. It does not add the full evaluation/calibration summary script, ablation table builder, paper-package integration, or final calibration comparison logic.

## Files Created Or Edited

- Create `scripts/train_paderborn_ablation.py`
- Create/update `reports/paderborn/ablation/paderborn_ablation_chunk2_plan.md`
- Do not create `scripts/eval_paderborn_ablation.py`
- Do not edit preprocessing, existing final model scripts, `artifacts/generative_upgrades/`, paper-package outputs, or LaTeX tables

## Training Harness

- Use existing Paderborn arrays unchanged:
  - `data/processed/paderborn/train/healthy_windows.npy`
  - `data/processed/paderborn/val/healthy_windows.npy`
  - `data/processed/paderborn/test/healthy_windows.npy`
  - `data/processed/paderborn/test/fault_windows.npy`
  - `data/processed/paderborn/test/fault_labels.npy`
- Default output root: `artifacts/paderborn_ablation/`
- Per-run layout: `artifacts/paderborn_ablation/{variant}/seed_{seed}/`
- Refuse to overwrite an existing run directory unless `--overwrite` is provided.
- With `--overwrite`, delete and recreate only the verified target run directory under the configured ablation root.
- CLI support:
  - `--variants`
  - `--seeds`
  - `--epochs`
  - `--learning-rate`
  - `--weight-decay`
  - `--batch-size-cuda`
  - `--batch-size-cpu`
  - `--patience`
  - `--train-subset`
  - `--val-subset`
  - `--test-subset`
  - `--artifacts-root`
  - `--overwrite`
- Default training settings mirror the final ResDilatedAE where applicable:
  - epochs `50`
  - learning rate `3e-4`
  - weight decay `1e-4`
  - patience `8`
  - CUDA batch size `256`
  - CPU batch size `128`
  - dropout `0.05`
  - base channels `16`

## Model Variants

- `compact_ae`: CompactAE-style model, no residual, no dilation, frequency loss weight `0.0`
- `dilated_ae`: ResDilatedAE topology with dilation schedule, no local residual additions, frequency loss weight `0.1`
- `res_ae`: same topology with local residual additions, all block dilations set to `1`, frequency loss weight `0.1`
- `resdilated_time`: same topology as final ResDilatedAE, residual plus dilation, frequency loss weight `0.0`
- `resdilated_full`: same topology as final ResDilatedAE, residual plus dilation, frequency loss weight `0.1`

For the non-compact variants, the encoder-decoder concatenation skips stay fixed across variants. "Residual" refers to the local residual addition inside convolutional blocks.

## Outputs Per Variant/Seed

Each run saves:

- `best.pt`
- `latest.pt`
- `history.json`
- `run_config.json`
- `val_healthy_scores.npy`
- `test_healthy_scores.npy`
- `test_fault_scores.npy`
- `sanity_metrics.json`

`sanity_metrics.json` is threshold-free only: AUROC, AUPRC, score means/stds, and counts. It must not include F1, thresholded table rows, or "no calibration" wording.

`run_config.json` logs variant, seed, model hyperparameters, parameter count, optimizer, learning rate, weight decay, batch size, epochs, early stopping, frequency loss weight, device, torch version, CUDA info, data paths, dataset sizes, and artifact path.

## Reproducibility And Safety

- Set Python `random`, NumPy, and PyTorch seeds.
- If CUDA is available, call `torch.cuda.manual_seed_all`.
- Set `torch.backends.cudnn.benchmark = False`.
- Use healthy validation loss for early stopping.
- Save `best.pt` by best validation total loss and `latest.pt` each epoch.
- Use per-window time-domain reconstruction MSE for saved anomaly scores, even when frequency loss is used for training.
- Do not call `mean_plus_3std` "no calibration." Future Chunk 3 will define the training-only threshold row as `train_mean_plus_3std`.

## Smoke Test Command

```powershell
python scripts/train_paderborn_ablation.py --variants compact_ae dilated_ae res_ae resdilated_time --seeds 42 --epochs 1 --train-subset 512 --val-subset 256 --test-subset 256 --artifacts-root artifacts/paderborn_ablation_smoke
```
