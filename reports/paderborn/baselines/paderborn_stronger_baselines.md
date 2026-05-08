# Paderborn Stronger Baselines

## ConvVAE Missing Seed Training Plan

This section tracks the ConvVAE three-seed baseline plan now that Paderborn reports are organized under `reports/paderborn/`.

### Current Status

- ConvVAE seed `42` has already been evaluated through the unified Paderborn runner from an existing checkpoint.
- Current seed `42` unified result:
  - AUROC: `0.529831`
  - AUPRC: `0.973019`
  - F1: `0.248265`
  - Precision: `0.998656`
  - Recall fault: `0.141752`
  - FAR: `0.005863`
- Missing ConvVAE training artifacts:
  - `artifacts/generative_upgrades/conv_vae/seed_7/`
  - `artifacts/generative_upgrades/conv_vae/seed_123/`

### Training Script Capability Check

`scripts/train_generative_upgrades.py` supports:

- `--model conv_vae`: yes. `MODEL_CHOICES` includes `conv_vae`.
- `--seed`: yes.
- `--resume`: yes.
- `--save-every-epochs`: yes.
- `--device cuda/cpu`: no. The script does not define a `--device` argument. It auto-selects `cuda` when `torch.cuda.is_available()` is true, otherwise `cpu`.

The ConvVAE implementation already exists in `scripts/train_generative_upgrades.py` as `class ConvVAE`, and `build_models()` registers it with `cli_name="conv_vae"`.

### Manual Training Commands

Run full ConvVAE training manually, not inside the agent.

Seed `7`:

```powershell
python scripts\train_generative_upgrades.py --model conv_vae --seed 7 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --batch-size-cuda 256 --batch-size-cpu 128 --patience 8 --freq-loss-weight 0.10 --vae-beta-max 0.001 --vae-kl-warmup-epochs 10 --dropout 0.05 --save-every-epochs 1
```

Seed `123`:

```powershell
python scripts\train_generative_upgrades.py --model conv_vae --seed 123 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --batch-size-cuda 256 --batch-size-cpu 128 --patience 8 --freq-loss-weight 0.10 --vae-beta-max 0.001 --vae-kl-warmup-epochs 10 --dropout 0.05 --save-every-epochs 1
```

These commands use the same effective defaults as the existing seed `42` ConvVAE run, with hyperparameters written explicitly for reproducibility.

### Expected Training Outputs

Seed `7` output directory:

- `artifacts/generative_upgrades/conv_vae/seed_7/`

Expected files:

- `artifacts/generative_upgrades/conv_vae/seed_7/checkpoints/best.pt`
- `artifacts/generative_upgrades/conv_vae/seed_7/checkpoints/latest.pt`
- `artifacts/generative_upgrades/conv_vae/seed_7/history.json`
- `artifacts/generative_upgrades/conv_vae/seed_7/status.json`
- `artifacts/generative_upgrades/conv_vae/seed_7/metrics.json`
- `artifacts/generative_upgrades/conv_vae/seed_7/report.md`
- `artifacts/generative_upgrades/conv_vae/seed_7/summary.png`
- `artifacts/generative_upgrades/conv_vae/seed_7/train.log`
- `artifacts/generative_upgrades/conv_vae/seed_7/val_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_7/test_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_7/test_fault_scores.npy`

Seed `123` output directory:

- `artifacts/generative_upgrades/conv_vae/seed_123/`

Expected files:

- `artifacts/generative_upgrades/conv_vae/seed_123/checkpoints/best.pt`
- `artifacts/generative_upgrades/conv_vae/seed_123/checkpoints/latest.pt`
- `artifacts/generative_upgrades/conv_vae/seed_123/history.json`
- `artifacts/generative_upgrades/conv_vae/seed_123/status.json`
- `artifacts/generative_upgrades/conv_vae/seed_123/metrics.json`
- `artifacts/generative_upgrades/conv_vae/seed_123/report.md`
- `artifacts/generative_upgrades/conv_vae/seed_123/summary.png`
- `artifacts/generative_upgrades/conv_vae/seed_123/train.log`
- `artifacts/generative_upgrades/conv_vae/seed_123/val_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_123/test_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_123/test_fault_scores.npy`

The current training script saves `checkpoints/best.pt` whenever validation loss improves. After completed training, it reloads `best.pt`, computes reconstruction scores, and saves validation/test score arrays. If score arrays are absent but `best.pt` exists, the unified runner can still infer scores from `best.pt`.

### Resume Commands

If seed `7` stops after creating `latest.pt`, resume with:

```powershell
python scripts\train_generative_upgrades.py --model conv_vae --seed 7 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --batch-size-cuda 256 --batch-size-cpu 128 --patience 8 --freq-loss-weight 0.10 --vae-beta-max 0.001 --vae-kl-warmup-epochs 10 --dropout 0.05 --save-every-epochs 1 --resume
```

If seed `123` stops after creating `latest.pt`, resume with:

```powershell
python scripts\train_generative_upgrades.py --model conv_vae --seed 123 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --batch-size-cuda 256 --batch-size-cpu 128 --patience 8 --freq-loss-weight 0.10 --vae-beta-max 0.001 --vae-kl-warmup-epochs 10 --dropout 0.05 --save-every-epochs 1 --resume
```

The script refuses to overwrite existing run artifacts without `--resume`. Resume requires the corresponding `checkpoints/latest.pt` file.

### Checkpoint Existence Checks

Check seed `7`:

```powershell
Test-Path artifacts\generative_upgrades\conv_vae\seed_7\checkpoints\best.pt
Test-Path artifacts\generative_upgrades\conv_vae\seed_7\checkpoints\latest.pt
```

Check seed `123`:

```powershell
Test-Path artifacts\generative_upgrades\conv_vae\seed_123\checkpoints\best.pt
Test-Path artifacts\generative_upgrades\conv_vae\seed_123\checkpoints\latest.pt
```

Check score arrays:

```powershell
Test-Path artifacts\generative_upgrades\conv_vae\seed_7\val_healthy_scores.npy
Test-Path artifacts\generative_upgrades\conv_vae\seed_7\test_healthy_scores.npy
Test-Path artifacts\generative_upgrades\conv_vae\seed_7\test_fault_scores.npy
Test-Path artifacts\generative_upgrades\conv_vae\seed_123\val_healthy_scores.npy
Test-Path artifacts\generative_upgrades\conv_vae\seed_123\test_healthy_scores.npy
Test-Path artifacts\generative_upgrades\conv_vae\seed_123\test_fault_scores.npy
```

### Unified Evaluation After Training

After seed `7` and seed `123` training are complete, run:

```powershell
python scripts\eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seeds 42 7 123 --device cuda --skip-train-if-artifacts-exist
```

Expected unified outputs:

- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_7/percentile_99_5/`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_123/percentile_99_5/`
- refreshed `artifacts/paderborn_unified_baselines/summary.csv`
- refreshed `artifacts/paderborn_unified_baselines/summary_by_model.csv`
- refreshed `artifacts/paderborn_unified_baselines/reproduction_check.md`

The unified runner will prefer existing ConvVAE score arrays when present. If arrays are missing but `checkpoints/best.pt` exists, it can infer scores from the checkpoint while preserving the same Paderborn protocol: healthy train only, healthy validation only for threshold calibration, and healthy plus fault test for final evaluation.
