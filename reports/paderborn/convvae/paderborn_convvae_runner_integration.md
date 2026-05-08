# Paderborn ConvVAE Runner Integration

## Code Changed

Updated `scripts/eval_paderborn_baselines_unified.py` to add ConvVAE as a unified Paderborn baseline option.

Changes made:

- Added `conv_vae` to `DEFAULT_MODELS`, so it is now accepted by `--models conv_vae` and included by `--models all`.
- Added `conv_vae_source_paths(seed)` for existing generative-upgrades artifacts.
- Added saved-score loading for:
  - `artifacts/generative_upgrades/conv_vae/seed_<seed>/val_healthy_scores.npy`
  - `artifacts/generative_upgrades/conv_vae/seed_<seed>/test_healthy_scores.npy`
  - `artifacts/generative_upgrades/conv_vae/seed_<seed>/test_fault_scores.npy`
- Added checkpoint inference from:
  - `artifacts/generative_upgrades/conv_vae/seed_<seed>/checkpoints/best.pt`
- Reused the existing `scripts/train_generative_upgrades.py` ConvVAE model factory and reconstruction scoring helper.
- Registered `conv_vae` in the unified runner model registry.
- Updated the runner-generated report text so ConvVAE is no longer described as omitted.

## CLI Status

`conv_vae` can now be selected from the unified runner CLI:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

The protocol is unchanged:

- training data remains healthy train windows only
- threshold calibration remains healthy validation scores only
- final evaluation remains healthy test windows plus fault test windows
- larger reconstruction MSE still means more abnormal
- `percentile_99_5` remains available as the p99.5 threshold rule

## Artifact Loading Status

ConvVAE loading now follows this order:

1. Load existing ConvVAE saved score arrays if all three split arrays exist.
2. If score arrays are missing, load the existing ConvVAE `best.pt` checkpoint and infer scores.
3. If neither score arrays nor checkpoint exist, return a clean per-model error row through the unified runner's existing error handling. The message says the artifact is missing and training is required.

Existing repository state before validation:

- seed `42` has a ConvVAE checkpoint.
- seed `42` does not have ConvVAE raw score arrays in `artifacts/generative_upgrades/conv_vae/seed_42/`.
- seeds `7` and `123` do not have ConvVAE checkpoints or score arrays, so they still require training before full three-seed evaluation.

## Training Status

No new ConvVAE training was implemented in the unified runner, and no full training should be started by this integration. The runner only reuses existing saved scores or existing checkpoints.

Training is still needed for ConvVAE seeds beyond seed `42` if the paper table should include the same three-seed coverage as other unified baselines.

## Exact Test Command

Requested validation command:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

Compile check:

```powershell
python -m py_compile scripts/eval_paderborn_baselines_unified.py
```

## Validation Results

Compile validation passed:

```powershell
python -m py_compile scripts/eval_paderborn_baselines_unified.py
```

The first literal requested `python ... --models conv_vae ...` command ran without a traceback, but that interpreter did not have PyTorch available before unified ConvVAE outputs existed. The runner recorded a clean error row with:

- model: `conv_vae`
- seed: `42`
- status: `error`
- error: `PyTorch is required for this neural baseline.`

The same smoke command was then run through the project PyTorch environment to verify checkpoint inference:

```powershell
python scripts\eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

That run passed and used the existing checkpoint:

- score source: `existing_conv_vae_checkpoint_inference`
- AUROC: `0.529831`
- AUPRC: `0.973019`
- F1: `0.248265`
- precision: `0.998656`
- recall fault: `0.141752`
- false alarm rate: `0.005863`
- threshold: `2.510491`
- reproduction check: `PASS`

After the unified ConvVAE outputs existed, the literal requested command was run again and passed by reusing complete artifacts:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

Final observed summary status for seed `42` is `reused` with score source `existing_conv_vae_checkpoint_inference`.

Unified ConvVAE artifacts were written to:

- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/metrics.json`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/run_config.json`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/val_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/test_scores.npy`
- `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/test_labels.npy`

Because the smoke command used the default output root, the top-level unified summary files under `artifacts/paderborn_unified_baselines/` were refreshed for this ConvVAE-only invocation.
