# Paderborn ConvVAE Readiness

## Scope

Chunk 1 was a read-only inspection of existing ConvVAE code, artifacts, and unified-runner compatibility. No training was run, and no paper files were edited.

## Files Inspected

- `scripts/train_generative_upgrades.py`
- `scripts/eval_paderborn_baselines_unified.py`
- `scripts/eval_paderborn_resdilated_mc_dropout.py`
- `scripts/eval_cwru_resdilated_load_shift.py`
- `scripts/build_paper_package.py`
- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md`
- `artifacts/generative_upgrades/conv_vae/seed_42/`
- `artifacts/generative_upgrades/resdilated_ae/seed_42/`
- `artifacts/paderborn_unified_baselines/`

Scripts/reports containing `vae`, `conv_vae`, `ConvVAE`, or `VAE`:

- `scripts/train_generative_upgrades.py`
- `scripts/eval_paderborn_baselines_unified.py`
- `scripts/eval_paderborn_resdilated_mc_dropout.py`
- `scripts/eval_cwru_resdilated_load_shift.py`
- `scripts/build_paper_package.py`
- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md`

## ConvVAE Implementation Status

ConvVAE is already implemented.

The model definition is in `scripts/train_generative_upgrades.py` as `class ConvVAE`. It is a 1D convolutional VAE with residual dilated encoder blocks, latent `mu` and `logvar` heads, reparameterization during training, and a transposed-convolution decoder. The `build_models()` factory registers it as:

- `name`: `ConvVAE`
- `cli_name`: `conv_vae`
- `output_stem`: `conv_vae`
- `model_kind`: `vae`
- architecture settings: `base_channels=16`, `latent_dim=48`, configurable dropout

The same script also contains the VAE-specific KL loss (`compute_vae_kl`), beta warmup (`calculate_beta`), training branch for `model_kind == "vae"`, and scoring branch that reconstructs with `reconstruction, _, _ = model(batch)`.

## Existing Paderborn ConvVAE Training Code

Paderborn ConvVAE training code already exists in `scripts/train_generative_upgrades.py`.

The script is explicitly Paderborn-oriented by default:

- processed root: `data/processed/paderborn`
- metadata root: `data/metadata/paderborn`
- CLI model choice includes `conv_vae`
- command shape: `python scripts/train_generative_upgrades.py --model conv_vae ...`
- data protocol: healthy train windows for training, healthy validation windows for threshold calibration, healthy plus fault test windows for evaluation

This is not integrated into the unified baseline runner yet. It is a separate generative-upgrades path.

## Available Artifacts

Existing ConvVAE artifact directory:

- `artifacts/generative_upgrades/conv_vae/seed_42/`

Files present:

- `history.json`
- `metrics.json`
- `report.md`
- `status.json`
- `summary.png`
- `train.log`
- `checkpoints/best.pt`
- `checkpoints/latest.pt`

Checkpoint status:

- `best.pt`: present, 5,432,003 bytes
- `latest.pt`: present, 16,311,499 bytes
- only seed `42` exists for ConvVAE
- no seed `7` or seed `123` ConvVAE checkpoints were found

The seed-42 run status says training and evaluation completed. It ran 50 epochs, with best epoch 49 and best validation total loss about `1.057086`.

Existing ConvVAE metric summary from `artifacts/generative_upgrades/conv_vae/seed_42/report.md`:

- threshold rule: `mean_plus_3std`
- AUROC: `0.529831`
- AUPRC: `0.973019`
- F1: `0.198182`
- precision: `0.999982`
- recall fault: `0.109990`
- false alarm rate: `0.000059`

## Score Arrays

No existing ConvVAE validation/test score arrays were found in `artifacts/generative_upgrades/conv_vae/seed_42/`.

Missing files:

- `artifacts/generative_upgrades/conv_vae/seed_42/val_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_42/test_healthy_scores.npy`
- `artifacts/generative_upgrades/conv_vae/seed_42/test_fault_scores.npy`

This differs from the ResDilatedAE artifacts, which do include these arrays under `artifacts/generative_upgrades/resdilated_ae/seed_42/`.

The current `scripts/train_generative_upgrades.py` has code paths that would save these arrays, but the existing ConvVAE artifact set does not contain them. The artifact appears to preserve the completed checkpoint and summary metrics, but not the raw score arrays needed for unified-runner validation.

## Unified Runner Compatibility

ConvVAE is not currently compatible with `scripts/eval_paderborn_baselines_unified.py` as a runnable model.

Current unified-runner state:

- `DEFAULT_MODELS` includes `ocsvm`, `isolation_forest`, `compact_ae`, and `resdilated_ae`.
- `conv_vae` is not in `DEFAULT_MODELS`.
- `expand_models()` rejects unsupported model names, so `--models conv_vae` would fail today.
- the registry in `main()` has no `conv_vae` entry.
- the report generator explicitly says ConvVAE was left out and can be added later.

The unified runner already has the right pattern for adding ConvVAE. `resdilated_ae` is implemented by first looking for saved score arrays and then falling back to checkpoint inference through `scripts.train_generative_upgrades` helpers. ConvVAE can reuse the same approach:

- add a `conv_vae_source_paths(seed)` helper
- add `run_conv_vae_from_saved_scores()`
- add `run_conv_vae_from_checkpoint()`
- load `build_models(context.dataset.window_size, dropout)`
- select `conv_vae`
- load `payload["state_dict"]`
- call `compute_reconstruction_scores(..., model_kind="vae")`
- return a `ScoreBundle`
- register `"conv_vae": run_conv_vae`

Existing seed-42 checkpoint compatibility looks likely because the checkpoint was produced by the same `train_generative_upgrades.py` model factory that the unified runner already imports for `resdilated_ae`. The raw score arrays are absent, so the first unified ConvVAE run should infer scores from `best.pt` and then write unified artifacts.

The existing ConvVAE metrics are not directly paper-ready for the current unified Paderborn table because:

- they use `mean_plus_3std`, while the current unified summary uses `percentile_99_5`
- they are outside `artifacts/paderborn_unified_baselines/`
- they do not include unified `test_scores.npy`, `test_labels.npy`, `run_config.json`, or reproduction-check outputs
- raw ConvVAE score arrays are missing, so the unified threshold checks cannot be reproduced from the existing artifact alone

## Reports Mentioning ConvVAE

- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md` mentions the unified runner status.
- No report under `reports/` contains ConvVAE result metrics.
- ConvVAE result metrics are present in `artifacts/generative_upgrades/conv_vae/seed_42/report.md`, but that is outside the `reports/` folder and outside the unified baseline artifact tree.
- `scripts/build_paper_package.py` says non-final generative side paths such as ConvVAE were intentionally omitted from the paper package.

## Missing Pieces

Required before ConvVAE can be included as a unified Paderborn baseline:

1. Add `conv_vae` to the unified runner's supported model list and registry.
2. Add ConvVAE source-path helpers parallel to the existing ResDilatedAE helpers.
3. Support loading existing ConvVAE score arrays when present.
4. Support checkpoint inference from `artifacts/generative_upgrades/conv_vae/seed_42/checkpoints/best.pt` when score arrays are missing.
5. Write unified outputs under `artifacts/paderborn_unified_baselines/conv_vae/seed_42/percentile_99_5/`.
6. Ensure reproduction checks pass using validation-only threshold calibration.
7. Decide whether ConvVAE should be run for seed 42 only first, or whether new training should later create seed 7 and seed 123 checkpoints for a three-seed paper table.

Optional cleanup after the first integration:

- Factor the duplicated generative checkpoint scoring logic between `resdilated_ae` and `conv_vae`.
- Backfill missing raw ConvVAE score arrays into `artifacts/generative_upgrades/conv_vae/seed_42/` if preserving the generative-upgrades artifact layout matters.

## Recommended Next Implementation Plan

Next chunk should implement unified-runner integration only, not training.

1. Edit `scripts/eval_paderborn_baselines_unified.py`.
2. Add `conv_vae` support using the existing `resdilated_ae` saved-score/checkpoint-inference pattern.
3. Keep the first validation run scoped to seed `42`, because that is the only existing ConvVAE checkpoint.
4. Use `percentile_99_5` so the result is comparable with `artifacts/paderborn_unified_baselines/summary.csv`.
5. Confirm the generated `conv_vae` unified row passes `reproduction_check.md`.
6. After checkpoint inference works, decide in a later chunk whether to train missing ConvVAE seeds `7` and `123`.

## Exact Suggested Command For The Next Chunk

After adding `conv_vae` to the unified runner, run this checkpoint-inference smoke command:

```powershell
python scripts\eval_paderborn_baselines_unified.py --models conv_vae --threshold-rule percentile_99_5 --seed 42 --device cuda --skip-train-if-artifacts-exist
```

Do not run `--seeds 42 7 123` for ConvVAE until seed `7` and seed `123` checkpoints exist or the next chunk intentionally allows those seeds to fail as documented missing-artifact rows.
