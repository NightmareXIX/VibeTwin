# Paderborn Deep SVDD Readiness

## Scope

This is a read-only readiness inspection for adding Deep SVDD as a stronger Paderborn baseline. No training was run, no code was changed, and no paper files were edited.

## Files And Directories Inspected

- `scripts/eval_paderborn_baselines_unified.py`
- `scripts/train_ae_baseline.py`
- `scripts/train_generative_upgrades.py`
- `scripts/train_paderborn_baselines.py`
- `scripts/train_paderborn_ablation.py`
- `scripts/eval_paderborn_ablation.py`
- `scripts/eval_paderborn_resdilated_threshold_calibration.py`
- `scripts/eval_paderborn_resdilated_mc_dropout.py`
- `scripts/eval_paderborn_deployment_metrics.py`
- `scripts/eval_cwru_load_shift.py`
- `scripts/train_shallow_baselines.py`
- `artifacts/`
- `artifacts/models/`
- `artifacts/metrics/`
- `artifacts/generative_upgrades/`
- `artifacts/paderborn_unified_baselines/`
- `reports/`
- `reports/README.md`
- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md`
- `reports/paderborn/baselines/paderborn_baseline_reproduction.md`
- `reports/paderborn/baselines/paderborn_stronger_baselines.md`
- `reports/paderborn/convvae/paderborn_convvae_readiness.md`
- `reports/paderborn/convvae/paderborn_convvae_runner_integration.md`
- `reports/paderborn/convvae/paderborn_convvae_baseline_results.md`

Searches were also run for `svdd`, `deep_svdd`, `DeepSVDD`, `deepsvdd`, `one_class`, and `oneclass` across `scripts/`, `artifacts/`, and `reports/`, excluding binary checkpoint/array/archive payloads where appropriate.

## Direct Answers

1. Deep SVDD is not implemented anywhere found in the inspected repository.
2. Yes, there are reusable 1D-CNN encoders and convolutional blocks, but none is currently packaged as a Deep SVDD encoder.
3. No saved Deep SVDD checkpoints, score arrays, metrics, or artifact directories were found.
4. Yes, the unified runner already has a modular baseline path that can support Deep SVDD with a new `deep_svdd` registry entry returning the existing `ScoreBundle`.
5. The next implementation should primarily modify `scripts/eval_paderborn_baselines_unified.py`. A small helper module/script is optional if keeping the SVDD model/training code outside the runner is preferred.

## Deep SVDD Existence

Deep SVDD does not currently exist as code or artifacts.

Evidence:

- No `svdd`, `deep_svdd`, `DeepSVDD`, or `deepsvdd` hits were found in `scripts/`, `artifacts/`, or `reports/`.
- The only one-class model code found is classical shallow OC-SVM:
  - `scripts/eval_paderborn_baselines_unified.py` uses `SGDOneClassSVM`.
  - `scripts/train_paderborn_baselines.py` uses `SGDOneClassSVM`.
  - `scripts/train_shallow_baselines.py` and `scripts/eval_cwru_load_shift.py` use `OneClassSVM`.
- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md` lists implemented unified models as `ocsvm`, `isolation_forest`, `compact_ae`, `resdilated_ae`, and `conv_vae`.

## Reusable Model And Training Code

Reusable pieces exist, but Deep SVDD-specific pieces are missing.

Reusable data/evaluation infrastructure:

- `TorchMemmapWindowDataset` and `make_torch_loader()` in `scripts/eval_paderborn_baselines_unified.py` already load Paderborn windows as `[batch, 1, window]` tensors.
- `RunContext`, `DatasetInfo`, `ScoreBundle`, `validate_score_bundle()`, `save_threshold_artifacts()`, `write_summaries()`, and `write_reproduction_check()` provide the runner protocol Deep SVDD should reuse.
- `save_threshold_artifacts()` already calibrates thresholds from healthy validation scores only, writes `val_healthy_scores.npy`, `test_scores.npy`, `test_labels.npy`, `metrics.json`, and `run_config.json`, and records the no-fault-training/thresholding protocol.
- `set_global_seed()`, `select_device()`, and `effective_batch_size()` are already available for deterministic neural runs.

Reusable 1D-CNN model code:

- `scripts/train_ae_baseline.py` defines `CompactConvAutoencoder.encoder`, a compact Conv1d stack ending in 32 channels after three pooling stages.
- `scripts/train_generative_upgrades.py` defines reusable residual/dilated building blocks and `ConvVAE.encoder`, which already has an `encode()` method returning latent `mu` and `logvar`.
- `scripts/train_generative_upgrades.py` defines `ResDilatedAE`, whose encoder/downsample/bottleneck path could inspire a stronger SVDD backbone, but it is not separated as a clean encoder class.
- `scripts/train_paderborn_ablation.py` contains additional configurable CompactAE and ResDilatedAE variants that could inform architecture choices, though they are ablation code rather than the main baseline path.

Best reuse candidate:

- For a first clean Deep SVDD baseline, reuse the runner infrastructure and create a dedicated `DeepSVDDEncoder1D` instead of trying to slice an autoencoder at runtime. The architecture can borrow the CompactAE encoder or the ConvVAE residual encoder pattern, then pool/project to a fixed embedding vector.

## Missing Components

Deep SVDD needs these pieces before it can be run:

- A 1D encoder that maps each window to a fixed embedding vector.
- Center initialization `c` from healthy training embeddings.
- Deep SVDD training objective, likely mean squared distance to center for one-class SVDD.
- Optional soft-boundary support with `nu` and radius `R`, if desired.
- Validation/test scoring as squared distance from embeddings to center, with larger score meaning more abnormal.
- Checkpoint format containing encoder state, center, radius/settings, seed, architecture settings, and training metadata.
- Unified runner CLI args for SVDD hyperparameters, for example embedding dimension, epochs, learning rate, weight decay, dropout, and possibly `nu`.
- A `run_deep_svdd()` path that trains or loads Deep SVDD, computes validation/test scores, and returns `ScoreBundle`.
- Reuse/skip behavior for existing Deep SVDD unified artifacts and saved checkpoints.
- Report text updates so generated runner reports include Deep SVDD.

## Artifact Status

No Deep SVDD artifacts were found.

Current `artifacts/paderborn_unified_baselines/` model directories:

- `compact_ae`
- `conv_vae`
- `isolation_forest`
- `ocsvm`
- `resdilated_ae`

Current top-level artifact families:

- `generative_upgrades`
- `metrics`
- `models`
- `paderborn_ablation`
- `paderborn_ablation_smoke`
- `paderborn_unified_baselines`
- `paper_package_v1`
- `plots`

`artifacts/models/` contains only:

- `cwru_ae_baseline.pt`
- `paderborn_ae_baseline.pt`

No artifact filename or directory name matching SVDD/DeepSVDD was found. No unified `deep_svdd` summary rows were found in `summary.csv` or `summary_by_model.csv`.

## Unified Runner Readiness

The unified runner is ready to host Deep SVDD structurally.

Relevant existing design:

- `DEFAULT_MODELS` currently includes `ocsvm`, `isolation_forest`, `compact_ae`, `resdilated_ae`, and `conv_vae`.
- `main()` registers models through a dictionary mapping model names to callables.
- Every model runner returns a `ScoreBundle` with `val_healthy_scores`, `test_healthy_scores`, `test_fault_scores`, `score_source`, `model_settings`, and `extra_artifacts`.
- `run_model_seed()` handles per-model/per-seed errors without stopping the whole run.
- `save_threshold_artifacts()` centralizes threshold calibration and final metric/report artifact writing.
- Neural examples already exist:
  - `compact_ae` trains or loads a compact AE and computes reconstruction errors.
  - `resdilated_ae` loads saved score arrays or infers scores from checkpoints.
  - `conv_vae` loads saved score arrays or infers scores from checkpoints.

Deep SVDD can fit the same path by returning distance-to-center scores instead of reconstruction errors.

## Recommended Next Implementation Steps

1. Add Deep SVDD support to `scripts/eval_paderborn_baselines_unified.py`.
2. Add `deep_svdd` to `DEFAULT_MODELS` and the `registry` in `main()`.
3. Add Deep SVDD CLI settings, keeping defaults conservative and comparable to the existing neural baselines.
4. Add a dedicated `DeepSVDDEncoder1D` using a compact 1D-CNN encoder with adaptive pooling and a small projection head.
5. Add helpers for:
   - `compute_deep_svdd_center()`
   - `train_deep_svdd()`
   - `compute_deep_svdd_scores()`
   - checkpoint save/load
6. Implement `run_deep_svdd()` so it trains on healthy training windows only, initializes/fixes center from healthy train embeddings, scores healthy validation/test/fault windows, and returns `ScoreBundle`.
7. Write Deep SVDD checkpoints under `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/deep_svdd.pt` or a similarly explicit path.
8. Let the existing `save_threshold_artifacts()` create per-threshold unified outputs under `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/`.
9. Run a compile check first, then a narrow CPU smoke run for one seed only. Full training should be a later chunk.

## Files Likely To Modify Next

Required:

- `scripts/eval_paderborn_baselines_unified.py`

Optional, depending on preferred organization:

- `scripts/train_deep_svdd_baseline.py` if Deep SVDD should live outside the unified runner and be imported like a reusable baseline module.
- `reports/README.md` only if a new model-specific report folder such as `reports/paderborn/deepsvdd/` is introduced.
- `reports/paderborn/baselines/paderborn_unified_baseline_runner.md` will be regenerated or updated after implementation/validation.

Do not modify the paper until Deep SVDD has a completed unified run, validated metrics, and a clear comparison against the existing unified baselines.
