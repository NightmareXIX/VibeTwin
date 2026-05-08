# Paderborn Deep SVDD Training Implementation

## Scope

Chunk 3 implements from-scratch 1D Deep SVDD training and evaluation inside the unified Paderborn baseline runner. No preprocessing, split, label, threshold, metric, normalization, or paper files were changed.

## Files Changed

- `scripts/eval_paderborn_baselines_unified.py`
- `reports/paderborn/deepsvdd/paderborn_deepsvdd_training_implementation.md`

## Runner Implementation

Added `DeepSVDDEncoder1D`, a compact 1D-CNN encoder:

- input: `[batch, 1, 2048]`
- Conv1d `1 -> 16`, kernel `9`, stride `2`
- GroupNorm, LeakyReLU, MaxPool1d
- Conv1d `16 -> 32`, kernel `7`
- GroupNorm, LeakyReLU, MaxPool1d
- Conv1d `32 -> 64`, kernel `5`
- GroupNorm, LeakyReLU
- AdaptiveAvgPool1d(1)
- flatten
- optional dropout
- linear projection to `embedding_dim`, default `64`

`deep_svdd` is now in `DEFAULT_MODELS` and the registry.

## CLI Support

Deep SVDD-specific args:

- `--svdd-embedding-dim`, default `64`
- `--svdd-lr`, default `1e-3`
- `--svdd-weight-decay`, default `1e-6`
- `--svdd-epochs`, default `8`
- `--svdd-dropout`, default `0.0`

Smoke-test epoch override:

- `--max-epochs`, optional. For Deep SVDD, the effective epoch count is `min(--svdd-epochs, --max-epochs)` when supplied.

## Training Protocol

Deep SVDD now uses the existing unified runner data path:

- training: healthy train windows only
- center initialization: mean encoder embedding over healthy train windows before SVDD optimization
- near-zero center dimensions are adjusted to `+/-0.1`
- center remains fixed during optimization
- loss: mean squared distance from `encoder(x)` to center `c`
- optimizer: Adam
- checkpoint selection: lowest healthy-validation mean SVDD score
- score direction: larger squared distance means more abnormal

## Scoring

Scores are computed as:

```text
score(x) = || encoder(x) - c ||^2
```

The returned `ScoreBundle` contains:

- healthy validation scores
- healthy test scores
- fault test scores
- `score_source`
- model settings
- extra artifact paths

The existing `save_threshold_artifacts()` then writes the official per-threshold artifacts and metrics.

## Checkpoint And Artifacts

Checkpoint path:

- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/deep_svdd.pt`

Checkpoint includes:

- model name
- model state dict
- center `c`
- seed
- embedding dimension
- model settings
- training settings
- history
- best epoch
- best validation mean score
- parameter count
- healthy-only data protocol marker

Model-level split score cache:

- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/val_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/test_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/test_fault_scores.npy`

Official threshold artifacts from the existing unified writer:

- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/metrics.json`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/run_config.json`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/val_healthy_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/test_scores.npy`
- `artifacts/paderborn_unified_baselines/deep_svdd/seed_<seed>/percentile_99_5/test_labels.npy`

## Score Sources

`run_deep_svdd()` can now return:

- `trained_deep_svdd`
- `loaded_deep_svdd_checkpoint`
- `existing_deep_svdd_scores`

With `--skip-train-if-artifacts-exist`, the runner first reuses complete threshold outputs through the existing top-level path. If it enters `run_deep_svdd()`, it can reuse cached split scores or load `deep_svdd.pt` and recompute scores.

## Validation

Compile check passed:

```powershell
python -m py_compile scripts/eval_paderborn_baselines_unified.py
```

The default `python` interpreter does not have PyTorch installed:

```text
C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
torch_import_error ModuleNotFoundError("No module named 'torch'")
```

CUDA is available in the project CUDA environment:

```text
python
torch 2.11.0+cu128
cuda_available True
```

One-epoch CUDA smoke command run:

```powershell
python scripts\eval_paderborn_baselines_unified.py --models deep_svdd --threshold-rule percentile_99_5 --seeds 42 --device cuda --max-epochs 1 --force
```

Observed training:

- Deep SVDD epoch `1/1`
- train loss: `0.0127795`
- healthy validation mean score: `0.00113662`
- summary rows: `1`
- successful/reused: `1`
- errors: `0`

Smoke metrics:

- AUROC: `0.837016`
- AUPRC: `0.994002`
- F1: `0.647591`
- precision: `0.999654`
- recall fault: `0.478922`
- FAR: `0.005093`
- threshold: `0.004823`
- score source: `trained_deep_svdd`

Artifact checks passed:

- `deep_svdd.pt` exists.
- `val_healthy_scores.npy` exists.
- `test_scores.npy` exists.
- `test_labels.npy` exists.
- `metrics.json` exists.
- Test labels contain both `0` and `1`.
- Threshold metadata records `fit_split = val_healthy` and `fit_count = 16884`.
- FAR recomputation matched `metrics.json` exactly.
- `reproduction_check.md` reports `PASS`.

Independent spot check:

- threshold delta: `7.22e-11`
- FAR delta: `0.0`
- labels: `[0, 1]`
- validation score count: `16884`
- test score count: `535968`

## Next Task

Do not run full three-seed training yet. The next chunk should decide the intended full-run command and whether to keep the current compact architecture/hyperparameters or tune Deep SVDD before producing the paper-ready three-seed baseline.
