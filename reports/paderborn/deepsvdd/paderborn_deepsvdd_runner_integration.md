# Paderborn Deep SVDD Runner Integration

## Scope

Chunk 2 adds Deep SVDD as a selectable unified-runner model placeholder. It does not implement Deep SVDD training, scoring, checkpointing, or paper updates.

## Files Changed

- `scripts/eval_paderborn_baselines_unified.py`
- `reports/paderborn/deepsvdd/paderborn_deepsvdd_runner_integration.md`

## Runner Changes

- Added `EXPERIMENTAL_MODELS = ("deep_svdd",)`.
- Added `SUPPORTED_MODELS = (*DEFAULT_MODELS, *EXPERIMENTAL_MODELS)`.
- Kept `DEFAULT_MODELS` unchanged so `--models all` still runs only implemented baselines.
- Updated `expand_models()` so `--models deep_svdd` is accepted explicitly.
- Registered `"deep_svdd": run_deep_svdd` in the model registry.
- Added `run_deep_svdd(context, seed, _feature_cache)` with the same signature style as the other model runners.

## CLI Args Added

- `--svdd-embedding-dim`, default `64`
- `--svdd-lr`, default `1e-3`
- `--svdd-weight-decay`, default `1e-6`
- `--svdd-epochs`, default `8`
- `--svdd-dropout`, default `0.0`

These arguments are parsed and included in the placeholder error message, but they do not trigger any training yet.

## Selection Status

`deep_svdd` can now be selected explicitly:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models deep_svdd --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

Because it is not in `DEFAULT_MODELS`, it is not included by `--models all` yet. That keeps routine reproduction runs from producing expected Deep SVDD placeholder failures before the implementation exists.

## Current Expected Behavior

Calling `deep_svdd` now reaches the unified runner's normal per-model execution path and raises a clear `NotImplementedError` from `run_deep_svdd()`.

The existing `run_model_seed()` error handler catches this exception and records a clean error row instead of crashing the full runner with a traceback. The row should report:

- `model`: `deep_svdd`
- `seed`: requested seed, for example `42`
- `threshold_rule`: requested threshold rule, for example `percentile_99_5`
- `status`: `error`
- `error`: message stating that Deep SVDD is selectable but training/scoring is not implemented yet and training is required

No Deep SVDD training is run, and no Deep SVDD score arrays/checkpoints are produced by this placeholder.

## Protocol Preservation

The placeholder does not alter existing protocol code. The eventual implementation should return a `ScoreBundle`, after which the existing centralized path will still enforce:

- healthy train windows only
- healthy validation scores only for threshold calibration
- healthy test plus fault test windows for final evaluation
- `percentile_99_5` threshold support
- larger score means more abnormal
- centralized metrics and threshold artifact saving
- reproduction checks based on validation-only thresholds and no fault data for training or thresholding

## Validation

Compile check passed:

```powershell
python -m py_compile scripts/eval_paderborn_baselines_unified.py
```

Placeholder selection command:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models deep_svdd --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

Observed behavior:

- The command accepted `deep_svdd` as a model name.
- The runner discovered the Paderborn data and entered the normal per-model/per-seed execution path.
- `run_deep_svdd()` raised the expected not-implemented/training-required message.
- The existing runner error handler recorded one clean error row.
- No traceback occurred.
- No Deep SVDD training ran.
- No Deep SVDD checkpoint or score arrays were produced.
- Process exit code was `1` because the selected run had zero successful/reused rows and one error row. This matches the runner's existing all-selected-models-failed return policy.

The validation command refreshed the default unified summary files, and `artifacts/paderborn_unified_baselines/summary.csv` now contains the expected placeholder row:

- `model`: `deep_svdd`
- `seed`: `42`
- `threshold_rule`: `percentile_99_5`
- `status`: `error`
- `error`: Deep SVDD is selectable, but training/scoring is not implemented yet and training is required.

## Next Implementation Task

Implement the real Deep SVDD model path:

1. Add a 1D-CNN encoder that maps each window to a fixed embedding.
2. Initialize the Deep SVDD center from healthy training embeddings.
3. Train on healthy training windows only with distance-to-center loss.
4. Score validation/test windows by squared distance to the center.
5. Save/load `deep_svdd.pt` checkpoints.
6. Return a real `ScoreBundle` from `run_deep_svdd()` so existing unified thresholding, metrics, and reports can run unchanged.
