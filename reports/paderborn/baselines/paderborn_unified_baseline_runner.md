# Paderborn Unified Baseline Runner

## What Was Implemented
- Added `scripts/eval_paderborn_baselines_unified.py` as a shared healthy-only runner for Paderborn baselines.
- Added common data discovery, deterministic seeding, scoring, validation-only threshold calibration, metrics, and reporting.
- Implemented `ocsvm`, `isolation_forest`, `compact_ae`, `resdilated_ae`, `conv_vae`, and `deep_svdd` through a model registry pattern.
- ConvVAE reuses existing `scripts/train_generative_upgrades.py` model and checkpoint-scoring helpers.
- Deep SVDD uses a compact 1D-CNN encoder, fixed healthy-train center, and squared distance-to-center anomaly scores.

## Existing Data And Project Paths Reused
- `train_healthy`: `data/processed/paderborn/train/healthy_windows.npy`
- `val_healthy`: `data/processed/paderborn/val/healthy_windows.npy`
- `test_healthy`: `data/processed/paderborn/test/healthy_windows.npy`
- `test_fault`: `data/processed/paderborn/test/fault_windows.npy`
- `fault_labels`: `data/processed/paderborn/test/fault_labels.npy`
- `preprocessing_config`: `data/metadata/paderborn/preprocessing_config.json`

## Models Ran Successfully
- compact_ae, conv_vae, deep_svdd, isolation_forest, memae, ocsvm, resdilated_ae

## Models Not Run In Latest Smoke Summary
- none

## Models Needing Checkpoints Or Further Work
- none in the latest run

## Smoke Test Commands
- Shallow smoke test: `python scripts/eval_paderborn_baselines_unified.py --models isolation_forest --threshold-rule percentile_99_5 --seed 42 --skip-train-if-artifacts-exist`
- Neural smoke test: `python scripts/eval_paderborn_baselines_unified.py --models resdilated_ae --threshold-rule percentile_99_5 --seed 42 --device cpu --skip-train-if-artifacts-exist`
- Combined summary refresh: `python scripts/eval_paderborn_baselines_unified.py --models isolation_forest resdilated_ae --threshold-rule percentile_99_5 --seed 42 --device cpu --skip-train-if-artifacts-exist`

## Latest Command Observed By Runner
- `python scripts/eval_paderborn_baselines_unified.py --models all --threshold-rule percentile_99_5 --seeds 42 7 123 --device auto`

## Output Artifacts
- Summary CSV: `artifacts/paderborn_unified_baselines/summary.csv`
- Summary by model CSV: `artifacts/paderborn_unified_baselines/summary_by_model.csv`
- Summary MD: `artifacts/paderborn_unified_baselines/summary.md`
- LaTeX table: `artifacts/paderborn_unified_baselines/latex_table.tex`
- LaTeX by model table: `artifacts/paderborn_unified_baselines/latex_table_by_model.tex`
- Reproduction check: `artifacts/paderborn_unified_baselines/reproduction_check.md`
