# Paderborn Ablation Chunk 3 Plan

## Summary

Create the evaluation-only Chunk 3 tooling for saved Chunk 2 runs. The evaluator loads per-run saved scores, computes consistent threshold-free and thresholded metrics, optionally computes missing training scores by inference from `best.pt`, and writes only evaluation outputs under `{artifacts_root}/evaluation/`. It does not train models or change preprocessing, final model artifacts, paper files, or package builders.

## Files Created Or Edited

- Create/update `reports/paderborn/ablation/paderborn_ablation_chunk3_plan.md`
- Create `scripts/eval_paderborn_ablation.py`
- Do not edit preprocessing scripts, final model artifacts, `artifacts/generative_upgrades/`, paper LaTeX tables, or `build_paper_package.py`

## Key Changes

- Add a CLI evaluator with:
  - `--artifacts-root`
  - `--variants`
  - `--seeds`
  - `--threshold-rules`
  - `--main-threshold-rule`
  - `--compute-train-scores-if-missing`
  - `--allow-missing`
- Reuse Chunk 2 model definitions by importing `VARIANT_CONFIGS`, `build_model`, `MemmapWindowDataset`, `compute_reconstruction_scores`, and `load_torch_payload` from `scripts/train_paderborn_ablation.py`.
- Load required per-run files from `{artifacts_root}/{variant}/seed_{seed}/`:
  - `best.pt`
  - `run_config.json`
  - `val_healthy_scores.npy`
  - `test_healthy_scores.npy`
  - `test_fault_scores.npy`
  - `sanity_metrics.json`
- Implement threshold rules:
  - `val_p99_5`: `np.percentile(val_healthy_scores, 99.5)`
  - `val_mean_plus_3std`: validation mean plus three std; this is validation-calibrated
  - `train_mean_plus_3std`: training-score mean plus three std; this is the no-validation-calibration comparator
- If `train_healthy_scores.npy` is missing and `--compute-train-scores-if-missing` is supplied, run inference only on `data/processed/paderborn/train/healthy_windows.npy`, save the training scores in the run directory, and continue.

## Metrics And Outputs

- Build binary labels from scores: test healthy `0`, test fault `1`.
- Compute AUROC/AUPRC once per run from raw scores.
- For each threshold rule, compute predictions with `score > threshold`.
- Compute F1, precision, recall fault, and FAR from confusion counts.
- Precision uses `1.0` when there are no predicted positives.
- Write or overwrite only these files inside `{artifacts_root}/evaluation/`:
  - `per_seed_metrics.csv`
  - `ablation_summary_val_p99_5.csv`
  - `calibration_comparison_resdilated_full.csv`
  - `evaluation_report.md`
  - `metrics_summary.json`
- Use sample std across seeds when at least two seeds are present; otherwise std is `0.0`.
- With `--allow-missing`, skip missing runs and report them clearly. Without it, fail on the first missing run or required score file.

## Test Plan

```powershell
python -m py_compile scripts\eval_paderborn_ablation.py
```

```powershell
python scripts\eval_paderborn_ablation.py --artifacts-root artifacts/paderborn_ablation_smoke --variants compact_ae dilated_ae res_ae resdilated_time --seeds 42 --threshold-rules val_p99_5 val_mean_plus_3std --main-threshold-rule val_p99_5 --allow-missing
```

Confirm the smoke command writes the five evaluation outputs under `artifacts/paderborn_ablation_smoke/evaluation/` and does not create `train_healthy_scores.npy` because the smoke command does not request `train_mean_plus_3std`.

## Assumptions

- Full production evaluation will be run later with all five variants and seeds `42 7 123`.
- `val_mean_plus_3std` is validation-calibrated, not a no-calibration result.
- `train_mean_plus_3std` is the no-validation-calibration comparator.
- `calibration_comparison_resdilated_full.csv` is written with headers even if `resdilated_full` is not available in a smoke run.
- Verification uses `local CUDA environment` because the local `.venv` currently has a PyTorch DLL load issue.
