# Paderborn Unified Baseline Reproduction

## What Was Updated
- `scripts/eval_paderborn_baselines_unified.py` now writes the original per-run `summary.csv` plus `summary_by_model.csv`, `latex_table_by_model.tex`, and `reproduction_check.md`.
- The runner keeps per-model/per-seed failure handling: one missing checkpoint, score file, or model failure is recorded as an error row while other runs continue.
- The reproduction check verifies threshold calibration from `val_healthy_scores.npy`, binary test labels, FAR recomputation on healthy test windows only, and the run-config flag that fault data was not used for training or thresholding.

## Manual Commands
Compile check:

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\eval_paderborn_baselines_unified.py
```

Quick smoke reuse check:

```powershell
.\.venv\Scripts\python.exe scripts\eval_paderborn_baselines_unified.py --models isolation_forest --threshold-rule percentile_99_5 --seeds 42 --device cpu --skip-train-if-artifacts-exist
```

Full manual reproduction run:

```powershell
.\.venv\Scripts\python.exe scripts\eval_paderborn_baselines_unified.py --models compact_ae ocsvm isolation_forest resdilated_ae --threshold-rule percentile_99_5 --seeds 42 7 123 --device cuda --skip-train-if-artifacts-exist
```

CPU fallback:

```powershell
.\.venv\Scripts\python.exe scripts\eval_paderborn_baselines_unified.py --models compact_ae ocsvm isolation_forest resdilated_ae --threshold-rule percentile_99_5 --seeds 42 7 123 --device cpu --skip-train-if-artifacts-exist
```

## Expected Output Files
- `artifacts/paderborn_unified_baselines/summary.csv`
- `artifacts/paderborn_unified_baselines/summary_by_model.csv`
- `artifacts/paderborn_unified_baselines/summary.md`
- `artifacts/paderborn_unified_baselines/latex_table.tex`
- `artifacts/paderborn_unified_baselines/latex_table_by_model.tex`
- `artifacts/paderborn_unified_baselines/reproduction_check.md`
- Per run: `metrics.json`, `val_healthy_scores.npy`, `test_scores.npy`, `test_labels.npy`, and `run_config.json`

## Resume If A Model Fails
- Rerun the same command first. Completed unified run directories are reused when `--skip-train-if-artifacts-exist` is present.
- To retry only one failed slice, run with a narrower target, for example:

```powershell
.\.venv\Scripts\python.exe scripts\eval_paderborn_baselines_unified.py --models resdilated_ae --threshold-rule percentile_99_5 --seeds 123 --device cuda --skip-train-if-artifacts-exist
```

- Use `--force` only when you intentionally want to recompute existing unified outputs for the selected model/seed/rule.

## Verify Results
- Open `artifacts/paderborn_unified_baselines/reproduction_check.md`; every completed run should show `PASS`.
- Confirm `summary_by_model.csv` has one aggregate row per model and threshold rule.
- Confirm each successful `run_config.json` records:
  - `training`: `healthy_train_windows_only`
  - `threshold_calibration`: `healthy_validation_scores_only`
  - `fault_data_used_for_training_or_threshold`: `false`
- For a spot check, recompute FAR as:

```python
import json
from pathlib import Path
import numpy as np

run_dir = Path("artifacts/paderborn_unified_baselines/isolation_forest/seed_42/percentile_99_5")
metrics = json.loads((run_dir / "metrics.json").read_text())
scores = np.load(run_dir / "test_scores.npy")
labels = np.load(run_dir / "test_labels.npy")
far = float(((scores >= metrics["threshold"]) & (labels == 0)).sum() / (labels == 0).sum())
print(far, metrics["far"])
```
