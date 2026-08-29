# Run Commands

Run commands from the repository root. Use Python 3.11 or 3.12 for the safest CPU-first setup; the release-prep machine reports Python 3.13.7, but Python 3.13 should be treated as unconfirmed unless the local dependency stack is tested.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Conda/mamba alternative:

```powershell
conda env create -f environment.yml
conda activate vibetwin-cpu
```

## Preprocessing

```powershell
python scripts/prepare_paderborn.py
python scripts/build_paderborn_label_map.py
python scripts/preprocess_cwru.py
python scripts/preprocess_paderborn.py
```

## CWRU Load-Shift Evaluation

```powershell
python scripts/eval_cwru_resdilated_load_shift.py --held-out-loads 0 1 2 3 --seed 42 --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --patience 8 --freq-loss-weight 0.1 --dropout 0.05 --save-every-epochs 1 --threshold-rule mean_plus_3std
```

## Paderborn Baseline Evaluation

```powershell
python scripts/eval_paderborn_baselines_unified.py --models all --threshold-rule percentile_99_5 --seeds 42 7 123 --device auto
```

For a single model:

```powershell
python scripts/eval_paderborn_baselines_unified.py --models isolation_forest --threshold-rule percentile_99_5 --seeds 42 7 123 --device cpu
```

## Paderborn ResDilatedAE Final Evaluation

```powershell
python scripts/train_generative_upgrades.py --model resdilated_ae --seed 42
python scripts/train_generative_upgrades.py --model resdilated_ae --seed 7
python scripts/train_generative_upgrades.py --model resdilated_ae --seed 123
python scripts/eval_paderborn_resdilated_threshold_calibration.py --seeds 42 7 123
```

## MemAE Comparator

```powershell
python scripts/verify_memae.py
python scripts/train_generative_upgrades.py --model memae --seed 42
python scripts/train_generative_upgrades.py --model memae --seed 7
python scripts/train_generative_upgrades.py --model memae --seed 123
```

`verify_memae.py` checks the memory mechanism and its trained behaviour, writing
`artifacts/metrics/memae_verification_metrics.json` and a matching report; it exits non-zero if
any check fails. Add `--skip-behavioural` to run the mechanism checks alone, which need no
processed windows.

## Paderborn Ablation

```powershell
python scripts/train_paderborn_ablation.py --seeds 42 7 123
python scripts/eval_paderborn_ablation.py
```

## Deployment Metrics

```powershell
python scripts/eval_paderborn_deployment_metrics.py
```

## Paper Package And Tables

```powershell
python scripts/build_paper_package.py
```
