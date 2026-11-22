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

`--models all` in the Paderborn Baseline Evaluation command above includes MemAE once the three
seeds have been trained; its rows land in `summary_by_model.csv` next to every other comparator.

Before the three-seed run, the `lambda` sweep (seed 42 only) and the memory-disabled control
(all three seeds) back the fairness paragraphs in the paper. Both are separate, non-canonical
`--artifacts-root` trees so they cannot overwrite the seed runs above:

```powershell
foreach ($lam in 0.002, 0.004, 0.006) {
    python scripts/train_generative_upgrades.py --model memae --seed 42 `
        --memae-shrink-threshold $lam --artifacts-root artifacts/memae_lambda_sweep/lam_$lam
}

foreach ($seed in 42, 7, 123) {
    python scripts/train_generative_upgrades.py --model memae --seed $seed `
        --memae-shrink-threshold 0 --memae-entropy-weight 0 --artifacts-root artifacts/memae_control
}
```

Comparative analysis against ResDilatedAE-T, after the unified evaluation has been run with
`--models all`:

```powershell
python scripts/eval_paderborn_deployment_metrics.py --include-memae --memae-seed 42 --output-dir artifacts/generative_upgrades/memae/analysis --output-stem memae_deployment
python scripts/compare_memae_resdilated.py
```

`compare_memae_resdilated.py` reads only saved per-window score arrays, so it retrains nothing.
It writes FAR-matched recall tables, the memory-disabled control contrast, the `λ` sweep, and an
analysis note to `artifacts/generative_upgrades/memae/analysis/`. Its FAR-matched thresholds come
in two flavours: `val_fitted` obeys the validation-only calibration rule and is the version that
may enter the paper, while `test_oracle` reads test healthy scores and is a diagnostic bound only.

## Paderborn Ablation

```powershell
python scripts/train_paderborn_ablation.py --seeds 42 7 123
python scripts/eval_paderborn_ablation.py
```

## Deployment Metrics

```powershell
python scripts/eval_paderborn_deployment_metrics.py
```

Add `--include-memae` to profile the MemAE comparator in the same process, on the same benchmark
pool and thread budget. Send those runs elsewhere with `--output-dir` / `--output-stem` so the
tracked ResDilatedAE-T deployment numbers are not overwritten by a comparison run.

## Paper Package And Tables

```powershell
python scripts/build_paper_package.py
```

## Camera-Ready Revision (ICCIT 2026)

These commands regenerate the camera-ready results under `artifacts/v3/`. They were run on Linux
with a CUDA GPU; the shallow baselines and the analysis scripts run on CPU.

Data and transfer folds (the main Paderborn split comes from the Preprocessing section above):

```bash
bash scripts/download_datasets.sh
python scripts/preprocess_cwru.py --file-map data/metadata/cwru48k/file_map.csv --raw-root data/raw/cwru48k --processed-root data/processed/cwru48k --metadata-root data/metadata/cwru48k
python scripts/build_paderborn_transfer_splits.py
```

CWRU, 48 kHz, leave-one-load-out (Table I):

```bash
D="--processed-root data/processed/cwru48k --metadata-root data/metadata/cwru48k"
R=artifacts/v3/cwru48k_full_recording
for s in 42 7 123 2024 2026; do
  python scripts/eval_cwru_resdilated_load_shift.py $D --held-out-loads 0 1 2 3 --seed $s --epochs 50 --learning-rate 0.0003 --weight-decay 0.0001 --patience 8 --freq-loss-weight 0.1 --dropout 0.05 --save-every-epochs 1 --threshold-rule mean_plus_3std --healthy-test-scope full_recording --artifacts-root $R
  python scripts/eval_cwru_load_shift.py $D --seed $s --healthy-test-scope full_recording --artifacts-root $R/baselines_seed_$s
  python scripts/eval_cwru_load_shift.py $D --models classical --classical-fs 48000 --classical-band-low-hz 2000 --classical-band-high-hz 10000 --healthy-test-scope full_recording --seed $s --artifacts-root artifacts/v3/cwru48k_full_recording_classical/seed_$s
done
python scripts/analyze_cwru_v3.py --root $R --classical-root artifacts/v3/cwru48k_full_recording_classical --output artifacts/v3/analysis/cwru48k_summary.json
```

Paderborn models (Tables II and IV):

```bash
S5="42 7 123 2024 2026"; S3="42 7 123"
for v in resdilated_time compact_ae; do for s in $S5; do
  python scripts/train_paderborn_ablation.py --variants $v --seeds $s --artifacts-root artifacts/v3/paderborn_ablation
done; done
for v in res_time dilated_time compact_ae_freq resdilated_full res_ae dilated_ae; do for s in $S3; do
  python scripts/train_paderborn_ablation.py --variants $v --seeds $s --artifacts-root artifacts/v3/paderborn_ablation
done; done
python scripts/eval_paderborn_ablation.py --variants resdilated_time --seeds $S5 --compute-train-scores-if-missing --artifacts-root artifacts/v3/paderborn_ablation
for s in $S5; do
  python scripts/train_generative_upgrades.py --model memae --seed $s --artifacts-root artifacts/v3/generative
  python scripts/train_generative_upgrades.py --model conv_vae --seed $s --artifacts-root artifacts/v3/generative
done
for s in $S3; do
  python scripts/train_generative_upgrades.py --model memae --seed $s --memae-shrink-threshold 0 --memae-entropy-weight 0 --artifacts-root artifacts/v3/generative_memae_control
done
python scripts/eval_paderborn_baselines_unified.py --models isolation_forest ocsvm ocsvm_rbf classical --threshold-rule percentile_99_5 --seeds $S5 --device cpu --output-root artifacts/v3/paderborn_baselines
python scripts/eval_paderborn_baselines_unified.py --models deep_svdd conv_vae memae --threshold-rule percentile_99_5 --seeds $S5 --device auto --output-root artifacts/v3/paderborn_baselines --source-artifacts-root artifacts/v3/generative
python scripts/eval_paderborn_baselines_unified.py --models memae --threshold-rule percentile_99_5 --seeds $S3 --device auto --output-root artifacts/v3/paderborn_baselines_memae_control --source-artifacts-root artifacts/v3/generative_memae_control
```

Paderborn transfer protocols (Table III):

```bash
for f in unseen_bearing/fold_0 unseen_bearing/fold_1 unseen_bearing/fold_2 loco/fold_0 loco/fold_1 loco/fold_2 loco/fold_3; do
  D="--processed-root data/processed/paderborn_v3/$f --metadata-root data/metadata/paderborn_v3/$f"
  A=artifacts/v3/paderborn_transfer/$f/paderborn_ablation
  for s in $S3; do
    python scripts/train_paderborn_ablation.py --variants resdilated_time --seeds $s $D --artifacts-root $A
  done
  python scripts/eval_paderborn_ablation.py --variants resdilated_time --seeds $S3 --compute-train-scores-if-missing $D --artifacts-root $A
  python scripts/eval_paderborn_baselines_unified.py --models isolation_forest classical --threshold-rule percentile_99_5 --seeds $S3 --device cpu $D --output-root artifacts/v3/paderborn_transfer/$f/baselines
done
python scripts/eval_paderborn_loco_commissioning.py
```

Summaries, statistics, Fig. 2, and checks. `--significance-exclude` limits the Holm family to the
baselines of Table II:

```bash
python scripts/analyze_paderborn_v3.py --alarm-rule gt --significance-exclude baseline/resdilated_ae baseline/ocsvm ablation/compact_ae_freq ablation/dilated_time ablation/dilated_ae ablation/res_time ablation/res_ae ablation/resdilated_full
python scripts/check_dc_offset_v3.py
python scripts/profile_v3_cpu.py
python scripts/make_fig2_v3.py
```
