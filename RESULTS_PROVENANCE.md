# Results Provenance

## CWRU Table I

- `paper_latex/tables/cwru_results_table.tex`: tracked manuscript table.
- `scripts/build_paper_package.py`: tracked package/table generation script.
- `artifacts/metrics/cwru_load_shift_metrics.json`: tracked compact baseline load-shift metrics.
- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_metrics.json`: generated locally and intentionally omitted from Git; copied into `artifacts/paper_package_v1/02_final_metrics/cwru/cwru_resdilated_load_shift_metrics.json` for compact provenance.

## Paderborn Table II

- `paper_latex/tables/paderborn_results_table.tex`: tracked manuscript table.
- `artifacts/metrics/paderborn_*_metrics.json`: tracked compact baseline metric JSON files.
- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`: generated locally and intentionally omitted from Git; copied into `artifacts/paper_package_v1/02_final_metrics/paderborn/paderborn_resdilated_threshold_calibration_metrics.json` for compact provenance.

## Paderborn Ablation Table III

- `artifacts/paderborn_ablation/evaluation/ablation_summary_val_p99_5.csv`: generated locally and intentionally omitted under the generated-artifact policy.
- `artifacts/paderborn_ablation/evaluation/metrics_summary.json`: generated locally and intentionally omitted under the generated-artifact policy.
- `artifacts/paderborn_ablation/evaluation/evaluation_report.md`: generated locally and intentionally omitted under the generated-artifact policy.
- `scripts/train_paderborn_ablation.py` and `scripts/eval_paderborn_ablation.py`: tracked scripts used to regenerate the ablation outputs.

## Deployment Metrics

- `scripts/eval_paderborn_deployment_metrics.py`: tracked deployment-metric script.
- `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json`: generated locally and intentionally omitted from Git.
- `artifacts/paper_package_v1/05_candidate_tables/table_deployment_summary.*`: tracked compact paper-package table files.

## Final Figures

- `paper_latex/figures/paderborn_threshold_calibration_clean.png`: tracked final figure.
- `paper_latex/figures/explanation_hardest_condition_clean.png`: tracked final figure.
- `scripts/clean_paper_figures.py`: tracked figure-cleanup script.
- `scripts/explain_paderborn_resdilated.py`: tracked explanation-generation script.
- Intermediate candidate figures under `artifacts/paper_package_v1/04_candidate_figures/` and model-run explanation outputs are generated locally and intentionally omitted.
