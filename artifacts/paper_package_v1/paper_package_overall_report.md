# VibeTwin Paper Package Overall Report

## Package Root
- Package folder: `artifacts/paper_package_v1`
- Generated at: `2026-04-13 21:57:52`
- Scope: finalized, paper-relevant evidence only. Originals were copied, not moved or edited.
- Intentionally omitted from this package: checkpoints, raw score arrays, full window manifests, and non-final generative side paths such as ConvVAE and DenoisingResDilatedAE. Their original artifacts remain in place if we need them later.

## Folder Inventory
- Per-file source mapping is also saved in `paper_package_manifest.csv`.

### 01_source_docs
- `01_source_docs/cwru_metadata_readme.md` <- `data/metadata/cwru/README.md`. CWRU metadata readme for writing context.
- `01_source_docs/cwru_preprocessing_audit.md` <- `data/metadata/cwru/preprocessing_audit.md`. CWRU preprocessing and split audit.
- `01_source_docs/paderborn_bearing_label_map.md` <- `data/metadata/paderborn/bearing_label_map.md`. Paderborn label provenance reference.
- `01_source_docs/paderborn_prepare_report.md` <- `data/metadata/paderborn/prepare_report.md`. Paderborn file inventory and preparation notes.
- `01_source_docs/paderborn_preprocessing_report.md` <- `data/metadata/paderborn/preprocessing_report.md`. Primary dataset preprocessing and split audit.
- `01_source_docs/readme_versions.txt` <- `readme_versions.txt`. Local version note captured for writing context.

### 02_final_metrics
- `02_final_metrics/cwru/cwru_ae_metrics.json` <- `artifacts/metrics/cwru_ae_metrics.json`. CWRU compact AE baseline metrics.
- `02_final_metrics/cwru/cwru_baseline_load_shift_metrics.json` <- `artifacts/metrics/cwru_load_shift_metrics.json`. Earlier CWRU baseline load-shift metrics for context.
- `02_final_metrics/cwru/cwru_baseline_threshold_calibration_metrics.json` <- `artifacts/metrics/cwru_threshold_calibration_metrics.json`. Earlier CWRU threshold-calibration metrics for context.
- `02_final_metrics/cwru/cwru_iforest_metrics.json` <- `artifacts/metrics/cwru_iforest_metrics.json`. CWRU Isolation Forest baseline metrics.
- `02_final_metrics/cwru/cwru_ocsvm_metrics.json` <- `artifacts/metrics/cwru_ocsvm_metrics.json`. CWRU OC-SVM baseline metrics.
- `02_final_metrics/cwru/cwru_resdilated_load_shift_metrics.json` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_metrics.json`. Final ResDilatedAE CWRU harder load-shift metrics.
- `02_final_metrics/cwru/cwru_resdilated_threshold_calibration_metrics.json` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json`. Final ResDilatedAE CWRU calibration metrics.
- `02_final_metrics/paderborn/paderborn_ae_metrics.json` <- `artifacts/metrics/paderborn_ae_metrics.json`. Paderborn compact AE baseline metrics.
- `02_final_metrics/paderborn/paderborn_iforest_metrics.json` <- `artifacts/metrics/paderborn_iforest_metrics.json`. Paderborn Isolation Forest baseline metrics.
- `02_final_metrics/paderborn/paderborn_ocsvm_metrics.json` <- `artifacts/metrics/paderborn_ocsvm_metrics.json`. Paderborn OC-SVM baseline metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_deployment_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json`. Final Paderborn deployment metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_explanation_cases.json` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_cases.json`. Selected explanation case metadata.
- `02_final_metrics/paderborn/paderborn_resdilated_mc_dropout_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json`. Final Paderborn MC-dropout uncertainty metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_seed_123_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/seed_123/metrics.json`. Seed 123 ResDilatedAE run metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_seed_42_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/seed_42/metrics.json`. Seed 42 ResDilatedAE run metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_seed_7_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/seed_7/metrics.json`. Seed 7 ResDilatedAE run metrics.
- `02_final_metrics/paderborn/paderborn_resdilated_threshold_calibration_metrics.json` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`. Final Paderborn ResDilatedAE threshold-calibration metrics.

### 03_final_reports
- `03_final_reports/cwru/cwru_baseline_load_shift_report.md` <- `artifacts/metrics/cwru_load_shift_report.md`. Earlier CWRU baseline load-shift report for context.
- `03_final_reports/cwru/cwru_baseline_threshold_calibration_report.md` <- `artifacts/metrics/cwru_threshold_calibration_report.md`. Earlier CWRU calibration report for context.
- `03_final_reports/cwru/cwru_resdilated_load_shift_report.md` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_report.md`. Final CWRU ResDilatedAE harder load-shift report.
- `03_final_reports/cwru/cwru_resdilated_threshold_calibration_report.md` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_report.md`. Final CWRU ResDilatedAE threshold-calibration report.
- `03_final_reports/cwru/cwru_shallow_baseline_report.md` <- `artifacts/metrics/cwru_shallow_report.md`. Earlier CWRU shallow-baseline report for context.
- `03_final_reports/paderborn/paderborn_baseline_report.md` <- `artifacts/metrics/paderborn_baseline_report.md`. Baseline comparison report for the primary dataset.
- `03_final_reports/paderborn/paderborn_resdilated_deployment_report.md` <- `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_report.md`. Final deployment report.
- `03_final_reports/paderborn/paderborn_resdilated_explanation_report.md` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_report.md`. Final explanation report.
- `03_final_reports/paderborn/paderborn_resdilated_mc_dropout_report.md` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_report.md`. Negative-result uncertainty report.
- `03_final_reports/paderborn/paderborn_resdilated_seed_123_report.md` <- `artifacts/generative_upgrades/resdilated_ae/seed_123/report.md`. Chosen single-seed run used for explanation and deployment assets.
- `03_final_reports/paderborn/paderborn_resdilated_seed_comparison_report.md` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_seed_comparison_report.md`. Seed stability summary for final Paderborn model.
- `03_final_reports/paderborn/paderborn_resdilated_threshold_calibration_report.md` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_report.md`. Final Paderborn calibration report.

### 04_candidate_figures
- `04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_summary.png`. Main CWRU ResDilatedAE load-shift plot candidate.
- `04_candidate_figures/cwru/cwru_threshold_calibration_summary.png` <- `artifacts/plots/cwru_threshold_calibration_summary.png`. CWRU threshold-calibration context plot candidate.
- `04_candidate_figures/paderborn/paderborn_baseline_summary.png` <- `artifacts/plots/paderborn_baseline_summary.png`. Paderborn baseline score-distribution plot candidate.
- `04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png` <- `artifacts/generative_upgrades/resdilated_ae/seed_123/summary.png`. Compact Paderborn comparison plot candidate from the chosen seed.
- `04_candidate_figures/paderborn_explanations/01_tp_ka.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/01_tp_ka.png`. Explanation case figure: KA true positive.
- `04_candidate_figures/paderborn_explanations/02_tp_kb.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/02_tp_kb.png`. Explanation case figure: KB true positive.
- `04_candidate_figures/paderborn_explanations/03_tp_ki.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/03_tp_ki.png`. Explanation case figure: KI true positive.
- `04_candidate_figures/paderborn_explanations/04_hardest_condition_tp.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/04_hardest_condition_tp.png`. Explanation case figure: hardest operating condition true positive.
- `04_candidate_figures/paderborn_explanations/05_healthy_true_negative.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/05_healthy_true_negative.png`. Explanation case figure: healthy true negative.
- `04_candidate_figures/paderborn_explanations/06_healthy_false_positive.png` <- `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/06_healthy_false_positive.png`. Explanation case figure: healthy false positive.

### 05_candidate_tables
- `05_candidate_tables/table_cwru_final_vs_baselines.csv` <- `artifacts/metrics/cwru_load_shift_metrics.json + artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/*.json`. Generated compact CWRU comparison table.
- `05_candidate_tables/table_cwru_final_vs_baselines.md` <- `artifacts/metrics/cwru_load_shift_metrics.json + artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/*.json`. Generated markdown version of the CWRU comparison table.
- `05_candidate_tables/table_cwru_threshold_rules.csv` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json`. Generated cwru threshold-rule CSV table.
- `05_candidate_tables/table_cwru_threshold_rules.md` <- `artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json`. Generated cwru threshold-rule markdown table.
- `05_candidate_tables/table_deployment_summary.csv` <- `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json`. Generated deployment CSV table.
- `05_candidate_tables/table_deployment_summary.md` <- `artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json`. Generated deployment markdown table.
- `05_candidate_tables/table_paderborn_final_vs_baselines.csv` <- `artifacts/metrics/paderborn_*_metrics.json + artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`. Generated compact Paderborn comparison table.
- `05_candidate_tables/table_paderborn_final_vs_baselines.md` <- `artifacts/metrics/paderborn_*_metrics.json + artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`. Generated markdown version of the Paderborn comparison table.
- `05_candidate_tables/table_paderborn_seed_stability.csv` <- `artifacts/generative_upgrades/resdilated_ae/seed_{42,7,123}/metrics.json`. Generated per-seed stability table.
- `05_candidate_tables/table_paderborn_seed_stability.md` <- `artifacts/generative_upgrades/resdilated_ae/seed_{42,7,123}/metrics.json`. Generated markdown version of the per-seed stability table.
- `05_candidate_tables/table_paderborn_threshold_rules.csv` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`. Generated paderborn threshold-rule CSV table.
- `05_candidate_tables/table_paderborn_threshold_rules.md` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json`. Generated paderborn threshold-rule markdown table.
- `05_candidate_tables/table_uncertainty_summary.csv` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json`. Generated uncertainty CSV table.
- `05_candidate_tables/table_uncertainty_summary.md` <- `artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json`. Generated uncertainty markdown table.

### 06_method_assets
- `06_method_assets/configs/cwru_fault_label_map.json` <- `data/metadata/cwru/fault_label_map.json`. Secondary-dataset fault label map.
- `06_method_assets/configs/cwru_normalization_stats.json` <- `data/metadata/cwru/normalization_stats.json`. Secondary-dataset normalization stats.
- `06_method_assets/configs/cwru_preprocessing_config.json` <- `data/metadata/cwru/preprocessing_config.json`. Secondary-dataset preprocessing config.
- `06_method_assets/configs/paderborn_bearing_label_map.json` <- `data/metadata/paderborn/bearing_label_map.json`. Primary-dataset label-map JSON.
- `06_method_assets/configs/paderborn_normalization_stats.json` <- `data/metadata/paderborn/normalization_stats.json`. Primary-dataset normalization stats.
- `06_method_assets/configs/paderborn_preprocessing_config.json` <- `data/metadata/paderborn/preprocessing_config.json`. Primary-dataset preprocessing config.
- `06_method_assets/scripts/build_paderborn_label_map.py` <- `scripts/build_paderborn_label_map.py`. Paderborn label-map helper script.
- `06_method_assets/scripts/eval_cwru_resdilated_load_shift.py` <- `scripts/eval_cwru_resdilated_load_shift.py`. Secondary-dataset harder load-shift evaluation script.
- `06_method_assets/scripts/eval_cwru_resdilated_threshold_calibration.py` <- `scripts/eval_cwru_resdilated_threshold_calibration.py`. Secondary-dataset calibration evaluation script.
- `06_method_assets/scripts/eval_paderborn_deployment_metrics.py` <- `scripts/eval_paderborn_deployment_metrics.py`. Deployment benchmark script.
- `06_method_assets/scripts/eval_paderborn_resdilated_mc_dropout.py` <- `scripts/eval_paderborn_resdilated_mc_dropout.py`. Uncertainty evaluation script.
- `06_method_assets/scripts/eval_paderborn_resdilated_threshold_calibration.py` <- `scripts/eval_paderborn_resdilated_threshold_calibration.py`. Primary-dataset threshold-calibration evaluation script.
- `06_method_assets/scripts/explain_paderborn_resdilated.py` <- `scripts/explain_paderborn_resdilated.py`. Explanation asset generation script.
- `06_method_assets/scripts/prepare_paderborn.py` <- `scripts/prepare_paderborn.py`. Paderborn file preparation script.
- `06_method_assets/scripts/preprocess_cwru.py` <- `scripts/preprocess_cwru.py`. Secondary-dataset preprocessing script.
- `06_method_assets/scripts/preprocess_paderborn.py` <- `scripts/preprocess_paderborn.py`. Primary-dataset preprocessing script.
- `06_method_assets/scripts/train_generative_upgrades.py` <- `scripts/train_generative_upgrades.py`. Core final-model training script.
- `06_method_assets/scripts/train_paderborn_baselines.py` <- `scripts/train_paderborn_baselines.py`. Primary-dataset baseline training and evaluation script.
- `06_method_assets/scripts/train_shallow_baselines.py` <- `scripts/train_shallow_baselines.py`. CWRU shallow baseline script.

### 07_notes_and_limitations
- `07_notes_and_limitations/figure_cleanup_watchlist.md` <- `Manual package note based on saved plot inspection`. Generated figure cleanup watchlist.
- `07_notes_and_limitations/writing_caveats.md` <- `Manual package note based on saved reports and metrics`. Generated writing caveat checklist.

### paper_package_manifest.csv
- `paper_package_manifest.csv` <- `Generated from copy specs and table/report assembly`. Per-file package manifest.

### paper_package_overall_report.md
- `paper_package_overall_report.md` <- `Generated from copy specs and saved metrics`. Overall paper-package report.

## Final Paper Story Summary
- Final model: `ResDilatedAE` is the supported final backbone. The compact AE is now a baseline, not the main method.
- Primary dataset/protocol: Paderborn is the main paper story. The finalized evidence covers healthy-only training, testing on healthy plus fault windows, four operating conditions, three saved seeds, calibration, explanations, deployment, and uncertainty analysis.
- Secondary dataset/protocol: CWRU is the secondary robustness story under the harder leave-one-load-out load-shift protocol.
- Strong positive findings: ResDilatedAE materially improves over CompactAE on Paderborn; calibrated `percentile_99_5` thresholding cuts Paderborn false alarms while keeping strong F1/recall; deployment is feasible on CPU gateway hardware; explanation case studies are available and interpretable.
- Mixed or negative findings: Isolation Forest still leads Paderborn AUROC; MC-dropout is not a successful reliability story; CWRU load-0 false alarms remain severe; deployment lacks a true Isolation Forest timing artifact.

## Recommended Items For The 6-Page Paper
### Best Candidate Figures
- `04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png` is the cleanest quick comparison figure, but it should be redrawn from the final calibrated 3-seed numbers before submission.
- `04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png` is the best saved figure for the secondary CWRU story, but it needs title cleanup.
- `04_candidate_figures/paderborn_explanations/04_hardest_condition_tp.png` is the strongest explanation figure for the main text because it aligns with the hardest operating condition claim.
- `04_candidate_figures/paderborn_explanations/06_healthy_false_positive.png` is a strong companion if the paper explicitly discusses the remaining false-alarm behavior.
- `04_candidate_figures/paderborn/paderborn_baseline_summary.png` is useful as appendix/context, not as a likely main-paper figure.

### Best Candidate Tables
- `05_candidate_tables/table_paderborn_final_vs_baselines.md`
- `05_candidate_tables/table_paderborn_threshold_rules.md`
- `05_candidate_tables/table_cwru_final_vs_baselines.md`
- `05_candidate_tables/table_deployment_summary.md`
- `05_candidate_tables/table_uncertainty_summary.md` if the negative MC-dropout result is summarized explicitly.

### Reports Most Likely To Be Used During Drafting
- `03_final_reports/paderborn/paderborn_resdilated_seed_comparison_report.md`
- `03_final_reports/paderborn/paderborn_resdilated_threshold_calibration_report.md`
- `03_final_reports/paderborn/paderborn_baseline_report.md`
- `03_final_reports/paderborn/paderborn_resdilated_explanation_report.md`
- `03_final_reports/paderborn/paderborn_resdilated_deployment_report.md`
- `03_final_reports/cwru/cwru_resdilated_load_shift_report.md`
- `03_final_reports/cwru/cwru_resdilated_threshold_calibration_report.md`

## Things Likely Need Editing Before Writing
- Several saved plots are research-ready but not publication-ready. The clearest cleanup targets are listed in `07_notes_and_limitations/figure_cleanup_watchlist.md`.
- Many report titles and subheadings are generic (`Practical Take`, `Saved Artifacts`, etc.) and should not be reused directly as paper prose or captions.
- The explanation report uses long metadata-rich case descriptions that are helpful internally but too verbose for a conference figure caption.
- There is no dedicated publication-style Paderborn calibration figure yet; the metrics are finalized, but the likely paper figure still needs to be redrawn from the saved calibration JSON.
- There is no publication-style deployment comparison plot yet; only the deployment report/table is finalized.

## Limitations / Caveats To Remember During Writing
- Paderborn labels are inferred rather than PDF-verified in the finalized pass.
- MC-dropout is not successful enough to present as a main positive contribution.
- The CWRU load-0 false-alarm issue remains a real limitation under operating-condition shift.
- Isolation Forest deployment timing is unavailable because the serialized estimator/scaler artifact was not saved.
- Paderborn should be written as the primary validated story, with CWRU as the secondary robustness check.

## Writing Readiness Judgment
- Judgment: the package is complete enough to begin drafting the 6-page conference paper now.
- Core evidence present: finalized model choice, primary/secondary dataset story, strongest quantitative results, calibration evidence, explanation assets, deployment evidence, shallow-baseline context, and explicit negative-result uncertainty evidence.
- Not essential for drafting, but still missing for submission polish: a publication-ready Paderborn calibration figure, a publication-ready multi-seed/final comparison figure, and a true Isolation Forest deployment benchmark artifact.
