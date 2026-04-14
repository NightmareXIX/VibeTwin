from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "artifacts" / "paper_package_v1"


@dataclass(frozen=True)
class CopySpec:
    source_rel: str
    dest_rel: str
    note: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_file(spec: CopySpec, manifest: list[dict[str, str]]) -> None:
    source_path = PROJECT_ROOT / spec.source_rel
    destination_path = PACKAGE_ROOT / spec.dest_rel
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path.as_posix()}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    manifest.append(
        {
            "kind": "copied",
            "packaged_path": spec.dest_rel,
            "source_path": spec.source_rel,
            "note": spec.note,
        }
    )


def fmt_num(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def add_generated_manifest_entry(
    manifest: list[dict[str, str]],
    packaged_path: str,
    source_path: str,
    note: str,
) -> None:
    manifest.append(
        {
            "kind": "generated",
            "packaged_path": packaged_path,
            "source_path": source_path,
            "note": note,
        }
    )


def build_copy_specs() -> list[CopySpec]:
    return [
        CopySpec("readme_versions.txt", "01_source_docs/readme_versions.txt", "Local version note captured for writing context."),
        CopySpec(
            "data/metadata/paderborn/preprocessing_report.md",
            "01_source_docs/paderborn_preprocessing_report.md",
            "Primary dataset preprocessing and split audit.",
        ),
        CopySpec(
            "data/metadata/paderborn/prepare_report.md",
            "01_source_docs/paderborn_prepare_report.md",
            "Paderborn file inventory and preparation notes.",
        ),
        CopySpec(
            "data/metadata/paderborn/bearing_label_map.md",
            "01_source_docs/paderborn_bearing_label_map.md",
            "Paderborn label provenance reference.",
        ),
        CopySpec(
            "data/metadata/cwru/preprocessing_audit.md",
            "01_source_docs/cwru_preprocessing_audit.md",
            "CWRU preprocessing and split audit.",
        ),
        CopySpec(
            "data/metadata/cwru/README.md",
            "01_source_docs/cwru_metadata_readme.md",
            "CWRU metadata readme for writing context.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_metrics.json",
            "02_final_metrics/cwru/cwru_resdilated_load_shift_metrics.json",
            "Final ResDilatedAE CWRU harder load-shift metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json",
            "02_final_metrics/cwru/cwru_resdilated_threshold_calibration_metrics.json",
            "Final ResDilatedAE CWRU calibration metrics.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_load_shift_metrics.json",
            "02_final_metrics/cwru/cwru_baseline_load_shift_metrics.json",
            "Earlier CWRU baseline load-shift metrics for context.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_threshold_calibration_metrics.json",
            "02_final_metrics/cwru/cwru_baseline_threshold_calibration_metrics.json",
            "Earlier CWRU threshold-calibration metrics for context.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_ae_metrics.json",
            "02_final_metrics/cwru/cwru_ae_metrics.json",
            "CWRU compact AE baseline metrics.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_iforest_metrics.json",
            "02_final_metrics/cwru/cwru_iforest_metrics.json",
            "CWRU Isolation Forest baseline metrics.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_ocsvm_metrics.json",
            "02_final_metrics/cwru/cwru_ocsvm_metrics.json",
            "CWRU OC-SVM baseline metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_threshold_calibration_metrics.json",
            "Final Paderborn ResDilatedAE threshold-calibration metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_mc_dropout_metrics.json",
            "Final Paderborn MC-dropout uncertainty metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_deployment_metrics.json",
            "Final Paderborn deployment metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/seed_42/metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_seed_42_metrics.json",
            "Seed 42 ResDilatedAE run metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/seed_7/metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_seed_7_metrics.json",
            "Seed 7 ResDilatedAE run metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/seed_123/metrics.json",
            "02_final_metrics/paderborn/paderborn_resdilated_seed_123_metrics.json",
            "Seed 123 ResDilatedAE run metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_cases.json",
            "02_final_metrics/paderborn/paderborn_resdilated_explanation_cases.json",
            "Selected explanation case metadata.",
        ),
        CopySpec(
            "artifacts/metrics/paderborn_ae_metrics.json",
            "02_final_metrics/paderborn/paderborn_ae_metrics.json",
            "Paderborn compact AE baseline metrics.",
        ),
        CopySpec(
            "artifacts/metrics/paderborn_iforest_metrics.json",
            "02_final_metrics/paderborn/paderborn_iforest_metrics.json",
            "Paderborn Isolation Forest baseline metrics.",
        ),
        CopySpec(
            "artifacts/metrics/paderborn_ocsvm_metrics.json",
            "02_final_metrics/paderborn/paderborn_ocsvm_metrics.json",
            "Paderborn OC-SVM baseline metrics.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_report.md",
            "03_final_reports/cwru/cwru_resdilated_load_shift_report.md",
            "Final CWRU ResDilatedAE harder load-shift report.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_report.md",
            "03_final_reports/cwru/cwru_resdilated_threshold_calibration_report.md",
            "Final CWRU ResDilatedAE threshold-calibration report.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_load_shift_report.md",
            "03_final_reports/cwru/cwru_baseline_load_shift_report.md",
            "Earlier CWRU baseline load-shift report for context.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_threshold_calibration_report.md",
            "03_final_reports/cwru/cwru_baseline_threshold_calibration_report.md",
            "Earlier CWRU calibration report for context.",
        ),
        CopySpec(
            "artifacts/metrics/cwru_shallow_report.md",
            "03_final_reports/cwru/cwru_shallow_baseline_report.md",
            "Earlier CWRU shallow-baseline report for context.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_seed_comparison_report.md",
            "03_final_reports/paderborn/paderborn_resdilated_seed_comparison_report.md",
            "Seed stability summary for final Paderborn model.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_report.md",
            "03_final_reports/paderborn/paderborn_resdilated_threshold_calibration_report.md",
            "Final Paderborn calibration report.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_report.md",
            "03_final_reports/paderborn/paderborn_resdilated_mc_dropout_report.md",
            "Negative-result uncertainty report.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_report.md",
            "03_final_reports/paderborn/paderborn_resdilated_deployment_report.md",
            "Final deployment report.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_report.md",
            "03_final_reports/paderborn/paderborn_resdilated_explanation_report.md",
            "Final explanation report.",
        ),
        CopySpec(
            "artifacts/metrics/paderborn_baseline_report.md",
            "03_final_reports/paderborn/paderborn_baseline_report.md",
            "Baseline comparison report for the primary dataset.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/seed_123/report.md",
            "03_final_reports/paderborn/paderborn_resdilated_seed_123_report.md",
            "Chosen single-seed run used for explanation and deployment assets.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_summary.png",
            "04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png",
            "Main CWRU ResDilatedAE load-shift plot candidate.",
        ),
        CopySpec(
            "artifacts/plots/cwru_threshold_calibration_summary.png",
            "04_candidate_figures/cwru/cwru_threshold_calibration_summary.png",
            "CWRU threshold-calibration context plot candidate.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/seed_123/summary.png",
            "04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png",
            "Compact Paderborn comparison plot candidate from the chosen seed.",
        ),
        CopySpec(
            "artifacts/plots/paderborn_baseline_summary.png",
            "04_candidate_figures/paderborn/paderborn_baseline_summary.png",
            "Paderborn baseline score-distribution plot candidate.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/01_tp_ka.png",
            "04_candidate_figures/paderborn_explanations/01_tp_ka.png",
            "Explanation case figure: KA true positive.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/02_tp_kb.png",
            "04_candidate_figures/paderborn_explanations/02_tp_kb.png",
            "Explanation case figure: KB true positive.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/03_tp_ki.png",
            "04_candidate_figures/paderborn_explanations/03_tp_ki.png",
            "Explanation case figure: KI true positive.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/04_hardest_condition_tp.png",
            "04_candidate_figures/paderborn_explanations/04_hardest_condition_tp.png",
            "Explanation case figure: hardest operating condition true positive.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/05_healthy_true_negative.png",
            "04_candidate_figures/paderborn_explanations/05_healthy_true_negative.png",
            "Explanation case figure: healthy true negative.",
        ),
        CopySpec(
            "artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/figures/06_healthy_false_positive.png",
            "04_candidate_figures/paderborn_explanations/06_healthy_false_positive.png",
            "Explanation case figure: healthy false positive.",
        ),
        CopySpec(
            "scripts/train_generative_upgrades.py",
            "06_method_assets/scripts/train_generative_upgrades.py",
            "Core final-model training script.",
        ),
        CopySpec(
            "scripts/eval_paderborn_resdilated_threshold_calibration.py",
            "06_method_assets/scripts/eval_paderborn_resdilated_threshold_calibration.py",
            "Primary-dataset threshold-calibration evaluation script.",
        ),
        CopySpec(
            "scripts/explain_paderborn_resdilated.py",
            "06_method_assets/scripts/explain_paderborn_resdilated.py",
            "Explanation asset generation script.",
        ),
        CopySpec(
            "scripts/eval_paderborn_deployment_metrics.py",
            "06_method_assets/scripts/eval_paderborn_deployment_metrics.py",
            "Deployment benchmark script.",
        ),
        CopySpec(
            "scripts/eval_paderborn_resdilated_mc_dropout.py",
            "06_method_assets/scripts/eval_paderborn_resdilated_mc_dropout.py",
            "Uncertainty evaluation script.",
        ),
        CopySpec(
            "scripts/eval_cwru_resdilated_load_shift.py",
            "06_method_assets/scripts/eval_cwru_resdilated_load_shift.py",
            "Secondary-dataset harder load-shift evaluation script.",
        ),
        CopySpec(
            "scripts/eval_cwru_resdilated_threshold_calibration.py",
            "06_method_assets/scripts/eval_cwru_resdilated_threshold_calibration.py",
            "Secondary-dataset calibration evaluation script.",
        ),
        CopySpec(
            "scripts/train_paderborn_baselines.py",
            "06_method_assets/scripts/train_paderborn_baselines.py",
            "Primary-dataset baseline training and evaluation script.",
        ),
        CopySpec(
            "scripts/train_shallow_baselines.py",
            "06_method_assets/scripts/train_shallow_baselines.py",
            "CWRU shallow baseline script.",
        ),
        CopySpec(
            "scripts/preprocess_paderborn.py",
            "06_method_assets/scripts/preprocess_paderborn.py",
            "Primary-dataset preprocessing script.",
        ),
        CopySpec(
            "scripts/preprocess_cwru.py",
            "06_method_assets/scripts/preprocess_cwru.py",
            "Secondary-dataset preprocessing script.",
        ),
        CopySpec(
            "scripts/prepare_paderborn.py",
            "06_method_assets/scripts/prepare_paderborn.py",
            "Paderborn file preparation script.",
        ),
        CopySpec(
            "scripts/build_paderborn_label_map.py",
            "06_method_assets/scripts/build_paderborn_label_map.py",
            "Paderborn label-map helper script.",
        ),
        CopySpec(
            "data/metadata/paderborn/preprocessing_config.json",
            "06_method_assets/configs/paderborn_preprocessing_config.json",
            "Primary-dataset preprocessing config.",
        ),
        CopySpec(
            "data/metadata/paderborn/normalization_stats.json",
            "06_method_assets/configs/paderborn_normalization_stats.json",
            "Primary-dataset normalization stats.",
        ),
        CopySpec(
            "data/metadata/paderborn/bearing_label_map.json",
            "06_method_assets/configs/paderborn_bearing_label_map.json",
            "Primary-dataset label-map JSON.",
        ),
        CopySpec(
            "data/metadata/cwru/preprocessing_config.json",
            "06_method_assets/configs/cwru_preprocessing_config.json",
            "Secondary-dataset preprocessing config.",
        ),
        CopySpec(
            "data/metadata/cwru/normalization_stats.json",
            "06_method_assets/configs/cwru_normalization_stats.json",
            "Secondary-dataset normalization stats.",
        ),
        CopySpec(
            "data/metadata/cwru/fault_label_map.json",
            "06_method_assets/configs/cwru_fault_label_map.json",
            "Secondary-dataset fault label map.",
        ),
    ]


def build_cwru_final_vs_baselines_table(manifest: list[dict[str, str]]) -> None:
    base = read_json(PROJECT_ROOT / "artifacts/metrics/cwru_load_shift_metrics.json")
    res = read_json(
        PROJECT_ROOT
        / "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_load_shift_metrics.json"
    )
    cal = read_json(
        PROJECT_ROOT
        / "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json"
    )
    best_rule = cal["best_practical_rule"]["rule"]
    rule_metrics = cal["best_practical_rule"]["metrics_mean_std"]

    rows_csv: list[dict[str, Any]] = []
    rows_md: list[list[str]] = []
    for model_name in ("AE", "OC-SVM", "Isolation Forest"):
        metrics = base["summary"][model_name]
        row = {
            "model": model_name,
            "threshold_setup": "saved_mean_plus_3std",
            "auroc_mean": metrics["auroc"]["mean"],
            "auroc_std": metrics["auroc"]["std"],
            "auprc_mean": metrics["auprc"]["mean"],
            "auprc_std": metrics["auprc"]["std"],
            "f1_mean": metrics["f1"]["mean"],
            "f1_std": metrics["f1"]["std"],
            "precision_mean": metrics["precision"]["mean"],
            "precision_std": metrics["precision"]["std"],
            "recall_mean": metrics["recall_fault"]["mean"],
            "recall_std": metrics["recall_fault"]["std"],
            "far_mean": metrics["false_alarm_rate"]["mean"],
            "far_std": metrics["false_alarm_rate"]["std"],
        }
        rows_csv.append(row)
        rows_md.append(
            [
                model_name,
                "saved mean_plus_3std",
                fmt_mean_std(row["auroc_mean"], row["auroc_std"]),
                fmt_mean_std(row["f1_mean"], row["f1_std"]),
                fmt_mean_std(row["precision_mean"], row["precision_std"]),
                fmt_mean_std(row["recall_mean"], row["recall_std"]),
                fmt_mean_std(row["far_mean"], row["far_std"]),
            ]
        )

    res_row = {
        "model": "ResDilatedAE",
        "threshold_setup": "saved_mean_plus_3std",
        "auroc_mean": res["summary"]["auroc"]["mean"],
        "auroc_std": res["summary"]["auroc"]["std"],
        "auprc_mean": res["summary"]["auprc"]["mean"],
        "auprc_std": res["summary"]["auprc"]["std"],
        "f1_mean": res["summary"]["f1"]["mean"],
        "f1_std": res["summary"]["f1"]["std"],
        "precision_mean": res["summary"]["precision"]["mean"],
        "precision_std": res["summary"]["precision"]["std"],
        "recall_mean": res["summary"]["recall_fault"]["mean"],
        "recall_std": res["summary"]["recall_fault"]["std"],
        "far_mean": res["summary"]["false_alarm_rate"]["mean"],
        "far_std": res["summary"]["false_alarm_rate"]["std"],
    }
    rows_csv.append(res_row)
    rows_md.append(
        [
            "ResDilatedAE",
            "saved mean_plus_3std",
            fmt_mean_std(res_row["auroc_mean"], res_row["auroc_std"]),
            fmt_mean_std(res_row["f1_mean"], res_row["f1_std"]),
            fmt_mean_std(res_row["precision_mean"], res_row["precision_std"]),
            fmt_mean_std(res_row["recall_mean"], res_row["recall_std"]),
            fmt_mean_std(res_row["far_mean"], res_row["far_std"]),
        ]
    )

    cal_row = {
        "model": "ResDilatedAE",
        "threshold_setup": best_rule,
        "auroc_mean": rule_metrics["auroc"]["mean"],
        "auroc_std": rule_metrics["auroc"]["std"],
        "auprc_mean": rule_metrics["auprc"]["mean"],
        "auprc_std": rule_metrics["auprc"]["std"],
        "f1_mean": rule_metrics["f1"]["mean"],
        "f1_std": rule_metrics["f1"]["std"],
        "precision_mean": rule_metrics["precision"]["mean"],
        "precision_std": rule_metrics["precision"]["std"],
        "recall_mean": rule_metrics["recall_fault"]["mean"],
        "recall_std": rule_metrics["recall_fault"]["std"],
        "far_mean": rule_metrics["false_alarm_rate"]["mean"],
        "far_std": rule_metrics["false_alarm_rate"]["std"],
    }
    rows_csv.append(cal_row)
    rows_md.append(
        [
            "ResDilatedAE",
            best_rule,
            fmt_mean_std(cal_row["auroc_mean"], cal_row["auroc_std"]),
            fmt_mean_std(cal_row["f1_mean"], cal_row["f1_std"]),
            fmt_mean_std(cal_row["precision_mean"], cal_row["precision_std"]),
            fmt_mean_std(cal_row["recall_mean"], cal_row["recall_std"]),
            fmt_mean_std(cal_row["far_mean"], cal_row["far_std"]),
        ]
    )

    csv_path = PACKAGE_ROOT / "05_candidate_tables/table_cwru_final_vs_baselines.csv"
    md_path = PACKAGE_ROOT / "05_candidate_tables/table_cwru_final_vs_baselines.md"
    fieldnames = list(rows_csv[0].keys())
    write_csv(csv_path, rows_csv, fieldnames)
    write_text(
        md_path,
        "\n".join(
            [
                "# Candidate Table: CWRU Final Model vs Baselines",
                "",
                "- Protocol: harder leave-one-load-out CWRU load shift.",
                "- ResDilatedAE rows use the saved final model family; the last row is the best practical calibrated rule.",
                "- Baseline rows come from the earlier saved CWRU baseline load-shift summary.",
                "",
                markdown_table(
                    ["Model", "Threshold Setup", "AUROC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
                    rows_md,
                ),
                "",
            ]
        ),
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_cwru_final_vs_baselines.csv",
        "artifacts/metrics/cwru_load_shift_metrics.json + artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/*.json",
        "Generated compact CWRU comparison table.",
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_cwru_final_vs_baselines.md",
        "artifacts/metrics/cwru_load_shift_metrics.json + artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/*.json",
        "Generated markdown version of the CWRU comparison table.",
    )


def build_paderborn_final_vs_baselines_table(manifest: list[dict[str, str]]) -> None:
    ae = read_json(PROJECT_ROOT / "artifacts/metrics/paderborn_ae_metrics.json")
    iforest = read_json(PROJECT_ROOT / "artifacts/metrics/paderborn_iforest_metrics.json")
    ocsvm = read_json(PROJECT_ROOT / "artifacts/metrics/paderborn_ocsvm_metrics.json")
    cal = read_json(
        PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json"
    )
    base_rule = "mean_plus_3std"
    best_rule = cal["best_practical_rule"]["rule"]

    rows_csv: list[dict[str, Any]] = []
    rows_md: list[list[str]] = []
    for model_name, payload in (
        ("CompactAE", ae),
        ("OC-SVM", ocsvm),
        ("Isolation Forest", iforest),
    ):
        row = {
            "model": model_name,
            "threshold_setup": payload.get("threshold_rule", "saved_mean_plus_3std"),
            "auroc_mean": payload["auroc"],
            "auroc_std": 0.0,
            "auprc_mean": payload["auprc"],
            "auprc_std": 0.0,
            "f1_mean": payload["f1"],
            "f1_std": 0.0,
            "precision_mean": payload["precision"],
            "precision_std": 0.0,
            "recall_mean": payload["recall_fault"],
            "recall_std": 0.0,
            "far_mean": payload["false_alarm_rate"],
            "far_std": 0.0,
        }
        rows_csv.append(row)
        rows_md.append(
            [
                model_name,
                row["threshold_setup"],
                fmt_num(row["auroc_mean"], 3),
                fmt_num(row["f1_mean"], 3),
                fmt_num(row["precision_mean"], 3),
                fmt_num(row["recall_mean"], 3),
                fmt_num(row["far_mean"], 4),
            ]
        )

    for rule_name in (base_rule, best_rule):
        metrics = cal["rule_summary"][rule_name]
        row = {
            "model": "ResDilatedAE",
            "threshold_setup": rule_name,
            "auroc_mean": metrics["auroc"]["mean"],
            "auroc_std": metrics["auroc"]["std"],
            "auprc_mean": 0.0,
            "auprc_std": 0.0,
            "f1_mean": metrics["f1"]["mean"],
            "f1_std": metrics["f1"]["std"],
            "precision_mean": metrics["precision"]["mean"],
            "precision_std": metrics["precision"]["std"],
            "recall_mean": metrics["recall_fault"]["mean"],
            "recall_std": metrics["recall_fault"]["std"],
            "far_mean": metrics["false_alarm_rate"]["mean"],
            "far_std": metrics["false_alarm_rate"]["std"],
        }
        rows_csv.append(row)
        rows_md.append(
            [
                "ResDilatedAE",
                rule_name,
                fmt_mean_std(row["auroc_mean"], row["auroc_std"]),
                fmt_mean_std(row["f1_mean"], row["f1_std"]),
                fmt_mean_std(row["precision_mean"], row["precision_std"]),
                fmt_mean_std(row["recall_mean"], row["recall_std"]),
                fmt_mean_std(row["far_mean"], row["far_std"], 4),
            ]
        )

    csv_path = PACKAGE_ROOT / "05_candidate_tables/table_paderborn_final_vs_baselines.csv"
    md_path = PACKAGE_ROOT / "05_candidate_tables/table_paderborn_final_vs_baselines.md"
    write_csv(csv_path, rows_csv, list(rows_csv[0].keys()))
    write_text(
        md_path,
        "\n".join(
            [
                "# Candidate Table: Paderborn Final Model vs Baselines",
                "",
                "- Primary dataset comparison table.",
                "- ResDilatedAE rows are 3-seed mean +/- std.",
                "- Baseline rows are the saved single-run references from the current codebase.",
                "",
                markdown_table(
                    ["Model", "Threshold Setup", "AUROC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
                    rows_md,
                ),
                "",
            ]
        ),
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_paderborn_final_vs_baselines.csv",
        "artifacts/metrics/paderborn_*_metrics.json + artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json",
        "Generated compact Paderborn comparison table.",
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_paderborn_final_vs_baselines.md",
        "artifacts/metrics/paderborn_*_metrics.json + artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json",
        "Generated markdown version of the Paderborn comparison table.",
    )


def build_paderborn_seed_stability_table(manifest: list[dict[str, str]]) -> None:
    paths = [
        PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/seed_42/metrics.json",
        PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/seed_7/metrics.json",
        PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/seed_123/metrics.json",
    ]
    rows_csv: list[dict[str, Any]] = []
    rows_md: list[list[str]] = []
    for path in paths:
        payload = read_json(path)
        seed = payload["seed"]
        metrics = payload["models"]["ResDilatedAE"]["metrics"]
        threshold = payload["models"]["ResDilatedAE"]["threshold"]
        row = {
            "seed": seed,
            "auroc": metrics["auroc"],
            "f1": metrics["f1"],
            "recall_fault": metrics["recall_fault"],
            "false_alarm_rate": metrics["false_alarm_rate"],
            "threshold": threshold,
        }
        rows_csv.append(row)
        rows_md.append(
            [
                str(seed),
                fmt_num(row["auroc"], 6),
                fmt_num(row["f1"], 6),
                fmt_num(row["recall_fault"], 6),
                fmt_num(row["false_alarm_rate"], 6),
                fmt_num(row["threshold"], 6),
            ]
        )

    csv_path = PACKAGE_ROOT / "05_candidate_tables/table_paderborn_seed_stability.csv"
    md_path = PACKAGE_ROOT / "05_candidate_tables/table_paderborn_seed_stability.md"
    write_csv(csv_path, rows_csv, list(rows_csv[0].keys()))
    write_text(
        md_path,
        "\n".join(
            [
                "# Candidate Table: Paderborn Seed Stability",
                "",
                "- Saved deterministic ResDilatedAE runs under the default `mean_plus_3std` threshold.",
                "",
                markdown_table(["Seed", "AUROC", "F1", "Recall Fault", "False Alarm Rate", "Threshold"], rows_md),
                "",
            ]
        ),
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_paderborn_seed_stability.csv",
        "artifacts/generative_upgrades/resdilated_ae/seed_{42,7,123}/metrics.json",
        "Generated per-seed stability table.",
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_paderborn_seed_stability.md",
        "artifacts/generative_upgrades/resdilated_ae/seed_{42,7,123}/metrics.json",
        "Generated markdown version of the per-seed stability table.",
    )


def build_threshold_rule_tables(manifest: list[dict[str, str]]) -> None:
    specs = [
        (
            PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_threshold_calibration_metrics.json",
            "paderborn",
            "Primary-dataset threshold rules.",
        ),
        (
            PROJECT_ROOT
            / "artifacts/generative_upgrades/cwru_load_shift/resdilated_ae/seed_42/cwru_resdilated_threshold_calibration_metrics.json",
            "cwru",
            "Secondary-dataset threshold rules.",
        ),
    ]
    for source_path, stem, note in specs:
        payload = read_json(source_path)
        rule_summary = payload["rule_summary"]
        rows_csv: list[dict[str, Any]] = []
        rows_md: list[list[str]] = []
        for rule_name, metrics in rule_summary.items():
            row = {
                "rule": rule_name,
                "threshold_mean": metrics["threshold"]["mean"],
                "threshold_std": metrics["threshold"]["std"],
                "auroc_mean": metrics["auroc"]["mean"],
                "auroc_std": metrics["auroc"]["std"],
                "f1_mean": metrics["f1"]["mean"],
                "f1_std": metrics["f1"]["std"],
                "precision_mean": metrics["precision"]["mean"],
                "precision_std": metrics["precision"]["std"],
                "recall_mean": metrics["recall_fault"]["mean"],
                "recall_std": metrics["recall_fault"]["std"],
                "far_mean": metrics["false_alarm_rate"]["mean"],
                "far_std": metrics["false_alarm_rate"]["std"],
            }
            rows_csv.append(row)
            rows_md.append(
                [
                    rule_name,
                    fmt_mean_std(row["threshold_mean"], row["threshold_std"], 6),
                    fmt_mean_std(row["f1_mean"], row["f1_std"]),
                    fmt_mean_std(row["precision_mean"], row["precision_std"]),
                    fmt_mean_std(row["recall_mean"], row["recall_std"]),
                    fmt_mean_std(row["far_mean"], row["far_std"], 4),
                ]
            )

        csv_rel = f"05_candidate_tables/table_{stem}_threshold_rules.csv"
        md_rel = f"05_candidate_tables/table_{stem}_threshold_rules.md"
        write_csv(PACKAGE_ROOT / csv_rel, rows_csv, list(rows_csv[0].keys()))
        write_text(
            PACKAGE_ROOT / md_rel,
            "\n".join(
                [
                    f"# Candidate Table: {stem.title()} Threshold Rules",
                    "",
                    f"- {note}",
                    "",
                    markdown_table(
                        ["Rule", "Threshold", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
                        rows_md,
                    ),
                    "",
                ]
            ),
        )
        add_generated_manifest_entry(
            manifest,
            csv_rel,
            source_path.relative_to(PROJECT_ROOT).as_posix(),
            f"Generated {stem} threshold-rule CSV table.",
        )
        add_generated_manifest_entry(
            manifest,
            md_rel,
            source_path.relative_to(PROJECT_ROOT).as_posix(),
            f"Generated {stem} threshold-rule markdown table.",
        )


def build_deployment_table(manifest: list[dict[str, str]]) -> None:
    payload = read_json(
        PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json"
    )
    rows_csv: list[dict[str, Any]] = []
    rows_md: list[list[str]] = []

    for model_name in ("ResDilatedAE", "CompactAE"):
        metrics = payload["models"][model_name]
        batch64 = metrics["cpu_benchmark"]["batch"]["64"]
        memory = metrics["cpu_benchmark"]["memory_rss"]["peak_delta_bytes"]
        saved = metrics.get("saved_detection_metrics") or {}
        row = {
            "model": model_name,
            "params": metrics["parameter_count"],
            "weights_mb": metrics["weights_only_size_bytes"] / (1024 * 1024),
            "checkpoint_mb": metrics["checkpoint_size_bytes"] / (1024 * 1024),
            "single_ms": metrics["cpu_benchmark"]["single_window"]["mean_ms"],
            "batch64_ms": batch64["mean_ms"],
            "batch64_windows_per_sec": batch64["throughput_windows_per_sec"],
            "peak_rss_delta_mb": "" if memory is None else memory / (1024 * 1024),
            "saved_f1": saved.get("f1", ""),
            "saved_auroc": saved.get("auroc", ""),
            "note": "",
        }
        rows_csv.append(row)
        rows_md.append(
            [
                model_name,
                str(row["params"]),
                fmt_num(row["weights_mb"], 3),
                fmt_num(row["checkpoint_mb"], 3),
                fmt_num(row["single_ms"], 3),
                fmt_num(row["batch64_ms"], 3),
                fmt_num(row["batch64_windows_per_sec"], 1),
                "n/a" if row["peak_rss_delta_mb"] == "" else fmt_num(float(row["peak_rss_delta_mb"]), 3),
                fmt_num(float(row["saved_f1"]), 3),
                fmt_num(float(row["saved_auroc"]), 3),
                "",
            ]
        )

    iforest_blocker = payload["comparisons"]["IsolationForest"]["benchmark_blocker"]
    iforest_saved = payload["comparisons"]["IsolationForest"]["saved_metrics"]
    iforest_row = {
        "model": "Isolation Forest",
        "params": "",
        "weights_mb": "",
        "checkpoint_mb": "",
        "single_ms": "",
        "batch64_ms": "",
        "batch64_windows_per_sec": "",
        "peak_rss_delta_mb": "",
        "saved_f1": iforest_saved["f1"] if iforest_saved else "",
        "saved_auroc": iforest_saved["auroc"] if iforest_saved else "",
        "note": iforest_blocker,
    }
    rows_csv.append(iforest_row)
    rows_md.append(
        [
            "Isolation Forest",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            fmt_num(float(iforest_row["saved_f1"]), 3),
            fmt_num(float(iforest_row["saved_auroc"]), 3),
            "benchmark blocked by missing serialized estimator",
        ]
    )

    csv_path = PACKAGE_ROOT / "05_candidate_tables/table_deployment_summary.csv"
    md_path = PACKAGE_ROOT / "05_candidate_tables/table_deployment_summary.md"
    write_csv(csv_path, rows_csv, list(rows_csv[0].keys()))
    write_text(
        md_path,
        "\n".join(
            [
                "# Candidate Table: Deployment Summary",
                "",
                "- CPU benchmark summary for the final saved deployment study.",
                "",
                markdown_table(
                    [
                        "Model",
                        "Params",
                        "Weights MB",
                        "Checkpoint MB",
                        "Single ms",
                        "Batch64 ms",
                        "Batch64 win/s",
                        "Peak RSS Delta MB",
                        "Saved F1",
                        "Saved AUROC",
                        "Note",
                    ],
                    rows_md,
                ),
                "",
            ]
        ),
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_deployment_summary.csv",
        "artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json",
        "Generated deployment CSV table.",
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_deployment_summary.md",
        "artifacts/generative_upgrades/resdilated_ae/deployment/resdilated_ae_deployment_metrics.json",
        "Generated deployment markdown table.",
    )


def build_uncertainty_table(manifest: list[dict[str, str]]) -> None:
    payload = read_json(PROJECT_ROOT / "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json")
    summary = payload["summary"]
    rows_csv: list[dict[str, Any]] = []
    rows_md: list[list[str]] = []
    rows = [
        ("Deterministic Baseline", summary["deterministic_baseline"], ""),
        ("MC No Defer", summary["mc_calibrated_no_defer"], ""),
        (
            "Uncertainty Aware",
            summary["uncertainty_aware"],
            f"{summary['deferred']['total_deferred_rate']['mean']:.4%}",
        ),
    ]
    for label, metrics, deferred_rate in rows:
        row = {
            "setting": label,
            "threshold_mean": metrics["threshold"]["mean"],
            "threshold_std": metrics["threshold"]["std"],
            "auroc_mean": metrics["auroc"]["mean"],
            "auroc_std": metrics["auroc"]["std"],
            "f1_mean": metrics["f1"]["mean"],
            "f1_std": metrics["f1"]["std"],
            "precision_mean": metrics["precision"]["mean"],
            "precision_std": metrics["precision"]["std"],
            "recall_mean": metrics["recall_fault"]["mean"],
            "recall_std": metrics["recall_fault"]["std"],
            "far_mean": metrics["false_alarm_rate"]["mean"],
            "far_std": metrics["false_alarm_rate"]["std"],
            "deferred_rate_mean": deferred_rate,
        }
        rows_csv.append(row)
        rows_md.append(
            [
                label,
                fmt_mean_std(row["auroc_mean"], row["auroc_std"]),
                fmt_mean_std(row["f1_mean"], row["f1_std"]),
                fmt_mean_std(row["precision_mean"], row["precision_std"]),
                fmt_mean_std(row["recall_mean"], row["recall_std"]),
                fmt_mean_std(row["far_mean"], row["far_std"], 4),
                deferred_rate or "n/a",
            ]
        )

    csv_path = PACKAGE_ROOT / "05_candidate_tables/table_uncertainty_summary.csv"
    md_path = PACKAGE_ROOT / "05_candidate_tables/table_uncertainty_summary.md"
    write_csv(csv_path, rows_csv, list(rows_csv[0].keys()))
    write_text(
        md_path,
        "\n".join(
            [
                "# Candidate Table: Uncertainty Summary",
                "",
                "- Negative-result summary for MC-dropout uncertainty.",
                "",
                markdown_table(
                    ["Setting", "AUROC", "F1", "Precision", "Recall Fault", "False Alarm Rate", "Deferred Rate"],
                    rows_md,
                ),
                "",
            ]
        ),
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_uncertainty_summary.csv",
        "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json",
        "Generated uncertainty CSV table.",
    )
    add_generated_manifest_entry(
        manifest,
        "05_candidate_tables/table_uncertainty_summary.md",
        "artifacts/generative_upgrades/resdilated_ae/resdilated_ae_mc_dropout_metrics.json",
        "Generated uncertainty markdown table.",
    )


def build_note_files(manifest: list[dict[str, str]]) -> None:
    figure_watchlist = "\n".join(
        [
            "# Figure Cleanup Watchlist",
            "",
            "1. `04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png` has overlapping title text at the top and should be redrawn before paper use.",
            "2. `04_candidate_figures/cwru/cwru_threshold_calibration_summary.png` is readable for internal review but has crowded x-axis labels, visually heavy error bars, and an awkward false-alarm axis range.",
            "3. `04_candidate_figures/paderborn/paderborn_baseline_summary.png` is dense and histogram-heavy; it is stronger as appendix/context than as a main 6-page figure.",
            "4. `04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png` is clean, but it reflects a single saved seed and default threshold rather than the final calibrated 3-seed story.",
            "5. The explanation figures are informative, but the full six-panel layouts and long metadata-heavy titles are too busy for a conference paper without trimming or redesign.",
            "",
        ]
    )
    caveats = "\n".join(
        [
            "# Writing Caveats",
            "",
            "1. Paderborn labels remain inferred from bearing-code families; the local support PDFs were not parsed automatically in this finalized pass.",
            "2. MC-dropout is a negative or mixed result, not a headline success: false alarms improve only by deferring many windows and sacrificing recall/F1.",
            "3. The CWRU harder load-shift study still has a severe held-out load-0 false-alarm problem, so threshold transfer should not be described as solved.",
            "4. Deployment benchmarking does not include a true Isolation Forest timing result because no serialized estimator/scaler artifact was saved.",
            "5. ResDilatedAE improves thresholded Paderborn F1/recall/FAR versus the saved Isolation Forest baseline, but it still trails Isolation Forest on AUROC.",
            "6. The finished paper story is Paderborn-primary and CWRU-secondary; older wording that centers CWRU first should be avoided.",
            "",
        ]
    )
    figure_rel = "07_notes_and_limitations/figure_cleanup_watchlist.md"
    caveat_rel = "07_notes_and_limitations/writing_caveats.md"
    write_text(PACKAGE_ROOT / figure_rel, figure_watchlist)
    write_text(PACKAGE_ROOT / caveat_rel, caveats)
    add_generated_manifest_entry(
        manifest,
        figure_rel,
        "Manual package note based on saved plot inspection",
        "Generated figure cleanup watchlist.",
    )
    add_generated_manifest_entry(
        manifest,
        caveat_rel,
        "Manual package note based on saved reports and metrics",
        "Generated writing caveat checklist.",
    )


def build_manifest_file(manifest: list[dict[str, str]]) -> None:
    write_csv(PACKAGE_ROOT / "paper_package_manifest.csv", manifest, ["kind", "packaged_path", "source_path", "note"])


def folder_inventory_lines(manifest: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        top_folder = row["packaged_path"].split("/", 1)[0]
        grouped.setdefault(top_folder, []).append(row)

    lines: list[str] = []
    for top_folder in sorted(grouped):
        lines.append(f"### {top_folder}")
        for row in sorted(grouped[top_folder], key=lambda item: item["packaged_path"]):
            lines.append(f"- `{row['packaged_path']}` <- `{row['source_path']}`. {row['note']}")
        lines.append("")
    return lines


def build_overall_report(manifest: list[dict[str, str]]) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# VibeTwin Paper Package Overall Report",
        "",
        "## Package Root",
        f"- Package folder: `{PACKAGE_ROOT.as_posix()}`",
        f"- Generated at: `{generated_at}`",
        "- Scope: finalized, paper-relevant evidence only. Originals were copied, not moved or edited.",
        "- Intentionally omitted from this package: checkpoints, raw score arrays, full window manifests, and non-final generative side paths such as ConvVAE and DenoisingResDilatedAE. Their original artifacts remain in place if we need them later.",
        "",
        "## Folder Inventory",
        "- Per-file source mapping is also saved in `paper_package_manifest.csv`.",
        "",
    ]
    lines.extend(folder_inventory_lines(manifest))
    lines.extend(
        [
            "## Final Paper Story Summary",
            "- Final model: `ResDilatedAE` is the supported final backbone. The compact AE is now a baseline, not the main method.",
            "- Primary dataset/protocol: Paderborn is the main paper story. The finalized evidence covers healthy-only training, testing on healthy plus fault windows, four operating conditions, three saved seeds, calibration, explanations, deployment, and uncertainty analysis.",
            "- Secondary dataset/protocol: CWRU is the secondary robustness story under the harder leave-one-load-out load-shift protocol.",
            "- Strong positive findings: ResDilatedAE materially improves over CompactAE on Paderborn; calibrated `percentile_99_5` thresholding cuts Paderborn false alarms while keeping strong F1/recall; deployment is feasible on CPU gateway hardware; explanation case studies are available and interpretable.",
            "- Mixed or negative findings: Isolation Forest still leads Paderborn AUROC; MC-dropout is not a successful reliability story; CWRU load-0 false alarms remain severe; deployment lacks a true Isolation Forest timing artifact.",
            "",
            "## Recommended Items For The 6-Page Paper",
            "### Best Candidate Figures",
            "- `04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png` is the cleanest quick comparison figure, but it should be redrawn from the final calibrated 3-seed numbers before submission.",
            "- `04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png` is the best saved figure for the secondary CWRU story, but it needs title cleanup.",
            "- `04_candidate_figures/paderborn_explanations/04_hardest_condition_tp.png` is the strongest explanation figure for the main text because it aligns with the hardest operating condition claim.",
            "- `04_candidate_figures/paderborn_explanations/06_healthy_false_positive.png` is a strong companion if the paper explicitly discusses the remaining false-alarm behavior.",
            "- `04_candidate_figures/paderborn/paderborn_baseline_summary.png` is useful as appendix/context, not as a likely main-paper figure.",
            "",
            "### Best Candidate Tables",
            "- `05_candidate_tables/table_paderborn_final_vs_baselines.md`",
            "- `05_candidate_tables/table_paderborn_threshold_rules.md`",
            "- `05_candidate_tables/table_cwru_final_vs_baselines.md`",
            "- `05_candidate_tables/table_deployment_summary.md`",
            "- `05_candidate_tables/table_uncertainty_summary.md` if the negative MC-dropout result is summarized explicitly.",
            "",
            "### Reports Most Likely To Be Used During Drafting",
            "- `03_final_reports/paderborn/paderborn_resdilated_seed_comparison_report.md`",
            "- `03_final_reports/paderborn/paderborn_resdilated_threshold_calibration_report.md`",
            "- `03_final_reports/paderborn/paderborn_baseline_report.md`",
            "- `03_final_reports/paderborn/paderborn_resdilated_explanation_report.md`",
            "- `03_final_reports/paderborn/paderborn_resdilated_deployment_report.md`",
            "- `03_final_reports/cwru/cwru_resdilated_load_shift_report.md`",
            "- `03_final_reports/cwru/cwru_resdilated_threshold_calibration_report.md`",
            "",
            "## Things Likely Need Editing Before Writing",
            "- Several saved plots are research-ready but not publication-ready. The clearest cleanup targets are listed in `07_notes_and_limitations/figure_cleanup_watchlist.md`.",
            "- Many report titles and subheadings are generic (`Practical Take`, `Saved Artifacts`, etc.) and should not be reused directly as paper prose or captions.",
            "- The explanation report uses long metadata-rich case descriptions that are helpful internally but too verbose for a conference figure caption.",
            "- There is no dedicated publication-style Paderborn calibration figure yet; the metrics are finalized, but the likely paper figure still needs to be redrawn from the saved calibration JSON.",
            "- There is no publication-style deployment comparison plot yet; only the deployment report/table is finalized.",
            "",
            "## Limitations / Caveats To Remember During Writing",
            "- Paderborn labels are inferred rather than PDF-verified in the finalized pass.",
            "- MC-dropout is not successful enough to present as a main positive contribution.",
            "- The CWRU load-0 false-alarm issue remains a real limitation under operating-condition shift.",
            "- Isolation Forest deployment timing is unavailable because the serialized estimator/scaler artifact was not saved.",
            "- Paderborn should be written as the primary validated story, with CWRU as the secondary robustness check.",
            "",
            "## Writing Readiness Judgment",
            "- Judgment: the package is complete enough to begin drafting the 6-page conference paper now.",
            "- Core evidence present: finalized model choice, primary/secondary dataset story, strongest quantitative results, calibration evidence, explanation assets, deployment evidence, shallow-baseline context, and explicit negative-result uncertainty evidence.",
            "- Not essential for drafting, but still missing for submission polish: a publication-ready Paderborn calibration figure, a publication-ready multi-seed/final comparison figure, and a true Isolation Forest deployment benchmark artifact.",
            "",
        ]
    )
    write_text(PACKAGE_ROOT / "paper_package_overall_report.md", "\n".join(lines))


def main() -> int:
    if PACKAGE_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package folder: {PACKAGE_ROOT.as_posix()}")

    for folder_name in (
        "01_source_docs",
        "02_final_metrics",
        "03_final_reports",
        "04_candidate_figures",
        "05_candidate_tables",
        "06_method_assets",
        "07_notes_and_limitations",
    ):
        (PACKAGE_ROOT / folder_name).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for spec in build_copy_specs():
        copy_file(spec, manifest)

    build_cwru_final_vs_baselines_table(manifest)
    build_paderborn_final_vs_baselines_table(manifest)
    build_paderborn_seed_stability_table(manifest)
    build_threshold_rule_tables(manifest)
    build_deployment_table(manifest)
    build_uncertainty_table(manifest)
    build_note_files(manifest)
    add_generated_manifest_entry(
        manifest,
        "paper_package_manifest.csv",
        "Generated from copy specs and table/report assembly",
        "Per-file package manifest.",
    )
    add_generated_manifest_entry(
        manifest,
        "paper_package_overall_report.md",
        "Generated from copy specs and saved metrics",
        "Overall paper-package report.",
    )
    build_manifest_file(manifest)
    build_overall_report(manifest)

    print(f"Paper package created at: {PACKAGE_ROOT.as_posix()}")
    print("Generated:")
    print("  - paper_package_manifest.csv")
    print("  - paper_package_overall_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
