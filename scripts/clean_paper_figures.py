from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

try:
    from explain_paderborn_resdilated import (
        RUN_CONFIG,
        collect_selected_metadata,
        ensure_required_files,
        load_label_array,
        load_resdilatedae_from_checkpoint,
        load_selected_windows,
        percentile_threshold,
        read_json,
        reconstruct_selected_windows,
        resolve_paths,
        select_cases,
        subgroup_metrics_from_manifest,
        torch,
    )
except ModuleNotFoundError:
    from scripts.explain_paderborn_resdilated import (
        RUN_CONFIG,
        collect_selected_metadata,
        ensure_required_files,
        load_label_array,
        load_resdilatedae_from_checkpoint,
        load_selected_windows,
        percentile_threshold,
        read_json,
        reconstruct_selected_windows,
        resolve_paths,
        select_cases,
        subgroup_metrics_from_manifest,
        torch,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "artifacts" / "paper_package_v1"
FIGURE_ROOT = PACKAGE_ROOT / "04_candidate_figures"
CLEAN_ROOT = FIGURE_ROOT / "cleaned"

CW_RULE_LABELS = {
    "mean_plus_3std": "mean+3std",
    "percentile_99": "p99",
    "percentile_99_5": "p99.5",
    "median_plus_3mad": "med+3mad",
    "median_plus_4mad": "med+4mad",
}
MODEL_COLORS = {
    "AE": "#4c78a8",
    "OC-SVM": "#59a14f",
    "Isolation Forest": "#f28e2b",
    "CompactAE": "#4c78a8",
    "IsolationForest": "#59a14f",
    "ResDilatedAE": "#1f4e79",
    "ResDilatedAE p99.5": "#d1495b",
}
EXPLANATION_CASE_ORDER = [
    "tp_ka",
    "tp_kb",
    "tp_ki",
    "hardest_condition_tp",
    "healthy_true_negative",
    "healthy_false_positive",
]


def configure_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": CLEAN_ROOT,
        "cwru": CLEAN_ROOT / "cwru",
        "paderborn": CLEAN_ROOT / "paderborn",
        "explanations": CLEAN_ROOT / "paderborn_explanations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_csv_scores(path: Path) -> dict[str, np.ndarray]:
    split_to_scores: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        score_field = "reconstruction_error" if "reconstruction_error" in reader.fieldnames else "anomaly_score"
        for row in reader:
            split_to_scores.setdefault(row["split"], []).append(float(row[score_field]))
    return {
        split: np.asarray(values, dtype=np.float64)
        for split, values in split_to_scores.items()
    }


def format_value(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}"


def metric_band(axis: plt.Axes, values: list[float], pad_low: float = 0.02, pad_high: float = 0.02) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-6)
    return min_value - span * pad_low, max_value + span * pad_high


def build_cwru_load_shift_clean(output_dir: Path) -> dict[str, str]:
    payload = read_json(PACKAGE_ROOT / "02_final_metrics" / "cwru" / "cwru_resdilated_load_shift_metrics.json")
    folds = payload["folds"]
    baseline_reference = payload["baseline_reference"]

    loads = [int(fold["held_out_load_hp"]) for fold in folds]
    f1_values = [float(fold["model"]["metrics"]["f1"]) for fold in folds]
    far_values = [float(fold["model"]["metrics"]["false_alarm_rate"]) for fold in folds]
    x = np.arange(len(loads), dtype=np.float64)
    bar_colors = ["#c65d3b" if load == 0 else "#2a6f97" for load in loads]

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=240, constrained_layout=True)
    ax_f1, ax_far = axes

    ax_f1.bar(x, f1_values, color=bar_colors, width=0.72)
    ax_f1.set_title("F1 by held-out load")
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels([f"L{load}" for load in loads])
    ax_f1.set_ylabel("F1")
    f1_reference_values = f1_values + [float(baseline_reference[name]["f1"]["mean"]) for name in ("AE", "OC-SVM", "Isolation Forest")]
    ymin, ymax = metric_band(ax_f1, f1_reference_values, pad_low=0.20, pad_high=0.10)
    ax_f1.set_ylim(max(0.90, ymin), min(1.01, ymax))
    for idx, value in enumerate(f1_values):
        ax_f1.text(idx, value + 0.002, format_value(value), ha="center", va="bottom", fontsize=8, color="#333333")

    for baseline_name in ("AE", "OC-SVM", "Isolation Forest"):
        ax_f1.axhline(
            float(baseline_reference[baseline_name]["f1"]["mean"]),
            color=MODEL_COLORS[baseline_name],
            linestyle="--",
            linewidth=1.2,
        )

    ax_far.bar(x, far_values, color=bar_colors, width=0.72)
    ax_far.set_title("False alarm rate by held-out load")
    ax_far.set_xticks(x)
    ax_far.set_xticklabels([f"L{load}" for load in loads])
    ax_far.set_ylabel("False alarm rate")
    ax_far.set_ylim(0.0, 1.02)
    for baseline_name in ("AE", "OC-SVM", "Isolation Forest"):
        baseline_far = float(baseline_reference[baseline_name]["false_alarm_rate"]["mean"])
        ax_far.axhline(baseline_far, color=MODEL_COLORS[baseline_name], linestyle="--", linewidth=1.2)

    inset = inset_axes(ax_far, width="46%", height="46%", loc="upper right", borderpad=1.2)
    inset.bar(x, far_values, color=bar_colors, width=0.72)
    for baseline_name in ("AE", "OC-SVM", "Isolation Forest"):
        baseline_far = float(baseline_reference[baseline_name]["false_alarm_rate"]["mean"])
        inset.axhline(baseline_far, color=MODEL_COLORS[baseline_name], linestyle="--", linewidth=1.0)
    inset.set_ylim(0.0, 0.05)
    inset.set_xticks(x)
    inset.set_xticklabels([f"L{load}" for load in loads], fontsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.set_title("Zoom: loads 1-3", fontsize=8)
    inset.grid(axis="y", alpha=0.18)
    mark_inset(ax_far, inset, loc1=2, loc2=4, fc="none", ec="#666666", lw=0.8)

    legend_handles = [
        Patch(facecolor="#2a6f97", edgecolor="none", label="ResDilatedAE fold"),
        Patch(facecolor="#c65d3b", edgecolor="none", label="Load 0"),
        Line2D([0], [0], color=MODEL_COLORS["AE"], linestyle="--", linewidth=1.2, label="AE mean"),
        Line2D([0], [0], color=MODEL_COLORS["OC-SVM"], linestyle="--", linewidth=1.2, label="OC-SVM mean"),
        Line2D([0], [0], color=MODEL_COLORS["Isolation Forest"], linestyle="--", linewidth=1.2, label="IF mean"),
    ]
    figure.legend(handles=legend_handles, ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.05))
    figure.suptitle("CWRU leave-one-load-out summary", y=1.08, fontsize=12)
    figure.text(0.01, 1.01, "AUROC = 1.00 for every held-out load.", fontsize=9, color="#444444")

    output_path = output_dir / "cwru_load_shift_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "04_candidate_figures/cwru/cwru_resdilated_load_shift_summary.png",
        "cleaned": f"04_candidate_figures/cleaned/cwru/{output_path.name}",
        "summary": "Removed the broken overlapping title, dropped the redundant AUROC panel, highlighted load 0, and added a FAR inset so loads 1-3 remain visible.",
        "recommendation": "Useful as a compact secondary-dataset figure or appendix panel.",
    }


def compute_cwru_threshold_summary(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    rules = payload["protocol"]["threshold_rules"]
    models = ("AE", "OC-SVM", "Isolation Forest")
    summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {model: {} for model in models}
    for model in models:
        for metric_name in ("false_alarm_rate", "f1"):
            summary[model][metric_name] = {}
            for rule in rules:
                values = [
                    float(fold["models"][model]["rules"][rule]["metrics"][metric_name])
                    for fold in payload["folds"]
                ]
                summary[model][metric_name][rule] = (float(np.mean(values)), float(np.std(values)))
    return summary


def build_cwru_threshold_calibration_clean(output_dir: Path) -> dict[str, str]:
    payload = read_json(PACKAGE_ROOT / "02_final_metrics" / "cwru" / "cwru_baseline_threshold_calibration_metrics.json")
    summary = compute_cwru_threshold_summary(payload)
    rules = payload["protocol"]["threshold_rules"]
    labels = [CW_RULE_LABELS[rule] for rule in rules]
    x = np.arange(len(rules), dtype=np.float64)
    offsets = {
        "AE": -0.18,
        "OC-SVM": 0.0,
        "Isolation Forest": 0.18,
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=240, constrained_layout=True)
    panel_specs = [
        ("false_alarm_rate", "Mean false alarm rate", (0.0, 0.90)),
        ("f1", "Mean F1", (0.94, 1.005)),
    ]
    for axis, (metric_name, title, ylim) in zip(axes, panel_specs, strict=True):
        for model_name in ("AE", "OC-SVM", "Isolation Forest"):
            means = [summary[model_name][metric_name][rule][0] for rule in rules]
            stds = [summary[model_name][metric_name][rule][1] for rule in rules]
            axis.errorbar(
                x + offsets[model_name],
                means,
                yerr=stds,
                color=MODEL_COLORS[model_name],
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                elinewidth=1.0,
                capsize=2.5,
                label=model_name,
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.20)
    axes[0].set_ylabel("Rate")
    axes[1].set_ylabel("F1")

    figure.legend(
        handles=[
            Line2D([0], [0], color=MODEL_COLORS["AE"], marker="o", linewidth=1.6, label="AE"),
            Line2D([0], [0], color=MODEL_COLORS["OC-SVM"], marker="o", linewidth=1.6, label="OC-SVM"),
            Line2D([0], [0], color=MODEL_COLORS["Isolation Forest"], marker="o", linewidth=1.6, label="Isolation Forest"),
        ],
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle("CWRU threshold rules under load shift", y=1.08, fontsize=12)

    output_path = output_dir / "cwru_threshold_calibration_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "04_candidate_figures/cwru/cwru_threshold_calibration_summary.png",
        "cleaned": f"04_candidate_figures/cleaned/cwru/{output_path.name}",
        "summary": "Shortened the rule labels, split FAR and F1 into two clean panels, moved the legend out of the plotting area, and replaced heavy styling with lighter error bars.",
        "recommendation": "Readable for review and possible appendix use; still secondary to the final ResDilatedAE calibration story.",
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    y = np.linspace(1.0 / max(sorted_values.size, 1), 1.0, max(sorted_values.size, 1))
    return sorted_values, y


def build_paderborn_baseline_distribution_clean(output_dir: Path) -> dict[str, str]:
    model_specs = [
        (
            "AE",
            REPO_ROOT / "artifacts" / "metrics" / "paderborn_ae_scores.csv",
            PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_ae_metrics.json",
        ),
        (
            "OC-SVM",
            REPO_ROOT / "artifacts" / "metrics" / "paderborn_ocsvm_scores.csv",
            PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_ocsvm_metrics.json",
        ),
        (
            "Isolation Forest",
            REPO_ROOT / "artifacts" / "metrics" / "paderborn_iforest_scores.csv",
            PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_iforest_metrics.json",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=240, sharey=True, constrained_layout=True)
    for axis, (label, score_path, metrics_path) in zip(axes, model_specs, strict=True):
        split_scores = load_csv_scores(score_path)
        metrics_payload = read_json(metrics_path)
        threshold = float(metrics_payload["threshold"])

        healthy_x, healthy_y = ecdf(split_scores["test_healthy"])
        fault_x, fault_y = ecdf(split_scores["test_fault"])
        axis.plot(healthy_x, healthy_y, color="#4c78a8", linewidth=1.6, label="Test healthy")
        axis.plot(fault_x, fault_y, color="#d1495b", linewidth=1.6, label="Test fault")
        axis.axvline(threshold, color="#444444", linestyle="--", linewidth=1.1, label="Threshold")
        axis.set_title(label)
        axis.set_xlabel("Score")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Cumulative fraction")

    figure.legend(
        handles=[
            Line2D([0], [0], color="#4c78a8", linewidth=1.6, label="Test healthy"),
            Line2D([0], [0], color="#d1495b", linewidth=1.6, label="Test fault"),
            Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.1, label="Threshold"),
        ],
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle("Paderborn baseline score distributions (ECDF)", y=1.08, fontsize=12)

    output_path = output_dir / "paderborn_baseline_summary_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "04_candidate_figures/paderborn/paderborn_baseline_summary.png",
        "cleaned": f"04_candidate_figures/cleaned/paderborn/{output_path.name}",
        "summary": "Replaced count histograms with ECDF curves so the score separation and thresholds stay readable despite very different score scales.",
        "recommendation": "Stronger as appendix/context than as a main-paper figure.",
    }


def extract_seed123_metrics() -> dict[str, dict[str, float]]:
    payload = read_json(PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_resdilated_seed_123_metrics.json")
    return {
        "CompactAE": payload["baseline_reference"]["CompactAE"],
        "IsolationForest": payload["baseline_reference"]["IsolationForest"],
        "ResDilatedAE": payload["models"]["ResDilatedAE"]["metrics"],
    }


def build_metric_panel(
    axis: plt.Axes,
    model_names: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    ylim: tuple[float, float] | None = None,
    value_precision: int = 3,
) -> None:
    bars = axis.bar(np.arange(len(model_names)), values, color=colors, width=0.72)
    axis.set_title(title)
    axis.set_xticks(np.arange(len(model_names)))
    axis.set_xticklabels(model_names, rotation=18, ha="right")
    if ylim is not None:
        axis.set_ylim(*ylim)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (0.01 if (ylim is None or ylim[1] > 0.2) else 0.0004),
            format_value(value, precision=value_precision),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_paderborn_seed123_clean(output_dir: Path) -> dict[str, str]:
    metrics = extract_seed123_metrics()
    model_names = ["CompactAE", "IsolationForest", "ResDilatedAE"]
    colors = [MODEL_COLORS[name] for name in model_names]

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), dpi=240, constrained_layout=True)
    panel_specs = [
        ("auroc", "AUROC", (0.50, 1.00), 3),
        ("f1", "F1", (0.20, 0.90), 3),
        ("recall_fault", "Recall on fault windows", (0.10, 0.75), 3),
        ("false_alarm_rate", "False alarm rate", (0.0, 0.0205), 4),
    ]
    for axis, (metric_key, title, ylim, precision) in zip(axes.reshape(-1), panel_specs, strict=True):
        values = [float(metrics[name][metric_key]) for name in model_names]
        build_metric_panel(axis, model_names, values, colors, title, ylim=ylim, value_precision=precision)

    figure.suptitle("Paderborn seed 123 comparison", y=1.06, fontsize=12)
    figure.text(
        0.01,
        1.015,
        "Threshold rule: mean_plus_3std. AUPRC is omitted because all three bars are near saturation.",
        fontsize=9,
        color="#444444",
    )

    output_path = output_dir / "paderborn_resdilated_seed_123_summary_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "04_candidate_figures/paderborn/paderborn_resdilated_seed_123_summary.png",
        "cleaned": f"04_candidate_figures/cleaned/paderborn/{output_path.name}",
        "summary": "Cut the chart from five panels to four, removed the near-saturated AUPRC panel, enlarged labels, and annotated the bar values.",
        "recommendation": "Cleaner than the original, but still not the best main-paper figure because it is single-seed and uses the default threshold.",
    }


def build_paderborn_final_comparison_clean(output_dir: Path) -> dict[str, str]:
    calibration_payload = read_json(
        PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_resdilated_threshold_calibration_metrics.json"
    )
    compact = read_json(PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_ae_metrics.json")
    iforest = read_json(PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_iforest_metrics.json")
    rule_summary = calibration_payload["rule_summary"]

    model_names = ["CompactAE", "IsolationForest", "ResDilatedAE", "ResDilatedAE p99.5"]
    colors = [
        MODEL_COLORS["CompactAE"],
        MODEL_COLORS["IsolationForest"],
        MODEL_COLORS["ResDilatedAE"],
        MODEL_COLORS["ResDilatedAE p99.5"],
    ]
    metric_values = {
        "auroc": [
            float(compact["auroc"]),
            float(iforest["auroc"]),
            float(rule_summary["mean_plus_3std"]["auroc"]["mean"]),
            float(rule_summary["percentile_99_5"]["auroc"]["mean"]),
        ],
        "f1": [
            float(compact["f1"]),
            float(iforest["f1"]),
            float(rule_summary["mean_plus_3std"]["f1"]["mean"]),
            float(rule_summary["percentile_99_5"]["f1"]["mean"]),
        ],
        "recall_fault": [
            float(compact["recall_fault"]),
            float(iforest["recall_fault"]),
            float(rule_summary["mean_plus_3std"]["recall_fault"]["mean"]),
            float(rule_summary["percentile_99_5"]["recall_fault"]["mean"]),
        ],
        "false_alarm_rate": [
            float(compact["false_alarm_rate"]),
            float(iforest["false_alarm_rate"]),
            float(rule_summary["mean_plus_3std"]["false_alarm_rate"]["mean"]),
            float(rule_summary["percentile_99_5"]["false_alarm_rate"]["mean"]),
        ],
    }

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), dpi=240, constrained_layout=True)
    panel_specs = [
        ("auroc", "AUROC", (0.50, 1.00), 3),
        ("f1", "F1", (0.20, 0.90), 3),
        ("recall_fault", "Recall on fault windows", (0.10, 0.75), 3),
        ("false_alarm_rate", "False alarm rate", (0.0, 0.0205), 4),
    ]
    for axis, (metric_key, title, ylim, precision) in zip(axes.reshape(-1), panel_specs, strict=True):
        build_metric_panel(axis, model_names, metric_values[metric_key], colors, title, ylim=ylim, value_precision=precision)

    figure.suptitle("Paderborn finalized comparison", y=1.06, fontsize=12)
    figure.text(
        0.01,
        1.015,
        "ResDilatedAE bars use the finalized 3-seed means. The p99.5 row is the calibrated operating point used in the final paper story.",
        fontsize=9,
        color="#444444",
    )

    output_path = output_dir / "paderborn_final_comparison_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "New figure from finalized metrics",
        "cleaned": f"04_candidate_figures/cleaned/paderborn/{output_path.name}",
        "summary": "Added a paper-oriented multi-seed comparison that matches the finalized story better than the saved seed-123 chart.",
        "recommendation": "Recommended main-paper comparison figure.",
    }


def build_paderborn_threshold_calibration_clean(output_dir: Path) -> dict[str, str]:
    payload = read_json(PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_resdilated_threshold_calibration_metrics.json")
    rules = ["mean_plus_3std", "percentile_99", "percentile_99_5", "median_plus_3mad", "median_plus_4mad"]
    labels = [CW_RULE_LABELS[rule] for rule in rules]
    x = np.arange(len(rules), dtype=np.float64)
    summary = payload["rule_summary"]

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=240, constrained_layout=True)
    panel_specs = [
        ("f1", "Mean F1", (0.73, 0.83)),
        ("false_alarm_rate", "Mean false alarm rate", (0.0, 0.08)),
    ]
    for axis, (metric_key, title, ylim) in zip(axes, panel_specs, strict=True):
        means = [float(summary[rule][metric_key]["mean"]) for rule in rules]
        stds = [float(summary[rule][metric_key]["std"]) for rule in rules]
        axis.errorbar(
            x,
            means,
            yerr=stds,
            color="#1f4e79",
            marker="o",
            linewidth=1.8,
            elinewidth=1.0,
            capsize=2.5,
        )
        axis.scatter(
            x[2],
            means[2],
            color="#d1495b",
            s=46,
            zorder=4,
            label="Chosen p99.5" if metric_key == "f1" else None,
        )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.20)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("False alarm rate")

    figure.legend(
        handles=[
            Line2D([0], [0], color="#1f4e79", marker="o", linewidth=1.8, label="3-seed mean +/- std"),
            Line2D([0], [0], color="#d1495b", marker="o", linewidth=0, label="Chosen p99.5"),
        ],
        ncol=2,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle("Paderborn threshold calibration", y=1.08, fontsize=12)

    output_path = output_dir / "paderborn_threshold_calibration_clean.png"
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "original": "New figure from finalized metrics",
        "cleaned": f"04_candidate_figures/cleaned/paderborn/{output_path.name}",
        "summary": "Added the publication-style calibration figure that was missing from the first paper package.",
        "recommendation": "Recommended main-paper calibration figure.",
    }


def case_title(case_payload: dict[str, Any]) -> str:
    metadata = case_payload["metadata"]
    if metadata["subset"] == "test_healthy":
        health_label = "Healthy"
    else:
        health_label = metadata["damage_group"] or "Fault"
    label = case_payload["label"]
    condition = metadata["condition_code"] or "-"
    return f"{label} ({health_label}, {condition})"


def build_explanation_case_figure(
    *,
    path: Path,
    case_payload: dict[str, Any],
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> None:
    residual = np.asarray(original, dtype=np.float64) - np.asarray(reconstruction, dtype=np.float64)
    time_axis = np.arange(original.shape[0], dtype=np.int64)
    freq_axis = np.fft.rfftfreq(original.shape[0], d=1.0)
    original_fft = np.log1p(np.abs(np.fft.rfft(original)))
    reconstruction_fft = np.log1p(np.abs(np.fft.rfft(reconstruction)))
    residual_fft = np.log1p(np.abs(np.fft.rfft(residual)))

    figure, axes = plt.subplots(2, 2, figsize=(9.8, 6.4), dpi=240, constrained_layout=True)
    ax_overlay_time, ax_residual_time, ax_overlay_fft, ax_residual_fft = axes.reshape(-1)

    ax_overlay_time.plot(time_axis, original, color="#1f3c88", linewidth=1.1, label="Original")
    ax_overlay_time.plot(time_axis, reconstruction, color="#1b998b", linewidth=1.1, label="Healthy reconstruction")
    ax_overlay_time.set_title("Waveform overlay")
    ax_overlay_time.set_xlabel("Sample")
    ax_overlay_time.set_ylabel("Amplitude")
    ax_overlay_time.legend(frameon=False, loc="upper left")

    ax_residual_time.plot(time_axis, residual, color="#d1495b", linewidth=1.0)
    ax_residual_time.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.6)
    ax_residual_time.set_title("Residual waveform")
    ax_residual_time.set_xlabel("Sample")
    ax_residual_time.set_ylabel("Amplitude")

    ax_overlay_fft.plot(freq_axis, original_fft, color="#1f3c88", linewidth=1.1, label="Original")
    ax_overlay_fft.plot(freq_axis, reconstruction_fft, color="#1b998b", linewidth=1.1, label="Healthy reconstruction")
    ax_overlay_fft.set_title("FFT log-magnitude")
    ax_overlay_fft.set_xlabel("Normalized frequency")
    ax_overlay_fft.set_ylabel("log(1 + |FFT|)")
    ax_overlay_fft.legend(frameon=False, loc="upper right")

    ax_residual_fft.plot(freq_axis, residual_fft, color="#d1495b", linewidth=1.0)
    ax_residual_fft.fill_between(freq_axis, 0.0, residual_fft, color="#d1495b", alpha=0.10)
    ax_residual_fft.set_title("Residual FFT")
    ax_residual_fft.set_xlabel("Normalized frequency")
    ax_residual_fft.set_ylabel("log(1 + |FFT|)")

    figure.suptitle(case_title(case_payload), y=1.02, fontsize=12)
    figure.text(
        0.01,
        0.995,
        f"pred={case_payload['prediction']} | score={case_payload['score']:.6f} | threshold={case_payload['threshold']:.6f}",
        fontsize=9,
        color="#444444",
    )
    figure.savefig(path)
    plt.close(figure)


def build_explanation_figures(output_dir: Path) -> list[dict[str, str]]:
    cases_payload = read_json(
        PACKAGE_ROOT / "02_final_metrics" / "paderborn" / "paderborn_resdilated_explanation_cases.json"
    )

    processed_root = REPO_ROOT / "data" / "processed" / "paderborn"
    metadata_root = REPO_ROOT / "data" / "metadata" / "paderborn"
    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)
    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    fault_label_map = {key: int(value) for key, value in preprocessing_config["fault_label_map"].items()}
    fault_labels = load_label_array(array_paths.fault_labels)

    seed = int(cases_payload["seed"])
    threshold = float(cases_payload["threshold"])
    checkpoint_path = Path(cases_payload["checkpoint_path"])
    score_paths = {name: Path(path) for name, path in cases_payload["score_paths"].items()}
    val_scores = np.load(score_paths["val_healthy_scores"]).astype(np.float32, copy=False)
    test_healthy_scores = np.load(score_paths["test_healthy_scores"]).astype(np.float32, copy=False)
    test_fault_scores = np.load(score_paths["test_fault_scores"]).astype(np.float32, copy=False)

    if not math.isclose(threshold, percentile_threshold(val_scores, 99.5), rel_tol=1e-8, abs_tol=1e-12):
        raise RuntimeError("Saved explanation threshold no longer matches the percentile_99_5 value from the saved scores.")

    subgroup_metrics = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        fault_labels=fault_labels,
        fault_label_map=fault_label_map,
        threshold=threshold,
    )
    hardest_condition = min(
        subgroup_metrics["by_condition"].items(),
        key=lambda item: float(item[1]["f1"]),
    )[0]
    selected_cases = select_cases(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        threshold=threshold,
        hardest_condition=hardest_condition,
    )
    selected_metadata = collect_selected_metadata(
        window_manifest_path=metadata_root / "window_manifest.csv",
        selected_cases=selected_cases,
    )

    device = torch.device("cpu")
    model, _checkpoint_payload = load_resdilatedae_from_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_seed=seed,
        expected_width=int(preprocessing_config["window_size"]),
        device=device,
    )
    original_windows = load_selected_windows(array_paths=array_paths, selected_cases=selected_cases)
    reconstructions = reconstruct_selected_windows(
        model=model,
        device=device,
        selected_cases=selected_cases,
        original_windows=original_windows,
    )

    payload_by_id = {case["case_id"]: case for case in cases_payload["cases"]}
    metadata_by_id = {case.case_id: selected_metadata[(case.subset, case.index)] for case in selected_cases}
    originals_by_id = {case.case_id: original_windows[(case.subset, case.index)] for case in selected_cases}
    recon_by_id = {case.case_id: reconstructions[(case.subset, case.index)] for case in selected_cases}

    figure_notes: list[dict[str, str]] = []
    for index, case_id in enumerate(EXPLANATION_CASE_ORDER, start=1):
        if case_id not in payload_by_id:
            continue
        case_payload = payload_by_id[case_id]
        case_payload["metadata"] = metadata_by_id[case_id]
        output_path = output_dir / f"{index:02d}_{case_id}_clean.png"
        build_explanation_case_figure(
            path=output_path,
            case_payload=case_payload,
            original=originals_by_id[case_id],
            reconstruction=recon_by_id[case_id],
        )
        figure_notes.append(
            {
                "original": f"04_candidate_figures/paderborn_explanations/{Path(case_payload['figure_relpath']).name}",
                "cleaned": f"04_candidate_figures/cleaned/paderborn_explanations/{output_path.name}",
                "summary": "Rebuilt the figure with a 2x2 overlay/residual layout and removed measurement-heavy metadata from the title.",
                "recommendation": "Use selectively. The hardest-condition TP and the healthy TN / false-positive examples are the strongest writing anchors.",
            }
        )
    return figure_notes


def build_cleanup_report(
    *,
    output_path: Path,
    cwru_load_shift_note: dict[str, str],
    cwru_threshold_note: dict[str, str],
    paderborn_baseline_note: dict[str, str],
    paderborn_seed_note: dict[str, str],
    paderborn_final_note: dict[str, str],
    paderborn_calibration_note: dict[str, str],
    explanation_notes: list[dict[str, str]],
) -> None:
    lines = [
        "# Figure Cleanup Report",
        "",
        "## Revised Figures",
        f"### `{cwru_load_shift_note['original']}`",
        "- Weak points in the original: the title overlapped with the legend, the AUROC panel added no useful information, and the FAR panel hid loads 1-3 because load 0 dominated the scale.",
        f"- Cleaned output: `{cwru_load_shift_note['cleaned']}`",
        f"- What changed: {cwru_load_shift_note['summary']}",
        f"- Recommendation: {cwru_load_shift_note['recommendation']}",
        "",
        f"### `{cwru_threshold_note['original']}`",
        "- Weak points in the original: long rule labels crowded the x-axis, the legend consumed plotting space, and the heavy error bars made the figure feel noisy.",
        f"- Cleaned output: `{cwru_threshold_note['cleaned']}`",
        f"- What changed: {cwru_threshold_note['summary']}",
        f"- Recommendation: {cwru_threshold_note['recommendation']}",
        "",
        f"### `{paderborn_baseline_note['original']}`",
        "- Weak points in the original: histogram counts depended strongly on sample volume, the score scales differed wildly across models, and the repeated legends made the figure visually dense.",
        f"- Cleaned output: `{paderborn_baseline_note['cleaned']}`",
        f"- What changed: {paderborn_baseline_note['summary']}",
        f"- Recommendation: {paderborn_baseline_note['recommendation']}",
        "",
        f"### `{paderborn_seed_note['original']}`",
        "- Weak points in the original: five small panels were too cramped, model names were hard to read, and the figure still reflected the single-seed default-threshold story rather than the final calibrated story.",
        f"- Cleaned output: `{paderborn_seed_note['cleaned']}`",
        f"- What changed: {paderborn_seed_note['summary']}",
        f"- Recommendation: {paderborn_seed_note['recommendation']}",
        "",
        "## Added Paper-Ready Figures",
        f"### `{paderborn_final_note['cleaned']}`",
        f"- Why it was added: {paderborn_final_note['summary']}",
        f"- Recommendation: {paderborn_final_note['recommendation']}",
        "",
        f"### `{paderborn_calibration_note['cleaned']}`",
        f"- Why it was added: {paderborn_calibration_note['summary']}",
        f"- Recommendation: {paderborn_calibration_note['recommendation']}",
        "",
        "## Explanation Figures",
        "- Original weaknesses across the current explanation set: the 2x3 layout used too many panels, the suptitles carried too much metadata, and the repeated single-signal panels made the central comparison harder to read.",
    ]

    for note in explanation_notes:
        lines.append(f"- `{note['original']}` -> `{note['cleaned']}`. {note['summary']}")

    lines.extend(
        [
            "",
            "## Recommended for the Main Paper",
            f"- `{paderborn_final_note['cleaned']}` as the main comparison figure.",
            f"- `{paderborn_calibration_note['cleaned']}` as the calibration figure.",
            f"- `{cwru_load_shift_note['cleaned']}` if the secondary CWRU story gets one figure.",
            "- For explanations, prefer the cleaned hardest-condition true positive together with the healthy true-negative and healthy false-positive examples.",
            "",
            "## Better for Appendix or Context Only",
            f"- `{paderborn_baseline_note['cleaned']}` remains more useful as background or appendix context than as a main 6-page figure.",
            f"- `{paderborn_seed_note['cleaned']}` is still weaker than the finalized multi-seed comparison because it uses a single saved seed and the default threshold.",
            f"- `{cwru_threshold_note['cleaned']}` is useful context, but it is not the strongest figure for the final CWRU paragraph.",
            "",
            "## Blockers",
            "- None. The cleaned explanation figures were regenerated from the saved checkpoint and processed arrays without retraining or modifying any saved metrics.",
        ]
    )
    write_text(output_path, "\n".join(lines) + "\n")


def main() -> int:
    configure_plot_style()
    dirs = ensure_dirs()

    cwru_load_shift_note = build_cwru_load_shift_clean(dirs["cwru"])
    cwru_threshold_note = build_cwru_threshold_calibration_clean(dirs["cwru"])
    paderborn_baseline_note = build_paderborn_baseline_distribution_clean(dirs["paderborn"])
    paderborn_seed_note = build_paderborn_seed123_clean(dirs["paderborn"])
    paderborn_final_note = build_paderborn_final_comparison_clean(dirs["paderborn"])
    paderborn_calibration_note = build_paderborn_threshold_calibration_clean(dirs["paderborn"])
    explanation_notes = build_explanation_figures(dirs["explanations"])

    build_cleanup_report(
        output_path=CLEAN_ROOT / "figure_cleanup_report.md",
        cwru_load_shift_note=cwru_load_shift_note,
        cwru_threshold_note=cwru_threshold_note,
        paderborn_baseline_note=paderborn_baseline_note,
        paderborn_seed_note=paderborn_seed_note,
        paderborn_final_note=paderborn_final_note,
        paderborn_calibration_note=paderborn_calibration_note,
        explanation_notes=explanation_notes,
    )
    print(f"Saved cleaned figures to {CLEAN_ROOT.as_posix()}")
    print(f"Saved cleanup report to {(CLEAN_ROOT / 'figure_cleanup_report.md').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
