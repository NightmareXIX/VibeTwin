from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_cwru_load_shift import (
    build_fold_dataset,
    build_indexed_windows,
    ensure_required_files,
    filter_manifest_rows,
    load_label_array,
    load_manifest_rows,
    load_window_array,
    read_json,
    recover_raw_windows,
    resolve_paths,
    train_and_evaluate_ae,
    train_and_evaluate_shallow_models,
)
from train_shallow_baselines import evaluate_scores as evaluate_thresholded_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
LOAD_SHIFT_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "cwru_load_shift_metrics.json"

THRESHOLD_RULES = (
    "mean_plus_3std",
    "percentile_99",
    "percentile_99_5",
    "median_plus_3mad",
    "median_plus_4mad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study threshold calibration rules for leave-one-load-out CWRU evaluation.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psd-band-count", type=int, default=5)
    parser.add_argument("--ocsvm-nu", type=float, default=0.05)
    parser.add_argument("--iforest-n-estimators", type=int, default=300)
    parser.add_argument("--iforest-max-samples", type=int, default=256)
    parser.add_argument(
        "--iforest-n-jobs",
        type=int,
        default=1,
        help="Defaulting to 1 avoids the Windows joblib access-denied issue seen in earlier runs.",
    )
    return parser.parse_args()


def select_threshold(scores: np.ndarray, rule: str) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError("Validation healthy scores are empty; cannot fit a threshold.")

    if rule == "mean_plus_3std":
        score_mean = float(values.mean())
        score_std = float(values.std())
        return {
            "rule": rule,
            "threshold": float(score_mean + (3.0 * score_std)),
            "validation_mean": score_mean,
            "validation_std": score_std,
            "fit_split": "val_healthy",
        }

    if rule == "percentile_99":
        percentile_value = float(np.percentile(values, 99.0))
        return {
            "rule": rule,
            "threshold": percentile_value,
            "percentile": 99.0,
            "fit_split": "val_healthy",
        }

    if rule == "percentile_99_5":
        percentile_value = float(np.percentile(values, 99.5))
        return {
            "rule": rule,
            "threshold": percentile_value,
            "percentile": 99.5,
            "fit_split": "val_healthy",
        }

    if rule == "median_plus_3mad":
        score_median = float(np.median(values))
        score_mad = float(np.median(np.abs(values - score_median)))
        return {
            "rule": rule,
            "threshold": float(score_median + (3.0 * score_mad)),
            "validation_median": score_median,
            "validation_mad": score_mad,
            "mad_definition": "raw_median_absolute_deviation",
            "fit_split": "val_healthy",
        }

    if rule == "median_plus_4mad":
        score_median = float(np.median(values))
        score_mad = float(np.median(np.abs(values - score_median)))
        return {
            "rule": rule,
            "threshold": float(score_median + (4.0 * score_mad)),
            "validation_median": score_median,
            "validation_mad": score_mad,
            "mad_definition": "raw_median_absolute_deviation",
            "fit_split": "val_healthy",
        }

    raise ValueError(f"Unsupported threshold rule: {rule}")


def evaluate_rule(
    *,
    val_scores: np.ndarray,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    rule: str,
) -> dict[str, Any]:
    threshold_meta = select_threshold(val_scores, rule)
    metrics = evaluate_thresholded_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_scores=np.asarray(test_healthy_scores, dtype=np.float32),
        test_fault_scores=np.asarray(test_fault_scores, dtype=np.float32),
    )
    return {
        "threshold_meta": threshold_meta,
        "metrics": {
            **metrics,
            "threshold": float(threshold_meta["threshold"]),
        },
    }


def summarize_rule_metrics(
    fold_results: list[dict[str, Any]],
    model_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = ["auroc", "auprc", "f1", "precision", "recall_fault", "false_alarm_rate", "threshold"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for rule in THRESHOLD_RULES:
        summary[rule] = {}
        for metric_name in metric_names:
            values = [
                float(fold["models"][model_name]["rules"][rule]["metrics"][metric_name])
                for fold in fold_results
            ]
            summary[rule][metric_name] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
            }
    return summary


def choose_best_false_alarm_rule(rule_summary: dict[str, dict[str, dict[str, float]]]) -> str:
    return min(
        THRESHOLD_RULES,
        key=lambda rule: (
            rule_summary[rule]["false_alarm_rate"]["mean"],
            -rule_summary[rule]["f1"]["mean"],
            -rule_summary[rule]["precision"]["mean"],
            -rule_summary[rule]["recall_fault"]["mean"],
        ),
    )


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["(none)" for _ in headers]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_fold_table(fold_result: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
        for rule in THRESHOLD_RULES:
            metrics = fold_result["models"][model_name]["rules"][rule]["metrics"]
            rows.append(
                [
                    model_name,
                    rule,
                    f"{metrics['threshold']:.6f}",
                    f"{metrics['auroc']:.6f}",
                    f"{metrics['auprc']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall_fault']:.6f}",
                    f"{metrics['false_alarm_rate']:.6f}",
                ]
            )
    return format_markdown_table(
        ["Model", "Threshold Rule", "Threshold", "AUROC", "AUPRC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
        rows,
    )


def build_model_summary_table(rule_summary: dict[str, dict[str, dict[str, float]]]) -> str:
    rows: list[list[Any]] = []
    for rule in THRESHOLD_RULES:
        rows.append(
            [
                rule,
                f"{rule_summary[rule]['threshold']['mean']:.6f} +/- {rule_summary[rule]['threshold']['std']:.6f}",
                f"{rule_summary[rule]['auroc']['mean']:.6f} +/- {rule_summary[rule]['auroc']['std']:.6f}",
                f"{rule_summary[rule]['auprc']['mean']:.6f} +/- {rule_summary[rule]['auprc']['std']:.6f}",
                f"{rule_summary[rule]['f1']['mean']:.6f} +/- {rule_summary[rule]['f1']['std']:.6f}",
                f"{rule_summary[rule]['precision']['mean']:.6f} +/- {rule_summary[rule]['precision']['std']:.6f}",
                f"{rule_summary[rule]['recall_fault']['mean']:.6f} +/- {rule_summary[rule]['recall_fault']['std']:.6f}",
                f"{rule_summary[rule]['false_alarm_rate']['mean']:.6f} +/- {rule_summary[rule]['false_alarm_rate']['std']:.6f}",
            ]
        )
    return format_markdown_table(
        [
            "Threshold Rule",
            "Threshold mean+/-std",
            "AUROC mean+/-std",
            "AUPRC mean+/-std",
            "F1 mean+/-std",
            "Precision mean+/-std",
            "Recall Fault mean+/-std",
            "False Alarm Rate mean+/-std",
        ],
        rows,
    )


def benefit_note(
    summary_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    best_rule_by_model: dict[str, str],
) -> str:
    deltas: dict[str, float] = {}
    f1_deltas: dict[str, float] = {}
    for model_name, rule_summary in summary_by_model.items():
        base_far = rule_summary["mean_plus_3std"]["false_alarm_rate"]["mean"]
        best_far = rule_summary[best_rule_by_model[model_name]]["false_alarm_rate"]["mean"]
        deltas[model_name] = float(base_far - best_far)

        base_f1 = rule_summary["mean_plus_3std"]["f1"]["mean"]
        best_f1 = rule_summary[best_rule_by_model[model_name]]["f1"]["mean"]
        f1_deltas[model_name] = float(best_f1 - base_f1)

    ae_delta = deltas["AE"]
    shallow_best_delta = max(deltas["OC-SVM"], deltas["Isolation Forest"])
    if ae_delta > shallow_best_delta + 1e-12:
        return (
            "AE benefits the most from improved calibration in terms of mean false-alarm reduction "
            f"({ae_delta:.6f}), with mean F1 change {f1_deltas['AE']:+.6f}."
        )
    if abs(ae_delta - shallow_best_delta) <= 1e-12:
        return (
            "AE benefits about as much as the strongest shallow baseline from improved calibration, "
            f"with mean false-alarm reduction {ae_delta:.6f} and mean F1 change {f1_deltas['AE']:+.6f}."
        )
    return (
        "AE benefits from improved calibration, but at least one shallow baseline benefits more in mean "
        f"false-alarm reduction. AE changes by {ae_delta:.6f} with mean F1 change {f1_deltas['AE']:+.6f}."
    )


def calibration_story_note(
    summary_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    best_rule_by_model: dict[str, str],
) -> str:
    ae_rule = best_rule_by_model["AE"]
    ae_far = summary_by_model["AE"][ae_rule]["false_alarm_rate"]["mean"]
    ae_f1 = summary_by_model["AE"][ae_rule]["f1"]["mean"]
    return (
        "The score ranking remains robust under load shift, but absolute score calibration drifts enough that a "
        "single validation-derived threshold can over-trigger on some unseen loads. For VibeTwin, this points to "
        "a calibration problem rather than a representation problem: the models can separate healthy and faulty "
        f"windows, yet deployment needs thresholding that is robust to operating-condition shift. Under this study, "
        f"AE's best calibration rule is `{ae_rule}`, yielding mean false alarm rate `{ae_far:.6f}` and mean F1 "
        f"`{ae_f1:.6f}` across loads."
    )


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    load_shift_source: str,
    fold_results: list[dict[str, Any]],
    summary_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    best_rule_by_model: dict[str, str],
    best_rule_notes: dict[str, str],
    ae_benefit_note: str,
    calibration_note: str,
    artifact_paths: dict[str, Path],
) -> str:
    lines = [
        "# CWRU Threshold Calibration Report",
        "",
        "## Protocol",
        "- Same leave-one-load-out folds as the load-shift study.",
        "- Same healthy-train-only model fitting.",
        "- Same healthy-val-only threshold fitting.",
        "- Same held-out healthy plus held-out fault testing.",
        "- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.",
        "- MAD uses the raw median absolute deviation over healthy validation scores.",
        f"- Load-shift base: {load_shift_source}",
        "",
        "## Dataset Defaults Reused",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Window stride: `{preprocessing_config['stride']}`",
        f"- Original preprocessing normalization: `{preprocessing_config['normalization']['method']}` fit on `{preprocessing_config['normalization']['fit_on']}`",
        "",
    ]

    for fold_result in fold_results:
        counts = fold_result["counts"]
        lines.extend(
            [
                f"## Held-Out Load {fold_result['held_out_load_hp']}",
                f"- Train healthy windows: `{counts['train_healthy']}`",
                f"- Val healthy windows: `{counts['val_healthy']}`",
                f"- Test healthy windows: `{counts['test_healthy']}`",
                f"- Test fault windows: `{counts['test_fault']}`",
                build_fold_table(fold_result),
                "",
            ]
        )

    lines.append("## Mean/Std Across Loads")
    for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
        lines.extend(
            [
                f"### {model_name}",
                build_model_summary_table(summary_by_model[model_name]),
                f"- Best false-alarm rule: `{best_rule_by_model[model_name]}`",
                f"- {best_rule_notes[model_name]}",
                "",
            ]
        )

    lines.extend(
        [
            "## Calibration Interpretation",
            f"- {ae_benefit_note}",
            f"- {calibration_note}",
            "",
            "## Saved Artifacts",
            f"- Metrics JSON: `{artifact_paths['metrics'].as_posix()}`",
            f"- Report: `{artifact_paths['report'].as_posix()}`",
            f"- Summary plot: `{artifact_paths['plot'].as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def plot_summary(
    summary_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), dpi=160, sharex=True)
    metric_specs = [
        ("false_alarm_rate", "Mean False Alarm Rate"),
        ("f1", "Mean F1"),
    ]
    model_styles = {
        "AE": {"color": "#1f4e79", "marker": "o"},
        "OC-SVM": {"color": "#6aa84f", "marker": "s"},
        "Isolation Forest": {"color": "#cc7a00", "marker": "^"},
    }

    x = np.arange(len(THRESHOLD_RULES), dtype=np.float64)
    for axis, (metric_name, ylabel) in zip(axes, metric_specs, strict=True):
        for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
            means = [summary_by_model[model_name][rule][metric_name]["mean"] for rule in THRESHOLD_RULES]
            stds = [summary_by_model[model_name][rule][metric_name]["std"] for rule in THRESHOLD_RULES]
            axis.errorbar(
                x,
                means,
                yerr=stds,
                label=model_name,
                linewidth=2.0,
                markersize=6,
                capsize=4,
                **model_styles[model_name],
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        if metric_name == "f1":
            axis.set_ylim(0.85, 1.01)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(THRESHOLD_RULES, rotation=20, ha="right")
    axes[0].legend(loc="upper center", ncol=3, frameon=False)
    figure.suptitle("CWRU Threshold Calibration Under Load Shift", fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    args = parse_args()

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    metrics_dir = artifacts_root / "metrics"
    plots_dir = artifacts_root / "plots"
    for directory in (metrics_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    normalization_stats = read_json(metadata_root / "normalization_stats.json")
    manifest_rows = load_manifest_rows(metadata_root / "window_manifest.csv")
    expected_width = int(preprocessing_config["window_size"])

    train_healthy = load_window_array(array_paths.train_healthy, expected_width)
    val_healthy = load_window_array(array_paths.val_healthy, expected_width)
    test_healthy = load_window_array(array_paths.test_healthy, expected_width)
    test_fault = load_window_array(array_paths.test_fault, expected_width)
    fault_labels = load_label_array(array_paths.fault_labels)

    train_rows = filter_manifest_rows(manifest_rows, split="train", condition="healthy")
    val_rows = filter_manifest_rows(manifest_rows, split="val", condition="healthy")
    test_healthy_rows = filter_manifest_rows(manifest_rows, split="test", condition="healthy")
    test_fault_rows = filter_manifest_rows(manifest_rows, split="test", condition="fault")

    indexed_train = build_indexed_windows("train_healthy", train_healthy, train_rows)
    indexed_val = build_indexed_windows("val_healthy", val_healthy, val_rows)
    indexed_test_healthy = build_indexed_windows("test_healthy", test_healthy, test_healthy_rows)
    indexed_test_fault = build_indexed_windows("test_fault", test_fault, test_fault_rows)

    if fault_labels.shape[0] != indexed_test_fault.windows.shape[0]:
        raise RuntimeError(
            f"fault_labels length does not match test_fault rows: {fault_labels.shape[0]} != "
            f"{indexed_test_fault.windows.shape[0]}"
        )

    original_mean = float(normalization_stats["mean"])
    original_std = float(normalization_stats["std"])
    raw_train = build_indexed_windows(
        "raw_train",
        recover_raw_windows(indexed_train.windows, original_mean, original_std),
        train_rows,
    )
    raw_val = build_indexed_windows(
        "raw_val",
        recover_raw_windows(indexed_val.windows, original_mean, original_std),
        val_rows,
    )
    raw_test_healthy = build_indexed_windows(
        "raw_test_healthy",
        recover_raw_windows(indexed_test_healthy.windows, original_mean, original_std),
        test_healthy_rows,
    )
    raw_test_fault = build_indexed_windows(
        "raw_test_fault",
        recover_raw_windows(indexed_test_fault.windows, original_mean, original_std),
        test_fault_rows,
    )

    held_out_loads = sorted(
        set(int(load) for load in raw_test_healthy.load_hp.tolist())
        & set(int(load) for load in raw_test_fault.load_hp.tolist())
    )
    if held_out_loads != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected held-out loads [0, 1, 2, 3], found {held_out_loads}")

    load_shift_source = (
        LOAD_SHIFT_METRICS_PATH.as_posix()
        if LOAD_SHIFT_METRICS_PATH.exists()
        else "load-shift artifact not found; study still reuses the same script helpers and fold logic"
    )

    fold_results: list[dict[str, Any]] = []
    for held_out_load_hp in held_out_loads:
        print(f"\nRunning threshold calibration sweep for held-out load {held_out_load_hp}")
        fold = build_fold_dataset(
            held_out_load_hp=held_out_load_hp,
            raw_train_healthy=raw_train,
            raw_val_healthy=raw_val,
            raw_test_healthy=raw_test_healthy,
            raw_test_fault=raw_test_fault,
            fault_labels=fault_labels,
        )

        ae_base = train_and_evaluate_ae(
            fold=fold,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold_rule="mean_plus_3std",
            seed=args.seed,
        )
        shallow_base = train_and_evaluate_shallow_models(
            fold=fold,
            window_size=expected_width,
            psd_band_count=args.psd_band_count,
            threshold_rule="mean_plus_3std",
            ocsvm_nu=args.ocsvm_nu,
            iforest_n_estimators=args.iforest_n_estimators,
            iforest_max_samples=args.iforest_max_samples,
            iforest_n_jobs=args.iforest_n_jobs,
            seed=args.seed,
        )

        model_payloads = {
            "AE": ae_base,
            "OC-SVM": shallow_base["OC-SVM"],
            "Isolation Forest": shallow_base["Isolation Forest"],
        }
        fold_model_results: dict[str, Any] = {}
        for model_name, payload in model_payloads.items():
            rules: dict[str, Any] = {}
            for rule in THRESHOLD_RULES:
                rules[rule] = evaluate_rule(
                    val_scores=payload["val_scores"],
                    test_healthy_scores=payload["test_healthy_scores"],
                    test_fault_scores=payload["test_fault_scores"],
                    rule=rule,
                )

            model_entry: dict[str, Any] = {
                "rules": rules,
            }
            if model_name == "AE":
                model_entry["training"] = {
                    "history": payload["history"],
                    "parameter_count": payload["metrics"]["parameter_count"],
                    "final_train_loss": payload["metrics"]["final_train_loss"],
                    "final_val_loss": payload["metrics"]["final_val_loss"],
                    "batch_shape_check": payload["batch_shape_check"],
                }
            else:
                model_entry["training"] = {
                    "feature_count": payload["metrics"]["feature_count"],
                    "feature_names": payload["feature_names"],
                    "hyperparameters": payload["hyperparameters"],
                }
            fold_model_results[model_name] = model_entry

        fold_results.append(
            {
                "held_out_load_hp": held_out_load_hp,
                "counts": fold.counts,
                "fold_normalization": {
                    "mean": fold.fold_mean,
                    "std": fold.fold_std,
                },
                "models": fold_model_results,
            }
        )

        for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
            mean3_far = fold_model_results[model_name]["rules"]["mean_plus_3std"]["metrics"]["false_alarm_rate"]
            best_rule = min(
                THRESHOLD_RULES,
                key=lambda rule: (
                    fold_model_results[model_name]["rules"][rule]["metrics"]["false_alarm_rate"],
                    -fold_model_results[model_name]["rules"][rule]["metrics"]["f1"],
                ),
            )
            best_far = fold_model_results[model_name]["rules"][best_rule]["metrics"]["false_alarm_rate"]
            print(
                f"  {model_name}: mean_plus_3std FAR={mean3_far:.6f}, "
                f"best rule={best_rule}, best FAR={best_far:.6f}"
            )

    summary_by_model = {
        model_name: summarize_rule_metrics(fold_results, model_name)
        for model_name in ["AE", "OC-SVM", "Isolation Forest"]
    }
    best_rule_by_model = {
        model_name: choose_best_false_alarm_rule(rule_summary)
        for model_name, rule_summary in summary_by_model.items()
    }
    best_rule_notes = {}
    for model_name, best_rule in best_rule_by_model.items():
        base_far = summary_by_model[model_name]["mean_plus_3std"]["false_alarm_rate"]["mean"]
        best_far = summary_by_model[model_name][best_rule]["false_alarm_rate"]["mean"]
        base_f1 = summary_by_model[model_name]["mean_plus_3std"]["f1"]["mean"]
        best_f1 = summary_by_model[model_name][best_rule]["f1"]["mean"]
        best_rule_notes[model_name] = (
            f"Mean false alarm rate improves from `{base_far:.6f}` to `{best_far:.6f}` "
            f"with mean F1 changing from `{base_f1:.6f}` to `{best_f1:.6f}`."
        )

    ae_benefit = benefit_note(summary_by_model, best_rule_by_model)
    calibration_note = calibration_story_note(summary_by_model, best_rule_by_model)

    artifact_paths = {
        "metrics": metrics_dir / "cwru_threshold_calibration_metrics.json",
        "report": metrics_dir / "cwru_threshold_calibration_report.md",
        "plot": plots_dir / "cwru_threshold_calibration_summary.png",
    }

    metrics_payload = {
        "protocol": {
            "name": "cwru_threshold_calibration_under_load_shift",
            "held_out_loads": held_out_loads,
            "threshold_rules": list(THRESHOLD_RULES),
            "threshold_fit_split": "healthy_val_only",
            "model_fit_split": "healthy_train_only",
            "mad_definition": "raw_median_absolute_deviation",
            "load_shift_source": load_shift_source,
        },
        "defaults": {
            "ae": {
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "device": "cpu",
            },
            "ocsvm": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": float(args.ocsvm_nu),
            },
            "iforest": {
                "n_estimators": int(args.iforest_n_estimators),
                "max_samples": int(args.iforest_max_samples),
                "contamination": "auto",
                "n_jobs": int(args.iforest_n_jobs),
            },
            "feature_pipeline": {
                "psd_band_count": int(args.psd_band_count),
            },
            "seed": int(args.seed),
            "source_normalization_stats": normalization_stats,
        },
        "folds": fold_results,
        "summary_by_model": summary_by_model,
        "best_rule_by_model": best_rule_by_model,
        "best_rule_notes": best_rule_notes,
        "ae_benefit_note": ae_benefit,
        "calibration_note": calibration_note,
    }
    with artifact_paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
        handle.write("\n")

    report_text = build_report(
        preprocessing_config=preprocessing_config,
        load_shift_source=load_shift_source,
        fold_results=fold_results,
        summary_by_model=summary_by_model,
        best_rule_by_model=best_rule_by_model,
        best_rule_notes=best_rule_notes,
        ae_benefit_note=ae_benefit,
        calibration_note=calibration_note,
        artifact_paths=artifact_paths,
    )
    artifact_paths["report"].write_text(report_text, encoding="utf-8")
    plot_summary(summary_by_model, artifact_paths["plot"])

    print("\nBest false-alarm rules by model")
    for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
        print(f"  {model_name}: {best_rule_by_model[model_name]} -> {best_rule_notes[model_name]}")
    print(f"  AE benefit note: {ae_benefit}")
    print(f"  Calibration note: {calibration_note}")
    print(f"  Saved metrics: {artifact_paths['metrics'].as_posix()}")
    print(f"  Saved report: {artifact_paths['report'].as_posix()}")
    print(f"  Saved plot: {artifact_paths['plot'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
