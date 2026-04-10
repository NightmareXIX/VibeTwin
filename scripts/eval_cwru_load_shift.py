from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_ae_baseline import (
    CompactConvAutoencoder,
    compute_reconstruction_errors,
    evaluate_scores as evaluate_ae_scores,
    make_loader,
    parameter_count,
    require_torch,
    select_threshold as select_ae_threshold,
    set_seed as set_ae_seed,
    train_model,
)
from train_shallow_baselines import (
    build_feature_config,
    compute_anomaly_scores,
    evaluate_scores as evaluate_shallow_scores,
    extract_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


@dataclass(frozen=True)
class IndexedWindows:
    windows: np.ndarray
    load_hp: np.ndarray
    classes: tuple[str, ...]
    source_ids: np.ndarray


@dataclass(frozen=True)
class FoldDataset:
    held_out_load_hp: int
    train_healthy: np.ndarray
    val_healthy: np.ndarray
    test_healthy: np.ndarray
    test_fault: np.ndarray
    fault_labels: np.ndarray
    fold_mean: float
    fold_std: float
    counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-load-out CWRU load-shift evaluation for AE, OC-SVM, and Isolation Forest.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-rule",
        choices=("mean_plus_3std",),
        default="mean_plus_3std",
        help="Threshold rule applied on healthy validation scores inside each fold.",
    )
    parser.add_argument(
        "--psd-band-count",
        type=int,
        default=5,
        help="Number of PSD band-energy ratio features used for shallow baselines.",
    )
    parser.add_argument(
        "--ocsvm-nu",
        type=float,
        default=0.05,
        help="OC-SVM upper bound on healthy-train outliers.",
    )
    parser.add_argument(
        "--iforest-n-estimators",
        type=int,
        default=300,
        help="Number of trees used by Isolation Forest.",
    )
    parser.add_argument(
        "--iforest-max-samples",
        type=int,
        default=256,
        help="Maximum healthy-train samples per Isolation Forest tree.",
    )
    parser.add_argument(
        "--iforest-n-jobs",
        type=int,
        default=1,
        help="Isolation Forest worker count. Defaulting to 1 avoids Windows access-denied IPC issues.",
    )
    return parser.parse_args()


def resolve_paths(processed_root: Path) -> ArrayPaths:
    return ArrayPaths(
        train_healthy=processed_root / "train" / "healthy_windows.npy",
        val_healthy=processed_root / "val" / "healthy_windows.npy",
        test_healthy=processed_root / "test" / "healthy_windows.npy",
        test_fault=processed_root / "test" / "fault_windows.npy",
        fault_labels=processed_root / "test" / "fault_labels.npy",
    )


def ensure_required_files(paths: ArrayPaths, metadata_root: Path) -> None:
    required = [
        paths.train_healthy,
        paths.val_healthy,
        paths.test_healthy,
        paths.test_fault,
        paths.fault_labels,
        metadata_root / "normalization_stats.json",
        metadata_root / "preprocessing_config.json",
        metadata_root / "fault_label_map.json",
        metadata_root / "window_manifest.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Required inputs are missing: {formatted}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_window_array(path: Path, expected_width: int) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"Unexpected array shape for {path.as_posix()}: {array.shape}")
    return np.asarray(array, dtype=np.float32)


def load_label_array(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D label array for {path.as_posix()}, got {array.shape}")
    return np.asarray(array, dtype=np.int64)


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    condition: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split and row["condition"] == condition]


def build_indexed_windows(name: str, windows: np.ndarray, rows: list[dict[str, Any]]) -> IndexedWindows:
    if windows.shape[0] != len(rows):
        raise RuntimeError(
            f"Manifest rows for {name} do not match array rows: {len(rows)} != {windows.shape[0]}"
        )
    return IndexedWindows(
        windows=windows,
        load_hp=np.asarray([int(row["load_hp"]) for row in rows], dtype=np.int64),
        classes=tuple(row["class"] for row in rows),
        source_ids=np.asarray([int(row["source_id"]) for row in rows], dtype=np.int64),
    )


def recover_raw_windows(normalized_windows: np.ndarray, original_mean: float, original_std: float) -> np.ndarray:
    return (
        np.asarray(normalized_windows, dtype=np.float64) * float(original_std) + float(original_mean)
    ).astype(np.float32)


def normalize_windows(raw_windows: np.ndarray, fit_mean: float, fit_std: float) -> np.ndarray:
    if fit_std <= 0.0:
        raise RuntimeError("Fold training windows have zero standard deviation; cannot z-score normalize.")
    normalized = (np.asarray(raw_windows, dtype=np.float64) - fit_mean) / fit_std
    return np.asarray(normalized, dtype=np.float32)


def build_fold_dataset(
    *,
    held_out_load_hp: int,
    raw_train_healthy: IndexedWindows,
    raw_val_healthy: IndexedWindows,
    raw_test_healthy: IndexedWindows,
    raw_test_fault: IndexedWindows,
    fault_labels: np.ndarray,
) -> FoldDataset:
    train_mask = raw_train_healthy.load_hp != held_out_load_hp
    val_mask = raw_val_healthy.load_hp != held_out_load_hp
    test_healthy_mask = raw_test_healthy.load_hp == held_out_load_hp
    test_fault_mask = raw_test_fault.load_hp == held_out_load_hp

    train_raw = raw_train_healthy.windows[train_mask]
    val_raw = raw_val_healthy.windows[val_mask]
    test_healthy_raw = raw_test_healthy.windows[test_healthy_mask]
    test_fault_raw = raw_test_fault.windows[test_fault_mask]
    fold_fault_labels = fault_labels[test_fault_mask]

    if train_raw.size == 0 or val_raw.size == 0:
        raise RuntimeError(f"Held-out load {held_out_load_hp}: empty healthy train or val fold.")
    if test_healthy_raw.size == 0 or test_fault_raw.size == 0:
        raise RuntimeError(f"Held-out load {held_out_load_hp}: empty held-out healthy or fault test fold.")
    if fold_fault_labels.shape[0] != test_fault_raw.shape[0]:
        raise RuntimeError(
            f"Held-out load {held_out_load_hp}: fault label length mismatch "
            f"{fold_fault_labels.shape[0]} != {test_fault_raw.shape[0]}"
        )

    fold_mean = float(train_raw.mean())
    fold_std = float(train_raw.std())

    return FoldDataset(
        held_out_load_hp=held_out_load_hp,
        train_healthy=normalize_windows(train_raw, fold_mean, fold_std),
        val_healthy=normalize_windows(val_raw, fold_mean, fold_std),
        test_healthy=normalize_windows(test_healthy_raw, fold_mean, fold_std),
        test_fault=normalize_windows(test_fault_raw, fold_mean, fold_std),
        fault_labels=fold_fault_labels,
        fold_mean=fold_mean,
        fold_std=fold_std,
        counts={
            "train_healthy": int(train_raw.shape[0]),
            "val_healthy": int(val_raw.shape[0]),
            "test_healthy": int(test_healthy_raw.shape[0]),
            "test_fault": int(test_fault_raw.shape[0]),
        },
    )


def train_and_evaluate_ae(
    *,
    fold: FoldDataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    threshold_rule: str,
    seed: int,
) -> dict[str, Any]:
    set_ae_seed(seed)
    device = torch.device("cpu")
    model = CompactConvAutoencoder().to(device)

    train_loader = make_loader(fold.train_healthy, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(fold.val_healthy, batch_size=batch_size, shuffle=False)
    test_healthy_loader = make_loader(fold.test_healthy, batch_size=batch_size, shuffle=False)
    test_fault_loader = make_loader(fold.test_fault, batch_size=batch_size, shuffle=False)

    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_output = model(sample_batch.to(device)).cpu()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    val_errors = compute_reconstruction_errors(model, val_loader, device)
    test_healthy_errors = compute_reconstruction_errors(model, test_healthy_loader, device)
    test_fault_errors = compute_reconstruction_errors(model, test_fault_loader, device)
    threshold_meta = select_ae_threshold(val_errors, threshold_rule)
    metrics = evaluate_ae_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_errors=test_healthy_errors,
        test_fault_errors=test_fault_errors,
    )
    return {
        "metrics": {
            **metrics,
            "parameter_count": int(parameter_count(model)),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "threshold": float(threshold_meta["threshold"]),
        },
        "threshold_meta": threshold_meta,
        "history": history,
        "batch_shape_check": {
            "input": [int(dim) for dim in sample_batch.shape],
            "output": [int(dim) for dim in sample_output.shape],
        },
        "val_scores": val_errors,
        "test_healthy_scores": test_healthy_errors,
        "test_fault_scores": test_fault_errors,
    }


def select_score_threshold(scores: np.ndarray, rule: str) -> dict[str, Any]:
    if scores.size == 0:
        raise RuntimeError("Validation healthy scores are empty; cannot select an anomaly threshold.")

    if rule == "mean_plus_3std":
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        threshold = score_mean + (3.0 * score_std)
        return {
            "rule": rule,
            "threshold": float(threshold),
            "validation_score_mean": score_mean,
            "validation_score_std": score_std,
            "fit_split": "val_healthy",
        }

    raise ValueError(f"Unsupported threshold rule: {rule}")


def train_and_evaluate_shallow_models(
    *,
    fold: FoldDataset,
    window_size: int,
    psd_band_count: int,
    threshold_rule: str,
    ocsvm_nu: float,
    iforest_n_estimators: int,
    iforest_max_samples: int,
    iforest_n_jobs: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    feature_config = build_feature_config(window_size, psd_band_count)
    train_features_raw = extract_features(fold.train_healthy, feature_config)
    val_features_raw = extract_features(fold.val_healthy, feature_config)
    test_healthy_features_raw = extract_features(fold.test_healthy, feature_config)
    test_fault_features_raw = extract_features(fold.test_fault, feature_config)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features_raw)
    val_features = scaler.transform(val_features_raw)
    test_healthy_features = scaler.transform(test_healthy_features_raw)
    test_fault_features = scaler.transform(test_fault_features_raw)

    models = {
        "OC-SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=ocsvm_nu),
        "Isolation Forest": IsolationForest(
            n_estimators=iforest_n_estimators,
            max_samples=min(iforest_max_samples, train_features.shape[0]),
            contamination="auto",
            random_state=seed,
            n_jobs=iforest_n_jobs,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(train_features)
        val_scores = compute_anomaly_scores(model, val_features)
        test_healthy_scores = compute_anomaly_scores(model, test_healthy_features)
        test_fault_scores = compute_anomaly_scores(model, test_fault_features)
        threshold_meta = select_score_threshold(val_scores, threshold_rule)
        metrics = evaluate_shallow_scores(
            threshold=float(threshold_meta["threshold"]),
            test_healthy_scores=test_healthy_scores,
            test_fault_scores=test_fault_scores,
        )
        hyperparameters = (
            {"kernel": "rbf", "gamma": "scale", "nu": float(ocsvm_nu)}
            if model_name == "OC-SVM"
            else {
                "n_estimators": int(iforest_n_estimators),
                "max_samples": int(min(iforest_max_samples, train_features.shape[0])),
                "contamination": "auto",
                "random_state": int(seed),
                "n_jobs": int(iforest_n_jobs),
            }
        )
        results[model_name] = {
            "metrics": {
                **metrics,
                "threshold": float(threshold_meta["threshold"]),
                "feature_count": int(train_features.shape[1]),
            },
            "threshold_meta": threshold_meta,
            "val_scores": val_scores,
            "test_healthy_scores": test_healthy_scores,
            "test_fault_scores": test_fault_scores,
            "feature_names": list(feature_config.feature_names),
            "hyperparameters": hyperparameters,
        }
    return results


def summarize_metrics_across_folds(
    fold_results: list[dict[str, Any]],
    model_order: list[str],
    metric_names: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in model_order:
        summary[model_name] = {}
        for metric_name in metric_names:
            values = [float(fold["models"][model_name]["metrics"][metric_name]) for fold in fold_results]
            summary[model_name][metric_name] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
            }
    return summary


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
        metrics = fold_result["models"][model_name]["metrics"]
        rows.append(
            [
                model_name,
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
        ["Model", "Threshold", "AUROC", "AUPRC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
        rows,
    )


def build_summary_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    rows: list[list[Any]] = []
    for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
        rows.append(
            [
                model_name,
                f"{summary[model_name]['auroc']['mean']:.6f} +/- {summary[model_name]['auroc']['std']:.6f}",
                f"{summary[model_name]['auprc']['mean']:.6f} +/- {summary[model_name]['auprc']['std']:.6f}",
                f"{summary[model_name]['f1']['mean']:.6f} +/- {summary[model_name]['f1']['std']:.6f}",
                f"{summary[model_name]['precision']['mean']:.6f} +/- {summary[model_name]['precision']['std']:.6f}",
                f"{summary[model_name]['recall_fault']['mean']:.6f} +/- {summary[model_name]['recall_fault']['std']:.6f}",
                f"{summary[model_name]['false_alarm_rate']['mean']:.6f} +/- {summary[model_name]['false_alarm_rate']['std']:.6f}",
            ]
        )
    return format_markdown_table(
        [
            "Model",
            "AUROC mean+/-std",
            "AUPRC mean+/-std",
            "F1 mean+/-std",
            "Precision mean+/-std",
            "Recall Fault mean+/-std",
            "False Alarm Rate mean+/-std",
        ],
        rows,
    )


def build_advantage_note(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    ae_f1 = summary["AE"]["f1"]["mean"]
    ae_far = summary["AE"]["false_alarm_rate"]["mean"]
    best_f1 = max(summary[model]["f1"]["mean"] for model in summary)
    best_far = min(summary[model]["false_alarm_rate"]["mean"] for model in summary)

    if abs(ae_f1 - best_f1) < 1e-12 and abs(ae_far - best_far) < 1e-12:
        return (
            "AE retains the clearest advantage under load shift: it matches or exceeds the other baselines on "
            "mean F1 while also keeping the lowest mean false alarm rate."
        )
    if abs(ae_f1 - best_f1) < 1e-12:
        return "AE keeps the best mean F1 under load shift, but its false alarm rate advantage is no longer unique."
    if abs(ae_far - best_far) < 1e-12:
        return (
            "AE keeps the lowest mean false alarm rate under load shift, but another shallow baseline matches or "
            "slightly exceeds it on mean F1."
        )
    return (
        "AE does not hold a clean overall advantage under load shift; the shallow baselines are at least competitive "
        "on both mean F1 and false alarm rate."
    )


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    source_note: str,
    fold_results: list[dict[str, Any]],
    summary: dict[str, dict[str, dict[str, float]]],
    advantage_note: str,
    artifact_paths: dict[str, Path],
) -> str:
    lines = [
        "# CWRU Load-Shift Report",
        "",
        "## Protocol",
        "- Leave-one-load-out evaluation across the four motor loads: `0`, `1`, `2`, and `3`.",
        "- Healthy train windows come only from non-held-out loads using the existing `train` split.",
        "- Healthy validation windows come only from non-held-out loads using the existing `val` split.",
        "- Test healthy windows come from the held-out load using the existing `test` split.",
        "- Test fault windows come from the held-out load using the existing fault-test windows.",
        "- Fold-specific z-score normalization is refit on healthy train windows only after reconstructing pre-z-score values from the saved preprocessing stats.",
        f"- {source_note}",
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

    lines.extend(
        [
            "## Mean/Std Across Folds",
            build_summary_table(summary),
            "",
            "## Model Comparison",
            f"- {advantage_note}",
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
    fold_results: list[dict[str, Any]],
    path: Path,
) -> None:
    metrics_to_plot = ["auroc", "auprc", "f1", "false_alarm_rate"]
    y_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
        "false_alarm_rate": "False Alarm Rate",
    }
    model_order = ["AE", "OC-SVM", "Isolation Forest"]
    colors = {
        "AE": "#1f4e79",
        "OC-SVM": "#6aa84f",
        "Isolation Forest": "#cc7a00",
    }

    loads = [fold["held_out_load_hp"] for fold in fold_results]
    x = np.arange(len(loads), dtype=np.float64)
    width = 0.22

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=160)
    axes = np.asarray(axes).reshape(-1)

    for axis, metric_name in zip(axes, metrics_to_plot, strict=True):
        for model_index, model_name in enumerate(model_order):
            values = [fold["models"][model_name]["metrics"][metric_name] for fold in fold_results]
            axis.bar(
                x + ((model_index - 1) * width),
                values,
                width=width,
                label=model_name,
                color=colors[model_name],
                alpha=0.9,
            )
        axis.set_title(y_labels[metric_name])
        axis.set_xticks(x)
        axis.set_xticklabels([f"Load {load}" for load in loads])
        axis.grid(axis="y", alpha=0.3)
        axis.set_ylabel(y_labels[metric_name])
        if metric_name != "false_alarm_rate":
            axis.set_ylim(0.0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("CWRU Leave-One-Load-Out Summary", fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path)
    plt.close(figure)


def make_json_ready_fold_results(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    json_ready: list[dict[str, Any]] = []
    for fold in fold_results:
        models: dict[str, Any] = {}
        for model_name, payload in fold["models"].items():
            model_entry: dict[str, Any] = {
                "metrics": payload["metrics"],
                "threshold_meta": payload["threshold_meta"],
            }
            if "history" in payload:
                model_entry["history"] = payload["history"]
            if "batch_shape_check" in payload:
                model_entry["batch_shape_check"] = payload["batch_shape_check"]
            if "hyperparameters" in payload:
                model_entry["hyperparameters"] = payload["hyperparameters"]
            if "feature_names" in payload:
                model_entry["feature_names"] = payload["feature_names"]
            models[model_name] = model_entry

        json_ready.append(
            {
                "held_out_load_hp": fold["held_out_load_hp"],
                "counts": fold["counts"],
                "fold_normalization": fold["fold_normalization"],
                "models": models,
            }
        )
    return json_ready


def main() -> int:
    require_torch()
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
    raw_train = IndexedWindows(
        windows=recover_raw_windows(indexed_train.windows, original_mean, original_std),
        load_hp=indexed_train.load_hp,
        classes=indexed_train.classes,
        source_ids=indexed_train.source_ids,
    )
    raw_val = IndexedWindows(
        windows=recover_raw_windows(indexed_val.windows, original_mean, original_std),
        load_hp=indexed_val.load_hp,
        classes=indexed_val.classes,
        source_ids=indexed_val.source_ids,
    )
    raw_test_healthy = IndexedWindows(
        windows=recover_raw_windows(indexed_test_healthy.windows, original_mean, original_std),
        load_hp=indexed_test_healthy.load_hp,
        classes=indexed_test_healthy.classes,
        source_ids=indexed_test_healthy.source_ids,
    )
    raw_test_fault = IndexedWindows(
        windows=recover_raw_windows(indexed_test_fault.windows, original_mean, original_std),
        load_hp=indexed_test_fault.load_hp,
        classes=indexed_test_fault.classes,
        source_ids=indexed_test_fault.source_ids,
    )

    held_out_loads = sorted(
        set(int(load) for load in raw_test_healthy.load_hp.tolist())
        & set(int(load) for load in raw_test_fault.load_hp.tolist())
    )
    if held_out_loads != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected held-out loads [0, 1, 2, 3], found {held_out_loads}")

    source_note = (
        "Existing `window_manifest.csv` was sufficient for load indexing; no additional preprocessing metadata file "
        "was needed."
    )
    fold_results: list[dict[str, Any]] = []

    for held_out_load_hp in held_out_loads:
        print(f"\nRunning leave-one-load-out fold with held-out load {held_out_load_hp}")
        fold = build_fold_dataset(
            held_out_load_hp=held_out_load_hp,
            raw_train_healthy=raw_train,
            raw_val_healthy=raw_val,
            raw_test_healthy=raw_test_healthy,
            raw_test_fault=raw_test_fault,
            fault_labels=fault_labels,
        )

        ae_result = train_and_evaluate_ae(
            fold=fold,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold_rule=args.threshold_rule,
            seed=args.seed,
        )
        shallow_results = train_and_evaluate_shallow_models(
            fold=fold,
            window_size=expected_width,
            psd_band_count=args.psd_band_count,
            threshold_rule=args.threshold_rule,
            ocsvm_nu=args.ocsvm_nu,
            iforest_n_estimators=args.iforest_n_estimators,
            iforest_max_samples=args.iforest_max_samples,
            iforest_n_jobs=args.iforest_n_jobs,
            seed=args.seed,
        )

        fold_result = {
            "held_out_load_hp": held_out_load_hp,
            "counts": fold.counts,
            "fold_normalization": {
                "mean": fold.fold_mean,
                "std": fold.fold_std,
            },
            "models": {
                "AE": ae_result,
                "OC-SVM": shallow_results["OC-SVM"],
                "Isolation Forest": shallow_results["Isolation Forest"],
            },
        }
        fold_results.append(fold_result)

        print(
            f"  AE: f1={ae_result['metrics']['f1']:.6f}, "
            f"false_alarm_rate={ae_result['metrics']['false_alarm_rate']:.6f}"
        )
        print(
            f"  OC-SVM: f1={shallow_results['OC-SVM']['metrics']['f1']:.6f}, "
            f"false_alarm_rate={shallow_results['OC-SVM']['metrics']['false_alarm_rate']:.6f}"
        )
        print(
            f"  Isolation Forest: f1={shallow_results['Isolation Forest']['metrics']['f1']:.6f}, "
            f"false_alarm_rate={shallow_results['Isolation Forest']['metrics']['false_alarm_rate']:.6f}"
        )

    metric_names = ["auroc", "auprc", "f1", "precision", "recall_fault", "false_alarm_rate"]
    summary = summarize_metrics_across_folds(
        fold_results=fold_results,
        model_order=["AE", "OC-SVM", "Isolation Forest"],
        metric_names=metric_names,
    )
    advantage_note = build_advantage_note(summary)

    artifact_paths = {
        "metrics": metrics_dir / "cwru_load_shift_metrics.json",
        "report": metrics_dir / "cwru_load_shift_report.md",
        "plot": plots_dir / "cwru_load_shift_summary.png",
    }

    metrics_payload = {
        "protocol": {
            "name": "cwru_leave_one_load_out",
            "held_out_loads": held_out_loads,
            "healthy_train_source": "existing train split from non-held-out loads only",
            "healthy_val_source": "existing val split from non-held-out loads only",
            "healthy_test_source": "existing test split from held-out load only",
            "fault_test_source": "existing fault test windows from held-out load only",
            "threshold_rule": args.threshold_rule,
            "fold_specific_renormalization": {
                "method": "zscore_global",
                "fit_on": "healthy_train_only_per_fold",
                "recovered_from_existing_processed_arrays": True,
                "source_normalization_stats": normalization_stats,
            },
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
        },
        "source_note": source_note,
        "folds": make_json_ready_fold_results(fold_results),
        "summary": summary,
        "advantage_note": advantage_note,
    }
    with artifact_paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
        handle.write("\n")

    report_text = build_report(
        preprocessing_config=preprocessing_config,
        source_note=source_note,
        fold_results=fold_results,
        summary=summary,
        advantage_note=advantage_note,
        artifact_paths=artifact_paths,
    )
    artifact_paths["report"].write_text(report_text, encoding="utf-8")
    plot_summary(fold_results, artifact_paths["plot"])

    print("\nLoad-shift mean/std summary")
    for model_name in ["AE", "OC-SVM", "Isolation Forest"]:
        print(
            f"  {model_name}: "
            f"F1={summary[model_name]['f1']['mean']:.6f}+/-{summary[model_name]['f1']['std']:.6f}, "
            f"FAR={summary[model_name]['false_alarm_rate']['mean']:.6f}+/-{summary[model_name]['false_alarm_rate']['std']:.6f}"
        )
    print(f"  Note: {advantage_note}")
    print(f"  Saved metrics: {artifact_paths['metrics'].as_posix()}")
    print(f"  Saved report: {artifact_paths['report'].as_posix()}")
    print(f"  Saved plot: {artifact_paths['plot'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
