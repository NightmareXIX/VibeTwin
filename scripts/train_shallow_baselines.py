from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
AE_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "cwru_ae_metrics.json"


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


@dataclass(frozen=True)
class FeatureConfig:
    feature_names: tuple[str, ...]
    band_bin_groups: tuple[tuple[int, ...], ...]
    band_frequency_ranges: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OC-SVM and Isolation Forest shallow anomaly-detection baselines on processed CWRU windows.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-rule",
        choices=("mean_plus_3std",),
        default="mean_plus_3std",
        help="Threshold selection rule applied on healthy validation anomaly scores.",
    )
    parser.add_argument(
        "--psd-band-count",
        type=int,
        default=5,
        help="Number of equal-width normalized PSD bands to summarize per window.",
    )
    parser.add_argument(
        "--ocsvm-nu",
        type=float,
        default=0.05,
        help="OC-SVM upper bound on training outliers and lower bound on support vectors.",
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
        help="Maximum number of healthy-train samples drawn per Isolation Forest tree.",
    )
    parser.add_argument(
        "--iforest-n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers used by Isolation Forest. Use 1 for the safest Windows setup.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        metadata_root / "preprocessing_config.json",
        metadata_root / "fault_label_map.json",
        metadata_root / "window_manifest.csv",
        AE_METRICS_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Required inputs are missing: {formatted}")


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


def build_feature_config(window_size: int, band_count: int) -> FeatureConfig:
    if band_count < 1:
        raise ValueError("psd_band_count must be at least 1.")

    usable_bins = np.arange(1, (window_size // 2) + 1, dtype=np.int64)
    if usable_bins.size < band_count:
        raise ValueError(
            f"Cannot build {band_count} PSD bands from only {usable_bins.size} non-DC FFT bins."
        )

    band_arrays = np.array_split(usable_bins, band_count)
    freqs = np.fft.rfftfreq(window_size, d=1.0)
    band_bin_groups = tuple(tuple(int(index) for index in band.tolist()) for band in band_arrays)
    band_frequency_ranges = tuple(
        (float(freqs[group[0]]), float(freqs[group[-1]]))
        for group in band_bin_groups
    )
    feature_names = (
        "mean",
        "std",
        "rms",
        "max",
        "min",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
        "signal_energy",
        "spectral_centroid",
        "spectral_entropy",
        *tuple(f"psd_band_{index:02d}_ratio" for index in range(band_count)),
    )
    return FeatureConfig(
        feature_names=feature_names,
        band_bin_groups=band_bin_groups,
        band_frequency_ranges=band_frequency_ranges,
    )


def extract_features(windows: np.ndarray, feature_config: FeatureConfig) -> np.ndarray:
    signals = np.asarray(windows, dtype=np.float64)
    squared = np.square(signals)

    mean = np.mean(signals, axis=1)
    std = np.std(signals, axis=1)
    rms = np.sqrt(np.mean(squared, axis=1))
    maximum = np.max(signals, axis=1)
    minimum = np.min(signals, axis=1)
    peak_to_peak = np.ptp(signals, axis=1)
    skewness = np.nan_to_num(skew(signals, axis=1, bias=False), nan=0.0, posinf=0.0, neginf=0.0)
    kurtosis_value = np.nan_to_num(
        kurtosis(signals, axis=1, fisher=True, bias=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    crest_factor = np.max(np.abs(signals), axis=1) / np.maximum(rms, 1e-12)
    signal_energy = np.sum(squared, axis=1)

    spectrum = np.fft.rfft(signals, axis=1)
    psd = np.square(np.abs(spectrum)) / signals.shape[1]
    psd[:, 0] = 0.0
    total_psd = np.sum(psd, axis=1)
    total_psd_safe = np.maximum(total_psd, 1e-12)

    freqs = np.fft.rfftfreq(signals.shape[1], d=1.0)
    spectral_centroid = np.sum(psd * freqs[None, :], axis=1) / total_psd_safe

    psd_prob = psd / total_psd_safe[:, None]
    spectral_entropy = -np.sum(psd_prob * np.log(psd_prob + 1e-12), axis=1) / np.log(psd.shape[1])

    band_energy_ratios = [
        np.sum(psd[:, band_indices], axis=1) / total_psd_safe
        for band_indices in feature_config.band_bin_groups
    ]

    feature_columns = [
        mean,
        std,
        rms,
        maximum,
        minimum,
        peak_to_peak,
        skewness,
        kurtosis_value,
        crest_factor,
        signal_energy,
        spectral_centroid,
        spectral_entropy,
        *band_energy_ratios,
    ]
    features = np.column_stack(feature_columns)
    return np.asarray(features, dtype=np.float32)


def select_threshold(scores: np.ndarray, rule: str) -> dict[str, Any]:
    if scores.size == 0:
        raise RuntimeError("Validation healthy scores are empty; cannot select an anomaly threshold.")

    if rule == "mean_plus_3std":
        mean = float(scores.mean())
        std = float(scores.std())
        threshold = mean + (3.0 * std)
        return {
            "rule": rule,
            "threshold": float(threshold),
            "validation_score_mean": mean,
            "validation_score_std": std,
            "fit_split": "val_healthy",
        }

    raise ValueError(f"Unsupported threshold rule: {rule}")


def compute_anomaly_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        normality_scores = model.decision_function(features)
    elif hasattr(model, "score_samples"):
        normality_scores = model.score_samples(features)
    else:
        raise TypeError(f"Model does not expose a scoring method: {type(model)!r}")
    anomaly_scores = -np.asarray(normality_scores, dtype=np.float64).reshape(-1)
    return np.asarray(anomaly_scores, dtype=np.float32)


def build_score_rows(
    *,
    split: str,
    true_label: int,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "true_label": true_label,
            "anomaly_score": float(score),
            "predicted_anomaly": int(score >= threshold),
        }
        for score in scores
    ]


def evaluate_scores(
    *,
    threshold: float,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
) -> dict[str, Any]:
    y_true = np.concatenate(
        [
            np.zeros(test_healthy_scores.shape[0], dtype=np.int64),
            np.ones(test_fault_scores.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    scores = np.concatenate([test_healthy_scores, test_fault_scores], axis=0)
    predictions = (scores >= threshold).astype(np.int64)

    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_fault": float(recall_score(y_true, predictions, zero_division=0)),
        "false_alarm_rate": float(predictions[: test_healthy_scores.shape[0]].mean()),
        "num_test_healthy": int(test_healthy_scores.shape[0]),
        "num_test_fault": int(test_fault_scores.shape[0]),
        "num_predicted_anomalies": int(predictions.sum()),
        "num_true_anomalies": int(y_true.sum()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "true_label", "anomaly_score", "predicted_anomaly"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fault_distribution(
    fault_labels: np.ndarray,
    fault_label_map: dict[str, Any],
) -> dict[str, int]:
    index_to_class = {
        int(index): name for index, name in fault_label_map["integer_to_class"].items()
    }
    counts = Counter(int(label) for label in fault_labels.tolist())
    return {
        index_to_class.get(index, str(index)): int(count)
        for index, count in sorted(counts.items())
    }


def compare_table(
    ae_metrics: dict[str, Any],
    ocsvm_metrics: dict[str, Any],
    iforest_metrics: dict[str, Any],
) -> str:
    rows = [
        ("AE", ae_metrics),
        ("OC-SVM", ocsvm_metrics),
        ("Isolation Forest", iforest_metrics),
    ]
    lines = [
        "| Model | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in rows:
        lines.append(
            "| "
            f"{label} | "
            f"{metrics['auroc']:.6f} | "
            f"{metrics['auprc']:.6f} | "
            f"{metrics['f1']:.6f} | "
            f"{metrics['precision']:.6f} | "
            f"{metrics['recall_fault']:.6f} | "
            f"{metrics['false_alarm_rate']:.6f} |"
        )
    return "\n".join(lines)


def plot_score_histograms(results: dict[str, dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for axis, (label, payload) in zip(axes, results.items(), strict=True):
        axis.hist(
            payload["test_healthy_scores"],
            bins=40,
            alpha=0.60,
            label="Test healthy",
        )
        axis.hist(
            payload["test_fault_scores"],
            bins=40,
            alpha=0.60,
            label="Test fault",
        )
        axis.axvline(
            payload["threshold_meta"]["threshold"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Threshold",
        )
        axis.set_title(f"{label} Anomaly Scores")
        axis.set_xlabel("Anomaly Score")
        axis.set_ylabel("Window Count")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Shallow Baseline Score Distributions", fontsize=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    feature_config: FeatureConfig,
    fault_counts: dict[str, int],
    ae_metrics: dict[str, Any],
    results: dict[str, dict[str, Any]],
    artifact_paths: dict[str, Path],
) -> str:
    band_lines = [
        f"- `{feature_name}` covers normalized frequency `{start:.4f}` to `{end:.4f}`"
        for feature_name, (start, end) in zip(
            feature_config.feature_names[12:],
            feature_config.band_frequency_ranges,
            strict=True,
        )
    ]

    lines = [
        "# Shallow Baseline Report",
        "",
        "## Setup",
        f"- Processed root: `{preprocessing_config['processed_root']}`",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Window stride: `{preprocessing_config['stride']}`",
        f"- Processed normalization: `{preprocessing_config['normalization']['method']}` fit on `{preprocessing_config['normalization']['fit_on']}`",
        f"- Feature count: `{len(feature_config.feature_names)}`",
        f"- Features: `{', '.join(feature_config.feature_names)}`",
        "",
        "## PSD Band Features",
        *band_lines,
        "",
        "## Fault Window Counts",
        *[f"- `{name}`: `{count}`" for name, count in fault_counts.items()],
        "",
    ]

    for label, payload in results.items():
        metrics = payload["metrics"]
        threshold_meta = payload["threshold_meta"]
        lines.extend(
            [
                f"## {label}",
                f"- Threshold rule: `{threshold_meta['rule']}`",
                f"- Threshold: `{threshold_meta['threshold']:.6f}`",
                f"- Val score mean: `{threshold_meta['validation_score_mean']:.6f}`",
                f"- Val score std: `{threshold_meta['validation_score_std']:.6f}`",
                f"- AUROC: `{metrics['auroc']:.6f}`",
                f"- AUPRC: `{metrics['auprc']:.6f}`",
                f"- F1: `{metrics['f1']:.6f}`",
                f"- Precision: `{metrics['precision']:.6f}`",
                f"- Recall on fault windows: `{metrics['recall_fault']:.6f}`",
                f"- False alarm rate on healthy test windows: `{metrics['false_alarm_rate']:.6f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## AE Comparison",
            compare_table(
                ae_metrics=ae_metrics,
                ocsvm_metrics=results["OC-SVM"]["metrics"],
                iforest_metrics=results["Isolation Forest"]["metrics"],
            ),
            "",
            "## Saved Artifacts",
            f"- OC-SVM metrics: `{artifact_paths['ocsvm_metrics'].as_posix()}`",
            f"- Isolation Forest metrics: `{artifact_paths['iforest_metrics'].as_posix()}`",
            f"- OC-SVM scores: `{artifact_paths['ocsvm_scores'].as_posix()}`",
            f"- Isolation Forest scores: `{artifact_paths['iforest_scores'].as_posix()}`",
            f"- Shared report: `{artifact_paths['report'].as_posix()}`",
            f"- Shared plot: `{artifact_paths['plot'].as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def train_and_evaluate_model(
    *,
    model_name: str,
    model: Any,
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_healthy_features: np.ndarray,
    test_fault_features: np.ndarray,
    threshold_rule: str,
) -> dict[str, Any]:
    model.fit(train_features)

    val_scores = compute_anomaly_scores(model, val_features)
    test_healthy_scores = compute_anomaly_scores(model, test_healthy_features)
    test_fault_scores = compute_anomaly_scores(model, test_fault_features)
    threshold_meta = select_threshold(val_scores, threshold_rule)
    metrics = evaluate_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
    )

    score_rows: list[dict[str, Any]] = []
    score_rows.extend(
        build_score_rows(
            split="val_healthy",
            true_label=0,
            scores=val_scores,
            threshold=float(threshold_meta["threshold"]),
        )
    )
    score_rows.extend(
        build_score_rows(
            split="test_healthy",
            true_label=0,
            scores=test_healthy_scores,
            threshold=float(threshold_meta["threshold"]),
        )
    )
    score_rows.extend(
        build_score_rows(
            split="test_fault",
            true_label=1,
            scores=test_fault_scores,
            threshold=float(threshold_meta["threshold"]),
        )
    )

    return {
        "model_name": model_name,
        "metrics": metrics,
        "threshold_meta": threshold_meta,
        "val_scores": val_scores,
        "test_healthy_scores": test_healthy_scores,
        "test_fault_scores": test_fault_scores,
        "score_rows": score_rows,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

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
    fault_label_map = read_json(metadata_root / "fault_label_map.json")
    ae_metrics = read_json(AE_METRICS_PATH)
    expected_width = int(preprocessing_config["window_size"])

    train_windows = load_window_array(array_paths.train_healthy, expected_width)
    val_windows = load_window_array(array_paths.val_healthy, expected_width)
    test_healthy_windows = load_window_array(array_paths.test_healthy, expected_width)
    test_fault_windows = load_window_array(array_paths.test_fault, expected_width)
    fault_labels = load_label_array(array_paths.fault_labels)

    if train_windows.size == 0 or val_windows.size == 0:
        raise RuntimeError("Healthy train and val arrays must both be non-empty.")
    if test_healthy_windows.size == 0 or test_fault_windows.size == 0:
        raise RuntimeError("Healthy test and fault test arrays must both be non-empty for evaluation.")
    if fault_labels.shape[0] != test_fault_windows.shape[0]:
        raise RuntimeError("fault_labels.npy must contain one label per fault test window.")

    feature_config = build_feature_config(expected_width, args.psd_band_count)
    train_features_raw = extract_features(train_windows, feature_config)
    val_features_raw = extract_features(val_windows, feature_config)
    test_healthy_features_raw = extract_features(test_healthy_windows, feature_config)
    test_fault_features_raw = extract_features(test_fault_windows, feature_config)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features_raw)
    val_features = scaler.transform(val_features_raw)
    test_healthy_features = scaler.transform(test_healthy_features_raw)
    test_fault_features = scaler.transform(test_fault_features_raw)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=args.ocsvm_nu)
    iforest = IsolationForest(
        n_estimators=args.iforest_n_estimators,
        max_samples=min(args.iforest_max_samples, train_features.shape[0]),
        contamination="auto",
        random_state=args.seed,
        n_jobs=args.iforest_n_jobs,
    )

    results = {
        "OC-SVM": train_and_evaluate_model(
            model_name="ocsvm",
            model=ocsvm,
            train_features=train_features,
            val_features=val_features,
            test_healthy_features=test_healthy_features,
            test_fault_features=test_fault_features,
            threshold_rule=args.threshold_rule,
        ),
        "Isolation Forest": train_and_evaluate_model(
            model_name="iforest",
            model=iforest,
            train_features=train_features,
            val_features=val_features,
            test_healthy_features=test_healthy_features,
            test_fault_features=test_fault_features,
            threshold_rule=args.threshold_rule,
        ),
    }

    artifact_paths = {
        "ocsvm_metrics": metrics_dir / "cwru_ocsvm_metrics.json",
        "iforest_metrics": metrics_dir / "cwru_iforest_metrics.json",
        "ocsvm_scores": metrics_dir / "cwru_ocsvm_scores.csv",
        "iforest_scores": metrics_dir / "cwru_iforest_scores.csv",
        "report": metrics_dir / "cwru_shallow_report.md",
        "plot": plots_dir / "cwru_shallow_score_hists.png",
    }

    fault_counts = fault_distribution(fault_labels, fault_label_map)

    for label, payload in results.items():
        metrics_payload = {
            **payload["metrics"],
            "model_name": payload["model_name"],
            "threshold": float(payload["threshold_meta"]["threshold"]),
            "threshold_rule": payload["threshold_meta"]["rule"],
            "validation_score_mean": float(payload["threshold_meta"]["validation_score_mean"]),
            "validation_score_std": float(payload["threshold_meta"]["validation_score_std"]),
            "feature_count": int(len(feature_config.feature_names)),
            "feature_names": list(feature_config.feature_names),
            "psd_band_frequency_ranges": [
                {
                    "feature": feature_name,
                    "normalized_frequency_start": start,
                    "normalized_frequency_end": end,
                }
                for feature_name, (start, end) in zip(
                    feature_config.feature_names[12:],
                    feature_config.band_frequency_ranges,
                    strict=True,
                )
            ],
            "train_feature_shape": list(train_features.shape),
            "val_feature_shape": list(val_features.shape),
            "test_healthy_feature_shape": list(test_healthy_features.shape),
            "test_fault_feature_shape": list(test_fault_features.shape),
            "seed": int(args.seed),
            "device": "cpu",
            "score_direction": "higher_is_more_anomalous",
            "scaler": "StandardScaler fit on healthy train only",
            "fault_label_counts": fault_counts,
            "normalization_metadata": preprocessing_config["normalization"],
        }
        if label == "OC-SVM":
            metrics_payload["hyperparameters"] = {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": float(args.ocsvm_nu),
            }
            write_json(artifact_paths["ocsvm_metrics"], metrics_payload)
            write_scores_csv(artifact_paths["ocsvm_scores"], payload["score_rows"])
        else:
            metrics_payload["hyperparameters"] = {
                "n_estimators": int(args.iforest_n_estimators),
                "max_samples": int(min(args.iforest_max_samples, train_features.shape[0])),
                "contamination": "auto",
                "random_state": int(args.seed),
                "n_jobs": int(args.iforest_n_jobs),
            }
            write_json(artifact_paths["iforest_metrics"], metrics_payload)
            write_scores_csv(artifact_paths["iforest_scores"], payload["score_rows"])

    plot_score_histograms(results, artifact_paths["plot"])

    report_text = build_report(
        preprocessing_config=preprocessing_config,
        feature_config=feature_config,
        fault_counts=fault_counts,
        ae_metrics=ae_metrics,
        results=results,
        artifact_paths=artifact_paths,
    )
    artifact_paths["report"].write_text(report_text, encoding="utf-8")

    for label, payload in results.items():
        metrics = payload["metrics"]
        threshold_meta = payload["threshold_meta"]
        print(f"\n{label} Summary")
        print(f"  threshold ({threshold_meta['rule']}): {threshold_meta['threshold']:.6f}")
        print(f"  auroc: {metrics['auroc']:.6f}")
        print(f"  auprc: {metrics['auprc']:.6f}")
        print(f"  f1: {metrics['f1']:.6f}")
        print(f"  precision: {metrics['precision']:.6f}")
        print(f"  recall_fault: {metrics['recall_fault']:.6f}")
        print(f"  false_alarm_rate: {metrics['false_alarm_rate']:.6f}")

    print(f"\nSaved OC-SVM metrics: {artifact_paths['ocsvm_metrics'].as_posix()}")
    print(f"Saved Isolation Forest metrics: {artifact_paths['iforest_metrics'].as_posix()}")
    print(f"Saved report: {artifact_paths['report'].as_posix()}")
    print(f"Saved plot: {artifact_paths['plot'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
