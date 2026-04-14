from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_ae_baseline import (
    CompactConvAutoencoder,
    DataLoader,
    Dataset,
    build_score_rows as build_ae_score_rows,
    compute_reconstruction_errors,
    evaluate_scores as evaluate_binary_scores,
    parameter_count,
    require_torch,
    select_threshold as select_ae_threshold,
    set_seed,
    torch,
    train_model,
)
from train_shallow_baselines import (
    build_feature_config,
    build_score_rows as build_shallow_score_rows,
    compute_anomaly_scores,
    extract_features,
    select_threshold as select_shallow_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


class MemmapWindowDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.windows = np.load(path, mmap_mode="r")
        if self.windows.ndim != 2:
            raise ValueError(f"Expected a 2D window array at {path.as_posix()}, got {self.windows.shape}")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        window = np.array(self.windows[index], dtype=np.float32, copy=True)
        return torch.from_numpy(window).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate AE, OC-SVM, and Isolation Forest baselines on processed Paderborn windows.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-rule", choices=("mean_plus_3std",), default="mean_plus_3std")
    parser.add_argument("--psd-band-count", type=int, default=5)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--ocsvm-nu", type=float, default=0.05)
    parser.add_argument("--ocsvm-max-iter", type=int, default=2000)
    parser.add_argument("--iforest-n-estimators", type=int, default=300)
    parser.add_argument("--iforest-max-samples", type=int, default=256)
    parser.add_argument("--iforest-n-jobs", type=int, default=1)
    return parser.parse_args()


def resolve_paths(processed_root: Path) -> ArrayPaths:
    return ArrayPaths(
        train_healthy=processed_root / "train" / "healthy_windows.npy",
        val_healthy=processed_root / "val" / "healthy_windows.npy",
        test_healthy=processed_root / "test" / "healthy_windows.npy",
        test_fault=processed_root / "test" / "fault_windows.npy",
        fault_labels=processed_root / "test" / "fault_labels.npy",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_required_files(paths: ArrayPaths, metadata_root: Path) -> None:
    required = [
        paths.train_healthy,
        paths.val_healthy,
        paths.test_healthy,
        paths.test_fault,
        paths.fault_labels,
        metadata_root / "preprocessing_config.json",
        metadata_root / "window_manifest.csv",
        metadata_root / "bearing_label_map.json",
        metadata_root / "bearing_label_map.md",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Paderborn inputs are missing: " + ", ".join(path.as_posix() for path in missing)
        )


def make_loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MemmapWindowDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def load_memmap(path: Path, expected_width: int) -> np.memmap:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"Unexpected window array shape for {path.as_posix()}: {array.shape}")
    return array


def load_label_array(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D label array for {path.as_posix()}, got {array.shape}")
    return np.asarray(array, dtype=np.int64)


def extract_features_chunked(windows: np.memmap, feature_config: Any, chunk_size: int) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("feature_chunk_size must be positive.")

    num_rows = int(windows.shape[0])
    num_features = len(feature_config.feature_names)
    features = np.empty((num_rows, num_features), dtype=np.float32)
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = np.asarray(windows[start:end], dtype=np.float32)
        features[start:end] = extract_features(chunk, feature_config)
    return features


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["split", "true_label", "score", "predicted_anomaly"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subgroup_metrics_from_manifest(
    *,
    window_manifest_path: Path,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    fault_labels: np.ndarray,
    fault_label_map: dict[str, int],
    threshold: float,
) -> dict[str, Any]:
    healthy_by_condition: dict[str, list[float]] = defaultdict(list)
    fault_by_condition: dict[str, list[float]] = defaultdict(list)
    fault_by_damage_group: dict[str, list[float]] = defaultdict(list)

    healthy_index = 0
    fault_index = 0

    with window_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subset = row["subset"]
            if subset == "test_healthy":
                if healthy_index >= test_healthy_scores.shape[0]:
                    raise RuntimeError("Window manifest has more test_healthy rows than test_healthy scores.")
                healthy_by_condition[row["condition_code"]].append(float(test_healthy_scores[healthy_index]))
                healthy_index += 1
            elif subset == "test_fault":
                if fault_index >= test_fault_scores.shape[0]:
                    raise RuntimeError("Window manifest has more test_fault rows than test_fault scores.")
                damage_group = row["damage_group"]
                expected_label = fault_label_map.get(damage_group)
                if expected_label is None:
                    raise RuntimeError(f"Unknown damage group in window manifest: {damage_group}")
                if row["fault_label_int"] and int(row["fault_label_int"]) != expected_label:
                    raise RuntimeError(
                        f"Manifest fault label mismatch for {row['relative_path']}: "
                        f"{row['fault_label_int']} != {expected_label}"
                    )
                if int(fault_labels[fault_index]) != expected_label:
                    raise RuntimeError(
                        f"fault_labels.npy mismatch at index {fault_index}: {fault_labels[fault_index]} != {expected_label}"
                    )
                score = float(test_fault_scores[fault_index])
                fault_by_condition[row["condition_code"]].append(score)
                fault_by_damage_group[damage_group].append(score)
                fault_index += 1

    if healthy_index != test_healthy_scores.shape[0]:
        raise RuntimeError(
            f"Window manifest test_healthy rows ({healthy_index}) do not match score count ({test_healthy_scores.shape[0]})."
        )
    if fault_index != test_fault_scores.shape[0]:
        raise RuntimeError(
            f"Window manifest test_fault rows ({fault_index}) do not match score count ({test_fault_scores.shape[0]})."
        )

    by_damage_group: dict[str, Any] = {}
    for damage_group in sorted(fault_by_damage_group):
        by_damage_group[damage_group] = {
            **evaluate_binary_scores(
                threshold=threshold,
                test_healthy_errors=np.asarray(test_healthy_scores, dtype=np.float32),
                test_fault_errors=np.asarray(fault_by_damage_group[damage_group], dtype=np.float32),
            ),
            "group_name": damage_group,
        }

    by_condition: dict[str, Any] = {}
    for condition_code in sorted(set(healthy_by_condition) | set(fault_by_condition)):
        healthy_scores = np.asarray(healthy_by_condition.get(condition_code, []), dtype=np.float32)
        fault_scores = np.asarray(fault_by_condition.get(condition_code, []), dtype=np.float32)
        if healthy_scores.size == 0 or fault_scores.size == 0:
            continue
        by_condition[condition_code] = {
            **evaluate_binary_scores(
                threshold=threshold,
                test_healthy_errors=healthy_scores,
                test_fault_errors=fault_scores,
            ),
            "condition_code": condition_code,
        }

    return {
        "by_damage_group": by_damage_group,
        "by_condition": by_condition,
    }


def format_metrics_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
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


def format_subgroup_table(group_name: str, payload: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"| {group_name} | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, metrics in payload.items():
        lines.append(
            "| "
            f"{key} | "
            f"{metrics['auroc']:.6f} | "
            f"{metrics['auprc']:.6f} | "
            f"{metrics['f1']:.6f} | "
            f"{metrics['precision']:.6f} | "
            f"{metrics['recall_fault']:.6f} | "
            f"{metrics['false_alarm_rate']:.6f} | "
            f"{metrics['num_test_fault']} |"
        )
    return "\n".join(lines)


def plot_summary(results: dict[str, dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for axis, (label, payload) in zip(axes, results.items(), strict=True):
        axis.hist(payload["test_healthy_scores"], bins=50, alpha=0.60, label="Test healthy")
        axis.hist(payload["test_fault_scores"], bins=50, alpha=0.60, label="Test fault")
        axis.axvline(payload["threshold"], color="black", linestyle="--", linewidth=1.4, label="Threshold")
        axis.set_title(label)
        axis.set_xlabel("Anomaly Score")
        axis.set_ylabel("Window Count")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Paderborn Baseline Score Distributions", fontsize=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def compare_to_cwru(paderborn_results: dict[str, dict[str, Any]], artifacts_root: Path) -> str:
    cwru_paths = {
        "AE": artifacts_root / "metrics" / "cwru_ae_metrics.json",
        "OC-SVM": artifacts_root / "metrics" / "cwru_ocsvm_metrics.json",
        "Isolation Forest": artifacts_root / "metrics" / "cwru_iforest_metrics.json",
    }
    if not all(path.exists() for path in cwru_paths.values()):
        return "CWRU baseline metrics were not all available locally, so no direct comparison note was generated."

    cwru_metrics = {label: read_json(path) for label, path in cwru_paths.items()}
    mean_cwru_far = float(np.mean([payload["false_alarm_rate"] for payload in cwru_metrics.values()]))
    mean_paderborn_far = float(np.mean([payload["metrics"]["false_alarm_rate"] for payload in paderborn_results.values()]))
    mean_cwru_f1 = float(np.mean([payload["f1"] for payload in cwru_metrics.values()]))
    mean_paderborn_f1 = float(np.mean([payload["metrics"]["f1"] for payload in paderborn_results.values()]))

    if mean_paderborn_f1 < mean_cwru_f1 - 0.05 and mean_paderborn_far < mean_cwru_far:
        return (
            "Compared with the current CWRU baselines, Paderborn looks materially harder in a more conservative way: "
            f"mean F1 dropped from {mean_cwru_f1:.4f} to {mean_paderborn_f1:.4f} "
            f"while mean false alarm changed from {mean_cwru_far:.4f} to {mean_paderborn_far:.4f}, "
            "so the bigger issue is missed faults rather than excess alarms."
        )
    if mean_paderborn_far > mean_cwru_far + 0.05 or mean_paderborn_f1 < mean_cwru_f1 - 0.05:
        return (
            "Compared with the current CWRU baselines, Paderborn looks materially harder: "
            f"mean false alarm changed from {mean_cwru_far:.4f} to {mean_paderborn_far:.4f} "
            f"and mean F1 shifted from {mean_cwru_f1:.4f} to {mean_paderborn_f1:.4f}."
        )
    return (
        "Compared with the current CWRU baselines, Paderborn looks broadly similar at a high level: "
        f"mean false alarm changed from {mean_cwru_far:.4f} to {mean_paderborn_far:.4f} "
        f"and mean F1 changed from {mean_cwru_f1:.4f} to {mean_paderborn_f1:.4f}."
    )


def build_report(
    *,
    label_map: dict[str, Any],
    preprocessing_config: dict[str, Any],
    results: dict[str, dict[str, Any]],
    cwru_note: str,
    artifact_paths: dict[str, Path],
) -> str:
    overall_rows = [(label, payload["metrics"]) for label, payload in results.items()]
    lines = [
        "# Paderborn Baseline Report",
        "",
        "## Label Provenance",
        f"- Total bearings: `{label_map['summary']['total_bearings']}`",
        f"- Verified bearings: `{label_map['summary']['verified_pdf_count']}`",
        f"- Inferred bearings: `{label_map['summary']['inferred_family_rule_count']}`",
        f"- Damage-group counts: `{json.dumps(label_map['summary']['damage_group_counts'], ensure_ascii=True)}`",
        "- All current evaluation labels remain family-rule inferences; no support PDFs were parsed automatically in this pass.",
        "",
        "## Setup",
        f"- Processed root: `{preprocessing_config['processed_root']}`",
        f"- Selected signal channel: `{preprocessing_config['channel']}`",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Stride: `{preprocessing_config['stride']}`",
        f"- Threshold rule: `mean_plus_3std`",
        f"- AE epochs: `{results['AE']['training']['epochs']}`",
        f"- AE batch size: `{results['AE']['training']['batch_size']}`",
        f"- OC-SVM variant: `{results['OC-SVM']['model_variant']}`",
        "",
        "## Overall Comparison",
        format_metrics_table(overall_rows),
        "",
    ]

    for label, payload in results.items():
        lines.extend(
            [
                f"## {label}",
                f"- Threshold: `{payload['threshold']:.6f}`",
                f"- Threshold rule: `{payload['threshold_rule']}`",
                f"- Precision: `{payload['metrics']['precision']:.6f}`",
                f"- Recall on fault windows: `{payload['metrics']['recall_fault']:.6f}`",
                f"- False alarm rate on healthy test windows: `{payload['metrics']['false_alarm_rate']:.6f}`",
                "",
                "### By Damage Group",
                format_subgroup_table("Damage Group", payload["by_damage_group"]),
                "",
                "### By Operating Condition",
                format_subgroup_table("Condition", payload["by_condition"]),
                "",
            ]
        )
        if label == "AE":
            lines.extend(
                [
                    f"- Final train loss: `{payload['training']['final_train_loss']:.6f}`",
                    f"- Final val loss: `{payload['training']['final_val_loss']:.6f}`",
                    f"- Parameter count: `{payload['training']['parameter_count']}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## CWRU Comparison Note",
            f"- {cwru_note}",
            "",
            "## Saved Artifacts",
            f"- AE model: `{artifact_paths['ae_model'].as_posix()}`",
            f"- AE metrics: `{artifact_paths['ae_metrics'].as_posix()}`",
            f"- OC-SVM metrics: `{artifact_paths['ocsvm_metrics'].as_posix()}`",
            f"- Isolation Forest metrics: `{artifact_paths['iforest_metrics'].as_posix()}`",
            f"- AE scores: `{artifact_paths['ae_scores'].as_posix()}`",
            f"- OC-SVM scores: `{artifact_paths['ocsvm_scores'].as_posix()}`",
            f"- Isolation Forest scores: `{artifact_paths['iforest_scores'].as_posix()}`",
            f"- Summary plot: `{artifact_paths['plot'].as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    require_torch()
    args = parse_args()
    set_seed(args.seed)

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    model_dir = artifacts_root / "models"
    metrics_dir = artifacts_root / "metrics"
    plots_dir = artifacts_root / "plots"
    for directory in (model_dir, metrics_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    label_map = read_json(metadata_root / "bearing_label_map.json")
    expected_width = int(preprocessing_config["window_size"])
    fault_label_map = {key: int(value) for key, value in preprocessing_config["fault_label_map"].items()}

    train_windows = load_memmap(array_paths.train_healthy, expected_width)
    val_windows = load_memmap(array_paths.val_healthy, expected_width)
    test_healthy_windows = load_memmap(array_paths.test_healthy, expected_width)
    test_fault_windows = load_memmap(array_paths.test_fault, expected_width)
    fault_labels = load_label_array(array_paths.fault_labels)

    if train_windows.shape[0] == 0 or val_windows.shape[0] == 0:
        raise RuntimeError("Healthy train and val arrays must both be non-empty.")
    if test_healthy_windows.shape[0] == 0 or test_fault_windows.shape[0] == 0:
        raise RuntimeError("Healthy test and fault test arrays must both be non-empty.")
    if fault_labels.shape[0] != test_fault_windows.shape[0]:
        raise RuntimeError("fault_labels.npy must contain one label per fault test window.")

    train_loader = make_loader(array_paths.train_healthy, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(array_paths.val_healthy, batch_size=args.batch_size, shuffle=False)
    test_healthy_loader = make_loader(array_paths.test_healthy, batch_size=args.batch_size, shuffle=False)
    test_fault_loader = make_loader(array_paths.test_fault, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = CompactConvAutoencoder().to(device)
    ae_parameter_count = parameter_count(ae_model)

    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_output = ae_model(sample_batch.to(device)).cpu()
    ae_batch_shape = {
        "input": list(sample_batch.shape),
        "output": list(sample_output.shape),
    }

    ae_history = train_model(
        model=ae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    ae_val_scores = compute_reconstruction_errors(ae_model, val_loader, device)
    ae_test_healthy_scores = compute_reconstruction_errors(ae_model, test_healthy_loader, device)
    ae_test_fault_scores = compute_reconstruction_errors(ae_model, test_fault_loader, device)
    ae_threshold_meta = select_ae_threshold(ae_val_scores, args.threshold_rule)
    ae_threshold = float(ae_threshold_meta["threshold"])
    ae_metrics = evaluate_binary_scores(
        threshold=ae_threshold,
        test_healthy_errors=ae_test_healthy_scores,
        test_fault_errors=ae_test_fault_scores,
    )
    ae_subgroups = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=ae_test_healthy_scores,
        test_fault_scores=ae_test_fault_scores,
        fault_labels=fault_labels,
        fault_label_map=fault_label_map,
        threshold=ae_threshold,
    )

    feature_config = build_feature_config(expected_width, args.psd_band_count)
    train_features_raw = extract_features_chunked(train_windows, feature_config, args.feature_chunk_size)
    val_features_raw = extract_features_chunked(val_windows, feature_config, args.feature_chunk_size)
    test_healthy_features_raw = extract_features_chunked(test_healthy_windows, feature_config, args.feature_chunk_size)
    test_fault_features_raw = extract_features_chunked(test_fault_windows, feature_config, args.feature_chunk_size)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features_raw)
    val_features = scaler.transform(val_features_raw)
    test_healthy_features = scaler.transform(test_healthy_features_raw)
    test_fault_features = scaler.transform(test_fault_features_raw)

    ocsvm = SGDOneClassSVM(
        nu=args.ocsvm_nu,
        random_state=args.seed,
        max_iter=args.ocsvm_max_iter,
        tol=1e-3,
        shuffle=True,
    )
    ocsvm.fit(train_features)
    ocsvm_val_scores = compute_anomaly_scores(ocsvm, val_features)
    ocsvm_test_healthy_scores = compute_anomaly_scores(ocsvm, test_healthy_features)
    ocsvm_test_fault_scores = compute_anomaly_scores(ocsvm, test_fault_features)
    ocsvm_threshold_meta = select_shallow_threshold(ocsvm_val_scores, args.threshold_rule)
    ocsvm_threshold = float(ocsvm_threshold_meta["threshold"])
    ocsvm_metrics = evaluate_binary_scores(
        threshold=ocsvm_threshold,
        test_healthy_errors=ocsvm_test_healthy_scores,
        test_fault_errors=ocsvm_test_fault_scores,
    )
    ocsvm_subgroups = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=ocsvm_test_healthy_scores,
        test_fault_scores=ocsvm_test_fault_scores,
        fault_labels=fault_labels,
        fault_label_map=fault_label_map,
        threshold=ocsvm_threshold,
    )

    iforest = IsolationForest(
        n_estimators=args.iforest_n_estimators,
        max_samples=min(args.iforest_max_samples, train_features.shape[0]),
        contamination="auto",
        random_state=args.seed,
        n_jobs=args.iforest_n_jobs,
    )
    iforest.fit(train_features)
    iforest_val_scores = compute_anomaly_scores(iforest, val_features)
    iforest_test_healthy_scores = compute_anomaly_scores(iforest, test_healthy_features)
    iforest_test_fault_scores = compute_anomaly_scores(iforest, test_fault_features)
    iforest_threshold_meta = select_shallow_threshold(iforest_val_scores, args.threshold_rule)
    iforest_threshold = float(iforest_threshold_meta["threshold"])
    iforest_metrics = evaluate_binary_scores(
        threshold=iforest_threshold,
        test_healthy_errors=iforest_test_healthy_scores,
        test_fault_errors=iforest_test_fault_scores,
    )
    iforest_subgroups = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=iforest_test_healthy_scores,
        test_fault_scores=iforest_test_fault_scores,
        fault_labels=fault_labels,
        fault_label_map=fault_label_map,
        threshold=iforest_threshold,
    )

    artifact_paths = {
        "ae_model": model_dir / "paderborn_ae_baseline.pt",
        "ae_metrics": metrics_dir / "paderborn_ae_metrics.json",
        "ocsvm_metrics": metrics_dir / "paderborn_ocsvm_metrics.json",
        "iforest_metrics": metrics_dir / "paderborn_iforest_metrics.json",
        "report": metrics_dir / "paderborn_baseline_report.md",
        "ae_scores": metrics_dir / "paderborn_ae_scores.csv",
        "ocsvm_scores": metrics_dir / "paderborn_ocsvm_scores.csv",
        "iforest_scores": metrics_dir / "paderborn_iforest_scores.csv",
        "plot": plots_dir / "paderborn_baseline_summary.png",
    }

    ae_checkpoint = {
        "model_state_dict": ae_model.state_dict(),
        "model_name": "CompactConvAutoencoder",
        "input_shape": [1, expected_width],
        "parameter_count": ae_parameter_count,
        "history": ae_history,
        "threshold": ae_threshold_meta,
        "metrics": ae_metrics,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    torch.save(ae_checkpoint, artifact_paths["ae_model"])

    ae_score_rows: list[dict[str, Any]] = []
    ae_score_rows.extend(build_ae_score_rows(split="val_healthy", true_label=0, errors=ae_val_scores, threshold=ae_threshold))
    ae_score_rows.extend(
        build_ae_score_rows(split="test_healthy", true_label=0, errors=ae_test_healthy_scores, threshold=ae_threshold)
    )
    ae_score_rows.extend(build_ae_score_rows(split="test_fault", true_label=1, errors=ae_test_fault_scores, threshold=ae_threshold))

    ocsvm_score_rows: list[dict[str, Any]] = []
    ocsvm_score_rows.extend(
        build_shallow_score_rows(split="val_healthy", true_label=0, scores=ocsvm_val_scores, threshold=ocsvm_threshold)
    )
    ocsvm_score_rows.extend(
        build_shallow_score_rows(
            split="test_healthy",
            true_label=0,
            scores=ocsvm_test_healthy_scores,
            threshold=ocsvm_threshold,
        )
    )
    ocsvm_score_rows.extend(
        build_shallow_score_rows(split="test_fault", true_label=1, scores=ocsvm_test_fault_scores, threshold=ocsvm_threshold)
    )

    iforest_score_rows: list[dict[str, Any]] = []
    iforest_score_rows.extend(
        build_shallow_score_rows(split="val_healthy", true_label=0, scores=iforest_val_scores, threshold=iforest_threshold)
    )
    iforest_score_rows.extend(
        build_shallow_score_rows(
            split="test_healthy",
            true_label=0,
            scores=iforest_test_healthy_scores,
            threshold=iforest_threshold,
        )
    )
    iforest_score_rows.extend(
        build_shallow_score_rows(
            split="test_fault",
            true_label=1,
            scores=iforest_test_fault_scores,
            threshold=iforest_threshold,
        )
    )

    write_scores_csv(artifact_paths["ae_scores"], ae_score_rows)
    write_scores_csv(artifact_paths["ocsvm_scores"], ocsvm_score_rows)
    write_scores_csv(artifact_paths["iforest_scores"], iforest_score_rows)

    results = {
        "AE": {
            "metrics": ae_metrics,
            "threshold": ae_threshold,
            "threshold_rule": ae_threshold_meta["rule"],
            "by_damage_group": ae_subgroups["by_damage_group"],
            "by_condition": ae_subgroups["by_condition"],
            "training": {
                "final_train_loss": float(ae_history["train_loss"][-1]),
                "final_val_loss": float(ae_history["val_loss"][-1]),
                "parameter_count": ae_parameter_count,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "batch_shape_check": ae_batch_shape,
            },
            "val_scores": ae_val_scores,
            "test_healthy_scores": ae_test_healthy_scores,
            "test_fault_scores": ae_test_fault_scores,
            "model_variant": "CompactConvAutoencoder",
        },
        "OC-SVM": {
            "metrics": ocsvm_metrics,
            "threshold": ocsvm_threshold,
            "threshold_rule": ocsvm_threshold_meta["rule"],
            "by_damage_group": ocsvm_subgroups["by_damage_group"],
            "by_condition": ocsvm_subgroups["by_condition"],
            "model_variant": "SGDOneClassSVM_linear_full_train",
            "test_healthy_scores": ocsvm_test_healthy_scores,
            "test_fault_scores": ocsvm_test_fault_scores,
        },
        "Isolation Forest": {
            "metrics": iforest_metrics,
            "threshold": iforest_threshold,
            "threshold_rule": iforest_threshold_meta["rule"],
            "by_damage_group": iforest_subgroups["by_damage_group"],
            "by_condition": iforest_subgroups["by_condition"],
            "model_variant": "IsolationForest",
            "test_healthy_scores": iforest_test_healthy_scores,
            "test_fault_scores": iforest_test_fault_scores,
        },
    }

    ae_metrics_payload = {
        **ae_metrics,
        "threshold": ae_threshold,
        "threshold_rule": ae_threshold_meta["rule"],
        "final_train_loss": float(ae_history["train_loss"][-1]),
        "final_val_loss": float(ae_history["val_loss"][-1]),
        "parameter_count": ae_parameter_count,
        "device": str(device),
        "batch_shape_check": ae_batch_shape,
        "by_damage_group": ae_subgroups["by_damage_group"],
        "by_condition": ae_subgroups["by_condition"],
        "label_provenance": label_map["summary"],
    }
    ocsvm_metrics_payload = {
        **ocsvm_metrics,
        "threshold": ocsvm_threshold,
        "threshold_rule": ocsvm_threshold_meta["rule"],
        "model_variant": "SGDOneClassSVM_linear_full_train",
        "by_damage_group": ocsvm_subgroups["by_damage_group"],
        "by_condition": ocsvm_subgroups["by_condition"],
        "label_provenance": label_map["summary"],
    }
    iforest_metrics_payload = {
        **iforest_metrics,
        "threshold": iforest_threshold,
        "threshold_rule": iforest_threshold_meta["rule"],
        "model_variant": "IsolationForest",
        "by_damage_group": iforest_subgroups["by_damage_group"],
        "by_condition": iforest_subgroups["by_condition"],
        "label_provenance": label_map["summary"],
    }
    write_json(artifact_paths["ae_metrics"], ae_metrics_payload)
    write_json(artifact_paths["ocsvm_metrics"], ocsvm_metrics_payload)
    write_json(artifact_paths["iforest_metrics"], iforest_metrics_payload)

    cwru_note = compare_to_cwru(results, artifacts_root)
    report_text = build_report(
        label_map=label_map,
        preprocessing_config=preprocessing_config,
        results=results,
        cwru_note=cwru_note,
        artifact_paths=artifact_paths,
    )
    artifact_paths["report"].write_text(report_text, encoding="utf-8")

    plot_summary(results, artifact_paths["plot"])

    print("\nPaderborn Baseline Summary")
    print(f"  AE final train loss: {ae_history['train_loss'][-1]:.6f}")
    print(f"  AE final val loss: {ae_history['val_loss'][-1]:.6f}")
    print(f"  AE threshold ({ae_threshold_meta['rule']}): {ae_threshold:.6f}")
    print(f"  AE auroc: {ae_metrics['auroc']:.6f}")
    print(f"  OC-SVM threshold ({ocsvm_threshold_meta['rule']}): {ocsvm_threshold:.6f}")
    print(f"  OC-SVM auroc: {ocsvm_metrics['auroc']:.6f}")
    print(f"  Isolation Forest threshold ({iforest_threshold_meta['rule']}): {iforest_threshold:.6f}")
    print(f"  Isolation Forest auroc: {iforest_metrics['auroc']:.6f}")
    print(f"  saved report: {artifact_paths['report'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
