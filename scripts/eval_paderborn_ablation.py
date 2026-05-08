from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

try:
    from train_paderborn_ablation import (
        MemmapWindowDataset,
        VARIANT_CONFIGS,
        build_model,
        compute_reconstruction_scores,
        load_torch_payload,
    )
except ModuleNotFoundError:
    from scripts.train_paderborn_ablation import (
        MemmapWindowDataset,
        VARIANT_CONFIGS,
        build_model,
        compute_reconstruction_scores,
        load_torch_payload,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
WINDOW_MANIFEST_PATH = METADATA_ROOT / "window_manifest.csv"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "paderborn_ablation"
DEFAULT_VARIANTS = ("compact_ae", "dilated_ae", "res_ae", "resdilated_time", "resdilated_full")
DEFAULT_SEEDS = (42, 7, 123)
DEFAULT_THRESHOLD_RULES = ("val_p99_5", "val_mean_plus_3std", "train_mean_plus_3std")
CALIBRATION_RULE_ORDER = ("train_mean_plus_3std", "val_mean_plus_3std", "val_p99_5")
METRIC_NAMES = ("auroc", "auprc", "f1", "precision", "recall_fault", "far")
SELECTED_CALIBRATION_VARIANT = "resdilated_time"
SELECTED_CALIBRATION_ALIAS = "resdilated_ae_time"
SELECTED_CALIBRATION_LABEL = "ResDilatedAE-T"
SELECTED_GROUP_ANALYSIS_RULE = "val_p99_5"
CONDITION_GROUP_METRICS = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall_fault",
    "far",
    "missed_fault_count",
    "missed_fault_percent",
)
DAMAGE_GROUP_METRICS = (
    "recall_fault",
    "missed_fault_count",
    "missed_fault_percent",
    "mean_score",
    "median_score",
)


@dataclass(frozen=True)
class RunFiles:
    run_dir: Path
    best_checkpoint: Path
    run_config_json: Path
    val_healthy_scores_npy: Path
    test_healthy_scores_npy: Path
    test_fault_scores_npy: Path
    sanity_metrics_json: Path
    train_healthy_scores_npy: Path


@dataclass(frozen=True)
class TestWindowMetadata:
    split: str
    label: int
    bearing_code: str
    operating_condition: str
    damage_group: str
    file_path: str
    window_index: int
    window_start: int
    window_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Paderborn ablation runs with shared threshold and metric definitions.",
    )
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_CONFIGS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--threshold-rules",
        nargs="+",
        choices=DEFAULT_THRESHOLD_RULES,
        default=list(DEFAULT_THRESHOLD_RULES),
    )
    parser.add_argument("--main-threshold-rule", choices=DEFAULT_THRESHOLD_RULES, default="val_p99_5")
    parser.add_argument("--compute-train-scores-if-missing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.main_threshold_rule not in args.threshold_rules:
        parser.error("--main-threshold-rule must also be listed in --threshold-rules.")
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_numpy_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float32))
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_int_field(value: str | None) -> int:
    text = (value or "").strip()
    if not text:
        return 0
    return int(text)


def load_test_window_metadata(manifest_path: Path) -> dict[str, list[TestWindowMetadata]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Paderborn window manifest: {manifest_path.as_posix()}")

    healthy_rows: list[TestWindowMetadata] = []
    fault_rows: list[TestWindowMetadata] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subset = (row.get("subset") or "").strip()
            if subset not in {"test_healthy", "test_fault"}:
                continue
            metadata = TestWindowMetadata(
                split=(row.get("split") or "").strip(),
                label=0 if subset == "test_healthy" else 1,
                bearing_code=(row.get("bearing_code") or "").strip(),
                operating_condition=(row.get("condition_code") or "").strip(),
                damage_group=(row.get("damage_group") or "").strip(),
                file_path=(row.get("relative_path") or "").strip(),
                window_index=parse_int_field(row.get("window_index")),
                window_start=parse_int_field(row.get("window_start")),
                window_end=parse_int_field(row.get("window_end")),
            )
            if subset == "test_healthy":
                healthy_rows.append(metadata)
            else:
                fault_rows.append(metadata)
    return {"test_healthy": healthy_rows, "test_fault": fault_rows}


def resolve_run_files(artifacts_root: Path, variant: str, seed: int) -> RunFiles:
    run_dir = artifacts_root / variant / f"seed_{seed}"
    return RunFiles(
        run_dir=run_dir,
        best_checkpoint=run_dir / "best.pt",
        run_config_json=run_dir / "run_config.json",
        val_healthy_scores_npy=run_dir / "val_healthy_scores.npy",
        test_healthy_scores_npy=run_dir / "test_healthy_scores.npy",
        test_fault_scores_npy=run_dir / "test_fault_scores.npy",
        sanity_metrics_json=run_dir / "sanity_metrics.json",
        train_healthy_scores_npy=run_dir / "train_healthy_scores.npy",
    )


def required_run_files(run_files: RunFiles) -> list[Path]:
    return [
        run_files.best_checkpoint,
        run_files.run_config_json,
        run_files.val_healthy_scores_npy,
        run_files.test_healthy_scores_npy,
        run_files.test_fault_scores_npy,
        run_files.sanity_metrics_json,
    ]


def load_score_array(path: Path, label: str) -> np.ndarray:
    values = np.load(path)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RuntimeError(f"{label} scores are empty: {path.as_posix()}")
    return array


def threshold_value(rule: str, *, val_scores: np.ndarray, train_scores: np.ndarray | None) -> float:
    if rule == "val_p99_5":
        return float(np.percentile(np.asarray(val_scores, dtype=np.float64), 99.5))
    if rule == "val_mean_plus_3std":
        values = np.asarray(val_scores, dtype=np.float64)
        return float(values.mean() + (3.0 * values.std()))
    if rule == "train_mean_plus_3std":
        if train_scores is None:
            raise RuntimeError("train_mean_plus_3std requires train_healthy_scores.npy.")
        values = np.asarray(train_scores, dtype=np.float64)
        return float(values.mean() + (3.0 * values.std()))
    raise ValueError(f"Unsupported threshold rule: {rule}")


def ranking_metrics(test_healthy_scores: np.ndarray, test_fault_scores: np.ndarray) -> dict[str, float]:
    y_true = np.concatenate(
        [
            np.zeros(test_healthy_scores.shape[0], dtype=np.int64),
            np.ones(test_fault_scores.shape[0], dtype=np.int64),
        ]
    )
    scores = np.concatenate([test_healthy_scores, test_fault_scores]).astype(np.float64, copy=False)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def thresholded_metrics(
    *,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    healthy_pred = np.asarray(test_healthy_scores, dtype=np.float64) > float(threshold)
    fault_pred = np.asarray(test_fault_scores, dtype=np.float64) > float(threshold)
    fp = int(healthy_pred.sum())
    tp = int(fault_pred.sum())
    n_healthy = int(test_healthy_scores.shape[0])
    n_fault = int(test_fault_scores.shape[0])
    predicted_positive = tp + fp
    precision = 1.0 if predicted_positive == 0 else float(tp / predicted_positive)
    recall_fault = float(tp / n_fault) if n_fault > 0 else 0.0
    f1 = 0.0 if (precision + recall_fault) == 0.0 else float((2.0 * precision * recall_fault) / (precision + recall_fault))
    far = float(fp / n_healthy) if n_healthy > 0 else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall_fault": recall_fault,
        "far": far,
        "tp": tp,
        "fp": fp,
        "n_test_healthy": n_healthy,
        "n_test_fault": n_fault,
    }


def validate_alignment(
    *,
    seed: int,
    healthy_metadata: list[TestWindowMetadata],
    fault_metadata: list[TestWindowMetadata],
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
) -> dict[str, Any]:
    healthy_metadata_count = int(len(healthy_metadata))
    fault_metadata_count = int(len(fault_metadata))
    healthy_score_count = int(test_healthy_scores.shape[0])
    fault_score_count = int(test_fault_scores.shape[0])
    healthy_aligned = healthy_metadata_count == healthy_score_count
    fault_aligned = fault_metadata_count == fault_score_count
    if not healthy_aligned or not fault_aligned:
        raise RuntimeError(
            "Paderborn score-to-metadata alignment failed for "
            f"seed {seed}: healthy metadata/scores {healthy_metadata_count}/{healthy_score_count}, "
            f"fault metadata/scores {fault_metadata_count}/{fault_score_count}."
        )
    return {
        "seed": int(seed),
        "healthy_metadata_count": healthy_metadata_count,
        "healthy_score_count": healthy_score_count,
        "healthy_aligned": healthy_aligned,
        "fault_metadata_count": fault_metadata_count,
        "fault_score_count": fault_score_count,
        "fault_aligned": fault_aligned,
    }


def compute_condition_rows_for_seed(
    *,
    seed: int,
    threshold: float,
    healthy_metadata: list[TestWindowMetadata],
    fault_metadata: list[TestWindowMetadata],
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
) -> list[dict[str, Any]]:
    healthy_conditions = np.asarray([item.operating_condition for item in healthy_metadata], dtype=object)
    fault_conditions = np.asarray([item.operating_condition for item in fault_metadata], dtype=object)
    conditions = sorted(set(healthy_conditions.tolist()) | set(fault_conditions.tolist()))
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        healthy_mask = healthy_conditions == condition
        fault_mask = fault_conditions == condition
        healthy_scores = np.asarray(test_healthy_scores[healthy_mask], dtype=np.float64)
        fault_scores = np.asarray(test_fault_scores[fault_mask], dtype=np.float64)
        if healthy_scores.size == 0 or fault_scores.size == 0:
            continue
        rank_metrics = ranking_metrics(healthy_scores, fault_scores)
        threshold_metrics = thresholded_metrics(
            test_healthy_scores=healthy_scores,
            test_fault_scores=fault_scores,
            threshold=threshold,
        )
        missed_fault_count = int(fault_scores.shape[0] - int(threshold_metrics["tp"]))
        missed_fault_percent = float(missed_fault_count / fault_scores.shape[0]) if fault_scores.size > 0 else 0.0
        rows.append(
            {
                "seed": int(seed),
                "operating_condition": condition,
                "n_healthy": int(healthy_scores.shape[0]),
                "n_fault": int(fault_scores.shape[0]),
                "auroc": rank_metrics["auroc"],
                "auprc": rank_metrics["auprc"],
                "f1": float(threshold_metrics["f1"]),
                "precision": float(threshold_metrics["precision"]),
                "recall_fault": float(threshold_metrics["recall_fault"]),
                "far": float(threshold_metrics["far"]),
                "missed_fault_count": missed_fault_count,
                "missed_fault_percent": missed_fault_percent,
            }
        )
    return rows


def compute_damage_rows_for_seed(
    *,
    seed: int,
    threshold: float,
    fault_metadata: list[TestWindowMetadata],
    test_fault_scores: np.ndarray,
) -> list[dict[str, Any]]:
    damage_groups = np.asarray([item.damage_group for item in fault_metadata], dtype=object)
    groups = sorted(set(group for group in damage_groups.tolist() if group))
    rows: list[dict[str, Any]] = []
    for group in groups:
        mask = damage_groups == group
        scores = np.asarray(test_fault_scores[mask], dtype=np.float64)
        if scores.size == 0:
            continue
        predicted_fault = scores > float(threshold)
        missed_fault_count = int(scores.shape[0] - int(predicted_fault.sum()))
        missed_fault_percent = float(missed_fault_count / scores.shape[0]) if scores.size > 0 else 0.0
        rows.append(
            {
                "seed": int(seed),
                "damage_group": group,
                "n_fault": int(scores.shape[0]),
                "recall_fault": float(predicted_fault.mean()),
                "missed_fault_count": missed_fault_count,
                "missed_fault_percent": missed_fault_percent,
                "mean_score": float(scores.mean()),
                "median_score": float(np.median(scores)),
            }
        )
    return rows


def aggregate_group_rows(
    *,
    per_seed_rows: list[dict[str, Any]],
    group_field: str,
    count_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[str(row[group_field])].append(row)

    output_rows: list[dict[str, Any]] = []
    for group_name in sorted(grouped):
        rows = grouped[group_name]
        output: dict[str, Any] = {group_field: group_name, "seed_count": int(len(rows))}
        for field in count_fields:
            values = [int(row[field]) for row in rows]
            output[field] = int(round(mean_value([float(value) for value in values])))
        for field in metric_fields:
            values = [float(row[field]) for row in rows]
            output[f"{field}_mean"] = mean_value(values)
            output[f"{field}_std"] = sample_std(values)
        output_rows.append(output)
    return output_rows


def write_per_window_predictions_csv(
    *,
    path: Path,
    per_seed_predictions: list[dict[str, Any]],
    healthy_metadata: list[TestWindowMetadata],
    fault_metadata: list[TestWindowMetadata],
) -> None:
    fieldnames = [
        "seed",
        "score",
        "label",
        "prediction",
        "bearing_code",
        "operating_condition",
        "damage_group",
        "file_path",
        "window_index",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in sorted(per_seed_predictions, key=lambda value: int(value["seed"])):
                seed = int(item["seed"])
                threshold = float(item["threshold_value"])
                healthy_scores = np.asarray(item["test_healthy_scores"], dtype=np.float64)
                fault_scores = np.asarray(item["test_fault_scores"], dtype=np.float64)
                for metadata, score in zip(healthy_metadata, healthy_scores, strict=True):
                    writer.writerow(
                        {
                            "seed": seed,
                            "score": f"{float(score):.10f}",
                            "label": metadata.label,
                            "prediction": int(score > threshold),
                            "bearing_code": metadata.bearing_code,
                            "operating_condition": metadata.operating_condition,
                            "damage_group": metadata.damage_group,
                            "file_path": metadata.file_path,
                            "window_index": metadata.window_index,
                        }
                    )
                for metadata, score in zip(fault_metadata, fault_scores, strict=True):
                    writer.writerow(
                        {
                            "seed": seed,
                            "score": f"{float(score):.10f}",
                            "label": metadata.label,
                            "prediction": int(score > threshold),
                            "bearing_code": metadata.bearing_code,
                            "operating_condition": metadata.operating_condition,
                            "damage_group": metadata.damage_group,
                            "file_path": metadata.file_path,
                            "window_index": metadata.window_index,
                        }
                    )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def device_for_inference() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_run(
    *,
    variant: str,
    seed: int,
    checkpoint_path: Path,
    device: torch.device,
    warnings: list[str],
) -> torch.nn.Module:
    variant_config = VARIANT_CONFIGS[variant]
    payload = load_torch_payload(checkpoint_path)
    payload_variant = payload.get("variant")
    payload_seed = payload.get("seed")
    if payload_variant is not None and payload_variant != variant:
        warnings.append(
            f"Checkpoint variant mismatch for {checkpoint_path.as_posix()}: expected {variant}, found {payload_variant}."
        )
    if payload_seed is not None and int(payload_seed) != int(seed):
        warnings.append(
            f"Checkpoint seed mismatch for {checkpoint_path.as_posix()}: expected {seed}, found {payload_seed}."
        )
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"Checkpoint has no state dict: {checkpoint_path.as_posix()}")
    model = build_model(variant_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def compute_train_scores_if_needed(
    *,
    variant: str,
    seed: int,
    run_files: RunFiles,
    run_config: dict[str, Any],
    compute_if_missing: bool,
    warnings: list[str],
    train_score_events: list[dict[str, Any]],
) -> np.ndarray:
    if run_files.train_healthy_scores_npy.exists():
        return load_score_array(run_files.train_healthy_scores_npy, "train_healthy")
    if not compute_if_missing:
        raise FileNotFoundError(
            "Missing train_healthy_scores.npy for train_mean_plus_3std. "
            f"Use --compute-train-scores-if-missing to infer it: {run_files.train_healthy_scores_npy.as_posix()}"
        )

    device = device_for_inference()
    batch_size = int(run_config.get("training", {}).get("batch_size") or (256 if device.type == "cuda" else 128))
    train_windows_path = PROCESSED_ROOT / "train" / "healthy_windows.npy"
    if not train_windows_path.exists():
        raise FileNotFoundError(f"Missing processed Paderborn training windows: {train_windows_path.as_posix()}")
    dataset = MemmapWindowDataset(train_windows_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    model = load_model_for_run(
        variant=variant,
        seed=seed,
        checkpoint_path=run_files.best_checkpoint,
        device=device,
        warnings=warnings,
    )
    scores = compute_reconstruction_scores(model=model, loader=loader, device=device)
    save_numpy_array(run_files.train_healthy_scores_npy, scores)
    train_score_events.append(
        {
            "variant": variant,
            "seed": int(seed),
            "path": run_files.train_healthy_scores_npy.as_posix(),
            "count": int(scores.shape[0]),
            "device": str(device),
            "batch_size": int(batch_size),
        }
    )
    return scores


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def mean_value(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def variant_flags(variant: str) -> dict[str, Any]:
    config = VARIANT_CONFIGS[variant]
    return {
        "residual": bool(config.use_residual),
        "dilation": bool(config.use_dilation),
        "frequency_loss": bool(config.frequency_loss_weight > 0),
    }


def summarize_by_variant(per_seed_rows: list[dict[str, Any]], main_rule: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = sorted({str(row["variant"]) for row in per_seed_rows})
    for variant in variants:
        variant_rows = [row for row in per_seed_rows if row["variant"] == variant and row["threshold_rule"] == main_rule]
        if not variant_rows:
            continue
        flags = variant_flags(variant)
        output: dict[str, Any] = {
            "variant": variant,
            "residual": flags["residual"],
            "dilation": flags["dilation"],
            "frequency_loss": flags["frequency_loss"],
            "threshold_rule": main_rule,
            "calibration": main_rule,
        }
        for metric_name in METRIC_NAMES:
            values = [float(row[metric_name]) for row in variant_rows]
            output[f"{metric_name}_mean"] = mean_value(values)
            output[f"{metric_name}_std"] = sample_std(values)
        rows.append(output)
    return rows


def append_selected_variant_alias_row(
    summary_rows: list[dict[str, Any]],
    *,
    per_seed_rows: list[dict[str, Any]],
    source_variant: str,
    alias_variant: str,
    threshold_rule: str,
) -> list[dict[str, Any]]:
    alias_rows = [
        row for row in per_seed_rows if row["variant"] == source_variant and row["threshold_rule"] == threshold_rule
    ]
    if not alias_rows:
        return list(summary_rows)
    flags = variant_flags(source_variant)
    output: dict[str, Any] = {
        "variant": alias_variant,
        "residual": flags["residual"],
        "dilation": flags["dilation"],
        "frequency_loss": flags["frequency_loss"],
        "threshold_rule": threshold_rule,
        "calibration": threshold_rule,
    }
    for metric_name in METRIC_NAMES:
        values = [float(row[metric_name]) for row in alias_rows]
        output[f"{metric_name}_mean"] = mean_value(values)
        output[f"{metric_name}_std"] = sample_std(values)
    return list(summary_rows) + [output]


def summarize_calibration(
    per_seed_rows: list[dict[str, Any]],
    *,
    selected_variant: str,
    selected_alias: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in CALIBRATION_RULE_ORDER:
        rule_rows = [
            row
            for row in per_seed_rows
            if row["variant"] == selected_variant and row["threshold_rule"] == rule
        ]
        if not rule_rows:
            continue
        output: dict[str, Any] = {"variant": selected_alias, "threshold_rule": rule}
        threshold_values = [float(row["threshold_value"]) for row in rule_rows]
        output["threshold_value_mean"] = mean_value(threshold_values)
        output["threshold_value_std"] = sample_std(threshold_values)
        for metric_name in METRIC_NAMES:
            values = [float(row[metric_name]) for row in rule_rows]
            output[f"{metric_name}_mean"] = mean_value(values)
            output[f"{metric_name}_std"] = sample_std(values)
        rows.append(output)
    return rows


def format_float(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    if not rows:
        lines.append("| " + " | ".join(["(none)"] + [""] * (len(headers) - 1)) + " |")
        return "\n".join(lines)
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def format_mean_std(mean_value_raw: Any, std_value_raw: Any) -> str:
    mean_raw = float(mean_value_raw)
    std_raw = float(std_value_raw)
    if math.isnan(mean_raw):
        return "nan"
    return f"{mean_raw:.3f} ± {std_raw:.3f}"


def summarize_group_findings(
    *,
    condition_rows: list[dict[str, Any]],
    damage_rows: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    if condition_rows:
        worst_condition = max(condition_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        best_condition = min(condition_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        findings.append(
            "Operating-condition misses are highest for "
            f"`{worst_condition['operating_condition']}` at `{float(worst_condition['missed_fault_percent_mean']) * 100.0:.2f}% ± "
            f"{float(worst_condition['missed_fault_percent_std']) * 100.0:.2f}%` missed fault windows, "
            f"versus `{best_condition['operating_condition']}` at `{float(best_condition['missed_fault_percent_mean']) * 100.0:.2f}% ± "
            f"{float(best_condition['missed_fault_percent_std']) * 100.0:.2f}%`."
        )
    if damage_rows:
        worst_damage = max(damage_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        best_damage = min(damage_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        findings.append(
            "Damage-family misses are highest for "
            f"`{worst_damage['damage_group']}` at `{float(worst_damage['missed_fault_percent_mean']) * 100.0:.2f}% ± "
            f"{float(worst_damage['missed_fault_percent_std']) * 100.0:.2f}%`, "
            f"while `{best_damage['damage_group']}` is lowest at `{float(best_damage['missed_fault_percent_mean']) * 100.0:.2f}% ± "
            f"{float(best_damage['missed_fault_percent_std']) * 100.0:.2f}%`."
        )
    if not findings:
        findings.append("No selected ResDilatedAE-T group-wise breakdown rows were available.")
    return findings


def build_group_breakdown_report(
    *,
    evaluation_root: Path,
    manifest_path: Path,
    alignment_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    damage_rows: list[dict[str, Any]],
    assumptions: list[str],
) -> str:
    condition_table_rows = [
        {
            "operating_condition": row["operating_condition"],
            "n_healthy": row["n_healthy"],
            "n_fault": row["n_fault"],
            "auroc": format_mean_std(row["auroc_mean"], row["auroc_std"]),
            "auprc": format_mean_std(row["auprc_mean"], row["auprc_std"]),
            "f1": format_mean_std(row["f1_mean"], row["f1_std"]),
            "precision": format_mean_std(row["precision_mean"], row["precision_std"]),
            "recall_fault": format_mean_std(row["recall_fault_mean"], row["recall_fault_std"]),
            "far": format_mean_std(row["far_mean"], row["far_std"]),
            "missed_fault_count": format_mean_std(row["missed_fault_count_mean"], row["missed_fault_count_std"]),
            "missed_fault_percent": format_mean_std(
                float(row["missed_fault_percent_mean"]) * 100.0,
                float(row["missed_fault_percent_std"]) * 100.0,
            )
            + "%",
        }
        for row in condition_rows
    ]
    damage_table_rows = [
        {
            "damage_group": row["damage_group"],
            "n_fault": row["n_fault"],
            "recall_fault": format_mean_std(row["recall_fault_mean"], row["recall_fault_std"]),
            "missed_fault_count": format_mean_std(row["missed_fault_count_mean"], row["missed_fault_count_std"]),
            "missed_fault_percent": format_mean_std(
                float(row["missed_fault_percent_mean"]) * 100.0,
                float(row["missed_fault_percent_std"]) * 100.0,
            )
            + "%",
            "mean_score": format_mean_std(row["mean_score_mean"], row["mean_score_std"]),
            "median_score": format_mean_std(row["median_score_mean"], row["median_score_std"]),
        }
        for row in damage_rows
    ]
    lines = [
        "# Paderborn Group-Wise Recall Breakdown",
        "",
        "## Source",
        f"- Selected model: `{SELECTED_CALIBRATION_LABEL}` (`{SELECTED_CALIBRATION_VARIANT}`)",
        f"- Threshold rule: `{SELECTED_GROUP_ANALYSIS_RULE}`",
        f"- Window manifest: `{manifest_path.as_posix()}`",
        f"- Evaluation root: `{evaluation_root.as_posix()}`",
        "",
        "## Alignment Audit",
        markdown_table(
            [
                "seed",
                "healthy_metadata_count",
                "healthy_score_count",
                "healthy_aligned",
                "fault_metadata_count",
                "fault_score_count",
                "fault_aligned",
            ],
            alignment_rows,
        ),
        "",
        "## Operating Condition Breakdown",
        markdown_table(
            [
                "operating_condition",
                "n_healthy",
                "n_fault",
                "auroc",
                "auprc",
                "f1",
                "precision",
                "recall_fault",
                "far",
                "missed_fault_count",
                "missed_fault_percent",
            ],
            condition_table_rows,
        ),
        "",
        "## Damage Group Breakdown",
        markdown_table(
            [
                "damage_group",
                "n_fault",
                "recall_fault",
                "missed_fault_count",
                "missed_fault_percent",
                "mean_score",
                "median_score",
            ],
            damage_table_rows,
        ),
        "",
        "## Findings",
    ]
    for finding in summarize_group_findings(condition_rows=condition_rows, damage_rows=damage_rows):
        lines.append(f"- {finding}")
    lines.extend(["", "## Alignment Assumptions"])
    for item in assumptions:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def format_mean_std(mean_value_raw: Any, std_value_raw: Any) -> str:
    mean_raw = float(mean_value_raw)
    std_raw = float(std_value_raw)
    if math.isnan(mean_raw):
        return "nan"
    return f"{mean_raw:.3f} +/- {std_raw:.3f}"


def summarize_group_findings(
    *,
    condition_rows: list[dict[str, Any]],
    damage_rows: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    if condition_rows:
        worst_condition = max(condition_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        best_condition = min(condition_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        findings.append(
            "Operating-condition misses are highest for "
            f"`{worst_condition['operating_condition']}` at `{float(worst_condition['missed_fault_percent_mean']) * 100.0:.2f}% +/- "
            f"{float(worst_condition['missed_fault_percent_std']) * 100.0:.2f}%` missed fault windows, "
            f"versus `{best_condition['operating_condition']}` at `{float(best_condition['missed_fault_percent_mean']) * 100.0:.2f}% +/- "
            f"{float(best_condition['missed_fault_percent_std']) * 100.0:.2f}%`."
        )
    if damage_rows:
        worst_damage = max(damage_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        best_damage = min(damage_rows, key=lambda row: float(row["missed_fault_percent_mean"]))
        findings.append(
            "Damage-family misses are highest for "
            f"`{worst_damage['damage_group']}` at `{float(worst_damage['missed_fault_percent_mean']) * 100.0:.2f}% +/- "
            f"{float(worst_damage['missed_fault_percent_std']) * 100.0:.2f}%`, "
            f"while `{best_damage['damage_group']}` is lowest at `{float(best_damage['missed_fault_percent_mean']) * 100.0:.2f}% +/- "
            f"{float(best_damage['missed_fault_percent_std']) * 100.0:.2f}%`."
        )
    if not findings:
        findings.append("No selected ResDilatedAE-T group-wise breakdown rows were available.")
    return findings


def build_report(
    *,
    artifacts_root: Path,
    evaluation_root: Path,
    requested_variants: list[str],
    requested_seeds: list[int],
    threshold_rules: list[str],
    main_threshold_rule: str,
    evaluated_runs: list[dict[str, Any]],
    missing_runs: list[dict[str, Any]],
    train_score_events: list[dict[str, Any]],
    warnings: list[str],
    main_summary_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    group_analysis_report_path: Path | None,
    group_analysis_condition_rows: list[dict[str, Any]],
    group_analysis_damage_rows: list[dict[str, Any]],
    group_alignment_rows: list[dict[str, Any]],
    group_analysis_assumptions: list[str],
) -> str:
    main_headers = [
        "variant",
        "residual",
        "dilation",
        "frequency_loss",
        "threshold_rule",
        "calibration",
        "auroc_mean",
        "auprc_mean",
        "f1_mean",
        "precision_mean",
        "recall_fault_mean",
        "far_mean",
    ]
    calibration_headers = [
        "threshold_rule",
        "threshold_value_mean",
        "auroc_mean",
        "auprc_mean",
        "f1_mean",
        "precision_mean",
        "recall_fault_mean",
        "far_mean",
    ]
    lines = [
        "# Paderborn Ablation Evaluation Report",
        "",
        "## Files Evaluated",
        f"- Artifacts root: `{artifacts_root.as_posix()}`",
        f"- Evaluation root: `{evaluation_root.as_posix()}`",
        f"- Requested variants: `{', '.join(requested_variants)}`",
        f"- Requested seeds: `{', '.join(str(seed) for seed in requested_seeds)}`",
        f"- Evaluated run count: `{len(evaluated_runs)}`",
    ]
    for item in evaluated_runs:
        lines.append(f"- `{item['run_dir']}`")

    lines.extend(
        [
            "",
            "## Missing Runs",
        ]
    )
    if missing_runs:
        for item in missing_runs:
            lines.append(
                f"- variant `{item['variant']}`, seed `{item['seed']}`: {item['reason']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Threshold Rules",
            "- `val_p99_5`: 99.5th percentile of healthy validation scores.",
            "- `val_mean_plus_3std`: healthy validation mean plus three standard deviations; validation-calibrated.",
            "- `train_mean_plus_3std`: healthy training-score mean plus three standard deviations; no-validation-calibration comparator.",
            f"- Rules evaluated in this run: `{', '.join(threshold_rules)}`",
            f"- Main ablation threshold: `{main_threshold_rule}`",
            "",
            "## Main Ablation Summary",
            markdown_table(main_headers, main_summary_rows),
            "",
            "## Calibration Comparison Summary",
            markdown_table(calibration_headers, calibration_rows),
            "",
            "## Selected Architecture Consistency",
            f"- Table IV calibration comparison is fixed to `{SELECTED_CALIBRATION_LABEL}` using internal ablation variant `{SELECTED_CALIBRATION_VARIANT}`.",
            f"- Paper-facing alias row: `{SELECTED_CALIBRATION_ALIAS}` with `threshold_rule = train_mean_plus_3std`.",
            "",
            "## Score Availability Audit",
            f"- Present for each `{SELECTED_CALIBRATION_VARIANT}` seed (`42`, `7`, `123`): `train_healthy_scores.npy`, `val_healthy_scores.npy`, `test_healthy_scores.npy`, `test_fault_scores.npy`, `best.pt`.",
            "- Absent by design in the ablation run directories: combined `test_scores.npy` and `test_labels.npy`.",
            "",
            "## Group-Wise Recall Analysis",
        ]
    )
    if group_analysis_report_path is not None and group_analysis_condition_rows and group_analysis_damage_rows:
        lines.append(f"- Detailed breakdown report: `{group_analysis_report_path.as_posix()}`")
        for finding in summarize_group_findings(
            condition_rows=group_analysis_condition_rows,
            damage_rows=group_analysis_damage_rows,
        ):
            lines.append(f"- {finding}")
        lines.append(
            "- Metadata alignment was count-verified for every selected seed, but row-level attachment still relies on manifest write order matching array write order."
        )
        for assumption in group_analysis_assumptions:
            lines.append(f"- Assumption: {assumption}")
        lines.append("")
        lines.append(
            markdown_table(
                [
                    "seed",
                    "healthy_metadata_count",
                    "healthy_score_count",
                    "healthy_aligned",
                    "fault_metadata_count",
                    "fault_score_count",
                    "fault_aligned",
                ],
                group_alignment_rows,
            )
        )
    else:
        lines.append("- Group-wise recall analysis was not generated in this run.")

    lines.extend(
        [
            "",
            "## Recompute Status",
        ]
    )
    if train_score_events:
        lines.append(
            "- Fresh train-score inference was used because at least one `train_healthy_scores.npy` file was missing."
        )
    else:
        lines.append(
            "- Retraining was not needed. Metrics were recomputed directly from saved score arrays, and no new train healthy scores were inferred."
        )

    lines.extend(
        [
            "",
            "## Train Score Inference",
        ]
    )
    if train_score_events:
        for item in train_score_events:
            lines.append(
                f"- Computed `{item['path']}` for `{item['variant']}` seed `{item['seed']}` "
                f"with `{item['count']}` scores on `{item['device']}`."
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings Or Reproducibility Issues"])
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    artifacts_root = args.artifacts_root.resolve()
    evaluation_root = artifacts_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    per_seed_rows: list[dict[str, Any]] = []
    evaluated_runs: list[dict[str, Any]] = []
    missing_runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    train_score_events: list[dict[str, Any]] = []
    selected_variant_seed_payloads: list[dict[str, Any]] = []

    for variant in args.variants:
        for seed in args.seeds:
            run_files = resolve_run_files(artifacts_root, variant, int(seed))
            missing_files = [path for path in required_run_files(run_files) if not path.exists()]
            if missing_files:
                reason = "missing files: " + ", ".join(path.name for path in missing_files)
                if args.allow_missing:
                    missing_runs.append({"variant": variant, "seed": int(seed), "reason": reason})
                    continue
                raise FileNotFoundError(
                    f"Requested run is incomplete for {variant} seed {seed}: "
                    + ", ".join(path.as_posix() for path in missing_files)
                )

            run_config = read_json(run_files.run_config_json)
            val_scores = load_score_array(run_files.val_healthy_scores_npy, "val_healthy")
            test_healthy_scores = load_score_array(run_files.test_healthy_scores_npy, "test_healthy")
            test_fault_scores = load_score_array(run_files.test_fault_scores_npy, "test_fault")
            if variant == SELECTED_CALIBRATION_VARIANT and int(seed) in DEFAULT_SEEDS:
                selected_variant_seed_payloads.append(
                    {
                        "seed": int(seed),
                        "val_scores": val_scores,
                        "test_healthy_scores": test_healthy_scores,
                        "test_fault_scores": test_fault_scores,
                        "run_dir": run_files.run_dir.as_posix(),
                    }
                )
            rank_metrics = ranking_metrics(test_healthy_scores, test_fault_scores)
            train_scores: np.ndarray | None = None
            if "train_mean_plus_3std" in args.threshold_rules:
                train_scores = compute_train_scores_if_needed(
                    variant=variant,
                    seed=int(seed),
                    run_files=run_files,
                    run_config=run_config,
                    compute_if_missing=args.compute_train_scores_if_missing,
                    warnings=warnings,
                    train_score_events=train_score_events,
                )

            evaluated_runs.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "run_dir": run_files.run_dir.as_posix(),
                    "n_val_healthy": int(val_scores.shape[0]),
                    "n_test_healthy": int(test_healthy_scores.shape[0]),
                    "n_test_fault": int(test_fault_scores.shape[0]),
                }
            )
            for rule in args.threshold_rules:
                threshold = threshold_value(rule, val_scores=val_scores, train_scores=train_scores)
                threshold_metrics = thresholded_metrics(
                    test_healthy_scores=test_healthy_scores,
                    test_fault_scores=test_fault_scores,
                    threshold=threshold,
                )
                per_seed_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "threshold_rule": rule,
                        "threshold_value": float(threshold),
                        "auroc": rank_metrics["auroc"],
                        "auprc": rank_metrics["auprc"],
                        "f1": threshold_metrics["f1"],
                        "precision": threshold_metrics["precision"],
                        "recall_fault": threshold_metrics["recall_fault"],
                        "far": threshold_metrics["far"],
                        "n_val_healthy": int(val_scores.shape[0]),
                        "n_test_healthy": int(test_healthy_scores.shape[0]),
                        "n_test_fault": int(test_fault_scores.shape[0]),
                        "run_dir": run_files.run_dir.as_posix(),
                    }
                )

    main_summary_rows = summarize_by_variant(per_seed_rows, args.main_threshold_rule)
    main_summary_rows = append_selected_variant_alias_row(
        main_summary_rows,
        per_seed_rows=per_seed_rows,
        source_variant=SELECTED_CALIBRATION_VARIANT,
        alias_variant=SELECTED_CALIBRATION_ALIAS,
        threshold_rule="train_mean_plus_3std",
    )
    calibration_rows = summarize_calibration(
        per_seed_rows,
        selected_variant=SELECTED_CALIBRATION_VARIANT,
        selected_alias=SELECTED_CALIBRATION_ALIAS,
    )
    if SELECTED_CALIBRATION_VARIANT in args.variants and not calibration_rows:
        warnings.append(
            f"No {SELECTED_CALIBRATION_VARIANT} calibration comparison rows were available for the selected architecture."
        )

    group_analysis_assumptions: list[str] = []
    group_alignment_rows: list[dict[str, Any]] = []
    group_condition_seed_rows: list[dict[str, Any]] = []
    group_damage_seed_rows: list[dict[str, Any]] = []
    group_condition_summary_rows: list[dict[str, Any]] = []
    group_damage_summary_rows: list[dict[str, Any]] = []
    group_analysis_report_text = ""
    group_analysis_report_path: Path | None = None

    if selected_variant_seed_payloads:
        manifest_rows = load_test_window_metadata(WINDOW_MANIFEST_PATH)
        healthy_metadata = manifest_rows["test_healthy"]
        fault_metadata = manifest_rows["test_fault"]
        group_analysis_assumptions = [
            "No explicit saved window IDs accompany the score arrays, so row-level attachment uses preprocessing order.",
            "That attachment order was source-verified from `scripts/preprocess_paderborn.py`, which writes test arrays and manifest rows in the same sequential loop with shared split offsets.",
            "Operating-condition values come from the manifest `condition_code` field and are exported as `operating_condition`.",
            "Damage groups remain bearing-family inferences (`KA`, `KB`, `KI`) from preprocessing and are not PDF-verified labels.",
        ]
        for item in sorted(selected_variant_seed_payloads, key=lambda value: int(value["seed"])):
            threshold = threshold_value(
                SELECTED_GROUP_ANALYSIS_RULE,
                val_scores=np.asarray(item["val_scores"], dtype=np.float32),
                train_scores=None,
            )
            group_alignment_rows.append(
                validate_alignment(
                    seed=int(item["seed"]),
                    healthy_metadata=healthy_metadata,
                    fault_metadata=fault_metadata,
                    test_healthy_scores=np.asarray(item["test_healthy_scores"], dtype=np.float32),
                    test_fault_scores=np.asarray(item["test_fault_scores"], dtype=np.float32),
                )
            )
            group_condition_seed_rows.extend(
                compute_condition_rows_for_seed(
                    seed=int(item["seed"]),
                    threshold=threshold,
                    healthy_metadata=healthy_metadata,
                    fault_metadata=fault_metadata,
                    test_healthy_scores=np.asarray(item["test_healthy_scores"], dtype=np.float32),
                    test_fault_scores=np.asarray(item["test_fault_scores"], dtype=np.float32),
                )
            )
            group_damage_seed_rows.extend(
                compute_damage_rows_for_seed(
                    seed=int(item["seed"]),
                    threshold=threshold,
                    fault_metadata=fault_metadata,
                    test_fault_scores=np.asarray(item["test_fault_scores"], dtype=np.float32),
                )
            )
            item["threshold_value"] = float(threshold)
        group_condition_summary_rows = aggregate_group_rows(
            per_seed_rows=group_condition_seed_rows,
            group_field="operating_condition",
            count_fields=("n_healthy", "n_fault"),
            metric_fields=CONDITION_GROUP_METRICS,
        )
        group_damage_summary_rows = aggregate_group_rows(
            per_seed_rows=group_damage_seed_rows,
            group_field="damage_group",
            count_fields=("n_fault",),
            metric_fields=DAMAGE_GROUP_METRICS,
        )
        group_analysis_report_text = build_group_breakdown_report(
            evaluation_root=evaluation_root,
            manifest_path=WINDOW_MANIFEST_PATH,
            alignment_rows=group_alignment_rows,
            condition_rows=group_condition_summary_rows,
            damage_rows=group_damage_summary_rows,
            assumptions=group_analysis_assumptions,
        )
        group_analysis_report_path = evaluation_root / "paderborn_group_breakdown_report.md"

    per_seed_fieldnames = [
        "variant",
        "seed",
        "threshold_rule",
        "threshold_value",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall_fault",
        "far",
        "n_val_healthy",
        "n_test_healthy",
        "n_test_fault",
        "run_dir",
    ]
    main_summary_fieldnames = [
        "variant",
        "residual",
        "dilation",
        "frequency_loss",
        "threshold_rule",
        "calibration",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_fault_mean",
        "recall_fault_std",
        "far_mean",
        "far_std",
    ]
    calibration_fieldnames = [
        "variant",
        "threshold_rule",
        "threshold_value_mean",
        "threshold_value_std",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_fault_mean",
        "recall_fault_std",
        "far_mean",
        "far_std",
    ]
    condition_breakdown_fieldnames = [
        "operating_condition",
        "seed_count",
        "n_healthy",
        "n_fault",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_fault_mean",
        "recall_fault_std",
        "far_mean",
        "far_std",
        "missed_fault_count_mean",
        "missed_fault_count_std",
        "missed_fault_percent_mean",
        "missed_fault_percent_std",
    ]
    damage_breakdown_fieldnames = [
        "damage_group",
        "seed_count",
        "n_fault",
        "recall_fault_mean",
        "recall_fault_std",
        "missed_fault_count_mean",
        "missed_fault_count_std",
        "missed_fault_percent_mean",
        "missed_fault_percent_std",
        "mean_score_mean",
        "mean_score_std",
        "median_score_mean",
        "median_score_std",
    ]

    output_paths = {
        "per_seed_metrics_csv": evaluation_root / "per_seed_metrics.csv",
        "ablation_summary_csv": evaluation_root / f"ablation_summary_{args.main_threshold_rule}.csv",
        "calibration_comparison_csv": evaluation_root / "calibration_comparison_resdilated_full.csv",
        "per_window_predictions_csv": evaluation_root / "per_window_predictions_resdilated_time.csv",
        "condition_breakdown_csv": evaluation_root / "paderborn_condition_breakdown.csv",
        "damage_group_breakdown_csv": evaluation_root / "paderborn_damage_group_breakdown.csv",
        "group_breakdown_report_md": evaluation_root / "paderborn_group_breakdown_report.md",
        "evaluation_report_md": evaluation_root / "evaluation_report.md",
        "metrics_summary_json": evaluation_root / "metrics_summary.json",
    }
    write_csv(output_paths["per_seed_metrics_csv"], per_seed_rows, per_seed_fieldnames)
    write_csv(output_paths["ablation_summary_csv"], main_summary_rows, main_summary_fieldnames)
    write_csv(output_paths["calibration_comparison_csv"], calibration_rows, calibration_fieldnames)
    if selected_variant_seed_payloads:
        write_per_window_predictions_csv(
            path=output_paths["per_window_predictions_csv"],
            per_seed_predictions=selected_variant_seed_payloads,
            healthy_metadata=healthy_metadata,
            fault_metadata=fault_metadata,
        )
        write_csv(
            output_paths["condition_breakdown_csv"],
            group_condition_summary_rows,
            condition_breakdown_fieldnames,
        )
        write_csv(
            output_paths["damage_group_breakdown_csv"],
            group_damage_summary_rows,
            damage_breakdown_fieldnames,
        )
        write_text(output_paths["group_breakdown_report_md"], group_analysis_report_text)
    report_text = build_report(
        artifacts_root=artifacts_root,
        evaluation_root=evaluation_root,
        requested_variants=list(args.variants),
        requested_seeds=[int(seed) for seed in args.seeds],
        threshold_rules=list(args.threshold_rules),
        main_threshold_rule=args.main_threshold_rule,
        evaluated_runs=evaluated_runs,
        missing_runs=missing_runs,
        train_score_events=train_score_events,
        warnings=warnings,
        main_summary_rows=main_summary_rows,
        calibration_rows=calibration_rows,
        group_analysis_report_path=group_analysis_report_path,
        group_analysis_condition_rows=group_condition_summary_rows,
        group_analysis_damage_rows=group_damage_summary_rows,
        group_alignment_rows=group_alignment_rows,
        group_analysis_assumptions=group_analysis_assumptions,
    )
    write_text(output_paths["evaluation_report_md"], report_text)
    write_json(
        output_paths["metrics_summary_json"],
        {
            "study": "paderborn_architectural_ablation_chunk3_evaluation",
            "artifacts_root": artifacts_root.as_posix(),
            "evaluation_root": evaluation_root.as_posix(),
            "requested": {
                "variants": list(args.variants),
                "seeds": [int(seed) for seed in args.seeds],
                "threshold_rules": list(args.threshold_rules),
                "main_threshold_rule": args.main_threshold_rule,
                "allow_missing": bool(args.allow_missing),
                "compute_train_scores_if_missing": bool(args.compute_train_scores_if_missing),
            },
            "evaluated_runs": evaluated_runs,
            "missing_runs": missing_runs,
            "train_score_events": train_score_events,
            "warnings": warnings,
            "per_seed_metrics": per_seed_rows,
            "main_summary": main_summary_rows,
            "selected_calibration_variant": SELECTED_CALIBRATION_VARIANT,
            "selected_calibration_alias": SELECTED_CALIBRATION_ALIAS,
            "calibration_comparison_resdilated_full": calibration_rows,
            "selected_group_analysis_threshold_rule": SELECTED_GROUP_ANALYSIS_RULE,
            "group_analysis_alignment": group_alignment_rows,
            "group_analysis_assumptions": group_analysis_assumptions,
            "group_analysis_condition_per_seed": group_condition_seed_rows,
            "group_analysis_damage_per_seed": group_damage_seed_rows,
            "group_analysis_condition_summary": group_condition_summary_rows,
            "group_analysis_damage_summary": group_damage_summary_rows,
            "output_files": {key: path.as_posix() for key, path in output_paths.items()},
        },
    )

    print("Paderborn ablation evaluation complete.")
    print(f"Evaluation root: {evaluation_root.as_posix()}")
    print(f"Evaluated runs: {len(evaluated_runs)}")
    print(f"Missing runs: {len(missing_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
