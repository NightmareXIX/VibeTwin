from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

try:
    from train_ae_baseline import evaluate_scores as evaluate_binary_scores, require_torch, set_seed, torch
    from train_paderborn_baselines import ensure_required_files, load_label_array, read_json, resolve_paths
    from train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_loader_bundle,
        build_models,
        build_run_paths,
        get_device,
        load_torch_payload,
        save_numpy_array,
        select_models,
        write_json,
        write_text,
    )
except ModuleNotFoundError:
    from scripts.train_ae_baseline import evaluate_scores as evaluate_binary_scores, require_torch, set_seed, torch
    from scripts.train_paderborn_baselines import ensure_required_files, load_label_array, read_json, resolve_paths
    from scripts.train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_loader_bundle,
        build_models,
        build_run_paths,
        get_device,
        load_torch_payload,
        save_numpy_array,
        select_models,
        write_json,
        write_text,
    )


RUN_CONFIG = ModelRunConfig(
    name="ResDilatedAE",
    cli_name="resdilated_ae",
    output_stem="resdilated_ae",
    model_kind="ae",
)
DEFAULT_SEEDS = (42, 7, 123)
SCORE_THRESHOLD_RULE = "percentile_99_5"
UNCERTAINTY_THRESHOLD_RULE = "variance_percentile_99_5"
DROPOUT_MODULE_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference-only MC-dropout uncertainty analysis for saved Paderborn ResDilatedAE checkpoints.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size-cuda", type=int, default=256)
    parser.add_argument("--batch-size-cpu", type=int, default=128)
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--uncertainty-percentile", type=float, default=99.5)
    args = parser.parse_args()
    if args.mc_passes < 2:
        parser.error("--mc-passes must be at least 2 for a variance estimate.")
    if not 0.0 < args.uncertainty_percentile < 100.0:
        parser.error("--uncertainty-percentile must be between 0 and 100.")
    return args


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def summarize_values(values: list[float]) -> dict[str, float]:
    numeric_values = [float(value) for value in values]
    return {
        "mean": float(mean(numeric_values)),
        "std": float(stdev(numeric_values)) if len(numeric_values) > 1 else 0.0,
    }


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError("Threshold fit scores are empty.")
    return float(np.percentile(values, percentile))


def load_resdilatedae_from_checkpoint(
    *,
    checkpoint_path: Path,
    expected_seed: int,
    expected_width: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path.as_posix()}")
    checkpoint_payload = load_torch_payload(checkpoint_path)

    model_cli_name = checkpoint_payload.get("model_cli_name")
    model_name = checkpoint_payload.get("model_name")
    checkpoint_seed = int(checkpoint_payload.get("seed", -1))
    if model_cli_name != RUN_CONFIG.cli_name or model_name != RUN_CONFIG.name:
        raise RuntimeError(
            f"Checkpoint mismatch at {checkpoint_path.as_posix()}: "
            f"expected {RUN_CONFIG.name}/{RUN_CONFIG.cli_name}, found {model_name}/{model_cli_name}"
        )
    if checkpoint_seed != expected_seed:
        raise RuntimeError(
            f"Checkpoint seed mismatch at {checkpoint_path.as_posix()}: "
            f"expected seed {expected_seed}, found {checkpoint_seed}"
        )

    checkpoint_settings = checkpoint_payload.get("training_settings", {})
    dropout = float(checkpoint_settings.get("dropout", 0.05))
    model_candidates = select_models(build_models(expected_width, dropout), RUN_CONFIG.cli_name)
    if len(model_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one model definition for {RUN_CONFIG.cli_name}, found {len(model_candidates)}."
        )
    _run_config, model = model_candidates[0]
    try:
        model.load_state_dict(checkpoint_payload["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint/model mismatch while loading {checkpoint_path.as_posix()}: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    return model, checkpoint_payload


def enable_mc_dropout(model: torch.nn.Module) -> dict[str, Any]:
    model.eval()
    dropout_probabilities: list[float] = []
    active_dropout_modules = 0
    for module in model.modules():
        if isinstance(module, DROPOUT_MODULE_TYPES):
            probability = float(getattr(module, "p", 0.0))
            if probability > 0.0:
                module.train(True)
                active_dropout_modules += 1
                dropout_probabilities.append(probability)
    return {
        "active_dropout_modules": int(active_dropout_modules),
        "dropout_probabilities": sorted({float(probability) for probability in dropout_probabilities}),
    }


def compute_batch_scores(
    *,
    model: torch.nn.Module,
    batch: torch.Tensor,
    model_kind: str,
) -> np.ndarray:
    if model_kind == "vae":
        reconstruction, _, _ = model(batch)
    else:
        reconstruction = model(batch)
    per_window_mse = torch.mean((reconstruction.float() - batch.float()) ** 2, dim=(1, 2))
    return per_window_mse.detach().cpu().numpy().astype(np.float64, copy=False)


def probe_mc_dropout_support(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    model_kind: str,
) -> dict[str, Any]:
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Validation loader is empty; cannot probe MC-dropout support.") from exc
    batch = batch.to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        first_scores = compute_batch_scores(model=model, batch=batch, model_kind=model_kind)
        second_scores = compute_batch_scores(model=model, batch=batch, model_kind=model_kind)
    absolute_diff = np.abs(first_scores - second_scores)
    return {
        "probe_batch_size": int(first_scores.shape[0]),
        "probe_max_abs_diff": float(absolute_diff.max()) if absolute_diff.size else 0.0,
        "probe_mean_abs_diff": float(absolute_diff.mean()) if absolute_diff.size else 0.0,
    }


def compute_mc_score_statistics(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    model_kind: str,
    mc_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_size = len(loader.dataset)
    score_sum = np.zeros(dataset_size, dtype=np.float64)
    score_sum_sq = np.zeros(dataset_size, dtype=np.float64)
    with torch.no_grad():
        for pass_index in range(mc_passes):
            offset = 0
            for batch in loader:
                batch = batch.to(device, non_blocking=device.type == "cuda")
                batch_scores = compute_batch_scores(model=model, batch=batch, model_kind=model_kind)
                next_offset = offset + batch_scores.shape[0]
                score_sum[offset:next_offset] += batch_scores
                score_sum_sq[offset:next_offset] += np.square(batch_scores)
                offset = next_offset
            if offset != dataset_size:
                raise RuntimeError(
                    f"MC score pass {pass_index + 1} produced {offset} windows, expected {dataset_size}."
                )
    mean_scores = score_sum / float(mc_passes)
    variance_scores = np.maximum((score_sum_sq / float(mc_passes)) - np.square(mean_scores), 0.0)
    return (
        mean_scores.astype(np.float32, copy=False),
        variance_scores.astype(np.float32, copy=False),
    )


def load_test_manifest_metadata(
    *,
    window_manifest_path: Path,
    fault_labels: np.ndarray,
) -> dict[str, list[str]]:
    healthy_conditions: list[str] = []
    fault_conditions: list[str] = []
    fault_damage_groups: list[str] = []
    manifest_fault_labels: list[int] = []

    with window_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subset = row["subset"]
            if subset == "test_healthy":
                healthy_conditions.append(row["condition_code"])
            elif subset == "test_fault":
                if not row["condition_code"]:
                    raise RuntimeError("Window manifest test_fault row is missing condition_code.")
                if not row["damage_group"]:
                    raise RuntimeError("Window manifest test_fault row is missing damage_group.")
                if not row["fault_label_int"]:
                    raise RuntimeError("Window manifest test_fault row is missing fault_label_int.")
                fault_conditions.append(row["condition_code"])
                fault_damage_groups.append(row["damage_group"])
                manifest_fault_labels.append(int(row["fault_label_int"]))

    manifest_fault_array = np.asarray(manifest_fault_labels, dtype=np.int64)
    expected_fault_array = np.asarray(fault_labels, dtype=np.int64).reshape(-1)
    if manifest_fault_array.shape[0] != expected_fault_array.shape[0]:
        raise RuntimeError(
            "Window manifest test_fault row count does not match fault_labels.npy: "
            f"{manifest_fault_array.shape[0]} != {expected_fault_array.shape[0]}"
        )
    mismatches = np.flatnonzero(manifest_fault_array != expected_fault_array)
    if mismatches.size:
        mismatch_index = int(mismatches[0])
        raise RuntimeError(
            f"fault_labels.npy mismatch at index {mismatch_index}: "
            f"manifest={manifest_fault_array[mismatch_index]}, fault_labels={expected_fault_array[mismatch_index]}"
        )

    return {
        "healthy_conditions": healthy_conditions,
        "fault_conditions": fault_conditions,
        "fault_damage_groups": fault_damage_groups,
    }


def subgroup_metrics_for_scores(
    *,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    healthy_conditions: list[str],
    fault_conditions: list[str],
    fault_damage_groups: list[str],
    threshold: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    if len(healthy_conditions) != int(test_healthy_scores.shape[0]):
        raise RuntimeError(
            "Test healthy manifest alignment mismatch: "
            f"{len(healthy_conditions)} rows vs {test_healthy_scores.shape[0]} scores."
        )
    if len(fault_conditions) != int(test_fault_scores.shape[0]):
        raise RuntimeError(
            "Test fault manifest alignment mismatch: "
            f"{len(fault_conditions)} rows vs {test_fault_scores.shape[0]} scores."
        )
    if len(fault_damage_groups) != int(test_fault_scores.shape[0]):
        raise RuntimeError(
            "Test fault damage-group alignment mismatch: "
            f"{len(fault_damage_groups)} rows vs {test_fault_scores.shape[0]} scores."
        )

    healthy_by_condition: dict[str, list[float]] = defaultdict(list)
    fault_by_condition: dict[str, list[float]] = defaultdict(list)
    fault_by_damage_group: dict[str, list[float]] = defaultdict(list)

    for condition_code, score in zip(healthy_conditions, test_healthy_scores, strict=True):
        healthy_by_condition[condition_code].append(float(score))
    for condition_code, score in zip(fault_conditions, test_fault_scores, strict=True):
        fault_by_condition[condition_code].append(float(score))
    for damage_group, score in zip(fault_damage_groups, test_fault_scores, strict=True):
        fault_by_damage_group[damage_group].append(float(score))

    by_condition: dict[str, dict[str, Any]] = {}
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

    by_damage_group: dict[str, dict[str, Any]] = {}
    for damage_group in sorted(fault_by_damage_group):
        group_scores = np.asarray(fault_by_damage_group[damage_group], dtype=np.float32)
        by_damage_group[damage_group] = {
            **evaluate_binary_scores(
                threshold=threshold,
                test_healthy_errors=np.asarray(test_healthy_scores, dtype=np.float32),
                test_fault_errors=group_scores,
            ),
            "group_name": damage_group,
        }

    return {
        "by_condition": by_condition,
        "by_damage_group": by_damage_group,
    }


def build_uncertainty_aware_metrics(
    *,
    test_healthy_mean_scores: np.ndarray,
    test_fault_mean_scores: np.ndarray,
    test_healthy_uncertainty: np.ndarray,
    test_fault_uncertainty: np.ndarray,
    score_threshold: float,
    uncertainty_threshold: float,
) -> dict[str, Any]:
    healthy_deferred_mask = np.asarray(test_healthy_uncertainty, dtype=np.float64) > float(uncertainty_threshold)
    fault_deferred_mask = np.asarray(test_fault_uncertainty, dtype=np.float64) > float(uncertainty_threshold)

    kept_healthy_scores = np.asarray(test_healthy_mean_scores, dtype=np.float32)[~healthy_deferred_mask]
    kept_fault_scores = np.asarray(test_fault_mean_scores, dtype=np.float32)[~fault_deferred_mask]
    if kept_healthy_scores.size == 0 or kept_fault_scores.size == 0:
        raise RuntimeError("Uncertainty-aware defer rule removed an entire class; cannot compute binary metrics.")

    metrics = evaluate_binary_scores(
        threshold=float(score_threshold),
        test_healthy_errors=kept_healthy_scores,
        test_fault_errors=kept_fault_scores,
    )
    metrics["threshold"] = float(score_threshold)
    deferred_healthy = int(healthy_deferred_mask.sum())
    deferred_fault = int(fault_deferred_mask.sum())
    total_windows = int(test_healthy_mean_scores.shape[0] + test_fault_mean_scores.shape[0])
    return {
        "metrics": metrics,
        "deferred": {
            "healthy_deferred": deferred_healthy,
            "fault_deferred": deferred_fault,
            "total_deferred": int(deferred_healthy + deferred_fault),
            "healthy_kept": int(kept_healthy_scores.shape[0]),
            "fault_kept": int(kept_fault_scores.shape[0]),
            "healthy_deferred_rate": float(deferred_healthy / max(int(test_healthy_mean_scores.shape[0]), 1)),
            "fault_deferred_rate": float(deferred_fault / max(int(test_fault_mean_scores.shape[0]), 1)),
            "total_deferred_rate": float((deferred_healthy + deferred_fault) / max(total_windows, 1)),
        },
    }


def analyze_uncertainty_patterns(
    *,
    score_threshold: float,
    test_healthy_mean_scores: np.ndarray,
    test_fault_mean_scores: np.ndarray,
    test_healthy_uncertainty: np.ndarray,
    test_fault_uncertainty: np.ndarray,
    healthy_conditions: list[str],
    fault_conditions: list[str],
    fault_damage_groups: list[str],
) -> dict[str, Any]:
    test_uncertainty = np.concatenate(
        [
            np.asarray(test_healthy_uncertainty, dtype=np.float64),
            np.asarray(test_fault_uncertainty, dtype=np.float64),
        ],
        axis=0,
    )
    prediction_labels = np.concatenate(
        [
            (np.asarray(test_healthy_mean_scores, dtype=np.float64) >= float(score_threshold)).astype(np.int8),
            (np.asarray(test_fault_mean_scores, dtype=np.float64) >= float(score_threshold)).astype(np.int8),
        ],
        axis=0,
    )
    true_labels = np.concatenate(
        [
            np.zeros(int(test_healthy_mean_scores.shape[0]), dtype=np.int8),
            np.ones(int(test_fault_mean_scores.shape[0]), dtype=np.int8),
        ],
        axis=0,
    )
    misclassified_mask = prediction_labels != true_labels
    correct_mask = ~misclassified_mask
    misclassified_uncertainty = test_uncertainty[misclassified_mask]
    correct_uncertainty = test_uncertainty[correct_mask]

    condition_uncertainty: dict[str, list[float]] = defaultdict(list)
    damage_group_uncertainty: dict[str, list[float]] = defaultdict(list)
    for condition_code, uncertainty in zip(healthy_conditions, test_healthy_uncertainty, strict=True):
        condition_uncertainty[condition_code].append(float(uncertainty))
    for condition_code, uncertainty in zip(fault_conditions, test_fault_uncertainty, strict=True):
        condition_uncertainty[condition_code].append(float(uncertainty))
    for damage_group, uncertainty in zip(fault_damage_groups, test_fault_uncertainty, strict=True):
        damage_group_uncertainty[damage_group].append(float(uncertainty))

    subgroup_metrics = subgroup_metrics_for_scores(
        test_healthy_scores=test_healthy_mean_scores,
        test_fault_scores=test_fault_mean_scores,
        healthy_conditions=healthy_conditions,
        fault_conditions=fault_conditions,
        fault_damage_groups=fault_damage_groups,
        threshold=float(score_threshold),
    )

    hardest_condition_code, hardest_condition_metrics = min(
        subgroup_metrics["by_condition"].items(),
        key=lambda item: float(item[1]["f1"]),
    )
    hardest_damage_group, hardest_damage_metrics = min(
        subgroup_metrics["by_damage_group"].items(),
        key=lambda item: float(item[1]["f1"]),
    )

    overall_test_mean_uncertainty = float(np.mean(test_uncertainty))
    overall_fault_mean_uncertainty = float(np.mean(np.asarray(test_fault_uncertainty, dtype=np.float64)))
    hardest_condition_mean_uncertainty = float(np.mean(condition_uncertainty[hardest_condition_code]))
    hardest_damage_mean_uncertainty = float(np.mean(damage_group_uncertainty[hardest_damage_group]))

    correct_mean_uncertainty = float(correct_uncertainty.mean()) if correct_uncertainty.size else 0.0
    misclassified_mean_uncertainty = float(misclassified_uncertainty.mean()) if misclassified_uncertainty.size else 0.0
    uncertainty_ratio = (
        float(misclassified_mean_uncertainty / correct_mean_uncertainty) if correct_mean_uncertainty > 0 else None
    )

    return {
        "baseline_mc_subgroups": subgroup_metrics,
        "misclassified_vs_correct": {
            "misclassified_count": int(misclassified_mask.sum()),
            "correct_count": int(correct_mask.sum()),
            "misclassified_mean_uncertainty": misclassified_mean_uncertainty,
            "correct_mean_uncertainty": correct_mean_uncertainty,
            "uncertainty_ratio": uncertainty_ratio,
            "higher_than_correct": bool(misclassified_mean_uncertainty > correct_mean_uncertainty),
        },
        "hardest_condition": {
            "condition_code": hardest_condition_code,
            "baseline_f1": float(hardest_condition_metrics["f1"]),
            "mean_uncertainty": hardest_condition_mean_uncertainty,
            "overall_test_mean_uncertainty": overall_test_mean_uncertainty,
            "uncertainty_gap": float(hardest_condition_mean_uncertainty - overall_test_mean_uncertainty),
            "higher_than_overall": bool(hardest_condition_mean_uncertainty > overall_test_mean_uncertainty),
        },
        "hardest_damage_group": {
            "damage_group": hardest_damage_group,
            "baseline_f1": float(hardest_damage_metrics["f1"]),
            "mean_uncertainty": hardest_damage_mean_uncertainty,
            "overall_fault_mean_uncertainty": overall_fault_mean_uncertainty,
            "uncertainty_gap": float(hardest_damage_mean_uncertainty - overall_fault_mean_uncertainty),
            "higher_than_overall_fault": bool(hardest_damage_mean_uncertainty > overall_fault_mean_uncertainty),
        },
    }


def summarize_metric_block(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = ["threshold", "auroc", "f1", "precision", "recall_fault", "false_alarm_rate"]
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        summary[metric_name] = summarize_values([float(entry[metric_name]) for entry in entries])
    return summary


def build_mc_score_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "val_healthy_mean": run_dir / "val_healthy_mc_mean_scores.npy",
        "val_healthy_variance": run_dir / "val_healthy_mc_score_variance.npy",
        "test_healthy_mean": run_dir / "test_healthy_mc_mean_scores.npy",
        "test_healthy_variance": run_dir / "test_healthy_mc_score_variance.npy",
        "test_fault_mean": run_dir / "test_fault_mc_mean_scores.npy",
        "test_fault_variance": run_dir / "test_fault_mc_score_variance.npy",
    }


def build_seed_baseline_table(seed_results: list[dict[str, Any]]) -> str:
    rows: list[list[Any]] = []
    for seed_result in seed_results:
        metrics = seed_result["deterministic_baseline"]["metrics"]
        rows.append(
            [
                seed_result["seed"],
                f"{metrics['threshold']:.6f}",
                f"{metrics['auroc']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall_fault']:.6f}",
                f"{metrics['false_alarm_rate']:.6f}",
            ]
        )
    return format_markdown_table(
        ["Seed", "Threshold", "AUROC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
        rows,
    )


def build_seed_uncertainty_table(seed_results: list[dict[str, Any]]) -> str:
    rows: list[list[Any]] = []
    for seed_result in seed_results:
        metrics = seed_result["uncertainty_aware"]["metrics"]
        deferred = seed_result["uncertainty_aware"]["deferred"]
        mc_metrics = seed_result["mc_calibrated_no_defer"]["metrics"]
        rows.append(
            [
                seed_result["seed"],
                f"{seed_result['mc_calibrated_no_defer']['threshold']:.6f}",
                f"{seed_result['uncertainty_aware']['uncertainty_threshold']:.8f}",
                f"{metrics['auroc']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall_fault']:.6f}",
                f"{metrics['false_alarm_rate']:.6f}",
                f"{deferred['total_deferred']} ({deferred['total_deferred_rate']:.4%})",
                f"{metrics['false_alarm_rate'] - mc_metrics['false_alarm_rate']:+.6f}",
            ]
        )
    return format_markdown_table(
        [
            "Seed",
            "MC Score Thr",
            "Uncertainty Thr",
            "AUROC",
            "F1",
            "Precision",
            "Recall Fault",
            "False Alarm Rate",
            "Deferred",
            "FAR Delta vs MC",
        ],
        rows,
    )


def build_seed_pattern_table(seed_results: list[dict[str, Any]]) -> str:
    rows: list[list[Any]] = []
    for seed_result in seed_results:
        misclassified = seed_result["uncertainty_patterns"]["misclassified_vs_correct"]
        hardest_condition = seed_result["uncertainty_patterns"]["hardest_condition"]
        hardest_damage_group = seed_result["uncertainty_patterns"]["hardest_damage_group"]
        ratio = misclassified["uncertainty_ratio"]
        rows.append(
            [
                seed_result["seed"],
                "yes" if misclassified["higher_than_correct"] else "no",
                f"{ratio:.3f}" if ratio is not None else "n/a",
                hardest_condition["condition_code"],
                f"{hardest_condition['baseline_f1']:.6f}",
                f"{hardest_condition['uncertainty_gap']:+.8f}",
                hardest_damage_group["damage_group"],
                f"{hardest_damage_group['baseline_f1']:.6f}",
                f"{hardest_damage_group['uncertainty_gap']:+.8f}",
            ]
        )
    return format_markdown_table(
        [
            "Seed",
            "Miscls Higher?",
            "Miscls/Correct Unc",
            "Hardest Condition",
            "Cond F1",
            "Cond Unc Gap",
            "Hardest Damage",
            "Damage F1",
            "Damage Unc Gap",
        ],
        rows,
    )


def build_summary_metric_table(summary: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for label, key in (
        ("Deterministic Baseline", "deterministic_baseline"),
        ("MC No Defer", "mc_calibrated_no_defer"),
        ("Uncertainty Aware", "uncertainty_aware"),
    ):
        metrics = summary[key]
        rows.append(
            [
                label,
                f"{metrics['auroc']['mean']:.6f} +/- {metrics['auroc']['std']:.6f}",
                f"{metrics['f1']['mean']:.6f} +/- {metrics['f1']['std']:.6f}",
                f"{metrics['precision']['mean']:.6f} +/- {metrics['precision']['std']:.6f}",
                f"{metrics['recall_fault']['mean']:.6f} +/- {metrics['recall_fault']['std']:.6f}",
                f"{metrics['false_alarm_rate']['mean']:.6f} +/- {metrics['false_alarm_rate']['std']:.6f}",
            ]
        )
    return format_markdown_table(
        ["Setting", "AUROC mean+/-std", "F1 mean+/-std", "Precision mean+/-std", "Recall mean+/-std", "FAR mean+/-std"],
        rows,
    )


def build_summary_uncertainty_table(summary: dict[str, Any]) -> str:
    rows = [
        [
            "Uncertainty threshold",
            f"{summary['uncertainty_threshold']['mean']:.8f} +/- {summary['uncertainty_threshold']['std']:.8f}",
        ],
        [
            "Total deferred rate",
            f"{summary['deferred']['total_deferred_rate']['mean']:.4%} +/- {summary['deferred']['total_deferred_rate']['std']:.4%}",
        ],
        [
            "Healthy deferred rate",
            f"{summary['deferred']['healthy_deferred_rate']['mean']:.4%} +/- {summary['deferred']['healthy_deferred_rate']['std']:.4%}",
        ],
        [
            "Fault deferred rate",
            f"{summary['deferred']['fault_deferred_rate']['mean']:.4%} +/- {summary['deferred']['fault_deferred_rate']['std']:.4%}",
        ],
        [
            "FAR delta vs MC no defer",
            f"{summary['false_alarm_delta_vs_mc_no_defer']['mean']:+.6f} +/- "
            f"{summary['false_alarm_delta_vs_mc_no_defer']['std']:.6f}",
        ],
    ]
    return format_markdown_table(["Metric", "Value"], rows)


def build_report(
    *,
    device: torch.device,
    batch_size: int,
    mc_passes: int,
    uncertainty_percentile: float,
    seed_results: list[dict[str, Any]],
    summary: dict[str, Any],
    metrics_path: Path,
    report_path: Path,
) -> str:
    mc_f1 = summary["mc_calibrated_no_defer"]["f1"]["mean"]
    ua_f1 = summary["uncertainty_aware"]["f1"]["mean"]
    mc_recall = summary["mc_calibrated_no_defer"]["recall_fault"]["mean"]
    ua_recall = summary["uncertainty_aware"]["recall_fault"]["mean"]
    mc_far = summary["mc_calibrated_no_defer"]["false_alarm_rate"]["mean"]
    ua_far = summary["uncertainty_aware"]["false_alarm_rate"]["mean"]
    return "\n".join(
        [
            "# Paderborn ResDilatedAE MC-Dropout Uncertainty Report",
            "",
            "## Protocol",
            "- Inference only from saved `best.pt` checkpoints.",
            "- Current backbone reused directly with inference-time dropout activation only.",
            f"- Seeds evaluated: `{', '.join(str(seed_result['seed']) for seed_result in seed_results)}`.",
            f"- MC passes per window: `{mc_passes}`.",
            f"- Score threshold rule: `{SCORE_THRESHOLD_RULE}` fit on validation healthy MC-mean scores.",
            f"- Uncertainty threshold rule: `{UNCERTAINTY_THRESHOLD_RULE}` using validation healthy score variance at `{uncertainty_percentile}` percentile.",
            f"- Device used: `{device}` with effective batch size `{batch_size}`.",
            "",
            "## Baseline Calibrated Metrics",
            build_seed_baseline_table(seed_results),
            "",
            "## Uncertainty-Aware Metrics",
            build_seed_uncertainty_table(seed_results),
            "",
            "## Uncertainty Concentration",
            build_seed_pattern_table(seed_results),
            "",
            "## Mean/Std Across Seeds",
            build_summary_metric_table(summary),
            "",
            build_summary_uncertainty_table(summary),
            "",
            "## Practical Take",
            f"- Deferring high-uncertainty windows moved mean false alarm rate from `{mc_far:.6f}` to `{ua_far:.6f}` "
            f"(`{ua_far - mc_far:+.6f}`), mean F1 from `{mc_f1:.6f}` to `{ua_f1:.6f}` "
            f"(`{ua_f1 - mc_f1:+.6f}`), and mean recall fault from `{mc_recall:.6f}` to `{ua_recall:.6f}` "
            f"(`{ua_recall - mc_recall:+.6f}`).",
            f"- Misclassified windows had higher uncertainty in `{summary['uncertainty_signals']['misclassified_higher_than_correct_count']}`/`{len(seed_results)}` seeds.",
            f"- The hardest operating condition had above-overall uncertainty in `{summary['uncertainty_signals']['hardest_condition_higher_than_overall_count']}`/`{len(seed_results)}` seeds.",
            f"- The hardest damage group had above-overall fault uncertainty in `{summary['uncertainty_signals']['hardest_damage_group_higher_than_overall_fault_count']}`/`{len(seed_results)}` seeds.",
            "",
            "## Saved Artifacts",
            f"- Metrics JSON: `{metrics_path.as_posix()}`",
            f"- Report: `{report_path.as_posix()}`",
            "",
        ]
    )


def main() -> int:
    require_torch()
    args = parse_args()
    set_seed(0)

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)
    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    expected_width = int(preprocessing_config["window_size"])
    fault_labels = load_label_array(array_paths.fault_labels)
    manifest_metadata = load_test_manifest_metadata(
        window_manifest_path=metadata_root / "window_manifest.csv",
        fault_labels=fault_labels,
    )

    device = get_device()
    batch_size = args.batch_size_cuda if device.type == "cuda" else args.batch_size_cpu
    print(f"Device selected: {device}")
    print(f"Effective batch size: {batch_size}")

    loaders = build_loader_bundle(
        array_paths=array_paths,
        batch_size=batch_size,
        train_subset=0,
        val_subset=0,
        test_subset=0,
    )

    metrics_path = artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_mc_dropout_metrics.json"
    report_path = artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_mc_dropout_report.md"

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        set_seed(seed)
        run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=RUN_CONFIG, seed=seed)
        mc_score_paths = build_mc_score_paths(run_paths.run_dir)

        print(f"\nRunning MC-dropout inference for seed {seed}")
        model, checkpoint_payload = load_resdilatedae_from_checkpoint(
            checkpoint_path=run_paths.best_checkpoint,
            expected_seed=seed,
            expected_width=expected_width,
            device=device,
        )
        dropout_support = enable_mc_dropout(model)
        if dropout_support["active_dropout_modules"] <= 0:
            raise RuntimeError(
                f"Seed {seed} checkpoint does not expose active dropout layers for meaningful MC inference: "
                f"{run_paths.best_checkpoint.as_posix()}"
            )
        stochastic_probe = probe_mc_dropout_support(
            model=model,
            loader=loaders["val"],
            device=device,
            model_kind=RUN_CONFIG.model_kind,
        )
        if stochastic_probe["probe_max_abs_diff"] <= 0.0:
            raise RuntimeError(
                f"Seed {seed} checkpoint loaded, but repeated stochastic passes produced identical scores at "
                f"{run_paths.best_checkpoint.as_posix()}. A code change or checkpoint rerun is needed before a "
                "meaningful MC-dropout study can be done."
            )

        deterministic_val_scores = np.load(run_paths.val_healthy_scores_npy).astype(np.float32, copy=False)
        deterministic_test_healthy_scores = np.load(run_paths.test_healthy_scores_npy).astype(np.float32, copy=False)
        deterministic_test_fault_scores = np.load(run_paths.test_fault_scores_npy).astype(np.float32, copy=False)
        deterministic_threshold = percentile_threshold(deterministic_val_scores, 99.5)
        deterministic_baseline_metrics = evaluate_binary_scores(
            threshold=deterministic_threshold,
            test_healthy_errors=deterministic_test_healthy_scores,
            test_fault_errors=deterministic_test_fault_scores,
        )
        deterministic_baseline_metrics["threshold"] = float(deterministic_threshold)

        split_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name, loader in (("val_healthy", loaders["val"]), ("test_healthy", loaders["test_healthy"]), ("test_fault", loaders["test_fault"])):
            print(f"  {split_name}: {args.mc_passes} stochastic passes")
            split_payloads[split_name] = compute_mc_score_statistics(
                model=model,
                loader=loader,
                device=device,
                model_kind=RUN_CONFIG.model_kind,
                mc_passes=args.mc_passes,
            )

        save_numpy_array(mc_score_paths["val_healthy_mean"], split_payloads["val_healthy"][0])
        save_numpy_array(mc_score_paths["val_healthy_variance"], split_payloads["val_healthy"][1])
        save_numpy_array(mc_score_paths["test_healthy_mean"], split_payloads["test_healthy"][0])
        save_numpy_array(mc_score_paths["test_healthy_variance"], split_payloads["test_healthy"][1])
        save_numpy_array(mc_score_paths["test_fault_mean"], split_payloads["test_fault"][0])
        save_numpy_array(mc_score_paths["test_fault_variance"], split_payloads["test_fault"][1])

        mc_threshold = percentile_threshold(split_payloads["val_healthy"][0], 99.5)
        mc_no_defer_metrics = evaluate_binary_scores(
            threshold=mc_threshold,
            test_healthy_errors=split_payloads["test_healthy"][0],
            test_fault_errors=split_payloads["test_fault"][0],
        )
        mc_no_defer_metrics["threshold"] = float(mc_threshold)

        uncertainty_threshold = percentile_threshold(split_payloads["val_healthy"][1], args.uncertainty_percentile)
        uncertainty_aware = build_uncertainty_aware_metrics(
            test_healthy_mean_scores=split_payloads["test_healthy"][0],
            test_fault_mean_scores=split_payloads["test_fault"][0],
            test_healthy_uncertainty=split_payloads["test_healthy"][1],
            test_fault_uncertainty=split_payloads["test_fault"][1],
            score_threshold=mc_threshold,
            uncertainty_threshold=uncertainty_threshold,
        )
        uncertainty_aware["uncertainty_threshold"] = float(uncertainty_threshold)
        uncertainty_aware["false_alarm_rate_delta_vs_mc_no_defer"] = float(
            uncertainty_aware["metrics"]["false_alarm_rate"] - mc_no_defer_metrics["false_alarm_rate"]
        )

        uncertainty_patterns = analyze_uncertainty_patterns(
            score_threshold=mc_threshold,
            test_healthy_mean_scores=split_payloads["test_healthy"][0],
            test_fault_mean_scores=split_payloads["test_fault"][0],
            test_healthy_uncertainty=split_payloads["test_healthy"][1],
            test_fault_uncertainty=split_payloads["test_fault"][1],
            healthy_conditions=manifest_metadata["healthy_conditions"],
            fault_conditions=manifest_metadata["fault_conditions"],
            fault_damage_groups=manifest_metadata["fault_damage_groups"],
        )

        seed_results.append(
            {
                "seed": int(seed),
                "run_dir": run_paths.run_dir.as_posix(),
                "checkpoint_path": run_paths.best_checkpoint.as_posix(),
                "checkpoint_training_settings": checkpoint_payload.get("training_settings", {}),
                "dropout_support": {
                    **dropout_support,
                    **stochastic_probe,
                },
                "saved_mc_score_paths": {key: path.as_posix() for key, path in mc_score_paths.items()},
                "deterministic_baseline": {
                    "threshold_rule": SCORE_THRESHOLD_RULE,
                    "metrics": deterministic_baseline_metrics,
                },
                "mc_calibrated_no_defer": {
                    "threshold_rule": SCORE_THRESHOLD_RULE,
                    "threshold": float(mc_threshold),
                    "metrics": mc_no_defer_metrics,
                },
                "uncertainty_aware": uncertainty_aware,
                "uncertainty_patterns": uncertainty_patterns,
            }
        )

    summary = {
        "deterministic_baseline": summarize_metric_block(
            [seed_result["deterministic_baseline"]["metrics"] for seed_result in seed_results]
        ),
        "mc_calibrated_no_defer": summarize_metric_block(
            [seed_result["mc_calibrated_no_defer"]["metrics"] for seed_result in seed_results]
        ),
        "uncertainty_aware": summarize_metric_block(
            [seed_result["uncertainty_aware"]["metrics"] for seed_result in seed_results]
        ),
        "uncertainty_threshold": summarize_values(
            [float(seed_result["uncertainty_aware"]["uncertainty_threshold"]) for seed_result in seed_results]
        ),
        "deferred": {
            "healthy_deferred_rate": summarize_values(
                [float(seed_result["uncertainty_aware"]["deferred"]["healthy_deferred_rate"]) for seed_result in seed_results]
            ),
            "fault_deferred_rate": summarize_values(
                [float(seed_result["uncertainty_aware"]["deferred"]["fault_deferred_rate"]) for seed_result in seed_results]
            ),
            "total_deferred_rate": summarize_values(
                [float(seed_result["uncertainty_aware"]["deferred"]["total_deferred_rate"]) for seed_result in seed_results]
            ),
        },
        "false_alarm_delta_vs_mc_no_defer": summarize_values(
            [float(seed_result["uncertainty_aware"]["false_alarm_rate_delta_vs_mc_no_defer"]) for seed_result in seed_results]
        ),
        "uncertainty_signals": {
            "misclassified_higher_than_correct_count": int(
                sum(
                    1
                    for seed_result in seed_results
                    if seed_result["uncertainty_patterns"]["misclassified_vs_correct"]["higher_than_correct"]
                )
            ),
            "hardest_condition_higher_than_overall_count": int(
                sum(
                    1
                    for seed_result in seed_results
                    if seed_result["uncertainty_patterns"]["hardest_condition"]["higher_than_overall"]
                )
            ),
            "hardest_damage_group_higher_than_overall_fault_count": int(
                sum(
                    1
                    for seed_result in seed_results
                    if seed_result["uncertainty_patterns"]["hardest_damage_group"]["higher_than_overall_fault"]
                )
            ),
        },
    }

    metrics_payload = {
        "study": "paderborn_resdilated_ae_mc_dropout_uncertainty",
        "model": RUN_CONFIG.name,
        "model_cli_name": RUN_CONFIG.cli_name,
        "checkpoint_source": "best.pt only",
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "device": str(device),
        "batch_size": int(batch_size),
        "mc_passes": int(args.mc_passes),
        "score_threshold_rule": SCORE_THRESHOLD_RULE,
        "uncertainty_threshold_rule": UNCERTAINTY_THRESHOLD_RULE,
        "uncertainty_percentile": float(args.uncertainty_percentile),
        "seeds": seed_results,
        "summary": summary,
    }
    report_text = build_report(
        device=device,
        batch_size=batch_size,
        mc_passes=args.mc_passes,
        uncertainty_percentile=args.uncertainty_percentile,
        seed_results=seed_results,
        summary=summary,
        metrics_path=metrics_path,
        report_path=report_path,
    )
    write_json(metrics_path, metrics_payload)
    write_text(report_path, report_text)
    print(f"\nSaved metrics to {metrics_path.as_posix()}")
    print(f"Saved report to {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
