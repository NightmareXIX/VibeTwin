from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

try:
    from train_ae_baseline import evaluate_scores as evaluate_binary_scores, require_torch, set_seed, torch
    from train_paderborn_baselines import ensure_required_files, load_label_array, read_json, resolve_paths
    from train_generative_upgrades import (
        ARTIFACTS_ROOT,
        BASELINE_IFOREST_METRICS_PATH,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_loader_bundle,
        build_models,
        build_run_paths,
        compute_reconstruction_scores,
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
        BASELINE_IFOREST_METRICS_PATH,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_loader_bundle,
        build_models,
        build_run_paths,
        compute_reconstruction_scores,
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
THRESHOLD_RULES = (
    "mean_plus_3std",
    "percentile_99",
    "percentile_99_5",
    "median_plus_3mad",
    "median_plus_4mad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference-only threshold calibration for saved Paderborn ResDilatedAE checkpoints.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size-cuda", type=int, default=256)
    parser.add_argument("--batch-size-cpu", type=int, default=128)
    return parser.parse_args()


def select_threshold_rule(scores: np.ndarray, rule: str) -> dict[str, Any]:
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
    threshold_meta = select_threshold_rule(val_scores, rule)
    metrics = evaluate_binary_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_errors=np.asarray(test_healthy_scores, dtype=np.float32),
        test_fault_errors=np.asarray(test_fault_scores, dtype=np.float32),
    )
    return {
        "threshold_meta": threshold_meta,
        "metrics": {
            **metrics,
            "threshold": float(threshold_meta["threshold"]),
        },
    }


def summarize_rule_metrics(seed_results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = ["threshold", "auroc", "f1", "precision", "recall_fault", "false_alarm_rate"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for rule in THRESHOLD_RULES:
        summary[rule] = {}
        for metric_name in metric_names:
            values = [float(seed_result["rules"][rule]["metrics"][metric_name]) for seed_result in seed_results]
            summary[rule][metric_name] = {
                "mean": float(mean(values)),
                "std": float(stdev(values)) if len(values) > 1 else 0.0,
            }
    return summary


def choose_best_practical_rule(rule_summary: dict[str, dict[str, dict[str, float]]]) -> tuple[str, str]:
    baseline_f1 = rule_summary["mean_plus_3std"]["f1"]["mean"]
    baseline_recall = rule_summary["mean_plus_3std"]["recall_fault"]["mean"]
    eligible_rules = [
        rule
        for rule in THRESHOLD_RULES
        if rule_summary[rule]["f1"]["mean"] >= (baseline_f1 - 0.015)
        and rule_summary[rule]["recall_fault"]["mean"] >= (baseline_recall - 0.02)
    ]
    if not eligible_rules:
        eligible_rules = list(THRESHOLD_RULES)
    best_rule = min(
        eligible_rules,
        key=lambda rule: (
            rule_summary[rule]["false_alarm_rate"]["mean"],
            -rule_summary[rule]["f1"]["mean"],
            -rule_summary[rule]["recall_fault"]["mean"],
            -rule_summary[rule]["precision"]["mean"],
        ),
    )
    note = (
        "Selected as the lowest-mean-false-alarm rule among threshold rules that keep mean F1 "
        "within 0.015 and mean recall fault within 0.02 of the current mean_plus_3std baseline."
    )
    return best_rule, note


def compare_with_saved_run(
    *,
    seed: int,
    run_paths: Any,
    recomputed_mean_plus_3std: dict[str, Any],
) -> None:
    if not run_paths.metrics_json.exists():
        raise FileNotFoundError(f"Saved run metrics are missing for seed {seed}: {run_paths.metrics_json.as_posix()}")
    saved_metrics_payload = read_json(run_paths.metrics_json)
    try:
        saved_model_payload = saved_metrics_payload["models"][RUN_CONFIG.name]
    except KeyError as exc:
        raise RuntimeError(
            f"Saved metrics for seed {seed} do not contain model key {RUN_CONFIG.name}: "
            f"{run_paths.metrics_json.as_posix()}"
        ) from exc

    comparisons = {
        "threshold": (
            float(saved_model_payload["threshold"]),
            float(recomputed_mean_plus_3std["metrics"]["threshold"]),
        ),
        "auroc": (
            float(saved_model_payload["metrics"]["auroc"]),
            float(recomputed_mean_plus_3std["metrics"]["auroc"]),
        ),
        "f1": (
            float(saved_model_payload["metrics"]["f1"]),
            float(recomputed_mean_plus_3std["metrics"]["f1"]),
        ),
        "precision": (
            float(saved_model_payload["metrics"]["precision"]),
            float(recomputed_mean_plus_3std["metrics"]["precision"]),
        ),
        "recall_fault": (
            float(saved_model_payload["metrics"]["recall_fault"]),
            float(recomputed_mean_plus_3std["metrics"]["recall_fault"]),
        ),
        "false_alarm_rate": (
            float(saved_model_payload["metrics"]["false_alarm_rate"]),
            float(recomputed_mean_plus_3std["metrics"]["false_alarm_rate"]),
        ),
    }
    mismatches: list[str] = []
    for metric_name, (saved_value, recomputed_value) in comparisons.items():
        if not math.isclose(saved_value, recomputed_value, rel_tol=1e-7, abs_tol=1e-8):
            mismatches.append(f"{metric_name}: saved={saved_value:.12f}, recomputed={recomputed_value:.12f}")
    if mismatches:
        raise RuntimeError(
            f"Checkpoint/inference mismatch for seed {seed} after recomputing mean_plus_3std from "
            f"{run_paths.best_checkpoint.as_posix()}: " + "; ".join(mismatches)
        )


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


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_seed_table(seed_result: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for rule in THRESHOLD_RULES:
        metrics = seed_result["rules"][rule]["metrics"]
        rows.append(
            [
                rule,
                f"{metrics['threshold']:.6f}",
                f"{metrics['auroc']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall_fault']:.6f}",
                f"{metrics['false_alarm_rate']:.6f}",
            ]
        )
    return format_markdown_table(
        ["Threshold Rule", "Threshold", "AUROC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
        rows,
    )


def build_summary_table(rule_summary: dict[str, dict[str, dict[str, float]]]) -> str:
    rows: list[list[Any]] = []
    for rule in THRESHOLD_RULES:
        rows.append(
            [
                rule,
                f"{rule_summary[rule]['threshold']['mean']:.6f} +/- {rule_summary[rule]['threshold']['std']:.6f}",
                f"{rule_summary[rule]['auroc']['mean']:.6f} +/- {rule_summary[rule]['auroc']['std']:.6f}",
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
            "F1 mean+/-std",
            "Precision mean+/-std",
            "Recall Fault mean+/-std",
            "False Alarm Rate mean+/-std",
        ],
        rows,
    )


def build_baseline_comparison_table(
    *,
    rule_summary: dict[str, dict[str, dict[str, float]]],
    best_rule: str,
    iforest_metrics: dict[str, Any],
) -> str:
    rows: list[list[Any]] = []
    for metric_name, label in (
        ("auroc", "AUROC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall_fault", "Recall Fault"),
        ("false_alarm_rate", "False Alarm Rate"),
    ):
        mean_value = rule_summary[best_rule][metric_name]["mean"]
        std_value = rule_summary[best_rule][metric_name]["std"]
        baseline_value = float(iforest_metrics[metric_name])
        rows.append(
            [
                label,
                f"{mean_value:.6f} +/- {std_value:.6f}",
                f"{baseline_value:.6f}",
                f"{mean_value - baseline_value:+.6f}",
            ]
        )
    return format_markdown_table(
        ["Metric", "Best Calibrated ResDilatedAE", "Isolation Forest", "Delta"],
        rows,
    )


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed_results: list[dict[str, Any]],
    rule_summary: dict[str, dict[str, dict[str, float]]],
    best_rule: str,
    best_rule_note: str,
    iforest_metrics: dict[str, Any],
    metrics_path: Path,
    report_path: Path,
) -> str:
    base_rule = "mean_plus_3std"
    base_far = rule_summary[base_rule]["false_alarm_rate"]["mean"]
    best_far = rule_summary[best_rule]["false_alarm_rate"]["mean"]
    base_f1 = rule_summary[base_rule]["f1"]["mean"]
    best_f1 = rule_summary[best_rule]["f1"]["mean"]
    base_recall = rule_summary[base_rule]["recall_fault"]["mean"]
    best_recall = rule_summary[best_rule]["recall_fault"]["mean"]
    lines = [
        "# Paderborn ResDilatedAE Threshold Calibration Report",
        "",
        "## Protocol",
        "- Inference only from saved `best.pt` checkpoints.",
        "- No retraining, no preprocessing changes, and no raw-data edits.",
        "- Seeds evaluated: `42`, `7`, `123`.",
        "- Splits scored: `val_healthy`, `test_healthy`, `test_fault`.",
        "- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.",
        "- MAD uses the raw median absolute deviation over healthy validation scores.",
        "",
        "## Dataset Defaults Reused",
        f"- Device used: `{device}`",
        f"- Effective batch size: `{batch_size}`",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Stride: `{preprocessing_config['stride']}`",
        "",
    ]

    for seed_result in seed_results:
        lines.extend(
            [
                f"## Seed {seed_result['seed']}",
                f"- Run directory: `{seed_result['run_dir']}`",
                f"- Checkpoint: `{seed_result['checkpoint_path']}`",
                f"- Saved scores: `{seed_result['score_paths']['val_healthy_scores']}`, "
                f"`{seed_result['score_paths']['test_healthy_scores']}`, "
                f"`{seed_result['score_paths']['test_fault_scores']}`",
                build_seed_table(seed_result),
                "",
            ]
        )

    lines.extend(
        [
            "## Mean/Std Across Seeds",
            build_summary_table(rule_summary),
            "",
            "## Practical Tradeoff",
            f"- Best practical rule: `{best_rule}`",
            f"- Selection rule: {best_rule_note}",
            f"- Mean false alarm rate moves from `{base_far:.6f}` to `{best_far:.6f}` "
            f"(`{best_far - base_far:+.6f}`), mean F1 moves from `{base_f1:.6f}` to `{best_f1:.6f}` "
            f"(`{best_f1 - base_f1:+.6f}`), and mean recall fault moves from `{base_recall:.6f}` "
            f"to `{best_recall:.6f}` (`{best_recall - base_recall:+.6f}`).",
            "",
            "## Comparison vs Isolation Forest Baseline",
            "- Threshold values are not directly comparable across models because the score scales differ.",
            build_baseline_comparison_table(rule_summary=rule_summary, best_rule=best_rule, iforest_metrics=iforest_metrics),
            "",
            "## Saved Artifacts",
            f"- Metrics JSON: `{metrics_path.as_posix()}`",
            f"- Report: `{report_path.as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


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
    if not BASELINE_IFOREST_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved Paderborn Isolation Forest baseline: {BASELINE_IFOREST_METRICS_PATH.as_posix()}"
        )

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    expected_width = int(preprocessing_config["window_size"])
    load_label_array(array_paths.fault_labels)

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

    metrics_path = (
        artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_threshold_calibration_metrics.json"
    )
    report_path = (
        artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_threshold_calibration_report.md"
    )

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=RUN_CONFIG, seed=seed)
        print(f"\nRunning inference-only score extraction for seed {seed}")
        model, checkpoint_payload = load_resdilatedae_from_checkpoint(
            checkpoint_path=run_paths.best_checkpoint,
            expected_seed=seed,
            expected_width=expected_width,
            device=device,
        )
        val_scores = compute_reconstruction_scores(
            model=model,
            loader=loaders["val"],
            device=device,
            model_kind=RUN_CONFIG.model_kind,
        )
        test_healthy_scores = compute_reconstruction_scores(
            model=model,
            loader=loaders["test_healthy"],
            device=device,
            model_kind=RUN_CONFIG.model_kind,
        )
        test_fault_scores = compute_reconstruction_scores(
            model=model,
            loader=loaders["test_fault"],
            device=device,
            model_kind=RUN_CONFIG.model_kind,
        )
        save_numpy_array(run_paths.val_healthy_scores_npy, val_scores)
        save_numpy_array(run_paths.test_healthy_scores_npy, test_healthy_scores)
        save_numpy_array(run_paths.test_fault_scores_npy, test_fault_scores)

        rule_results = {
            rule: evaluate_rule(
                val_scores=val_scores,
                test_healthy_scores=test_healthy_scores,
                test_fault_scores=test_fault_scores,
                rule=rule,
            )
            for rule in THRESHOLD_RULES
        }
        compare_with_saved_run(
            seed=seed,
            run_paths=run_paths,
            recomputed_mean_plus_3std=rule_results["mean_plus_3std"],
        )
        seed_results.append(
            {
                "seed": int(seed),
                "run_dir": run_paths.run_dir.as_posix(),
                "checkpoint_path": run_paths.best_checkpoint.as_posix(),
                "checkpoint_training_settings": checkpoint_payload.get("training_settings", {}),
                "score_paths": {
                    "val_healthy_scores": run_paths.val_healthy_scores_npy.as_posix(),
                    "test_healthy_scores": run_paths.test_healthy_scores_npy.as_posix(),
                    "test_fault_scores": run_paths.test_fault_scores_npy.as_posix(),
                },
                "score_counts": {
                    "val_healthy": int(val_scores.shape[0]),
                    "test_healthy": int(test_healthy_scores.shape[0]),
                    "test_fault": int(test_fault_scores.shape[0]),
                },
                "rules": rule_results,
            }
        )

    rule_summary = summarize_rule_metrics(seed_results)
    best_rule, best_rule_note = choose_best_practical_rule(rule_summary)
    iforest_metrics = read_json(BASELINE_IFOREST_METRICS_PATH)

    metrics_payload = {
        "study": "paderborn_resdilated_ae_threshold_calibration",
        "model": RUN_CONFIG.name,
        "model_cli_name": RUN_CONFIG.cli_name,
        "checkpoint_source": "best.pt only",
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "device": str(device),
        "batch_size": int(batch_size),
        "seeds": seed_results,
        "threshold_rules": list(THRESHOLD_RULES),
        "rule_summary": rule_summary,
        "best_practical_rule": {
            "rule": best_rule,
            "selection_note": best_rule_note,
            "metrics_mean_std": rule_summary[best_rule],
        },
        "isolation_forest_baseline": iforest_metrics,
        "comparison_vs_isolation_forest": {
            metric_name: {
                "resdilated_ae_mean": float(rule_summary[best_rule][metric_name]["mean"]),
                "resdilated_ae_std": float(rule_summary[best_rule][metric_name]["std"]),
                "isolation_forest": float(iforest_metrics[metric_name]),
                "delta": float(rule_summary[best_rule][metric_name]["mean"] - float(iforest_metrics[metric_name])),
            }
            for metric_name in ("auroc", "f1", "precision", "recall_fault", "false_alarm_rate")
        },
    }
    write_json(metrics_path, metrics_payload)
    write_text(
        report_path,
        build_report(
            preprocessing_config=preprocessing_config,
            batch_size=batch_size,
            device=device,
            seed_results=seed_results,
            rule_summary=rule_summary,
            best_rule=best_rule,
            best_rule_note=best_rule_note,
            iforest_metrics=iforest_metrics,
            metrics_path=metrics_path,
            report_path=report_path,
        ),
    )

    print("\nThreshold calibration complete.")
    print(f"Best practical rule: {best_rule}")
    print(f"Metrics JSON: {metrics_path.as_posix()}")
    print(f"Report: {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
