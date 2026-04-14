from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

try:
    from train_shallow_baselines import evaluate_scores as evaluate_thresholded_scores
except ModuleNotFoundError:
    from scripts.train_shallow_baselines import evaluate_scores as evaluate_thresholded_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACTS_ROOT / "generative_upgrades" / "cwru_load_shift" / "resdilated_ae"
BASELINE_LOAD_SHIFT_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "cwru_load_shift_metrics.json"
DEFAULT_HELD_OUT_LOADS = (0, 1, 2, 3)
THRESHOLD_RULES = (
    "mean_plus_3std",
    "percentile_99",
    "percentile_99_5",
    "median_plus_3mad",
    "median_plus_4mad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an inference-free threshold calibration sweep for saved CWRU ResDilatedAE load-shift folds.",
    )
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-loads", type=int, nargs="+", default=list(DEFAULT_HELD_OUT_LOADS))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def build_seed_root(*, artifacts_root: Path, seed: int) -> Path:
    return artifacts_root / "generative_upgrades" / "cwru_load_shift" / "resdilated_ae" / f"seed_{seed}"


def build_fold_dir(*, seed_root: Path, held_out_load_hp: int) -> Path:
    return seed_root / f"load_{held_out_load_hp}"


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


def load_scores(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved score array: {path.as_posix()}")
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D score array at {path.as_posix()}, got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def validate_saved_mean_plus_3std(
    *,
    held_out_load_hp: int,
    fold_metrics_payload: dict[str, Any],
    recomputed_rule_result: dict[str, Any],
) -> None:
    saved_metrics = fold_metrics_payload["model"]["metrics"]
    saved_threshold = float(saved_metrics["threshold"])
    comparisons = {
        "threshold": (saved_threshold, float(recomputed_rule_result["metrics"]["threshold"])),
        "auroc": (float(saved_metrics["auroc"]), float(recomputed_rule_result["metrics"]["auroc"])),
        "auprc": (float(saved_metrics["auprc"]), float(recomputed_rule_result["metrics"]["auprc"])),
        "f1": (float(saved_metrics["f1"]), float(recomputed_rule_result["metrics"]["f1"])),
        "precision": (float(saved_metrics["precision"]), float(recomputed_rule_result["metrics"]["precision"])),
        "recall_fault": (
            float(saved_metrics["recall_fault"]),
            float(recomputed_rule_result["metrics"]["recall_fault"]),
        ),
        "false_alarm_rate": (
            float(saved_metrics["false_alarm_rate"]),
            float(recomputed_rule_result["metrics"]["false_alarm_rate"]),
        ),
    }
    mismatches: list[str] = []
    for metric_name, (saved_value, recomputed_value) in comparisons.items():
        if not math.isclose(saved_value, recomputed_value, rel_tol=1e-7, abs_tol=1e-8):
            mismatches.append(f"{metric_name}: saved={saved_value:.12f}, recomputed={recomputed_value:.12f}")
    if mismatches:
        raise RuntimeError(
            f"Saved score arrays do not match the recorded mean_plus_3std metrics for held-out load "
            f"{held_out_load_hp}: " + "; ".join(mismatches)
        )


def summarize_rule_metrics(fold_results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = ["threshold", "auroc", "auprc", "f1", "precision", "recall_fault", "false_alarm_rate"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for rule in THRESHOLD_RULES:
        summary[rule] = {}
        for metric_name in metric_names:
            values = [float(fold_result["rules"][rule]["metrics"][metric_name]) for fold_result in fold_results]
            summary[rule][metric_name] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
            }
    return summary


def choose_best_practical_rule(rule_summary: dict[str, dict[str, dict[str, float]]]) -> tuple[str, str]:
    baseline_f1 = float(rule_summary["mean_plus_3std"]["f1"]["mean"])
    baseline_recall = float(rule_summary["mean_plus_3std"]["recall_fault"]["mean"])
    eligible_rules = [
        rule
        for rule in THRESHOLD_RULES
        if float(rule_summary[rule]["f1"]["mean"]) >= (baseline_f1 - 0.01)
        and float(rule_summary[rule]["recall_fault"]["mean"]) >= (baseline_recall - 0.02)
    ]
    if not eligible_rules:
        eligible_rules = list(THRESHOLD_RULES)
    best_rule = min(
        eligible_rules,
        key=lambda rule: (
            float(rule_summary[rule]["false_alarm_rate"]["mean"]),
            -float(rule_summary[rule]["f1"]["mean"]),
            -float(rule_summary[rule]["precision"]["mean"]),
            -float(rule_summary[rule]["recall_fault"]["mean"]),
        ),
    )
    note = (
        "Selected as the lowest-mean-false-alarm rule among threshold rules that keep mean F1 within 0.01 "
        "and mean recall fault within 0.02 of the current mean_plus_3std baseline."
    )
    return best_rule, note


def choose_best_load0_far_rule(load0_result: dict[str, Any]) -> str:
    return min(
        THRESHOLD_RULES,
        key=lambda rule: (
            float(load0_result["rules"][rule]["metrics"]["false_alarm_rate"]),
            -float(load0_result["rules"][rule]["metrics"]["f1"]),
            -float(load0_result["rules"][rule]["metrics"]["precision"]),
        ),
    )


def load_baseline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved CWRU load-shift baseline metrics: {path.as_posix()}")
    payload = read_json(path)
    if "summary" not in payload:
        raise RuntimeError(f"Saved CWRU baseline metrics are missing the summary block: {path.as_posix()}")
    return payload


def build_fold_table(fold_result: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for rule in THRESHOLD_RULES:
        metrics = fold_result["rules"][rule]["metrics"]
        rows.append(
            [
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
        ["Threshold Rule", "Threshold", "AUROC", "AUPRC", "F1", "Precision", "Recall Fault", "False Alarm Rate"],
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


def build_baseline_comparison_table(
    *,
    best_rule: str,
    rule_summary: dict[str, dict[str, dict[str, float]]],
    baseline_summary: dict[str, Any],
) -> str:
    rows: list[list[Any]] = []
    metric_rows = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall_fault", "Recall Fault"),
        ("false_alarm_rate", "False Alarm Rate"),
    ]
    resdilated = rule_summary[best_rule]
    for baseline_model in ("AE", "OC-SVM", "Isolation Forest"):
        for metric_key, label in metric_rows:
            baseline_mean = float(baseline_summary["summary"][baseline_model][metric_key]["mean"])
            rows.append(
                [
                    baseline_model,
                    label,
                    f"{resdilated[metric_key]['mean']:.6f} +/- {resdilated[metric_key]['std']:.6f}",
                    f"{baseline_mean:.6f}",
                    f"{resdilated[metric_key]['mean'] - baseline_mean:+.6f}",
                ]
            )
    return format_markdown_table(
        ["Baseline", "Metric", "Best Calibrated ResDilatedAE", "Saved Baseline Mean", "Delta"],
        rows,
    )


def build_load0_note(
    *,
    load0_result: dict[str, Any] | None,
    best_rule: str,
    load0_best_far_rule: str | None,
) -> str:
    if load0_result is None or load0_best_far_rule is None:
        return "Held-out load 0 was not included in this sweep, so the false-alarm-transfer question could not be checked."

    base_metrics = load0_result["rules"]["mean_plus_3std"]["metrics"]
    best_far_metrics = load0_result["rules"][load0_best_far_rule]["metrics"]
    overall_best_metrics = load0_result["rules"][best_rule]["metrics"]
    clearly_improves = float(best_far_metrics["false_alarm_rate"]) <= (float(base_metrics["false_alarm_rate"]) - 0.25)
    verdict = "does" if clearly_improves else "does not"
    if float(best_far_metrics["false_alarm_rate"]) > 0.25:
        practical = (
            f"The best load-0 rule is `{load0_best_far_rule}`, which lowers FAR from "
            f"`{base_metrics['false_alarm_rate']:.6f}` to `{best_far_metrics['false_alarm_rate']:.6f}`, "
            f"but the residual FAR is still too high to call the issue mostly solved by threshold transfer alone."
        )
    else:
        practical = (
            f"The best load-0 rule is `{load0_best_far_rule}`, which lowers FAR from "
            f"`{base_metrics['false_alarm_rate']:.6f}` to `{best_far_metrics['false_alarm_rate']:.6f}` "
            f"while keeping F1 at `{best_far_metrics['f1']:.6f}`."
        )
    return (
        f"Held-out load 0 {verdict} show a clear threshold effect. Under `{best_rule}`, its FAR is "
        f"`{overall_best_metrics['false_alarm_rate']:.6f}` versus `{base_metrics['false_alarm_rate']:.6f}` for "
        f"`mean_plus_3std`. {practical}"
    )


def build_report(
    *,
    seed: int,
    seed_root: Path,
    fold_results: list[dict[str, Any]],
    rule_summary: dict[str, dict[str, dict[str, float]]],
    best_rule: str,
    best_rule_note: str,
    load0_note: str,
    baseline_summary: dict[str, Any],
    metrics_path: Path,
    report_path: Path,
) -> str:
    base_rule = "mean_plus_3std"
    base_far = float(rule_summary[base_rule]["false_alarm_rate"]["mean"])
    best_far = float(rule_summary[best_rule]["false_alarm_rate"]["mean"])
    base_f1 = float(rule_summary[base_rule]["f1"]["mean"])
    best_f1 = float(rule_summary[best_rule]["f1"]["mean"])
    base_precision = float(rule_summary[base_rule]["precision"]["mean"])
    best_precision = float(rule_summary[best_rule]["precision"]["mean"])
    lines = [
        "# CWRU ResDilatedAE Threshold Calibration Report",
        "",
        "## Protocol",
        "- Inference free from saved score arrays only.",
        "- No retraining, no new model inference, no preprocessing changes, and no raw-data edits.",
        f"- Seed evaluated: `{seed}`.",
        f"- Held-out loads evaluated: `{', '.join(str(item['held_out_load_hp']) for item in fold_results)}`.",
        "- Score splits reused per fold: `val_healthy`, `test_healthy`, `test_fault`.",
        "- Threshold rules compared: `mean_plus_3std`, `percentile_99`, `percentile_99_5`, `median_plus_3mad`, `median_plus_4mad`.",
        "- MAD uses the raw median absolute deviation over healthy validation scores.",
        f"- Saved run root: `{seed_root.as_posix()}`",
        "",
    ]

    for fold_result in fold_results:
        lines.extend(
            [
                f"## Held-Out Load {fold_result['held_out_load_hp']}",
                f"- Fold directory: `{fold_result['fold_dir']}`",
                f"- Saved scores: `{fold_result['score_paths']['val_healthy_scores']}`, "
                f"`{fold_result['score_paths']['test_healthy_scores']}`, "
                f"`{fold_result['score_paths']['test_fault_scores']}`",
                build_fold_table(fold_result),
                "",
            ]
        )

    lines.extend(
        [
            "## Mean/Std Across Held-Out Loads",
            build_summary_table(rule_summary),
            "",
            "## Practical Takeaway",
            f"- Best practical rule: `{best_rule}`",
            f"- Selection rule: {best_rule_note}",
            f"- Mean false alarm rate moves from `{base_far:.6f}` to `{best_far:.6f}` "
            f"(`{best_far - base_far:+.6f}`), mean F1 moves from `{base_f1:.6f}` to `{best_f1:.6f}` "
            f"(`{best_f1 - base_f1:+.6f}`), and mean precision moves from `{base_precision:.6f}` "
            f"to `{best_precision:.6f}` (`{best_precision - base_precision:+.6f}`).",
            f"- Load-0 check: {load0_note}",
            "",
            "## Comparison vs Earlier Saved CWRU Load-Shift References",
            "- Threshold values are not directly comparable across models because the score scales differ.",
            build_baseline_comparison_table(
                best_rule=best_rule,
                rule_summary=rule_summary,
                baseline_summary=baseline_summary,
            ),
            "",
            "## Saved Artifacts",
            f"- Metrics JSON: `{metrics_path.as_posix()}`",
            f"- Report: `{report_path.as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    artifacts_root = args.artifacts_root.resolve()
    held_out_loads = sorted(set(int(load) for load in args.held_out_loads))
    if not held_out_loads:
        raise RuntimeError("No held-out loads were requested.")

    seed_root = build_seed_root(artifacts_root=artifacts_root, seed=args.seed)
    if not seed_root.exists():
        raise FileNotFoundError(f"Missing saved CWRU ResDilatedAE run root: {seed_root.as_posix()}")

    baseline_summary = load_baseline_summary(BASELINE_LOAD_SHIFT_METRICS_PATH)

    fold_results: list[dict[str, Any]] = []
    for held_out_load_hp in held_out_loads:
        fold_dir = build_fold_dir(seed_root=seed_root, held_out_load_hp=held_out_load_hp)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing saved held-out load directory: {fold_dir.as_posix()}")

        val_scores_path = fold_dir / "val_healthy_scores.npy"
        test_healthy_scores_path = fold_dir / "test_healthy_scores.npy"
        test_fault_scores_path = fold_dir / "test_fault_scores.npy"
        fold_metrics_path = fold_dir / "fold_metrics.json"
        missing = [
            path.as_posix()
            for path in (val_scores_path, test_healthy_scores_path, test_fault_scores_path, fold_metrics_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing saved score artifacts for held-out load {held_out_load_hp}: " + ", ".join(missing)
            )

        val_scores = load_scores(val_scores_path)
        test_healthy_scores = load_scores(test_healthy_scores_path)
        test_fault_scores = load_scores(test_fault_scores_path)
        fold_metrics_payload = read_json(fold_metrics_path)

        rule_results = {
            rule: evaluate_rule(
                val_scores=val_scores,
                test_healthy_scores=test_healthy_scores,
                test_fault_scores=test_fault_scores,
                rule=rule,
            )
            for rule in THRESHOLD_RULES
        }
        validate_saved_mean_plus_3std(
            held_out_load_hp=held_out_load_hp,
            fold_metrics_payload=fold_metrics_payload,
            recomputed_rule_result=rule_results["mean_plus_3std"],
        )

        fold_results.append(
            {
                "held_out_load_hp": held_out_load_hp,
                "fold_dir": fold_dir.as_posix(),
                "counts": fold_metrics_payload["counts"],
                "score_paths": {
                    "val_healthy_scores": val_scores_path.as_posix(),
                    "test_healthy_scores": test_healthy_scores_path.as_posix(),
                    "test_fault_scores": test_fault_scores_path.as_posix(),
                },
                "score_counts": {
                    "val_healthy": int(val_scores.shape[0]),
                    "test_healthy": int(test_healthy_scores.shape[0]),
                    "test_fault": int(test_fault_scores.shape[0]),
                },
                "rules": rule_results,
            }
        )

    rule_summary = summarize_rule_metrics(fold_results)
    best_rule, best_rule_note = choose_best_practical_rule(rule_summary)
    load0_result = next((item for item in fold_results if int(item["held_out_load_hp"]) == 0), None)
    load0_best_far_rule = choose_best_load0_far_rule(load0_result) if load0_result is not None else None
    load0_note = build_load0_note(
        load0_result=load0_result,
        best_rule=best_rule,
        load0_best_far_rule=load0_best_far_rule,
    )

    metrics_path = seed_root / "cwru_resdilated_threshold_calibration_metrics.json"
    report_path = seed_root / "cwru_resdilated_threshold_calibration_report.md"
    metrics_payload = {
        "study": "cwru_resdilated_ae_threshold_calibration",
        "model": "ResDilatedAE",
        "model_cli_name": "resdilated_ae",
        "protocol": "leave_one_load_out",
        "checkpoint_source": "saved score arrays only",
        "seed": int(args.seed),
        "held_out_loads": held_out_loads,
        "threshold_rules": list(THRESHOLD_RULES),
        "folds": fold_results,
        "rule_summary": rule_summary,
        "best_practical_rule": {
            "rule": best_rule,
            "selection_note": best_rule_note,
            "metrics_mean_std": rule_summary[best_rule],
        },
        "load0_false_alarm_check": {
            "note": load0_note,
            "best_far_rule": load0_best_far_rule,
        },
        "saved_baseline_summary": baseline_summary["summary"],
        "comparison_vs_saved_baselines": {
            baseline_model: {
                metric_name: {
                    "resdilated_ae_mean": float(rule_summary[best_rule][metric_name]["mean"]),
                    "resdilated_ae_std": float(rule_summary[best_rule][metric_name]["std"]),
                    "baseline_mean": float(baseline_summary["summary"][baseline_model][metric_name]["mean"]),
                    "delta": float(
                        rule_summary[best_rule][metric_name]["mean"]
                        - baseline_summary["summary"][baseline_model][metric_name]["mean"]
                    ),
                }
                for metric_name in ("auroc", "auprc", "f1", "precision", "recall_fault", "false_alarm_rate")
            }
            for baseline_model in ("AE", "OC-SVM", "Isolation Forest")
        },
    }
    write_json(metrics_path, metrics_payload)
    write_text(
        report_path,
        build_report(
            seed=args.seed,
            seed_root=seed_root,
            fold_results=fold_results,
            rule_summary=rule_summary,
            best_rule=best_rule,
            best_rule_note=best_rule_note,
            load0_note=load0_note,
            baseline_summary=baseline_summary,
            metrics_path=metrics_path,
            report_path=report_path,
        ),
    )

    print("CWRU ResDilatedAE threshold calibration complete.")
    print(f"Best practical rule: {best_rule}")
    print(f"Metrics JSON: {metrics_path.as_posix()}")
    print(f"Report: {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
