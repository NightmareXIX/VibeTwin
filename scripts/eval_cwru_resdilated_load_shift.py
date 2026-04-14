from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from train_ae_baseline import (
        evaluate_scores as evaluate_binary_scores,
        make_loader,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from train_generative_upgrades import (
        ARTIFACTS_ROOT,
        ModelRunConfig,
        build_models,
        compute_reconstruction_scores,
        ensure_run_is_ready,
        get_device,
        load_torch_payload,
        save_numpy_array,
        select_models,
        train_single_model,
        write_json,
        write_text,
    )
    from eval_cwru_load_shift import (
        build_fold_dataset,
        build_indexed_windows,
        filter_manifest_rows,
        format_markdown_table,
        load_label_array,
        load_manifest_rows,
        load_window_array,
        read_json,
        recover_raw_windows,
        resolve_paths,
    )
except ModuleNotFoundError:
    from scripts.train_ae_baseline import (
        evaluate_scores as evaluate_binary_scores,
        make_loader,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from scripts.train_generative_upgrades import (
        ARTIFACTS_ROOT,
        ModelRunConfig,
        build_models,
        compute_reconstruction_scores,
        ensure_run_is_ready,
        get_device,
        load_torch_payload,
        save_numpy_array,
        select_models,
        train_single_model,
        write_json,
        write_text,
    )
    from scripts.eval_cwru_load_shift import (
        build_fold_dataset,
        build_indexed_windows,
        filter_manifest_rows,
        format_markdown_table,
        load_label_array,
        load_manifest_rows,
        load_window_array,
        read_json,
        recover_raw_windows,
        resolve_paths,
    )


RUN_CONFIG = ModelRunConfig(
    name="ResDilatedAE",
    cli_name="resdilated_ae",
    output_stem="resdilated_ae",
    model_kind="ae",
)
DEFAULT_HELD_OUT_LOADS = (0, 1, 2, 3)
BASELINE_LOAD_SHIFT_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "cwru_load_shift_metrics.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"


@dataclass(frozen=True)
class FoldRunPaths:
    run_dir: Path
    checkpoints_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    val_healthy_scores_npy: Path
    test_healthy_scores_npy: Path
    test_fault_scores_npy: Path
    history_json: Path
    status_json: Path
    metrics_json: Path
    report_md: Path
    plot_png: Path
    train_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable leave-one-load-out CWRU evaluation for ResDilatedAE under the harder load-shift protocol.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--held-out-loads", type=int, nargs="+", default=list(DEFAULT_HELD_OUT_LOADS))
    parser.add_argument("--batch-size-cuda", type=int, default=128)
    parser.add_argument("--batch-size-cpu", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--freq-loss-weight", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-rule", choices=("mean_plus_3std",), default="mean_plus_3std")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted fold or skip already-completed folds.")
    args = parser.parse_args()
    if args.save_every_epochs <= 0:
        parser.error("--save-every-epochs must be positive.")
    return args


def build_fold_run_paths(*, artifacts_root: Path, held_out_load_hp: int, seed: int) -> FoldRunPaths:
    run_dir = (
        artifacts_root
        / "generative_upgrades"
        / "cwru_load_shift"
        / RUN_CONFIG.output_stem
        / f"seed_{seed}"
        / f"load_{held_out_load_hp}"
    )
    checkpoints_dir = run_dir / "checkpoints"
    return FoldRunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        latest_checkpoint=checkpoints_dir / "latest.pt",
        best_checkpoint=checkpoints_dir / "best.pt",
        val_healthy_scores_npy=run_dir / "val_healthy_scores.npy",
        test_healthy_scores_npy=run_dir / "test_healthy_scores.npy",
        test_fault_scores_npy=run_dir / "test_fault_scores.npy",
        history_json=run_dir / "history.json",
        status_json=run_dir / "status.json",
        metrics_json=run_dir / "fold_metrics.json",
        report_md=run_dir / "fold_report.md",
        plot_png=run_dir / "fold_summary.png",
        train_log=run_dir / "train.log",
    )


def build_root_output_dir(*, artifacts_root: Path, seed: int) -> Path:
    return artifacts_root / "generative_upgrades" / "cwru_load_shift" / RUN_CONFIG.output_stem / f"seed_{seed}"


def build_manual_command(args: argparse.Namespace, held_out_load_hp: int, *, resume: bool = False) -> str:
    command = [
        r".\.venv-cuda\Scripts\python.exe",
        "scripts/eval_cwru_resdilated_load_shift.py",
        "--held-out-loads",
        str(held_out_load_hp),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--freq-loss-weight",
        str(args.freq_loss_weight),
        "--dropout",
        str(args.dropout),
        "--save-every-epochs",
        str(args.save_every_epochs),
        "--threshold-rule",
        args.threshold_rule,
    ]
    if resume:
        command.append("--resume")
    return " ".join(command)


def print_manual_commands(args: argparse.Namespace, *, artifacts_root: Path) -> None:
    print("\nManual GPU commands", flush=True)
    for held_out_load_hp in sorted(set(int(load) for load in args.held_out_loads)):
        run_paths = build_fold_run_paths(artifacts_root=artifacts_root, held_out_load_hp=held_out_load_hp, seed=args.seed)
        print(f"  Held-out load {held_out_load_hp}:", flush=True)
        print(f"    train : {build_manual_command(args, held_out_load_hp)}", flush=True)
        print(f"    resume: {build_manual_command(args, held_out_load_hp, resume=True)}", flush=True)
        print(f"    output: {run_paths.run_dir.as_posix()}", flush=True)


def make_json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: make_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_ready(item) for item in value]
    return value


def load_saved_baselines(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved CWRU load-shift baseline metrics: {path.as_posix()}")
    return read_json(path)


def choose_model(window_size: int, dropout: float) -> tuple[ModelRunConfig, torch.nn.Module]:
    selected_models = select_models(build_models(window_size, dropout), RUN_CONFIG.cli_name)
    if len(selected_models) != 1:
        raise RuntimeError(f"Expected exactly one {RUN_CONFIG.cli_name} model definition, found {len(selected_models)}.")
    return selected_models[0]


def load_best_model_for_evaluation(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    best_payload = load_torch_payload(checkpoint_path)
    model.load_state_dict(best_payload["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def save_fold_status(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def fold_is_completed(run_paths: FoldRunPaths) -> bool:
    return run_paths.metrics_json.exists()


def ensure_fold_is_ready(run_paths: FoldRunPaths, *, resume: bool) -> str:
    if resume and fold_is_completed(run_paths):
        return "skip_completed"
    ensure_run_is_ready(run_paths, resume=resume)
    return "resume" if resume else "fresh"


def train_and_evaluate_fold(
    *,
    fold: Any,
    held_out_load_hp: int,
    run_paths: FoldRunPaths,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    preprocessing_config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    run_config, model = choose_model(window_size=int(preprocessing_config["window_size"]), dropout=float(args.dropout))
    loaders = {
        "train": make_loader(fold.train_healthy, batch_size=batch_size, shuffle=True),
        "val": make_loader(fold.val_healthy, batch_size=batch_size, shuffle=False),
        "test_healthy": make_loader(fold.test_healthy, batch_size=batch_size, shuffle=False),
        "test_fault": make_loader(fold.test_fault, batch_size=batch_size, shuffle=False),
    }

    training_settings = {
        "dataset": "cwru",
        "protocol": "leave_one_load_out",
        "held_out_load_hp": int(held_out_load_hp),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "freq_loss_weight": float(args.freq_loss_weight),
        "vae_beta_max": 0.0,
        "vae_kl_warmup_epochs": 0,
        "save_every_epochs": int(args.save_every_epochs),
        "batch_size": int(batch_size),
        "threshold_rule": args.threshold_rule,
        "dropout": float(args.dropout),
        "normalization": {
            "method": "zscore_global",
            "fit_on": "healthy_train_only_per_fold",
        },
    }

    training_summary = train_single_model(
        run_config=run_config,
        model=model,
        loaders=loaders,
        device=device,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        patience=int(args.patience),
        freq_loss_weight=float(args.freq_loss_weight),
        vae_beta_max=0.0,
        vae_warmup_epochs=0,
        run_paths=run_paths,
        seed=int(args.seed),
        save_every_epochs=int(args.save_every_epochs),
        resume=resume,
        training_settings=training_settings,
    )

    if training_summary["status"] != "completed":
        save_fold_status(
            run_paths.status_json,
            {
                "dataset": "cwru",
                "protocol": "leave_one_load_out",
                "held_out_load_hp": int(held_out_load_hp),
                "model_name": run_config.name,
                "model_cli_name": run_config.cli_name,
                "status": training_summary["status"],
                "stage": "training",
                "message": "Training was interrupted before evaluation completed.",
                "run_dir": run_paths.run_dir.as_posix(),
                "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
                "best_checkpoint": run_paths.best_checkpoint.as_posix(),
                "training_summary": training_summary,
            },
        )
        return {
            "status": training_summary["status"],
            "held_out_load_hp": int(held_out_load_hp),
            "run_dir": run_paths.run_dir.as_posix(),
        }

    model = load_best_model_for_evaluation(model=model, checkpoint_path=run_paths.best_checkpoint, device=device)
    val_scores = compute_reconstruction_scores(model=model, loader=loaders["val"], device=device, model_kind=run_config.model_kind)
    test_healthy_scores = compute_reconstruction_scores(
        model=model,
        loader=loaders["test_healthy"],
        device=device,
        model_kind=run_config.model_kind,
    )
    test_fault_scores = compute_reconstruction_scores(
        model=model,
        loader=loaders["test_fault"],
        device=device,
        model_kind=run_config.model_kind,
    )
    save_numpy_array(run_paths.val_healthy_scores_npy, val_scores)
    save_numpy_array(run_paths.test_healthy_scores_npy, test_healthy_scores)
    save_numpy_array(run_paths.test_fault_scores_npy, test_fault_scores)

    threshold_meta = select_threshold(val_scores, args.threshold_rule)
    metrics = evaluate_binary_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_errors=test_healthy_scores,
        test_fault_errors=test_fault_scores,
    )
    metrics["threshold"] = float(threshold_meta["threshold"])

    fold_payload = {
        "dataset": "cwru",
        "protocol": "leave_one_load_out",
        "held_out_load_hp": int(held_out_load_hp),
        "counts": fold.counts,
        "fold_normalization": {
            "mean": float(fold.fold_mean),
            "std": float(fold.fold_std),
        },
        "model": {
            "model_name": run_config.name,
            "model_cli_name": run_config.cli_name,
            "model_kind": run_config.model_kind,
            "denoising": bool(run_config.denoising),
            "threshold_rule": args.threshold_rule,
            "training_settings": training_settings,
            "training": {
                "epochs_ran": len(training_summary["history"]["train_total_loss"]),
                "best_epoch": int(training_summary["best_epoch"]),
                "best_val_total_loss": float(training_summary["best_val_total_loss"]),
                "elapsed_seconds": float(training_summary["elapsed_seconds"]),
                "history": training_summary["history"],
            },
            "metrics": metrics,
            "threshold_meta": threshold_meta,
        },
        "artifacts": {
            "run_dir": run_paths.run_dir.as_posix(),
            "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "val_healthy_scores_npy": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores_npy": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores_npy": run_paths.test_fault_scores_npy.as_posix(),
            "history_json": run_paths.history_json.as_posix(),
            "status_json": run_paths.status_json.as_posix(),
            "metrics_json": run_paths.metrics_json.as_posix(),
            "train_log": run_paths.train_log.as_posix(),
        },
    }
    write_json(run_paths.metrics_json, make_json_ready(fold_payload))
    fold_report = "\n".join(
        [
            f"# CWRU ResDilatedAE Fold Report: Held-Out Load {held_out_load_hp}",
            "",
            f"- Run directory: `{run_paths.run_dir.as_posix()}`",
            f"- Threshold: `{metrics['threshold']:.6f}`",
            f"- AUROC: `{metrics['auroc']:.6f}`",
            f"- AUPRC: `{metrics['auprc']:.6f}`",
            f"- F1: `{metrics['f1']:.6f}`",
            f"- Precision: `{metrics['precision']:.6f}`",
            f"- Recall fault: `{metrics['recall_fault']:.6f}`",
            f"- False alarm rate: `{metrics['false_alarm_rate']:.6f}`",
            "",
        ]
    )
    write_text(run_paths.report_md, fold_report)
    save_fold_status(
        run_paths.status_json,
        {
            "dataset": "cwru",
            "protocol": "leave_one_load_out",
            "held_out_load_hp": int(held_out_load_hp),
            "model_name": run_config.name,
            "model_cli_name": run_config.cli_name,
            "status": "completed",
            "stage": "evaluation",
            "message": "Fold evaluation completed.",
            "run_dir": run_paths.run_dir.as_posix(),
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "metrics_json": run_paths.metrics_json.as_posix(),
        },
    )
    return fold_payload


def load_fold_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved fold metrics for resume skip: {path.as_posix()}")
    return read_json(path)


def summarize_resdilated_metrics(fold_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = ["auroc", "auprc", "f1", "precision", "recall_fault", "false_alarm_rate"]
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(fold["model"]["metrics"][metric_name]) for fold in fold_results]
        summary[metric_name] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
        }
    return summary


def build_fold_table(fold_results: list[dict[str, Any]]) -> str:
    rows: list[list[Any]] = []
    for fold in fold_results:
        metrics = fold["model"]["metrics"]
        rows.append(
            [
                fold["held_out_load_hp"],
                fold["counts"]["train_healthy"],
                fold["counts"]["val_healthy"],
                fold["counts"]["test_healthy"],
                fold["counts"]["test_fault"],
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
        [
            "Held-Out Load",
            "Train H",
            "Val H",
            "Test H",
            "Test F",
            "Threshold",
            "AUROC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall Fault",
            "False Alarm Rate",
        ],
        rows,
    )


def build_summary_table(
    resdilated_summary: dict[str, dict[str, float]],
    baseline_summary: dict[str, dict[str, dict[str, float]]],
) -> str:
    rows = [
        [
            "ResDilatedAE",
            f"{resdilated_summary['auroc']['mean']:.6f} +/- {resdilated_summary['auroc']['std']:.6f}",
            f"{resdilated_summary['auprc']['mean']:.6f} +/- {resdilated_summary['auprc']['std']:.6f}",
            f"{resdilated_summary['f1']['mean']:.6f} +/- {resdilated_summary['f1']['std']:.6f}",
            f"{resdilated_summary['precision']['mean']:.6f} +/- {resdilated_summary['precision']['std']:.6f}",
            f"{resdilated_summary['recall_fault']['mean']:.6f} +/- {resdilated_summary['recall_fault']['std']:.6f}",
            f"{resdilated_summary['false_alarm_rate']['mean']:.6f} +/- {resdilated_summary['false_alarm_rate']['std']:.6f}",
        ]
    ]
    for baseline_name in ["AE", "OC-SVM", "Isolation Forest"]:
        if baseline_name not in baseline_summary:
            continue
        rows.append(
            [
                baseline_name,
                f"{baseline_summary[baseline_name]['auroc']['mean']:.6f} +/- {baseline_summary[baseline_name]['auroc']['std']:.6f}",
                f"{baseline_summary[baseline_name]['auprc']['mean']:.6f} +/- {baseline_summary[baseline_name]['auprc']['std']:.6f}",
                f"{baseline_summary[baseline_name]['f1']['mean']:.6f} +/- {baseline_summary[baseline_name]['f1']['std']:.6f}",
                f"{baseline_summary[baseline_name]['precision']['mean']:.6f} +/- {baseline_summary[baseline_name]['precision']['std']:.6f}",
                f"{baseline_summary[baseline_name]['recall_fault']['mean']:.6f} +/- {baseline_summary[baseline_name]['recall_fault']['std']:.6f}",
                f"{baseline_summary[baseline_name]['false_alarm_rate']['mean']:.6f} +/- {baseline_summary[baseline_name]['false_alarm_rate']['std']:.6f}",
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


def build_comparison_note(
    *,
    resdilated_summary: dict[str, dict[str, float]],
    baseline_summary: dict[str, dict[str, dict[str, float]]],
) -> str:
    notes: list[str] = []
    ae_summary = baseline_summary.get("AE")
    if ae_summary is not None:
        notes.append(
            "Against the earlier CompactAE load-shift result, ResDilatedAE "
            f"{'improves' if resdilated_summary['f1']['mean'] >= ae_summary['f1']['mean'] else 'does not improve'} "
            f"mean F1 ({resdilated_summary['f1']['mean']:.3f} vs {ae_summary['f1']['mean']:.3f}) and "
            f"{'reduces' if resdilated_summary['false_alarm_rate']['mean'] <= ae_summary['false_alarm_rate']['mean'] else 'increases'} "
            f"mean false alarm rate ({resdilated_summary['false_alarm_rate']['mean']:.3f} vs {ae_summary['false_alarm_rate']['mean']:.3f})."
        )
    best_baseline_name = None
    best_baseline_f1 = None
    for baseline_name in ("AE", "OC-SVM", "Isolation Forest"):
        if baseline_name not in baseline_summary:
            continue
        baseline_f1 = float(baseline_summary[baseline_name]["f1"]["mean"])
        if best_baseline_f1 is None or baseline_f1 > best_baseline_f1:
            best_baseline_f1 = baseline_f1
            best_baseline_name = baseline_name
    if best_baseline_name is not None and best_baseline_f1 is not None:
        notes.append(
            f"The strongest earlier shallow/baseline reference on mean F1 was `{best_baseline_name}` at `{best_baseline_f1:.3f}`; "
            f"ResDilatedAE lands at `{resdilated_summary['f1']['mean']:.3f}` under the same leakage-safe load-shift protocol."
        )
    return " ".join(notes)


def plot_summary(
    *,
    fold_results: list[dict[str, Any]],
    baseline_summary: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    metrics_to_plot = ["auroc", "f1", "false_alarm_rate"]
    titles = {
        "auroc": "AUROC by Held-Out Load",
        "f1": "F1 by Held-Out Load",
        "false_alarm_rate": "False Alarm Rate by Held-Out Load",
    }
    loads = [int(fold["held_out_load_hp"]) for fold in fold_results]
    x_positions = np.arange(len(loads), dtype=np.float64)
    values_by_metric = {
        metric_name: [float(fold["model"]["metrics"][metric_name]) for fold in fold_results]
        for metric_name in metrics_to_plot
    }

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=160)
    for axis, metric_name in zip(axes, metrics_to_plot, strict=True):
        axis.bar(x_positions, values_by_metric[metric_name], color="#1f4e79", alpha=0.9)
        axis.set_title(titles[metric_name])
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"Load {load}" for load in loads])
        axis.grid(axis="y", alpha=0.3)
        if metric_name != "false_alarm_rate":
            axis.set_ylim(0.0, 1.05)
        for baseline_name, color in (("AE", "#999999"), ("OC-SVM", "#6aa84f"), ("Isolation Forest", "#cc7a00")):
            if baseline_name not in baseline_summary:
                continue
            baseline_mean = float(baseline_summary[baseline_name][metric_name]["mean"])
            axis.axhline(baseline_mean, color=color, linestyle="--", linewidth=1.0, label=f"{baseline_name} mean")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    figure.suptitle("CWRU ResDilatedAE Leave-One-Load-Out Summary", fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_path)
    plt.close(figure)


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    fold_results: list[dict[str, Any]],
    resdilated_summary: dict[str, dict[str, float]],
    baseline_payload: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> str:
    baseline_summary = baseline_payload.get("summary", {})
    comparison_note = build_comparison_note(
        resdilated_summary=resdilated_summary,
        baseline_summary=baseline_summary,
    )
    lines = [
        "# CWRU ResDilatedAE Load-Shift Report",
        "",
        "## Protocol",
        "- Leave-one-load-out evaluation across the four motor loads: `0`, `1`, `2`, and `3`.",
        "- Healthy train windows come only from non-held-out loads using the existing `train` split.",
        "- Healthy validation windows come only from non-held-out loads using the existing `val` split.",
        "- Test healthy windows come from the held-out load using the existing `test` split.",
        "- Test fault windows come from the held-out load using the existing fault-test windows.",
        "- Fold-specific z-score normalization is refit on healthy train windows only after reconstructing pre-z-score values from the saved preprocessing stats.",
        "- This run keeps the harder CWRU load-shift protocol intact and adds only the final chosen generative model family.",
        "",
        "## ResDilatedAE Settings",
        f"- Device used: `{device}`",
        f"- Effective batch size: `{batch_size}`",
        f"- Epoch budget: `{args.epochs}` with patience `{args.patience}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- Weight decay: `{args.weight_decay}`",
        f"- Dropout: `{args.dropout}`",
        f"- Frequency-loss weight: `{args.freq_loss_weight}`",
        f"- Threshold rule: `{args.threshold_rule}`",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Window stride: `{preprocessing_config['stride']}`",
        "",
        "## Per-Load Results",
        build_fold_table(fold_results),
        "",
        "## Mean/Std Comparison",
        build_summary_table(resdilated_summary, baseline_summary),
        "",
        "## Practical Comparison",
        f"- {comparison_note}",
        "",
        "## Manual Commands",
        f"- All loads: `{build_manual_command(args, 0).replace('--held-out-loads 0', '--held-out-loads 0 1 2 3')}`",
        f"- Resume all: `{build_manual_command(args, 0, resume=True).replace('--held-out-loads 0', '--held-out-loads 0 1 2 3')}`",
        "",
        "## Saved Artifacts",
        f"- Metrics JSON: `{artifact_paths['metrics'].as_posix()}`",
        f"- Report: `{artifact_paths['report'].as_posix()}`",
        f"- Summary plot: `{artifact_paths['plot'].as_posix()}`",
        f"- Root run directory: `{artifact_paths['root'].as_posix()}`",
        "",
    ]
    return "\n".join(lines)


def save_progress(
    *,
    path: Path,
    held_out_loads: list[int],
    completed_loads: list[int],
    pending_loads: list[int],
    status: str,
    last_message: str,
) -> None:
    write_json(
        path,
        {
            "dataset": "cwru",
            "protocol": "leave_one_load_out",
            "model_name": RUN_CONFIG.name,
            "held_out_loads": [int(load) for load in held_out_loads],
            "completed_loads": [int(load) for load in completed_loads],
            "pending_loads": [int(load) for load in pending_loads],
            "status": status,
            "message": last_message,
        },
    )


def main() -> int:
    require_torch()
    args = parse_args()
    set_seed(int(args.seed))

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    array_paths = resolve_paths(processed_root)

    required = [
        array_paths.train_healthy,
        array_paths.val_healthy,
        array_paths.test_healthy,
        array_paths.test_fault,
        array_paths.fault_labels,
        metadata_root / "normalization_stats.json",
        metadata_root / "preprocessing_config.json",
        metadata_root / "fault_label_map.json",
        metadata_root / "window_manifest.csv",
        BASELINE_LOAD_SHIFT_METRICS_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required CWRU inputs are missing: " + ", ".join(path.as_posix() for path in missing))

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    normalization_stats = read_json(metadata_root / "normalization_stats.json")
    baseline_payload = load_saved_baselines(BASELINE_LOAD_SHIFT_METRICS_PATH)
    expected_width = int(preprocessing_config["window_size"])

    train_healthy = load_window_array(array_paths.train_healthy, expected_width)
    val_healthy = load_window_array(array_paths.val_healthy, expected_width)
    test_healthy = load_window_array(array_paths.test_healthy, expected_width)
    test_fault = load_window_array(array_paths.test_fault, expected_width)
    fault_labels = load_label_array(array_paths.fault_labels)
    manifest_rows = load_manifest_rows(metadata_root / "window_manifest.csv")

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
            f"fault_labels length does not match test_fault rows: {fault_labels.shape[0]} != {indexed_test_fault.windows.shape[0]}"
        )

    original_mean = float(normalization_stats["mean"])
    original_std = float(normalization_stats["std"])
    raw_train = type(indexed_train)(
        windows=recover_raw_windows(indexed_train.windows, original_mean, original_std),
        load_hp=indexed_train.load_hp,
        classes=indexed_train.classes,
        source_ids=indexed_train.source_ids,
    )
    raw_val = type(indexed_val)(
        windows=recover_raw_windows(indexed_val.windows, original_mean, original_std),
        load_hp=indexed_val.load_hp,
        classes=indexed_val.classes,
        source_ids=indexed_val.source_ids,
    )
    raw_test_healthy = type(indexed_test_healthy)(
        windows=recover_raw_windows(indexed_test_healthy.windows, original_mean, original_std),
        load_hp=indexed_test_healthy.load_hp,
        classes=indexed_test_healthy.classes,
        source_ids=indexed_test_healthy.source_ids,
    )
    raw_test_fault = type(indexed_test_fault)(
        windows=recover_raw_windows(indexed_test_fault.windows, original_mean, original_std),
        load_hp=indexed_test_fault.load_hp,
        classes=indexed_test_fault.classes,
        source_ids=indexed_test_fault.source_ids,
    )

    available_loads = sorted(
        set(int(load) for load in raw_test_healthy.load_hp.tolist())
        & set(int(load) for load in raw_test_fault.load_hp.tolist())
    )
    requested_loads = sorted(set(int(load) for load in args.held_out_loads))
    missing_loads = [load for load in requested_loads if load not in available_loads]
    if missing_loads:
        raise RuntimeError(
            f"Requested held-out loads are not available in the saved CWRU protocol: {missing_loads}; available={available_loads}"
        )

    device = get_device()
    batch_size = args.batch_size_cuda if device.type == "cuda" else args.batch_size_cpu
    root_output_dir = build_root_output_dir(artifacts_root=artifacts_root, seed=args.seed)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print_manual_commands(args, artifacts_root=artifacts_root)
    print(f"\nDevice selected: {device}")
    print(f"Effective batch size: {batch_size}")

    artifact_paths = {
        "root": root_output_dir,
        "metrics": root_output_dir / "cwru_resdilated_load_shift_metrics.json",
        "report": root_output_dir / "cwru_resdilated_load_shift_report.md",
        "plot": root_output_dir / "cwru_resdilated_load_shift_summary.png",
        "progress": root_output_dir / "cwru_resdilated_load_shift_progress.json",
    }

    fold_results: list[dict[str, Any]] = []
    completed_loads: list[int] = []
    for held_out_load_hp in requested_loads:
        run_paths = build_fold_run_paths(artifacts_root=artifacts_root, held_out_load_hp=held_out_load_hp, seed=args.seed)
        readiness = ensure_fold_is_ready(run_paths, resume=args.resume)

        if readiness == "skip_completed":
            print(f"\nSkipping held-out load {held_out_load_hp} because a completed fold result already exists.")
            fold_results.append(load_fold_result(run_paths.metrics_json))
            completed_loads.append(int(held_out_load_hp))
            save_progress(
                path=artifact_paths["progress"],
                held_out_loads=requested_loads,
                completed_loads=completed_loads,
                pending_loads=[load for load in requested_loads if load not in completed_loads],
                status="running",
                last_message=f"Skipped completed fold load {held_out_load_hp}.",
            )
            continue

        print(f"\nRunning ResDilatedAE leave-one-load-out fold with held-out load {held_out_load_hp}")
        fold = build_fold_dataset(
            held_out_load_hp=held_out_load_hp,
            raw_train_healthy=raw_train,
            raw_val_healthy=raw_val,
            raw_test_healthy=raw_test_healthy,
            raw_test_fault=raw_test_fault,
            fault_labels=fault_labels,
        )

        fold_result = train_and_evaluate_fold(
            fold=fold,
            held_out_load_hp=held_out_load_hp,
            run_paths=run_paths,
            device=device,
            batch_size=int(batch_size),
            args=args,
            preprocessing_config=preprocessing_config,
            resume=args.resume,
        )
        if fold_result.get("status") != "completed" and "model" not in fold_result:
            save_progress(
                path=artifact_paths["progress"],
                held_out_loads=requested_loads,
                completed_loads=completed_loads,
                pending_loads=[load for load in requested_loads if load not in completed_loads],
                status="interrupted",
                last_message=f"Training interrupted on held-out load {held_out_load_hp}. Resume with --resume.",
            )
            print(
                f"\nHeld-out load {held_out_load_hp} interrupted. Resume with: {build_manual_command(args, held_out_load_hp, resume=True)}"
            )
            return 1

        fold_results.append(fold_result)
        completed_loads.append(int(held_out_load_hp))
        metrics = fold_result["model"]["metrics"]
        print(
            f"  ResDilatedAE: F1={metrics['f1']:.6f}, AUROC={metrics['auroc']:.6f}, FAR={metrics['false_alarm_rate']:.6f}"
        )
        save_progress(
            path=artifact_paths["progress"],
            held_out_loads=requested_loads,
            completed_loads=completed_loads,
            pending_loads=[load for load in requested_loads if load not in completed_loads],
            status="running",
            last_message=f"Completed held-out load {held_out_load_hp}.",
        )

    if len(fold_results) != len(requested_loads):
        raise RuntimeError(f"Expected {len(requested_loads)} completed folds, found {len(fold_results)}.")

    fold_results = sorted(fold_results, key=lambda item: int(item["held_out_load_hp"]))
    resdilated_summary = summarize_resdilated_metrics(fold_results)
    plot_summary(
        fold_results=fold_results,
        baseline_summary=baseline_payload.get("summary", {}),
        output_path=artifact_paths["plot"],
    )

    metrics_payload = {
        "protocol": {
            "name": "cwru_leave_one_load_out",
            "held_out_loads": requested_loads,
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
        "model": {
            "model_name": RUN_CONFIG.name,
            "model_cli_name": RUN_CONFIG.cli_name,
            "seed": int(args.seed),
            "device": str(device),
            "batch_size": int(batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "freq_loss_weight": float(args.freq_loss_weight),
            "dropout": float(args.dropout),
            "save_every_epochs": int(args.save_every_epochs),
        },
        "folds": make_json_ready(fold_results),
        "summary": resdilated_summary,
        "baseline_reference": baseline_payload.get("summary", {}),
        "baseline_reference_path": BASELINE_LOAD_SHIFT_METRICS_PATH.as_posix(),
        "manual_commands": {
            "all_loads": build_manual_command(args, requested_loads[0]).replace(
                f"--held-out-loads {requested_loads[0]}",
                "--held-out-loads " + " ".join(str(load) for load in requested_loads),
            ),
            "resume_all_loads": build_manual_command(args, requested_loads[0], resume=True).replace(
                f"--held-out-loads {requested_loads[0]}",
                "--held-out-loads " + " ".join(str(load) for load in requested_loads),
            ),
        },
    }
    write_json(artifact_paths["metrics"], make_json_ready(metrics_payload))
    report_text = build_report(
        preprocessing_config=preprocessing_config,
        device=device,
        batch_size=int(batch_size),
        args=args,
        fold_results=fold_results,
        resdilated_summary=resdilated_summary,
        baseline_payload=baseline_payload,
        artifact_paths=artifact_paths,
    )
    write_text(artifact_paths["report"], report_text)
    save_progress(
        path=artifact_paths["progress"],
        held_out_loads=requested_loads,
        completed_loads=completed_loads,
        pending_loads=[],
        status="completed",
        last_message="All requested held-out loads completed.",
    )

    print("\nResDilatedAE load-shift mean/std summary")
    print(
        f"  F1={resdilated_summary['f1']['mean']:.6f}+/-{resdilated_summary['f1']['std']:.6f}, "
        f"FAR={resdilated_summary['false_alarm_rate']['mean']:.6f}+/-{resdilated_summary['false_alarm_rate']['std']:.6f}"
    )
    print(f"  Saved metrics: {artifact_paths['metrics'].as_posix()}")
    print(f"  Saved report: {artifact_paths['report'].as_posix()}")
    print(f"  Saved plot: {artifact_paths['plot'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
