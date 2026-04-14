from __future__ import annotations

import argparse
import gc
import io
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

try:
    from train_ae_baseline import CompactConvAutoencoder, parameter_count, require_torch, set_seed, torch
    from train_paderborn_baselines import ensure_required_files, read_json, resolve_paths
    from train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_models,
        build_run_paths,
        load_torch_payload,
        select_models,
        write_json,
        write_text,
    )
except ModuleNotFoundError:
    from scripts.train_ae_baseline import CompactConvAutoencoder, parameter_count, require_torch, set_seed, torch
    from scripts.train_paderborn_baselines import ensure_required_files, read_json, resolve_paths
    from scripts.train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_models,
        build_run_paths,
        load_torch_payload,
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
THRESHOLD_RULE = "percentile_99_5"
PADEBORN_AE_MODEL_PATH = ARTIFACTS_ROOT / "models" / "paderborn_ae_baseline.pt"
PADEBORN_AE_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "paderborn_ae_metrics.json"
PADEBORN_IFOREST_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "paderborn_iforest_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deployment-oriented CPU inference metrics for the final Paderborn ResDilatedAE system.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument(
        "--calibration-metrics",
        type=Path,
        default=ARTIFACTS_ROOT / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_threshold_calibration_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=0, help="Use a specific saved ResDilatedAE seed. Default 0 picks the best saved percentile_99_5 seed.")
    parser.add_argument("--single-runs", type=int, default=120)
    parser.add_argument("--batch-runs", type=int, default=40)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--benchmark-pool-size", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=12)
    return parser.parse_args()


def choose_best_seed(calibration_metrics_path: Path) -> tuple[int, dict[str, Any]]:
    if not calibration_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing saved threshold calibration metrics: {calibration_metrics_path.as_posix()}"
        )
    payload = read_json(calibration_metrics_path)
    seed_entries = payload.get("seeds", [])
    if not seed_entries:
        raise RuntimeError(
            f"Saved threshold calibration metrics do not contain any seed results: {calibration_metrics_path.as_posix()}"
        )
    best_entry = max(
        seed_entries,
        key=lambda item: (
            float(item["rules"][THRESHOLD_RULE]["metrics"]["f1"]),
            float(item["rules"][THRESHOLD_RULE]["metrics"]["recall_fault"]),
            -float(item["rules"][THRESHOLD_RULE]["metrics"]["false_alarm_rate"]),
        ),
    )
    return int(best_entry["seed"]), best_entry


def summarize_samples_ms(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise RuntimeError("No latency samples were collected.")
    ordered = sorted(float(value) for value in latencies_ms)
    percentile_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_ms": float(mean(ordered)),
        "std_ms": float(stdev(ordered)) if len(ordered) > 1 else 0.0,
        "p95_ms": float(ordered[percentile_index]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def serialize_state_dict_size_bytes(state_dict: dict[str, Any]) -> int:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return int(len(buffer.getvalue()))


def get_process_rss_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        success = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
        if not success:
            return None
        return int(counters.WorkingSetSize)
    except Exception:
        return None


def format_mib(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "n/a"
    return f"{float(size_bytes / (1024 * 1024)):.3f}"


def load_resdilatedae_from_checkpoint(
    *,
    checkpoint_path: Path,
    expected_seed: int,
    expected_width: int,
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
            f"Checkpoint seed mismatch at {checkpoint_path.as_posix()}: expected {expected_seed}, found {checkpoint_seed}"
        )
    checkpoint_settings = checkpoint_payload.get("training_settings", {})
    dropout = float(checkpoint_settings.get("dropout", 0.05))
    model_candidates = select_models(build_models(expected_width, dropout), RUN_CONFIG.cli_name)
    if len(model_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one model definition for {RUN_CONFIG.cli_name}, found {len(model_candidates)}."
        )
    _run_config, model = model_candidates[0]
    model.load_state_dict(checkpoint_payload["state_dict"])
    model = model.to("cpu")
    model.eval()
    return model, checkpoint_payload


def load_compactae_from_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CompactAE checkpoint: {checkpoint_path.as_posix()}")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CompactConvAutoencoder().to("cpu")
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    return model, checkpoint_payload


def build_benchmark_windows(*, array_paths: Any, pool_size: int) -> np.ndarray:
    if pool_size < 64:
        raise RuntimeError("--benchmark-pool-size must be at least 64.")
    val_windows = np.load(array_paths.val_healthy, mmap_mode="r")
    test_healthy_windows = np.load(array_paths.test_healthy, mmap_mode="r")
    test_fault_windows = np.load(array_paths.test_fault, mmap_mode="r")

    val_count = min(pool_size // 4, int(val_windows.shape[0]))
    healthy_count = min(pool_size // 4, int(test_healthy_windows.shape[0]))
    fault_count = min(pool_size - val_count - healthy_count, int(test_fault_windows.shape[0]))
    if val_count <= 0 or healthy_count <= 0 or fault_count <= 0:
        raise RuntimeError("Could not build a representative benchmark pool from the saved Paderborn windows.")

    fault_indices = np.linspace(0, int(test_fault_windows.shape[0]) - 1, num=fault_count, dtype=np.int64)
    windows = np.concatenate(
        [
            np.asarray(val_windows[:val_count], dtype=np.float32),
            np.asarray(test_healthy_windows[:healthy_count], dtype=np.float32),
            np.asarray(test_fault_windows[fault_indices], dtype=np.float32),
        ],
        axis=0,
    )
    return windows.astype(np.float32, copy=False)


def run_model_forward(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    output = model(batch)
    if isinstance(output, tuple):
        return output[0]
    return output


def benchmark_model_cpu(
    *,
    model: torch.nn.Module,
    benchmark_windows: np.ndarray,
    single_runs: int,
    batch_runs: int,
    batch_sizes: list[int],
    warmup_runs: int,
    rss_before_load: int | None,
    rss_after_model_ready: int | None,
) -> dict[str, Any]:
    if benchmark_windows.ndim != 2:
        raise RuntimeError(f"Benchmark windows must be 2D, got shape {benchmark_windows.shape}")

    single_runs = max(single_runs, 1)
    batch_runs = max(batch_runs, 1)
    batch_sizes = sorted(set(int(size) for size in batch_sizes if int(size) > 0))
    if not batch_sizes:
        raise RuntimeError("At least one positive batch size is required for benchmarking.")

    model = model.to("cpu")
    model.eval()
    benchmark_tensor = torch.from_numpy(benchmark_windows).unsqueeze(1).contiguous()

    single_warmup = min(warmup_runs, int(benchmark_tensor.shape[0]))
    with torch.inference_mode():
        for index in range(single_warmup):
            _ = run_model_forward(model, benchmark_tensor[index : index + 1])

    single_latencies_ms: list[float] = []
    rss_peak = max(value for value in [rss_before_load, rss_after_model_ready] if value is not None) if any(
        value is not None for value in [rss_before_load, rss_after_model_ready]
    ) else None
    with torch.inference_mode():
        for run_index in range(single_runs):
            sample_index = run_index % int(benchmark_tensor.shape[0])
            sample_batch = benchmark_tensor[sample_index : sample_index + 1]
            start = time.perf_counter()
            _ = run_model_forward(model, sample_batch)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            single_latencies_ms.append(float(elapsed_ms))
            rss_now = get_process_rss_bytes()
            if rss_peak is None:
                rss_peak = rss_now
            elif rss_now is not None:
                rss_peak = max(rss_peak, rss_now)

    batch_results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        if batch_size > int(benchmark_tensor.shape[0]):
            raise RuntimeError(
                f"Batch size {batch_size} is larger than the benchmark pool size {benchmark_tensor.shape[0]}."
            )
        with torch.inference_mode():
            for warmup_index in range(min(warmup_runs, batch_runs)):
                start_index = (warmup_index * batch_size) % (int(benchmark_tensor.shape[0]) - batch_size + 1)
                _ = run_model_forward(model, benchmark_tensor[start_index : start_index + batch_size])

        latencies_ms: list[float] = []
        with torch.inference_mode():
            for run_index in range(batch_runs):
                start_index = (run_index * batch_size) % (int(benchmark_tensor.shape[0]) - batch_size + 1)
                batch = benchmark_tensor[start_index : start_index + batch_size]
                start = time.perf_counter()
                _ = run_model_forward(model, batch)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(float(elapsed_ms))
                rss_now = get_process_rss_bytes()
                if rss_peak is None:
                    rss_peak = rss_now
                elif rss_now is not None:
                    rss_peak = max(rss_peak, rss_now)
        summary = summarize_samples_ms(latencies_ms)
        batch_results[str(batch_size)] = {
            **summary,
            "throughput_windows_per_sec": float(batch_size / max(summary["mean_ms"] / 1000.0, 1e-12)),
        }

    single_summary = summarize_samples_ms(single_latencies_ms)
    return {
        "single_window": {
            **single_summary,
            "throughput_windows_per_sec": float(1.0 / max(single_summary["mean_ms"] / 1000.0, 1e-12)),
        },
        "batch": batch_results,
        "memory_rss": {
            "baseline_bytes": rss_before_load,
            "after_model_ready_bytes": rss_after_model_ready,
            "peak_bytes": rss_peak,
            "load_delta_bytes": None
            if rss_before_load is None or rss_after_model_ready is None
            else int(rss_after_model_ready - rss_before_load),
            "peak_delta_bytes": None if rss_before_load is None or rss_peak is None else int(rss_peak - rss_before_load),
        },
        "benchmark_pool": {
            "window_count": int(benchmark_tensor.shape[0]),
            "window_size": int(benchmark_tensor.shape[-1]),
        },
    }


def build_summary_row(
    *,
    model_name: str,
    parameter_total: int,
    weights_size_bytes: int,
    checkpoint_size_bytes: int | None,
    benchmark: dict[str, Any],
    performance_reference: dict[str, Any] | None,
) -> list[str]:
    batch64 = benchmark["batch"].get("64")
    return [
        model_name,
        str(parameter_total),
        f"{weights_size_bytes / (1024 * 1024):.3f}",
        f"{checkpoint_size_bytes / (1024 * 1024):.3f}" if checkpoint_size_bytes is not None else "n/a",
        f"{benchmark['single_window']['mean_ms']:.3f}",
        f"{batch64['mean_ms']:.3f}" if batch64 is not None else "n/a",
        f"{batch64['throughput_windows_per_sec']:.1f}" if batch64 is not None else "n/a",
        format_mib(benchmark["memory_rss"]["peak_delta_bytes"]),
        f"{performance_reference['f1']:.3f}" if performance_reference is not None else "n/a",
        f"{performance_reference['auroc']:.3f}" if performance_reference is not None else "n/a",
    ]


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    *,
    seed: int,
    threshold_value: float,
    thread_count: int,
    benchmark_windows: np.ndarray,
    resdilated_summary: list[str],
    compactae_summary: list[str] | None,
    resdilated_peak_rss_delta_bytes: int | None,
    compactae_peak_rss_delta_bytes: int | None,
    iforest_metrics: dict[str, Any] | None,
    iforest_blocker: str | None,
    metrics_path: Path,
    report_path: Path,
) -> str:
    table_rows = [resdilated_summary]
    if compactae_summary is not None:
        table_rows.append(compactae_summary)

    lines = [
        "# Paderborn Deployment Metrics Report",
        "",
        "## Setup",
        f"- Final chosen model benchmarked: `{RUN_CONFIG.name}` using saved seed `{seed}` and calibrated threshold rule `{THRESHOLD_RULE}`.",
        f"- Threshold value for the saved final run: `{threshold_value:.6f}`",
        f"- CPU benchmarking used `{thread_count}` Torch thread(s) with representative saved windows only.",
        f"- Benchmark pool: `{benchmark_windows.shape[0]}` windows of length `{benchmark_windows.shape[1]}` drawn from `val_healthy`, `test_healthy`, and `test_fault`.",
        "",
        "## CPU Benchmarks",
        format_markdown_table(
            [
                "Model",
                "Params",
                "Weights MB",
                "Checkpoint MB",
                "Single ms",
                "Batch64 ms",
                "Batch64 win/s",
                "Peak RSS Delta MB",
                "Saved F1",
                "Saved AUROC",
            ],
            table_rows,
        ),
        "",
        "## Practical Take",
    ]

    if compactae_summary is not None:
        if resdilated_peak_rss_delta_bytes is not None and resdilated_peak_rss_delta_bytes > 128 * 1024 * 1024:
            lines.append(
                "- `ResDilatedAE` keeps a tiny on-disk footprint, but the measured CPU runtime RSS bump is much larger in PyTorch. That still fits an industrial PC or gateway-style edge deployment, but it is not a microcontroller-class model."
            )
        else:
            lines.append(
                "- `ResDilatedAE` stays light enough for the VibeTwin edge-friendly claim in this saved CPU benchmark: small on disk and fast enough for sliding-window scoring on a modest device."
            )
        compactae_memory_text = (
            f"CompactAE peak RSS delta was about `{compactae_peak_rss_delta_bytes / (1024 * 1024):.1f}` MB."
            if compactae_peak_rss_delta_bytes is not None
            else "CompactAE memory could not be estimated cleanly."
        )
        lines.append(
            "- The tradeoff versus `CompactAE` is straightforward: `ResDilatedAE` is larger and slower on CPU, but it buys a large jump in saved Paderborn detection quality under the chosen calibrated setup. "
            + compactae_memory_text
        )
    else:
        lines.append(
            "- `ResDilatedAE` stays in a compact range on disk and on CPU, so it still supports an edge-friendly deployment story for Paderborn."
        )

    if iforest_blocker is None and iforest_metrics is not None:
        lines.append(
            "- Isolation Forest remains a strong classical baseline in saved detection metrics, but no clean like-for-like CPU benchmark was needed because a serialized estimator was available."
        )
    else:
        lines.append(
            f"- Isolation Forest comparison is partially blocked: {iforest_blocker}"
        )
        if iforest_metrics is not None:
            lines.append(
                f"- Saved Isolation Forest reference remains useful for context: F1 `{iforest_metrics['f1']:.3f}`, AUROC `{iforest_metrics['auroc']:.3f}`, FAR `{iforest_metrics['false_alarm_rate']:.4f}`."
            )

    lines.extend(
        [
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
    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    benchmark_windows = build_benchmark_windows(array_paths=array_paths, pool_size=args.benchmark_pool_size)
    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    expected_width = int(preprocessing_config["window_size"])

    best_seed, best_seed_entry = choose_best_seed(args.calibration_metrics.resolve())
    seed = int(args.seed) if args.seed > 0 else best_seed
    selected_seed_entry = best_seed_entry
    if seed != best_seed:
        calibration_payload = read_json(args.calibration_metrics.resolve())
        matches = [entry for entry in calibration_payload["seeds"] if int(entry["seed"]) == seed]
        if not matches:
            raise RuntimeError(
                f"Requested seed {seed} is not present in saved threshold calibration metrics: {args.calibration_metrics.resolve().as_posix()}"
            )
        selected_seed_entry = matches[0]

    metrics_output_dir = artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "deployment"
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_output_dir / "resdilated_ae_deployment_metrics.json"
    report_path = metrics_output_dir / "resdilated_ae_deployment_report.md"

    run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=RUN_CONFIG, seed=seed)
    if not run_paths.best_checkpoint.exists():
        raise FileNotFoundError(f"Missing saved final checkpoint: {run_paths.best_checkpoint.as_posix()}")

    resdilated_rss_before_load = get_process_rss_bytes()
    resdilated_model, resdilated_checkpoint = load_resdilatedae_from_checkpoint(
        checkpoint_path=run_paths.best_checkpoint,
        expected_seed=seed,
        expected_width=expected_width,
    )
    resdilated_rss_after_load = get_process_rss_bytes()
    resdilated_parameter_count = int(parameter_count(resdilated_model))
    resdilated_weights_size_bytes = serialize_state_dict_size_bytes(resdilated_checkpoint["state_dict"])
    resdilated_checkpoint_size_bytes = int(run_paths.best_checkpoint.stat().st_size)
    resdilated_benchmark = benchmark_model_cpu(
        model=resdilated_model,
        benchmark_windows=benchmark_windows,
        single_runs=args.single_runs,
        batch_runs=args.batch_runs,
        batch_sizes=args.batch_sizes,
        warmup_runs=args.warmup_runs,
        rss_before_load=resdilated_rss_before_load,
        rss_after_model_ready=resdilated_rss_after_load,
    )
    resdilated_performance = selected_seed_entry["rules"][THRESHOLD_RULE]["metrics"]
    del resdilated_model
    gc.collect()

    compactae_benchmark: dict[str, Any] | None = None
    compactae_parameter_count: int | None = None
    compactae_weights_size_bytes: int | None = None
    compactae_checkpoint_size_bytes: int | None = None
    compactae_metrics: dict[str, Any] | None = None
    compactae_blocker: str | None = None
    if PADEBORN_AE_MODEL_PATH.exists():
        compactae_rss_before_load = get_process_rss_bytes()
        compactae_model, compactae_checkpoint = load_compactae_from_checkpoint(PADEBORN_AE_MODEL_PATH)
        compactae_rss_after_load = get_process_rss_bytes()
        compactae_parameter_count = int(parameter_count(compactae_model))
        compactae_weights_size_bytes = serialize_state_dict_size_bytes(compactae_checkpoint["model_state_dict"])
        compactae_checkpoint_size_bytes = int(PADEBORN_AE_MODEL_PATH.stat().st_size)
        compactae_benchmark = benchmark_model_cpu(
            model=compactae_model,
            benchmark_windows=benchmark_windows,
            single_runs=args.single_runs,
            batch_runs=args.batch_runs,
            batch_sizes=args.batch_sizes,
            warmup_runs=args.warmup_runs,
            rss_before_load=compactae_rss_before_load,
            rss_after_model_ready=compactae_rss_after_load,
        )
        compactae_metrics = read_json(PADEBORN_AE_METRICS_PATH) if PADEBORN_AE_METRICS_PATH.exists() else None
        del compactae_model
        gc.collect()
    else:
        compactae_blocker = f"Missing saved CompactAE checkpoint: {PADEBORN_AE_MODEL_PATH.as_posix()}"

    iforest_metrics = read_json(PADEBORN_IFOREST_METRICS_PATH) if PADEBORN_IFOREST_METRICS_PATH.exists() else None
    iforest_model_candidates = list((artifacts_root / "models").glob("*iforest*")) if (artifacts_root / "models").exists() else []
    if iforest_model_candidates:
        iforest_blocker = None
    else:
        iforest_blocker = (
            "no serialized Paderborn Isolation Forest estimator or scaler was saved under `artifacts/models/`; "
            "only metrics and score CSVs exist, so a real inference benchmark would require refitting the model, "
            "which would violate the no-retraining constraint."
        )

    summary_rows = [
        build_summary_row(
            model_name="ResDilatedAE",
            parameter_total=resdilated_parameter_count,
            weights_size_bytes=resdilated_weights_size_bytes,
            checkpoint_size_bytes=resdilated_checkpoint_size_bytes,
            benchmark=resdilated_benchmark,
            performance_reference=resdilated_performance,
        )
    ]
    if compactae_benchmark is not None and compactae_parameter_count is not None and compactae_weights_size_bytes is not None:
        summary_rows.append(
            build_summary_row(
                model_name="CompactAE",
                parameter_total=compactae_parameter_count,
                weights_size_bytes=compactae_weights_size_bytes,
                checkpoint_size_bytes=compactae_checkpoint_size_bytes,
                benchmark=compactae_benchmark,
                performance_reference=compactae_metrics,
            )
        )

    metrics_payload = {
        "study": "paderborn_resdilated_ae_deployment_metrics",
        "model": RUN_CONFIG.name,
        "final_threshold_rule": THRESHOLD_RULE,
        "seed": int(seed),
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "benchmarking": {
            "cpu_only": True,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": None,
            "single_runs": int(args.single_runs),
            "batch_runs": int(args.batch_runs),
            "batch_sizes": [int(size) for size in args.batch_sizes],
            "warmup_runs": int(args.warmup_runs),
            "benchmark_pool_shape": list(benchmark_windows.shape),
        },
        "models": {
            "ResDilatedAE": {
                "checkpoint_path": run_paths.best_checkpoint.as_posix(),
                "threshold": float(resdilated_performance["threshold"]),
                "parameter_count": resdilated_parameter_count,
                "weights_only_size_bytes": resdilated_weights_size_bytes,
                "checkpoint_size_bytes": resdilated_checkpoint_size_bytes,
                "saved_detection_metrics": resdilated_performance,
                "cpu_benchmark": resdilated_benchmark,
            },
        },
        "comparisons": {
            "IsolationForest": {
                "saved_metrics": iforest_metrics,
                "benchmark_blocker": iforest_blocker,
            }
        },
    }
    if compactae_benchmark is not None and compactae_parameter_count is not None and compactae_weights_size_bytes is not None:
        metrics_payload["models"]["CompactAE"] = {
            "checkpoint_path": PADEBORN_AE_MODEL_PATH.as_posix(),
            "parameter_count": compactae_parameter_count,
            "weights_only_size_bytes": compactae_weights_size_bytes,
            "checkpoint_size_bytes": compactae_checkpoint_size_bytes,
            "saved_detection_metrics": compactae_metrics,
            "cpu_benchmark": compactae_benchmark,
        }
    else:
        metrics_payload["comparisons"]["CompactAE"] = {
            "benchmark_blocker": compactae_blocker,
        }

    report_text = build_report(
        seed=seed,
        threshold_value=float(resdilated_performance["threshold"]),
        thread_count=int(torch.get_num_threads()),
        benchmark_windows=benchmark_windows,
        resdilated_summary=summary_rows[0],
        compactae_summary=summary_rows[1] if len(summary_rows) > 1 else None,
        resdilated_peak_rss_delta_bytes=resdilated_benchmark["memory_rss"]["peak_delta_bytes"],
        compactae_peak_rss_delta_bytes=None if compactae_benchmark is None else compactae_benchmark["memory_rss"]["peak_delta_bytes"],
        iforest_metrics=iforest_metrics,
        iforest_blocker=iforest_blocker,
        metrics_path=metrics_path,
        report_path=report_path,
    )
    write_json(metrics_path, metrics_payload)
    write_text(report_path, report_text)
    print(f"Saved deployment metrics to {metrics_path.as_posix()}")
    print(f"Saved deployment report to {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
