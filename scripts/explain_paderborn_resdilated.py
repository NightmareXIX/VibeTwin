from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from train_ae_baseline import require_torch, set_seed, torch
    from train_paderborn_baselines import (
        ensure_required_files,
        load_label_array,
        read_json,
        resolve_paths,
        subgroup_metrics_from_manifest,
    )
    from train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_models,
        build_run_paths,
        get_device,
        load_torch_payload,
        select_models,
        write_json,
        write_text,
    )
except ModuleNotFoundError:
    from scripts.train_ae_baseline import require_torch, set_seed, torch
    from scripts.train_paderborn_baselines import (
        ensure_required_files,
        load_label_array,
        read_json,
        resolve_paths,
        subgroup_metrics_from_manifest,
    )
    from scripts.train_generative_upgrades import (
        ARTIFACTS_ROOT,
        METADATA_ROOT,
        PROCESSED_ROOT,
        ModelRunConfig,
        build_models,
        build_run_paths,
        get_device,
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


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    subset: str
    index: int
    label: str
    score: float
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate inference-only explanation figures for the best calibrated Paderborn ResDilatedAE checkpoint.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument(
        "--calibration-metrics",
        type=Path,
        default=ARTIFACTS_ROOT / "generative_upgrades" / RUN_CONFIG.output_stem / "resdilated_ae_threshold_calibration_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=0, help="Use a specific saved seed. Default 0 chooses the best saved percentile_99_5 seed.")
    parser.add_argument("--figure-dpi", type=int, default=170)
    return parser.parse_args()


def configure_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
        }
    )


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError("Healthy validation scores are empty; cannot fit percentile threshold.")
    return float(np.percentile(values, percentile))


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


def choose_index_close_to_reference(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    prefer_above_threshold: bool,
    exclude: set[int] | None = None,
) -> int:
    if candidate_indices.size == 0:
        raise RuntimeError("Cannot choose a representative case from an empty candidate set.")
    if exclude:
        candidate_indices = np.asarray([index for index in candidate_indices.tolist() if index not in exclude], dtype=np.int64)
        if candidate_indices.size == 0:
            raise RuntimeError("Candidate set became empty after removing already selected indices.")
    candidate_scores = np.asarray(scores, dtype=np.float64)[candidate_indices]
    reference_score = float(np.median(candidate_scores))
    sort_key = np.lexsort(
        (
            np.where(prefer_above_threshold, -candidate_scores, candidate_scores),
            np.abs(candidate_scores - reference_score),
        )
    )
    return int(candidate_indices[int(sort_key[0])])


def select_cases(
    *,
    window_manifest_path: Path,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    threshold: float,
    hardest_condition: str,
) -> list[SelectedCase]:
    true_positive_by_group: dict[str, list[int]] = {"KA": [], "KB": [], "KI": []}
    true_positive_by_condition: dict[str, list[int]] = {}

    fault_index = 0
    with window_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["subset"] != "test_fault":
                continue
            if fault_index >= int(test_fault_scores.shape[0]):
                raise RuntimeError("Window manifest has more test_fault rows than saved fault scores.")
            score = float(test_fault_scores[fault_index])
            if score >= threshold:
                true_positive_by_group.setdefault(row["damage_group"], []).append(fault_index)
                true_positive_by_condition.setdefault(row["condition_code"], []).append(fault_index)
            fault_index += 1

    if fault_index != int(test_fault_scores.shape[0]):
        raise RuntimeError(
            f"Window manifest test_fault rows ({fault_index}) do not match saved fault score count ({test_fault_scores.shape[0]})."
        )

    selected_fault_indices: set[int] = set()
    selected_cases: list[SelectedCase] = []
    for damage_group in ("KA", "KB", "KI"):
        candidate_indices = np.asarray(true_positive_by_group.get(damage_group, []), dtype=np.int64)
        if candidate_indices.size == 0:
            raise RuntimeError(
                f"Current best checkpoint has no true-positive windows for damage group {damage_group} under {THRESHOLD_RULE}."
            )
        case_index = choose_index_close_to_reference(
            test_fault_scores,
            candidate_indices,
            prefer_above_threshold=True,
            exclude=selected_fault_indices,
        )
        selected_fault_indices.add(case_index)
        selected_cases.append(
            SelectedCase(
                case_id=f"tp_{damage_group.lower()}",
                subset="test_fault",
                index=case_index,
                label=f"True Positive {damage_group}",
                score=float(test_fault_scores[case_index]),
                selection_reason=f"Representative true-positive example from damage group {damage_group}.",
            )
        )

    hardest_condition_candidates = np.asarray(true_positive_by_condition.get(hardest_condition, []), dtype=np.int64)
    if hardest_condition_candidates.size == 0:
        raise RuntimeError(
            f"Current best checkpoint has no true-positive windows in hardest operating condition {hardest_condition}."
        )
    try:
        hardest_condition_index = choose_index_close_to_reference(
            test_fault_scores,
            hardest_condition_candidates,
            prefer_above_threshold=True,
            exclude=selected_fault_indices,
        )
    except RuntimeError:
        hardest_condition_index = choose_index_close_to_reference(
            test_fault_scores,
            hardest_condition_candidates,
            prefer_above_threshold=True,
            exclude=None,
        )
    selected_cases.append(
        SelectedCase(
            case_id="hardest_condition_tp",
            subset="test_fault",
            index=hardest_condition_index,
            label="Hardest Condition TP",
            score=float(test_fault_scores[hardest_condition_index]),
            selection_reason=f"True-positive example from hardest operating condition {hardest_condition}.",
        )
    )

    healthy_true_negative_indices = np.flatnonzero(np.asarray(test_healthy_scores, dtype=np.float64) < float(threshold))
    if healthy_true_negative_indices.size == 0:
        raise RuntimeError("Current best checkpoint has no healthy true-negative windows under the calibrated threshold.")
    healthy_true_negative_index = choose_index_close_to_reference(
        test_healthy_scores,
        healthy_true_negative_indices,
        prefer_above_threshold=False,
        exclude=None,
    )
    selected_cases.append(
        SelectedCase(
            case_id="healthy_true_negative",
            subset="test_healthy",
            index=healthy_true_negative_index,
            label="Healthy True Negative",
            score=float(test_healthy_scores[healthy_true_negative_index]),
            selection_reason="Representative healthy window with a comfortably sub-threshold score.",
        )
    )

    false_positive_indices = np.flatnonzero(np.asarray(test_healthy_scores, dtype=np.float64) >= float(threshold))
    if false_positive_indices.size > 0:
        margins = np.asarray(test_healthy_scores[false_positive_indices], dtype=np.float64) - float(threshold)
        false_positive_index = int(false_positive_indices[int(np.argmin(margins))])
        selected_cases.append(
            SelectedCase(
                case_id="healthy_false_positive",
                subset="test_healthy",
                index=false_positive_index,
                label="Healthy False Positive",
                score=float(test_healthy_scores[false_positive_index]),
                selection_reason="Borderline healthy false-positive with the smallest score margin above threshold.",
            )
        )
    else:
        borderline_true_negative_index = int(
            healthy_true_negative_indices[
                int(np.argmax(np.asarray(test_healthy_scores[healthy_true_negative_indices], dtype=np.float64)))
            ]
        )
        if borderline_true_negative_index != healthy_true_negative_index:
            selected_cases.append(
                SelectedCase(
                    case_id="healthy_borderline_true_negative",
                    subset="test_healthy",
                    index=borderline_true_negative_index,
                    label="Healthy Borderline TN",
                    score=float(test_healthy_scores[borderline_true_negative_index]),
                    selection_reason="Healthy window closest to the threshold from below because no false positive was available.",
                )
            )

    return selected_cases


def collect_selected_metadata(
    *,
    window_manifest_path: Path,
    selected_cases: list[SelectedCase],
) -> dict[tuple[str, int], dict[str, str]]:
    healthy_targets = {case.index for case in selected_cases if case.subset == "test_healthy"}
    fault_targets = {case.index for case in selected_cases if case.subset == "test_fault"}
    metadata: dict[tuple[str, int], dict[str, str]] = {}

    healthy_index = 0
    fault_index = 0
    with window_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subset = row["subset"]
            if subset == "test_healthy":
                if healthy_index in healthy_targets:
                    metadata[(subset, healthy_index)] = {
                        "subset": subset,
                        "health_status": row["health_status"],
                        "damage_group": row["damage_group"],
                        "bearing_code": row["bearing_code"],
                        "condition_code": row["condition_code"],
                        "relative_path": row["relative_path"],
                        "measurement_id": row["measurement_id"],
                        "window_index": row["window_index"],
                        "selected_signal": row["selected_signal"],
                        "fault_label_name": row["fault_label_name"],
                    }
                healthy_index += 1
            elif subset == "test_fault":
                if fault_index in fault_targets:
                    metadata[(subset, fault_index)] = {
                        "subset": subset,
                        "health_status": row["health_status"],
                        "damage_group": row["damage_group"],
                        "bearing_code": row["bearing_code"],
                        "condition_code": row["condition_code"],
                        "relative_path": row["relative_path"],
                        "measurement_id": row["measurement_id"],
                        "window_index": row["window_index"],
                        "selected_signal": row["selected_signal"],
                        "fault_label_name": row["fault_label_name"],
                    }
                fault_index += 1

    expected_targets = {(case.subset, case.index) for case in selected_cases}
    missing = sorted(expected_targets - set(metadata))
    if missing:
        raise RuntimeError(
            "Window manifest did not yield metadata for selected cases: "
            + ", ".join(f"{subset}[{index}]" for subset, index in missing)
        )
    return metadata


def load_selected_windows(
    *,
    array_paths: Any,
    selected_cases: list[SelectedCase],
) -> dict[tuple[str, int], np.ndarray]:
    healthy_windows = np.load(array_paths.test_healthy, mmap_mode="r")
    fault_windows = np.load(array_paths.test_fault, mmap_mode="r")
    windows: dict[tuple[str, int], np.ndarray] = {}
    for case in selected_cases:
        if case.subset == "test_healthy":
            window = np.asarray(healthy_windows[case.index], dtype=np.float32)
        else:
            window = np.asarray(fault_windows[case.index], dtype=np.float32)
        windows[(case.subset, case.index)] = window
    return windows


def reconstruct_selected_windows(
    *,
    model: torch.nn.Module,
    device: torch.device,
    selected_cases: list[SelectedCase],
    original_windows: dict[tuple[str, int], np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    ordered_windows = [original_windows[(case.subset, case.index)] for case in selected_cases]
    batch = torch.from_numpy(np.stack(ordered_windows, axis=0)).unsqueeze(1).to(
        device,
        non_blocking=device.type == "cuda",
    )
    with torch.no_grad():
        reconstruction = model(batch)
    reconstruction_np = reconstruction.detach().cpu().numpy().astype(np.float32, copy=False)[:, 0, :]
    return {
        (case.subset, case.index): reconstruction_np[index]
        for index, case in enumerate(selected_cases)
    }


def signal_rms(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(np.asarray(values, dtype=np.float64))))))


def describe_time_domain(original: np.ndarray, reconstruction: np.ndarray, residual: np.ndarray) -> str:
    original_rms = signal_rms(original)
    residual_rms = signal_rms(residual)
    residual_ratio = residual_rms / max(original_rms, 1e-12)
    correlation = 1.0
    if float(np.std(original)) > 1e-12 and float(np.std(reconstruction)) > 1e-12:
        correlation = float(np.corrcoef(original, reconstruction)[0, 1])
    peak_ratio = float(np.max(np.abs(residual)) / max(float(np.max(np.abs(original))), 1e-12))

    if residual_ratio < 0.08 and correlation > 0.97:
        return "The reconstruction closely follows the waveform and leaves only small residual ripples."
    if peak_ratio > 0.55:
        return "The largest mismatch appears as sharp local bursts, so the model is missing transient structure rather than only a small amplitude shift."
    if correlation < 0.90:
        return "The reconstruction drifts away from the waveform shape across much of the window, leaving a broad residual mismatch."
    return "The reconstruction captures the coarse envelope but leaves a moderate oscillatory mismatch in the finer waveform detail."


def describe_frequency_domain(original: np.ndarray, reconstruction: np.ndarray, residual: np.ndarray) -> str:
    freqs = np.fft.rfftfreq(original.shape[0], d=1.0)
    original_mag = np.log1p(np.abs(np.fft.rfft(original)))
    reconstruction_mag = np.log1p(np.abs(np.fft.rfft(reconstruction)))
    residual_mag = np.log1p(np.abs(np.fft.rfft(residual)))
    spectral_gap = float(np.mean(np.abs(original_mag - reconstruction_mag)) / max(np.mean(original_mag), 1e-12))

    band_masks = {
        "low": freqs <= 0.10,
        "mid": (freqs > 0.10) & (freqs <= 0.25),
        "high": freqs > 0.25,
    }
    band_energy = {
        band_name: float(np.sum(np.square(residual_mag[mask])))
        for band_name, mask in band_masks.items()
    }
    dominant_band = max(band_energy, key=band_energy.get)
    top_bins = np.argsort(residual_mag[1:])[-3:] + 1 if residual_mag.shape[0] > 4 else np.arange(min(3, residual_mag.shape[0]))
    dominant_freqs = ", ".join(f"{float(freqs[index]):.3f}" for index in sorted(top_bins.tolist()))

    if spectral_gap < 0.05:
        return (
            f"FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is "
            f"small and is concentrated most in the {dominant_band}-frequency band around {dominant_freqs} cycles/sample."
        )
    return (
        f"The residual spectrum is strongest in the {dominant_band}-frequency band, with the largest gaps near "
        f"{dominant_freqs} cycles/sample in log-magnitude FFT."
    )


def build_case_interpretation(
    *,
    metadata: dict[str, str],
    score: float,
    threshold: float,
    original: np.ndarray,
    reconstruction: np.ndarray,
    residual: np.ndarray,
) -> str:
    prediction = "abnormal" if score >= threshold else "healthy"
    actual_label = "healthy" if metadata["subset"] == "test_healthy" else f"fault ({metadata['damage_group']})"
    time_summary = describe_time_domain(original, reconstruction, residual)
    frequency_summary = describe_frequency_domain(original, reconstruction, residual)
    if metadata["subset"] == "test_healthy" and prediction == "healthy":
        decision_summary = "Because both the waveform and spectrum stay close to the learned healthy template, the score remains below the calibrated threshold."
    elif metadata["subset"] == "test_healthy" and prediction == "abnormal":
        decision_summary = "Even though the window is labeled healthy, the remaining mismatch is large enough to push the anomaly score above threshold, so this behaves like a false alarm."
    elif metadata["subset"] == "test_fault" and prediction == "abnormal":
        decision_summary = "That persistent mismatch against the healthy reconstruction is what likely pushed the anomaly score above threshold."
    else:
        decision_summary = "This is a miss: the reconstruction still matches too much of the faulted waveform, so the score stays below threshold."
    return " ".join(
        [
            f"Actual label: {actual_label}; predicted label: {prediction}.",
            time_summary,
            frequency_summary,
            decision_summary,
        ]
    )


def save_case_figure(
    *,
    path: Path,
    case: SelectedCase,
    metadata: dict[str, str],
    original: np.ndarray,
    reconstruction: np.ndarray,
    threshold: float,
    dpi: int,
) -> None:
    residual = original - reconstruction
    time_axis = np.arange(original.shape[0], dtype=np.int64)
    freq_axis = np.fft.rfftfreq(original.shape[0], d=1.0)
    original_fft = np.log1p(np.abs(np.fft.rfft(original)))
    reconstruction_fft = np.log1p(np.abs(np.fft.rfft(reconstruction)))
    residual_fft = np.log1p(np.abs(np.fft.rfft(residual)))
    prediction = "abnormal" if case.score >= threshold else "healthy"

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), dpi=dpi)
    (ax_original, ax_reconstruction, ax_residual), (ax_original_fft, ax_reconstruction_fft, ax_residual_fft) = axes

    ax_original.plot(time_axis, original, color="#1f3c88", linewidth=1.4)
    ax_original.set_title("Original Window")
    ax_original.set_xlabel("Sample")
    ax_original.set_ylabel("Amplitude")

    ax_reconstruction.plot(time_axis, reconstruction, color="#1b998b", linewidth=1.4)
    ax_reconstruction.set_title("Reconstructed Healthy Signal")
    ax_reconstruction.set_xlabel("Sample")
    ax_reconstruction.set_ylabel("Amplitude")

    ax_residual.plot(time_axis, residual, color="#d1495b", linewidth=1.2)
    ax_residual.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax_residual.set_title("Residual (Original - Reconstruction)")
    ax_residual.set_xlabel("Sample")
    ax_residual.set_ylabel("Amplitude")

    ax_original_fft.plot(freq_axis, original_fft, color="#1f3c88", linewidth=1.4)
    ax_original_fft.set_title("Original FFT Magnitude")
    ax_original_fft.set_xlabel("Normalized Frequency")
    ax_original_fft.set_ylabel("log(1 + |FFT|)")

    ax_reconstruction_fft.plot(freq_axis, reconstruction_fft, color="#1b998b", linewidth=1.4)
    ax_reconstruction_fft.set_title("Reconstruction FFT Magnitude")
    ax_reconstruction_fft.set_xlabel("Normalized Frequency")
    ax_reconstruction_fft.set_ylabel("log(1 + |FFT|)")

    ax_residual_fft.plot(freq_axis, residual_fft, color="#d1495b", linewidth=1.2)
    ax_residual_fft.set_title("Residual FFT Magnitude")
    ax_residual_fft.set_xlabel("Normalized Frequency")
    ax_residual_fft.set_ylabel("log(1 + |FFT|)")

    figure.suptitle(
        " | ".join(
            [
                case.label,
                f"prediction={prediction}",
                f"score={case.score:.6f}",
                f"threshold={threshold:.6f}",
                f"condition={metadata['condition_code'] or '-'}",
                f"damage_group={metadata['damage_group'] or '-'}",
            ]
        ),
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path)
    plt.close(figure)


def build_report(
    *,
    seed: int,
    checkpoint_path: Path,
    threshold: float,
    hardest_condition: str,
    selected_cases: list[SelectedCase],
    case_payloads: list[dict[str, Any]],
    report_path: Path,
    cases_json_path: Path,
) -> str:
    lines = [
        "# Paderborn ResDilatedAE Explanation Report",
        "",
        "## Setup",
        f"- Model: `{RUN_CONFIG.name}`",
        f"- Seed used: `{seed}`",
        f"- Checkpoint: `{checkpoint_path.as_posix()}`",
        f"- Threshold rule: `{THRESHOLD_RULE}`",
        f"- Threshold value: `{threshold:.6f}`",
        f"- Hardest operating condition under this setup: `{hardest_condition}`",
        "- Explanations are inference-only and use the saved deterministic score outputs plus the saved `best.pt` checkpoint.",
        "",
        "## Selected Cases",
        "| Case | Split | Score | Prediction | Condition | Damage Group | Why Selected |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for case_payload in case_payloads:
        lines.append(
            "| "
            f"{case_payload['case_id']} | "
            f"{case_payload['metadata']['subset']} | "
            f"{case_payload['score']:.6f} | "
            f"{case_payload['prediction']} | "
            f"{case_payload['metadata']['condition_code'] or '-'} | "
            f"{case_payload['metadata']['damage_group'] or '-'} | "
            f"{case_payload['selection_reason']} |"
        )

    for case_payload in case_payloads:
        lines.extend(
            [
                "",
                f"## {case_payload['label']}",
                f"![{case_payload['label']}]({case_payload['figure_relpath']})",
                f"- Score vs threshold: `{case_payload['score']:.6f}` vs `{case_payload['threshold']:.6f}`",
                f"- Prediction: `{case_payload['prediction']}`",
                f"- Metadata: subset=`{case_payload['metadata']['subset']}`, health_status=`{case_payload['metadata']['health_status'] or '-'}`, "
                f"damage_group=`{case_payload['metadata']['damage_group'] or '-'}`, condition=`{case_payload['metadata']['condition_code'] or '-'}`, "
                f"bearing=`{case_payload['metadata']['bearing_code'] or '-'}`, measurement=`{case_payload['metadata']['measurement_id'] or '-'}`, "
                f"window_index=`{case_payload['metadata']['window_index'] or '-'}`, signal=`{case_payload['metadata']['selected_signal'] or '-'}`",
                f"- Source window: `{case_payload['metadata']['relative_path']}`",
                f"- Interpretation: {case_payload['interpretation']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Saved Artifacts",
            f"- Report: `{report_path.as_posix()}`",
            f"- Cases JSON: `{cases_json_path.as_posix()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    require_torch()
    configure_plot_style()
    args = parse_args()
    set_seed(0)

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    fault_label_map = {key: int(value) for key, value in preprocessing_config["fault_label_map"].items()}
    window_size = int(preprocessing_config["window_size"])
    fault_labels = load_label_array(array_paths.fault_labels)

    best_seed, best_seed_entry = choose_best_seed(args.calibration_metrics.resolve())
    seed = int(args.seed) if args.seed > 0 else best_seed
    if args.seed <= 0:
        print(
            f"Using best calibrated seed {seed} from {args.calibration_metrics.resolve().as_posix()} "
            f"(best saved {THRESHOLD_RULE} F1)."
        )

    run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=RUN_CONFIG, seed=seed)
    required_outputs = [
        run_paths.best_checkpoint,
        run_paths.val_healthy_scores_npy,
        run_paths.test_healthy_scores_npy,
        run_paths.test_fault_scores_npy,
    ]
    missing_outputs = [path for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "Required saved explanation inputs are missing: "
            + ", ".join(path.as_posix() for path in missing_outputs)
        )

    val_scores = np.load(run_paths.val_healthy_scores_npy).astype(np.float32, copy=False)
    test_healthy_scores = np.load(run_paths.test_healthy_scores_npy).astype(np.float32, copy=False)
    test_fault_scores = np.load(run_paths.test_fault_scores_npy).astype(np.float32, copy=False)
    threshold = percentile_threshold(val_scores, 99.5)

    subgroup_metrics = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        fault_labels=fault_labels,
        fault_label_map=fault_label_map,
        threshold=threshold,
    )
    hardest_condition = min(
        subgroup_metrics["by_condition"].items(),
        key=lambda item: float(item[1]["f1"]),
    )[0]

    selected_cases = select_cases(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        threshold=threshold,
        hardest_condition=hardest_condition,
    )
    selected_metadata = collect_selected_metadata(
        window_manifest_path=metadata_root / "window_manifest.csv",
        selected_cases=selected_cases,
    )

    device = get_device()
    model, _checkpoint_payload = load_resdilatedae_from_checkpoint(
        checkpoint_path=run_paths.best_checkpoint,
        expected_seed=seed,
        expected_width=window_size,
        device=device,
    )
    original_windows = load_selected_windows(array_paths=array_paths, selected_cases=selected_cases)
    reconstructed_windows = reconstruct_selected_windows(
        model=model,
        device=device,
        selected_cases=selected_cases,
        original_windows=original_windows,
    )

    output_dir = artifacts_root / "generative_upgrades" / RUN_CONFIG.output_stem / "explanations" / f"seed_{seed}_{THRESHOLD_RULE}"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "resdilated_ae_explanation_report.md"
    cases_json_path = output_dir / "resdilated_ae_explanation_cases.json"

    case_payloads: list[dict[str, Any]] = []
    for order_index, case in enumerate(selected_cases, start=1):
        metadata = selected_metadata[(case.subset, case.index)]
        original = original_windows[(case.subset, case.index)]
        reconstruction = reconstructed_windows[(case.subset, case.index)]
        residual = original - reconstruction
        prediction = "abnormal" if case.score >= threshold else "healthy"
        figure_name = f"{order_index:02d}_{case.case_id}.png"
        figure_path = figures_dir / figure_name
        save_case_figure(
            path=figure_path,
            case=case,
            metadata=metadata,
            original=original,
            reconstruction=reconstruction,
            threshold=threshold,
            dpi=args.figure_dpi,
        )
        case_payloads.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "selection_reason": case.selection_reason,
                "score": float(case.score),
                "threshold": float(threshold),
                "prediction": prediction,
                "metadata": metadata,
                "figure_path": figure_path.as_posix(),
                "figure_relpath": figure_path.relative_to(report_path.parent).as_posix(),
                "interpretation": build_case_interpretation(
                    metadata=metadata,
                    score=float(case.score),
                    threshold=float(threshold),
                    original=original,
                    reconstruction=reconstruction,
                    residual=residual,
                ),
            }
        )

    report_text = build_report(
        seed=seed,
        checkpoint_path=run_paths.best_checkpoint,
        threshold=threshold,
        hardest_condition=hardest_condition,
        selected_cases=selected_cases,
        case_payloads=case_payloads,
        report_path=report_path,
        cases_json_path=cases_json_path,
    )
    write_json(
        cases_json_path,
        {
            "study": "paderborn_resdilated_ae_explanations",
            "model": RUN_CONFIG.name,
            "seed": int(seed),
            "threshold_rule": THRESHOLD_RULE,
            "threshold": float(threshold),
            "hardest_condition": hardest_condition,
            "source_run_dir": run_paths.run_dir.as_posix(),
            "checkpoint_path": run_paths.best_checkpoint.as_posix(),
            "score_paths": {
                "val_healthy_scores": run_paths.val_healthy_scores_npy.as_posix(),
                "test_healthy_scores": run_paths.test_healthy_scores_npy.as_posix(),
                "test_fault_scores": run_paths.test_fault_scores_npy.as_posix(),
            },
            "cases": case_payloads,
        },
    )
    write_text(report_path, report_text)
    print(f"Saved explanation figures to {figures_dir.as_posix()}")
    print(f"Saved explanation report to {report_path.as_posix()}")
    print(f"Saved case metadata to {cases_json_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
