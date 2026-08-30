from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from train_ae_baseline import (
        evaluate_scores as evaluate_binary_scores,
        parameter_count,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from train_generative_upgrades import (
        MEMAE_ADDRESSING,
        MEMAE_ADDRESSING_CHOICES,
        MEMAE_ENTROPY_WEIGHT,
        MEMAE_MEMORY_SIZE,
        MemAE,
        MemoryModule,
        ResDilatedAE,
        default_shrink_threshold,
        memory_entropy_loss,
    )
    from train_paderborn_baselines import resolve_paths
except ModuleNotFoundError:
    from scripts.train_ae_baseline import (
        evaluate_scores as evaluate_binary_scores,
        parameter_count,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from scripts.train_generative_upgrades import (
        MEMAE_ADDRESSING,
        MEMAE_ADDRESSING_CHOICES,
        MEMAE_ENTROPY_WEIGHT,
        MEMAE_MEMORY_SIZE,
        MemAE,
        MemoryModule,
        ResDilatedAE,
        default_shrink_threshold,
        memory_entropy_loss,
    )
    from scripts.train_paderborn_baselines import resolve_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

nn = torch.nn if torch is not None else None
F = torch.nn.functional if torch is not None else None

WINDOW_SIZE = 2048
REFERENCE_PARAMETER_COUNT = 222_657
PARAMETER_TOLERANCE = 0.15


@dataclass
class CheckResult:
    number: int
    name: str
    group: str
    expectation: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    kind: str = "assertion"

    @property
    def is_assertion(self) -> bool:
        """Assertions are claims about the implementation and gate the exit code.

        Findings record how the trained model behaved. A finding that comes out
        the other way is a result about MemAE on this benchmark, not a bug, so it
        is reported loudly and left out of the exit code.
        """
        return self.kind == "assertion"

    @property
    def status(self) -> str:
        if self.is_assertion:
            return "pass" if self.passed else "fail"
        return "as expected" if self.passed else "CONTRARY"

    def to_payload(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "name": self.name,
            "group": self.group,
            "kind": self.kind,
            "expectation": self.expectation,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness cross-check for the MemAE comparator (mechanism and behavioural tests).",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-size", type=int, default=MEMAE_MEMORY_SIZE)
    parser.add_argument("--shrink-threshold", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=MEMAE_ENTROPY_WEIGHT)
    parser.add_argument("--addressing", choices=MEMAE_ADDRESSING_CHOICES, default=MEMAE_ADDRESSING)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--overfit-windows", type=int, default=100)
    parser.add_argument("--overfit-epochs", type=int, default=200)
    parser.add_argument("--overfit-learning-rate", type=float, default=1e-3)
    # The behavioural probe defaults are sized for a resolvable answer, not for speed:
    # at 10 epochs on 8192 windows the check-12 delta flipped sign across repeats
    # (-0.019, -0.014, +0.005), i.e. the probe could not separate the two variants
    # from its own run-to-run spread. At these settings the gap is unambiguous.
    # Cost is roughly fifteen minutes on an RTX 4060; reduce them for a fast smoke
    # check, but do not read check 12 from a reduced run.
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--probe-train-windows", type=int, default=32768)
    parser.add_argument("--probe-val-windows", type=int, default=4096)
    parser.add_argument("--probe-test-windows", type=int, default=4096)
    parser.add_argument(
        "--skip-behavioural",
        action="store_true",
        help="Run only the mechanism tests (1-6); no processed windows required.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def load_window_subset(path: Path, count: int) -> torch.Tensor:
    """Load ``count`` evenly spaced windows as a (N, 1, L) float32 tensor.

    The arrays are stored file by file, so the leading windows come from one or
    two recordings; striding keeps every operating condition represented in the
    probe subsets.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required window array is missing: {path.as_posix()}")
    windows = np.load(path, mmap_mode="r")
    if windows.ndim != 2:
        raise ValueError(f"Expected a 2D window array at {path.as_posix()}, got {windows.shape}")
    total = int(windows.shape[0])
    limit = min(count, total)
    indices = np.linspace(0, total - 1, limit).astype(np.int64)
    subset = np.array(windows[indices], dtype=np.float32, copy=True)
    return torch.from_numpy(subset).unsqueeze(1)


# ---------------------------------------------------------------------------
# Mechanism tests (1-6)
# ---------------------------------------------------------------------------


def build_memory_module(args: argparse.Namespace, feature_dim: int, *, shrink_threshold: float | None = None) -> MemoryModule:
    return MemoryModule(
        memory_size=args.memory_size,
        feature_dim=feature_dim,
        shrink_threshold=args.shrink_threshold if shrink_threshold is None else shrink_threshold,
        addressing=args.addressing,
    )


def sharp_attention(rows: int, memory_size: int, concentration: float, generator: torch.Generator) -> torch.Tensor:
    """Sample normalized addressing rows that are sharper than the near-uniform init."""
    logits = torch.randn(rows, memory_size, generator=generator) * concentration
    return torch.softmax(logits, dim=-1)


def check_renormalized_mass(args: argparse.Namespace, generator: torch.Generator) -> CheckResult:
    module = build_memory_module(args, feature_dim=96)
    latent = torch.randn(4, 96, 128, generator=generator)
    _, attention = module(latent)
    deviation = float((attention.sum(dim=-1) - 1.0).abs().max().item())

    sharp = sharp_attention(256, args.memory_size, concentration=4.0, generator=generator)
    sharp_deviation = float((module.shrink_and_renormalize(sharp).sum(dim=-1) - 1.0).abs().max().item())

    worst = max(deviation, sharp_deviation)
    return CheckResult(
        number=1,
        name="Addressing weights sum to one per position",
        group="mechanism",
        expectation="max |sum - 1| < 1e-5",
        passed=worst < 1e-5,
        message=f"max deviation {worst:.3e} (init path {deviation:.3e}, sharp path {sharp_deviation:.3e})",
        details={
            "max_abs_deviation_init_addressing": deviation,
            "max_abs_deviation_sharp_addressing": sharp_deviation,
        },
    )


def check_zero_threshold_identity(args: argparse.Namespace, generator: torch.Generator) -> CheckResult:
    module = build_memory_module(args, feature_dim=96, shrink_threshold=0.0)
    attention = sharp_attention(512, args.memory_size, concentration=4.0, generator=generator)
    shrunk = module.shrink_and_renormalize(attention)
    max_delta = float((shrunk - attention).abs().max().item())
    return CheckResult(
        number=2,
        name="Shrinkage is the identity at lambda = 0",
        group="mechanism",
        expectation="max |w_hat - w| < 1e-6",
        passed=max_delta < 1e-6,
        message=f"max |w_hat - w| = {max_delta:.3e}",
        details={"max_abs_delta": max_delta},
    )


def nonzero_slot_count(module: MemoryModule, attention: torch.Tensor) -> float:
    shrunk = module.shrink_and_renormalize(attention)
    return float((shrunk > 0).sum(dim=-1).float().mean().item())


def check_threshold_monotonicity(args: argparse.Namespace, generator: torch.Generator) -> CheckResult:
    """Sweeping lambda over [0, 3/N] must not increase the number of surviving slots.

    Sampled on sharp (trained-like) addressing: at initialization cosine
    similarities sit in a narrow band, every softmax weight is near 1/N, and no
    slot survives lambda >= 1/N, so the collapse fallback masks the sweep.
    """
    attention = sharp_attention(512, args.memory_size, concentration=4.0, generator=generator)
    multipliers = (0.0, 1.0, 2.0, 3.0)
    counts: list[float] = []
    for multiplier in multipliers:
        module = build_memory_module(args, feature_dim=96, shrink_threshold=multiplier / float(args.memory_size))
        counts.append(nonzero_slot_count(module, attention))
    monotone = all(counts[index + 1] <= counts[index] + 1e-9 for index in range(len(counts) - 1))
    strict = counts[-1] < counts[0]
    return CheckResult(
        number=3,
        name="Raising lambda shrinks the active slot count",
        group="mechanism",
        expectation="mean non-zero slots per position is non-increasing in lambda, and strictly lower at 3/N than at 0",
        passed=monotone and strict,
        message=" -> ".join(f"{count:.1f}" for count in counts)
        + f" slots for lambda in {{0, 1/N, 2/N, 3/N}} (N={args.memory_size})",
        details={
            "lambda_multipliers_of_inverse_n": list(multipliers),
            "mean_nonzero_slots": counts,
            "monotone_non_increasing": monotone,
            "strictly_lower_at_3_over_n": strict,
        },
    )


def check_memory_gradient(args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> CheckResult:
    set_seed(args.seed)
    model = MemAE(
        base_channels=24,
        memory_size=args.memory_size,
        shrink_threshold=args.shrink_threshold,
        addressing=args.addressing,
    ).to(device)
    batch = torch.randn(4, 1, WINDOW_SIZE, generator=generator).to(device)
    reconstruction, attention = model(batch)
    loss = F.mse_loss(reconstruction, batch) + (args.entropy_weight * memory_entropy_loss(attention))
    loss.backward()
    grad = model.memory.memory.grad
    grad_norm = float(grad.norm().item()) if grad is not None else 0.0
    nonzero_rows = int((grad.abs().sum(dim=-1) > 0).sum().item()) if grad is not None else 0
    return CheckResult(
        number=4,
        name="Gradient reaches the memory bank",
        group="mechanism",
        expectation="M.grad is not None and has non-zero norm",
        passed=grad is not None and grad_norm > 0.0,
        message=f"||dL/dM|| = {grad_norm:.3e} over {nonzero_rows}/{args.memory_size} slots with non-zero gradient",
        details={
            "grad_is_none": grad is None,
            "grad_norm": grad_norm,
            "slots_with_nonzero_grad": nonzero_rows,
            "memory_size": int(args.memory_size),
        },
    )


def check_cosine_bounds(args: argparse.Namespace, generator: torch.Generator) -> CheckResult:
    module = build_memory_module(args, feature_dim=96)
    latent = torch.randn(8, 96, 128, generator=generator) * 25.0
    queries = F.normalize(latent.transpose(1, 2), p=2.0, dim=-1, eps=module.epsilon)
    memory = F.normalize(module.memory, p=2.0, dim=-1, eps=module.epsilon)
    similarity = torch.matmul(queries, memory.t())
    minimum = float(similarity.min().item())
    maximum = float(similarity.max().item())
    tolerance = 1e-6
    return CheckResult(
        number=5,
        name="Cosine similarity stays in [-1, 1]",
        group="mechanism",
        expectation="-1 - 1e-6 <= d <= 1 + 1e-6 for the cosine addressing path, whichever mode is the default",
        passed=minimum >= -1.0 - tolerance and maximum <= 1.0 + tolerance,
        message=f"similarity range [{minimum:.6f}, {maximum:.6f}]",
        details={"min_similarity": minimum, "max_similarity": maximum},
    )


def check_addressing_shape(args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> CheckResult:
    set_seed(args.seed)
    model = MemAE(
        base_channels=24,
        memory_size=args.memory_size,
        shrink_threshold=args.shrink_threshold,
        addressing=args.addressing,
    ).to(device)
    batch = torch.randn(4, 1, WINDOW_SIZE, generator=generator).to(device)
    with torch.no_grad():
        reconstruction, attention = model(batch)
    expected = (4, WINDOW_SIZE // 16, args.memory_size)
    actual = tuple(int(value) for value in attention.shape)
    shape_ok = actual == expected
    reconstruction_ok = tuple(int(value) for value in reconstruction.shape) == (4, 1, WINDOW_SIZE)
    return CheckResult(
        number=6,
        name="Addressing is per latent position",
        group="mechanism",
        expectation=f"attention shape {expected}, reconstruction shape (4, 1, {WINDOW_SIZE})",
        passed=shape_ok and reconstruction_ok,
        message=f"attention {actual}, reconstruction {tuple(int(value) for value in reconstruction.shape)}",
        details={
            "attention_shape": list(actual),
            "expected_attention_shape": list(expected),
            "reconstruction_shape": [int(value) for value in reconstruction.shape],
        },
    )


def check_parameter_budget(args: argparse.Namespace) -> CheckResult:
    memae = MemAE(
        base_channels=24,
        memory_size=args.memory_size,
        shrink_threshold=args.shrink_threshold,
        addressing=args.addressing,
    )
    reference = ResDilatedAE(base_channels=16, dropout=0.05)
    memae_params = parameter_count(memae)
    reference_params = parameter_count(reference)
    relative = (memae_params - reference_params) / float(reference_params)
    return CheckResult(
        number=0,
        name="Capacity matched to ResDilatedAE",
        group="mechanism",
        expectation=f"|delta| <= {PARAMETER_TOLERANCE:.0%} of {REFERENCE_PARAMETER_COUNT}",
        passed=abs(relative) <= PARAMETER_TOLERANCE,
        message=f"MemAE {memae_params:,} vs ResDilatedAE {reference_params:,} ({relative:+.2%})",
        details={
            "memae_parameters": memae_params,
            "resdilated_ae_parameters": reference_params,
            "reference_parameter_count": REFERENCE_PARAMETER_COUNT,
            "relative_difference": relative,
        },
    )


# ---------------------------------------------------------------------------
# Behavioural tests (7-11)
# ---------------------------------------------------------------------------


def iterate_batches(tensor: torch.Tensor, batch_size: int, *, shuffle: bool, generator: torch.Generator | None = None):
    count = int(tensor.shape[0])
    order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        yield tensor[order[start : start + batch_size]]


def train_probe(
    *,
    model: nn.Module,
    train_windows: torch.Tensor,
    val_windows: torch.Tensor | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    entropy_weight: float,
    device: torch.device,
    seed: int,
) -> dict[str, list[float]]:
    """Minimal training loop mirroring the MemAE branch of the production trainer."""
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_time_loss": [],
        "train_mem_loss": [],
        "val_time_loss": [],
        "val_mem_loss": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        time_total = 0.0
        mem_total = 0.0
        seen = 0
        for batch in iterate_batches(train_windows, batch_size, shuffle=True, generator=generator):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, attention = model(batch)
            time_loss = F.mse_loss(reconstruction, batch)
            mem_loss = memory_entropy_loss(attention)
            (time_loss + (entropy_weight * mem_loss)).backward()
            optimizer.step()
            count = int(batch.shape[0])
            time_total += float(time_loss.item()) * count
            mem_total += float(mem_loss.item()) * count
            seen += count
        history["epoch"].append(epoch)
        history["train_time_loss"].append(time_total / max(seen, 1))
        history["train_mem_loss"].append(mem_total / max(seen, 1))
        if val_windows is None:
            history["val_time_loss"].append(float("nan"))
            history["val_mem_loss"].append(float("nan"))
            continue
        val_stats = evaluate_probe(model=model, windows=val_windows, batch_size=batch_size, device=device)
        history["val_time_loss"].append(val_stats["time_loss"])
        history["val_mem_loss"].append(val_stats["mem_loss"])
    return history


def evaluate_probe(*, model: nn.Module, windows: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    time_total = 0.0
    mem_total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in iterate_batches(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            reconstruction, attention = model(batch)
            count = int(batch.shape[0])
            time_total += float(F.mse_loss(reconstruction, batch).item()) * count
            mem_total += float(memory_entropy_loss(attention).item()) * count
            seen += count
    divisor = max(seen, 1)
    return {"time_loss": time_total / divisor, "mem_loss": mem_total / divisor}


def score_windows(*, model: nn.Module, windows: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in iterate_batches(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            per_window = torch.mean((reconstruction.float() - batch.float()) ** 2, dim=(1, 2))
            scores.append(per_window.cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def collect_memory_usage(*, model: nn.Module, windows: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, Any]:
    """Average the addressing weights over every latent position of ``windows``."""
    model.eval()
    memory_size = int(model.memory.memory_size)
    usage = torch.zeros(memory_size, dtype=torch.float64)
    nonzero_total = 0.0
    entropy_total = 0.0
    positions = 0
    with torch.no_grad():
        for batch in iterate_batches(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            _, attention = model(batch)
            flattened = attention.reshape(-1, memory_size).double().cpu()
            usage += flattened.sum(dim=0)
            nonzero_total += float((flattened > 0).sum(dim=-1).sum().item())
            entropy_total += float((-(flattened * torch.log(flattened + 1e-12))).sum(dim=-1).sum().item())
            positions += int(flattened.shape[0])
    divisor = max(positions, 1)
    mean_usage = (usage / divisor).numpy()
    uniform = 1.0 / float(memory_size)
    utilized = int((mean_usage > (uniform / 10.0)).sum())
    order = np.argsort(mean_usage)[::-1]
    return {
        "memory_size": memory_size,
        "positions_scored": positions,
        "mean_usage_per_slot": mean_usage.tolist(),
        "utilized_slots": utilized,
        "utilized_fraction": utilized / float(memory_size),
        "top_slot_share": float(mean_usage[order[0]]),
        "top10_slot_share": float(mean_usage[order[:10]].sum()),
        "mean_nonzero_slots_per_position": nonzero_total / divisor,
        "mean_addressing_entropy": entropy_total / divisor,
        "uniform_entropy": math.log(memory_size),
    }


def count_histogram_modes(values: np.ndarray, bins: int = 60) -> int:
    """Count local maxima of a lightly smoothed histogram, ignoring small ripples."""
    counts, _ = np.histogram(values, bins=bins)
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(counts.astype(np.float64), kernel, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return 0
    modes = 0
    for index in range(1, len(smoothed) - 1):
        if smoothed[index] >= smoothed[index - 1] and smoothed[index] > smoothed[index + 1] and smoothed[index] > 0.05 * peak:
            modes += 1
    return modes


def check_overfit_capacity(
    args: argparse.Namespace,
    train_windows: torch.Tensor,
    device: torch.device,
) -> tuple[CheckResult, dict[str, Any]]:
    windows = train_windows[: args.overfit_windows].to(device)
    set_seed(args.seed)
    model = MemAE(
        base_channels=24,
        memory_size=args.memory_size,
        shrink_threshold=args.shrink_threshold,
        addressing=args.addressing,
    ).to(device)
    history = train_probe(
        model=model,
        train_windows=windows,
        val_windows=None,
        epochs=args.overfit_epochs,
        batch_size=min(args.batch_size, int(windows.shape[0])),
        learning_rate=args.overfit_learning_rate,
        entropy_weight=args.entropy_weight,
        device=device,
        seed=args.seed,
    )
    initial = history["train_time_loss"][0]
    final = history["train_time_loss"][-1]
    variance = float(windows.float().var().item())
    explained = 1.0 - (final / max(variance, 1e-12))
    passed = final < 0.3 and final < 0.5 * initial
    details = {
        "windows": int(windows.shape[0]),
        "epochs": args.overfit_epochs,
        "initial_recon_mse": initial,
        "final_recon_mse": final,
        "window_variance": variance,
        "fraction_variance_explained": explained,
        "final_addressing_entropy": history["train_mem_loss"][-1],
    }
    return (
        CheckResult(
            number=7,
            name="Overfits a small healthy subset",
            group="behavioural",
            expectation="final recon MSE < 0.3 and < half of the first epoch",
            passed=passed,
            message=f"recon MSE {initial:.4f} -> {final:.4f} over {args.overfit_epochs} epochs "
            f"({explained:.1%} of window variance explained)",
            details=details,
        ),
        details,
    )


def build_probe_model(args: argparse.Namespace, *, memory_enabled: bool, device: torch.device) -> nn.Module:
    set_seed(args.seed)
    return MemAE(
        base_channels=24,
        memory_size=args.memory_size,
        shrink_threshold=args.shrink_threshold if memory_enabled else 0.0,
        addressing=args.addressing,
    ).to(device)


def run_probe_variant(
    args: argparse.Namespace,
    *,
    memory_enabled: bool,
    train_windows: torch.Tensor,
    val_windows: torch.Tensor,
    test_healthy: torch.Tensor,
    test_fault: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Train one probe variant and score it under the production protocol."""
    model = build_probe_model(args, memory_enabled=memory_enabled, device=device)
    started = time.perf_counter()
    history = train_probe(
        model=model,
        train_windows=train_windows,
        val_windows=val_windows,
        epochs=args.probe_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight if memory_enabled else 0.0,
        device=device,
        seed=args.seed,
    )
    val_scores = score_windows(model=model, windows=val_windows, batch_size=args.batch_size, device=device)
    healthy_scores = score_windows(model=model, windows=test_healthy, batch_size=args.batch_size, device=device)
    fault_scores = score_windows(model=model, windows=test_fault, batch_size=args.batch_size, device=device)
    threshold = select_threshold(val_scores, "mean_plus_3std")
    metrics = evaluate_binary_scores(
        threshold=threshold["threshold"],
        test_healthy_errors=healthy_scores,
        test_fault_errors=fault_scores,
    )
    usage = collect_memory_usage(model=model, windows=val_windows, batch_size=args.batch_size, device=device)
    return {
        "memory_enabled": memory_enabled,
        "shrink_threshold": float(model.memory.shrink_threshold),
        "entropy_weight": float(args.entropy_weight if memory_enabled else 0.0),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "threshold": threshold,
        "metrics": metrics,
        "memory_usage": usage,
        "score_summary": {
            "val_healthy_mean": float(val_scores.mean()),
            "val_healthy_median": float(np.median(val_scores)),
            "test_healthy_median": float(np.median(healthy_scores)),
            "test_fault_median": float(np.median(fault_scores)),
            "val_healthy_modes": count_histogram_modes(val_scores),
        },
        "score_arrays": {
            "val_healthy": val_scores,
            "test_healthy": healthy_scores,
            "test_fault": fault_scores,
        },
    }


def check_memory_utilization(probe: dict[str, Any]) -> CheckResult:
    usage = probe["memory_usage"]
    memory_size = usage["memory_size"]
    fraction = usage["utilized_fraction"]
    top_share = usage["top_slot_share"]
    active = usage["mean_nonzero_slots_per_position"]
    # Sparsity must be real: if every slot survives shrinkage the addressing is
    # still near-uniform and the memory is a low-rank linear layer, which the
    # utilization and collapse bounds alone would not catch.
    sparse = active < 0.9 * memory_size
    passed = sparse and fraction > 0.10 and top_share < 0.5
    return CheckResult(
        number=8,
        name="Memory utilization after training",
        group="behavioural",
        expectation="shrinkage active (mean surviving slots < 0.9N), >10% of slots carry mean weight "
        "above 1/(10N), and no slot holds >50% of the mass",
        passed=passed,
        message=f"{active:.1f}/{memory_size} slots survive shrinkage per position; "
        f"{usage['utilized_slots']}/{memory_size} slots utilized ({fraction:.1%}); "
        f"top slot holds {top_share:.3%} of the addressing mass",
        details={
            key: value for key, value in usage.items() if key != "mean_usage_per_slot"
        },
    )


def check_entropy_trend(probe: dict[str, Any]) -> CheckResult:
    history = probe["history"]
    series = history["train_mem_loss"]
    first = series[0]
    last = series[-1]
    uniform = probe["memory_usage"]["uniform_entropy"]
    # Sharpening is measured against the uniform reference log(N) rather than
    # against epoch 1: most of the drop happens inside the first epoch, which is
    # already averaged into series[0], so a "strictly decreasing" reading would
    # miss addressing that sharpened immediately and then held.
    sharpened = last < 0.95 * uniform
    stable = last <= first + (0.05 * uniform)
    passed = sharpened and stable
    return CheckResult(
        number=9,
        name="Addressing entropy sharpens during training",
        group="behavioural",
        expectation="final train_mem_loss below 0.95 log(N) and not materially above the first epoch",
        passed=passed,
        message=f"train_mem_loss {first:.4f} -> {last:.4f} "
        f"(uniform reference log(N) = {uniform:.4f})",
        details={
            "train_mem_loss_first": first,
            "train_mem_loss_last": last,
            "sharpened_below_uniform": sharpened,
            "stable_across_training": stable,
            "train_mem_loss": series,
            "val_mem_loss": history["val_mem_loss"],
            "uniform_entropy": probe["memory_usage"]["uniform_entropy"],
        },
    )


def check_degenerate_control(full: dict[str, Any], degenerate: dict[str, Any]) -> CheckResult:
    full_auroc = full["metrics"]["auroc"]
    degenerate_auroc = degenerate["metrics"]["auroc"]
    delta = full_auroc - degenerate_auroc
    memory_size = degenerate["memory_usage"]["memory_size"]
    full_active = full["memory_usage"]["mean_nonzero_slots_per_position"]
    control_active = degenerate["memory_usage"]["mean_nonzero_slots_per_position"]
    # The control is only a control if it computes something different: with
    # lambda = 0 every slot must survive, and with the mechanism live in the full
    # variant far fewer must. Equal slot counts mean the two runs differ by seed
    # noise alone and the comparison says nothing about the memory.
    distinguishable = control_active > 0.99 * memory_size and full_active < 0.9 * control_active
    return CheckResult(
        number=10,
        name="Degenerate control (alpha = 0, lambda = 0)",
        group="behavioural",
        expectation="the control must differ mechanically (all slots survive at lambda = 0, far fewer with "
        "shrinkage live)",
        passed=distinguishable,
        message=f"AUROC full {full_auroc:.4f} vs control {degenerate_auroc:.4f} (delta {delta:+.4f}); "
        f"recall {full['metrics']['recall_fault']:.4f} vs {degenerate['metrics']['recall_fault']:.4f}; "
        f"surviving slots {full_active:.1f} vs {control_active:.1f}",
        details={
            "full_metrics": full["metrics"],
            "control_metrics": degenerate["metrics"],
            "auroc_delta": delta,
            "full_memory_entropy": full["history"]["train_mem_loss"][-1],
            "control_memory_entropy": degenerate["history"]["train_mem_loss"][-1],
            "full_mean_nonzero_slots": full_active,
            "control_mean_nonzero_slots": control_active,
            "control_is_mechanically_distinct": distinguishable,
            "memory_outperforms_control": delta > 0.0,
        },
    )


def check_score_distribution(probe: dict[str, Any]) -> CheckResult:
    summary = probe["score_summary"]
    modes = summary["val_healthy_modes"]
    shifted = summary["test_fault_median"] > summary["test_healthy_median"]
    return CheckResult(
        number=11,
        name="Score distribution sanity",
        group="behavioural",
        kind="finding",
        expectation="validation healthy scores unimodal and fault median above healthy median",
        passed=modes == 1 and shifted,
        message=f"val healthy modes = {modes}; median fault {summary['test_fault_median']:.4f} vs "
        f"median healthy {summary['test_healthy_median']:.4f}",
        details=summary,
    )


def report_attribution(full: dict[str, Any], degenerate: dict[str, Any]) -> CheckResult:
    """Which of the two variants detects better — a result about MemAE, not a bug.

    Kept separate from test 10 so that a contrary outcome is reported rather than
    swallowed by an implementation gate. Test 10 establishes that the comparison
    is meaningful; this says how it came out.
    """
    full_metrics = full["metrics"]
    control_metrics = degenerate["metrics"]
    delta = full_metrics["auroc"] - control_metrics["auroc"]
    return CheckResult(
        number=12,
        name="Memory attribution: live mechanism vs disabled control",
        group="behavioural",
        kind="finding",
        expectation="the memory-enabled model outperforms the control; otherwise any advantage MemAE shows "
        "is architectural and the paper must attribute it that way",
        passed=delta > 0.0,
        message=f"AUROC {full_metrics['auroc']:.4f} (memory live) vs {control_metrics['auroc']:.4f} (control), "
        f"delta {delta:+.4f}{' — within the reduced probe repeat spread, inconclusive' if abs(delta) < 0.03 else ''}; "
        f"recon {full['history']['val_time_loss'][-1]:.4f} vs "
        f"{degenerate['history']['val_time_loss'][-1]:.4f}",
        details={
            "auroc_delta": delta,
            "delta_within_default_probe_noise": abs(delta) < 0.03,
            "full_metrics": full_metrics,
            "control_metrics": control_metrics,
            "full_val_recon": full["history"]["val_time_loss"][-1],
            "control_val_recon": degenerate["history"]["val_time_loss"][-1],
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(*, results: list[CheckResult], payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MemAE correctness cross-check\n")
    lines.append(f"Device: `{payload['device']}` | seed {payload['seed']} | "
                 f"addressing `{payload['memae_settings']['addressing']}`, "
                 f"N = {payload['memae_settings']['memory_size']}, "
                 f"lambda = {payload['memae_settings']['shrink_threshold']}, "
                 f"alpha = {payload['memae_settings']['entropy_weight']}\n")
    assertions = [result for result in results if result.is_assertion]
    findings = [result for result in results if not result.is_assertion]
    passed = sum(1 for result in assertions if result.passed)
    lines.append(f"**{passed}/{len(assertions)} implementation checks passed.**\n")
    contrary = [result for result in findings if not result.passed]
    if contrary:
        lines.append(
            "Contrary findings — results about the method under this protocol, not implementation faults: "
            + ", ".join(f"check {result.number} ({result.name})" for result in contrary)
            + ".\n"
        )

    def render(rows: list[CheckResult], title: str) -> None:
        if not rows:
            return
        lines.append(f"### {title}\n")
        lines.append("| # | Check | Group | Status | Observation |")
        lines.append("|---|---|---|---|---|")
        for result in rows:
            label = "-" if result.number == 0 else str(result.number)
            message = result.message.replace("|", r"\|")
            lines.append(f"| {label} | {result.name} | {result.group} | {result.status} | {message} |")
        lines.append("")

    render(assertions, "Implementation checks")
    render(findings, "Findings")

    lines.append("## Expectations\n")
    for result in results:
        label = "capacity" if result.number == 0 else f"check {result.number}"
        kind = "assertion" if result.is_assertion else "finding"
        lines.append(f"- **{label} — {result.name}** ({kind}): {result.expectation}")
    lines.append("")

    if "probe" in payload:
        probe = payload["probe"]
        lines.append("## Probe run\n")
        lines.append(
            f"{probe['train_windows']} healthy train windows, {probe['val_windows']} validation, "
            f"{probe['test_healthy_windows']} healthy test, {probe['test_fault_windows']} fault test, "
            f"{probe['epochs']} epochs, batch {probe['batch_size']}.\n"
        )
        lines.append("| Variant | recon MSE (val) | addressing entropy | AUROC | recall | FAR |")
        lines.append("|---|---|---|---|---|---|")
        for key, label in (("full", "MemAE (memory live)"), ("control", "Control (lambda = 0, alpha = 0)")):
            variant = probe[key]
            lines.append(
                f"| {label} | {variant['history']['val_time_loss'][-1]:.4f} | "
                f"{variant['history']['val_mem_loss'][-1]:.4f} | "
                f"{variant['metrics']['auroc']:.4f} | {variant['metrics']['recall_fault']:.4f} | "
                f"{variant['metrics']['false_alarm_rate']:.4f} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- The probe run is a reduced-scale smoke check, not a headline result. Paper numbers come from the "
        "full three-seed run and `eval_paderborn_baselines_unified.py`."
    )
    lines.append(
        "- Check 12 is only readable at the default probe size. Reduced runs cannot separate the two "
        "variants from their own run-to-run spread."
    )
    lines.append(
        "- Deviations from the reference release (`donggong1/memae-anomaly-detection`) are recorded in "
        "`implementation_docs/memae_phase3_notes.md`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    require_torch()
    device = resolve_device(args.device)
    if args.shrink_threshold is None:
        args.shrink_threshold = default_shrink_threshold(args.memory_size)

    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    results: list[CheckResult] = [
        check_parameter_budget(args),
        check_renormalized_mass(args, generator),
        check_zero_threshold_identity(args, generator),
        check_threshold_monotonicity(args, generator),
        check_memory_gradient(args, generator, device),
        check_cosine_bounds(args, generator),
        check_addressing_shape(args, generator, device),
    ]

    payload: dict[str, Any] = {
        "script": "verify_memae.py",
        "device": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": args.seed,
        "memae_settings": {
            "memory_size": int(args.memory_size),
            "addressing": args.addressing,
            "shrink_threshold": float(args.shrink_threshold),
            "entropy_weight": float(args.entropy_weight),
            "base_channels": 24,
        },
        "processed_root": args.processed_root.as_posix(),
    }

    if not args.skip_behavioural:
        array_paths = resolve_paths(args.processed_root)
        train_windows = load_window_subset(array_paths.train_healthy, args.probe_train_windows)
        val_windows = load_window_subset(array_paths.val_healthy, args.probe_val_windows)
        test_healthy = load_window_subset(array_paths.test_healthy, args.probe_test_windows)
        test_fault = load_window_subset(array_paths.test_fault, args.probe_test_windows)

        overfit_result, _ = check_overfit_capacity(args, train_windows, device)
        results.append(overfit_result)

        probe_kwargs = {
            "train_windows": train_windows,
            "val_windows": val_windows,
            "test_healthy": test_healthy,
            "test_fault": test_fault,
            "device": device,
        }
        full = run_probe_variant(args, memory_enabled=True, **probe_kwargs)
        control = run_probe_variant(args, memory_enabled=False, **probe_kwargs)

        results.append(check_memory_utilization(full))
        results.append(check_entropy_trend(full))
        results.append(check_degenerate_control(full, control))
        results.append(check_score_distribution(full))
        results.append(report_attribution(full, control))

        for variant in (full, control):
            variant.pop("score_arrays", None)
            variant["memory_usage"].pop("mean_usage_per_slot", None)

        payload["probe"] = {
            "train_windows": int(train_windows.shape[0]),
            "val_windows": int(val_windows.shape[0]),
            "test_healthy_windows": int(test_healthy.shape[0]),
            "test_fault_windows": int(test_fault.shape[0]),
            "epochs": int(args.probe_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "threshold_rule": "mean_plus_3std",
            "full": full,
            "control": control,
        }

    assertion_results = [result for result in results if result.is_assertion]
    payload["checks"] = [result.to_payload() for result in results]
    payload["checks_passed"] = sum(1 for result in assertion_results if result.passed)
    payload["checks_total"] = len(assertion_results)
    payload["contrary_findings"] = [
        result.number for result in results if not result.is_assertion and not result.passed
    ]

    metrics_path = args.artifacts_root / "metrics" / "memae_verification_metrics.json"
    report_path = args.artifacts_root / "metrics" / "memae_verification_report.md"
    write_json(metrics_path, payload)
    write_text(report_path, build_report(results=results, payload=payload))

    for result in results:
        label = "capacity" if result.number == 0 else f"test {result.number:>2}"
        if result.is_assertion:
            marker = "PASS" if result.passed else "FAIL"
        else:
            marker = "AS EXPECTED" if result.passed else "CONTRARY"
        print(f"[{marker}] {label}: {result.name} - {result.message}")
    print(f"\n{payload['checks_passed']}/{payload['checks_total']} implementation checks passed.")
    if payload["contrary_findings"]:
        print(
            "Contrary findings (results about the method, not implementation faults): "
            + ", ".join(str(number) for number in payload["contrary_findings"])
        )
    print(f"Metrics: {metrics_path.as_posix()}")
    print(f"Report:  {report_path.as_posix()}")

    return 0 if payload["checks_passed"] == payload["checks_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
