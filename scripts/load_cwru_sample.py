from __future__ import annotations

from pathlib import Path

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - dependency guard
    loadmat = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cwru"
SAMPLE_PATHS = {
    "normal": RAW_ROOT / "normal" / "normal_0.mat",
    "inner_race": RAW_ROOT / "ir_007" / "ir007_0.mat",
    "ball": RAW_ROOT / "ball_007" / "ball007_0.mat",
    "outer_race_6": RAW_ROOT / "or_007_6" / "or007_6_0.mat",
}


def extract_primary_signal(data: dict[str, object]) -> tuple[str, np.ndarray]:
    keys = [key for key in data.keys() if not key.startswith("__")]
    priority_suffixes = ("_DE_time", "_FE_time", "_BA_time")
    for suffix in priority_suffixes:
        for key in keys:
            if key.endswith(suffix):
                return key, np.asarray(data[key]).squeeze()

    for key in keys:
        value = data[key]
        if hasattr(value, "shape"):
            return key, np.asarray(value).squeeze()

    raise ValueError("No array-like signal found in MAT file.")


def print_signal_stats(label: str, key: str, signal: np.ndarray) -> None:
    print(f"\n[{label}]")
    print(f"  signal key: {key}")
    print(f"  length: {signal.size}")
    print(f"  dtype: {signal.dtype}")
    print(f"  mean: {signal.mean():.6f}")
    print(f"  std: {signal.std():.6f}")
    print(f"  min: {signal.min():.6f}")
    print(f"  max: {signal.max():.6f}")


def plot_signals(signals: dict[str, np.ndarray], segment_length: int = 2048) -> None:
    if plt is None:
        print("\nmatplotlib is not available; skipping plots.")
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print("\nmatplotlib is available, but the current backend is non-interactive; skipping plot display.")
        return

    figure, axes = plt.subplots(len(signals), 1, figsize=(10, 8), sharex=False)
    if len(signals) == 1:
        axes = [axes]

    for axis, (label, signal) in zip(axes, signals.items()):
        segment = signal[: min(segment_length, signal.size)]
        axis.plot(segment, linewidth=1.0)
        axis.set_title(label)
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index")
    figure.suptitle("CWRU Sample Signals")
    figure.tight_layout()
    plt.show()


def main() -> int:
    if loadmat is None:
        print("scipy is not available; install scipy to load CWRU MAT files.")
        return 0

    loaded_signals: dict[str, np.ndarray] = {}
    for label, path in SAMPLE_PATHS.items():
        if not path.exists():
            print(f"Missing sample file for {label}: {path.relative_to(PROJECT_ROOT).as_posix()}")
            continue

        try:
            data = loadmat(path)
            key, signal = extract_primary_signal(data)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Failed to load {path.relative_to(PROJECT_ROOT).as_posix()}: {exc}")
            continue

        print_signal_stats(label, key, signal)
        loaded_signals[label] = signal

    if not loaded_signals:
        print("\nNo sample signals were loaded.")
        return 0

    plot_signals(loaded_signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
