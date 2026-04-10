from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - runtime dependency guard
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
ModuleBase = nn.Module if nn is not None else object


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray) -> None:
        self.windows = np.asarray(windows, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        window = self.windows[index]
        return torch.from_numpy(window).unsqueeze(0)


class CompactConvAutoencoder(ModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=7, padding=3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a compact 1D-CNN autoencoder baseline on processed CWRU windows.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-rule",
        choices=("mean_plus_3std",),
        default="mean_plus_3std",
        help="Threshold selection rule applied on healthy validation reconstruction errors.",
    )
    return parser.parse_args()


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed in the current environment. "
            "Install torch before running scripts/train_ae_baseline.py."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(processed_root: Path) -> ArrayPaths:
    return ArrayPaths(
        train_healthy=processed_root / "train" / "healthy_windows.npy",
        val_healthy=processed_root / "val" / "healthy_windows.npy",
        test_healthy=processed_root / "test" / "healthy_windows.npy",
        test_fault=processed_root / "test" / "fault_windows.npy",
        fault_labels=processed_root / "test" / "fault_labels.npy",
    )


def ensure_required_files(paths: ArrayPaths, metadata_root: Path) -> None:
    required = [
        paths.train_healthy,
        paths.val_healthy,
        paths.test_healthy,
        paths.test_fault,
        paths.fault_labels,
        metadata_root / "normalization_stats.json",
        metadata_root / "preprocessing_config.json",
        metadata_root / "fault_label_map.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Required inputs are missing: {formatted}")


def load_array(path: Path, expected_width: int = 2048) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 1:
        return array
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"Unexpected array shape for {path.as_posix()}: {array.shape}")
    return np.asarray(array, dtype=np.float32)


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = WindowDataset(windows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def average_epoch_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            total_loss += float(loss.item()) * batch.size(0)
            total_samples += int(batch.size(0))
    return total_loss / max(total_samples, 1)


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="mean")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch.size(0)
            total_samples += int(batch.size(0))

        train_loss = total_loss / max(total_samples, 1)
        val_loss = average_epoch_loss(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} "
            f"- train_loss={train_loss:.6f} "
            f"- val_loss={val_loss:.6f}"
        )

    return history


def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            per_window_mse = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            errors.append(per_window_mse.cpu().numpy())
    if not errors:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(errors, axis=0).astype(np.float32, copy=False)


def select_threshold(errors: np.ndarray, rule: str) -> dict[str, Any]:
    if errors.size == 0:
        raise RuntimeError("Validation healthy errors are empty; cannot select an anomaly threshold.")

    if rule == "mean_plus_3std":
        mean = float(errors.mean())
        std = float(errors.std())
        threshold = mean + (3.0 * std)
        return {
            "rule": rule,
            "threshold": float(threshold),
            "validation_error_mean": mean,
            "validation_error_std": std,
            "fit_split": "val_healthy",
        }

    raise ValueError(f"Unsupported threshold rule: {rule}")


def build_score_rows(
    *,
    split: str,
    true_label: int,
    errors: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "true_label": true_label,
            "reconstruction_error": float(error),
            "predicted_anomaly": int(error >= threshold),
        }
        for error in errors
    ]


def evaluate_scores(
    *,
    threshold: float,
    test_healthy_errors: np.ndarray,
    test_fault_errors: np.ndarray,
) -> dict[str, Any]:
    y_true = np.concatenate(
        [
            np.zeros(test_healthy_errors.shape[0], dtype=np.int64),
            np.ones(test_fault_errors.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    scores = np.concatenate([test_healthy_errors, test_fault_errors], axis=0)
    predictions = (scores >= threshold).astype(np.int64)

    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_fault": float(recall_score(y_true, predictions, zero_division=0)),
        "false_alarm_rate": float(predictions[: test_healthy_errors.shape[0]].mean()),
        "num_test_healthy": int(test_healthy_errors.shape[0]),
        "num_test_fault": int(test_fault_errors.shape[0]),
        "num_predicted_anomalies": int(predictions.sum()),
        "num_true_anomalies": int(y_true.sum()),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def plot_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2.0)
    plt.plot(epochs, history["val_loss"], label="Val", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AE Baseline Training History")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_histogram(
    *,
    series: list[tuple[str, np.ndarray]],
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, values in series:
        plt.hist(values, bins=40, alpha=0.55, label=label, density=False)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Window Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "true_label", "reconstruction_error", "predicted_anomaly"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    *,
    history: dict[str, list[float]],
    threshold_meta: dict[str, Any],
    metrics: dict[str, Any],
    parameter_total: int,
    batch_shape_check: dict[str, Any],
    paths: dict[str, Path],
    device: torch.device,
) -> str:
    lines = [
        "# AE Baseline Report",
        "",
        "## Training Summary",
        f"- Device: `{device}`",
        f"- Model parameter count: `{parameter_total}`",
        f"- Final train loss: `{history['train_loss'][-1]:.6f}`",
        f"- Final val loss: `{history['val_loss'][-1]:.6f}`",
        "",
        "## Threshold",
        f"- Rule: `{threshold_meta['rule']}`",
        f"- Threshold: `{threshold_meta['threshold']:.6f}`",
        f"- Val error mean: `{threshold_meta['validation_error_mean']:.6f}`",
        f"- Val error std: `{threshold_meta['validation_error_std']:.6f}`",
        "",
        "## Evaluation",
        f"- AUROC: `{metrics['auroc']:.6f}`",
        f"- AUPRC: `{metrics['auprc']:.6f}`",
        f"- F1: `{metrics['f1']:.6f}`",
        f"- Precision: `{metrics['precision']:.6f}`",
        f"- Recall on fault windows: `{metrics['recall_fault']:.6f}`",
        f"- False alarm rate on healthy test windows: `{metrics['false_alarm_rate']:.6f}`",
        "",
        "## Shape Check",
        f"- Input batch shape: `{tuple(batch_shape_check['input'])}`",
        f"- Output batch shape: `{tuple(batch_shape_check['output'])}`",
        "",
        "## Saved Artifacts",
        f"- Model: `{paths['model'].as_posix()}`",
        f"- History JSON: `{paths['history'].as_posix()}`",
        f"- Threshold JSON: `{paths['threshold'].as_posix()}`",
        f"- Metrics JSON: `{paths['metrics'].as_posix()}`",
        f"- Scores CSV: `{paths['scores'].as_posix()}`",
        f"- Loss curve: `{paths['loss_plot'].as_posix()}`",
        f"- Val histogram: `{paths['val_hist'].as_posix()}`",
        f"- Test histogram: `{paths['test_hist'].as_posix()}`",
        "",
    ]
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

    train_windows = load_array(array_paths.train_healthy)
    val_windows = load_array(array_paths.val_healthy)
    test_healthy_windows = load_array(array_paths.test_healthy)
    test_fault_windows = load_array(array_paths.test_fault)
    fault_labels = load_array(array_paths.fault_labels)

    if train_windows.size == 0 or val_windows.size == 0:
        raise RuntimeError("Healthy train and val arrays must both be non-empty.")
    if test_healthy_windows.size == 0 or test_fault_windows.size == 0:
        raise RuntimeError("Healthy test and fault test arrays must both be non-empty for evaluation.")
    if fault_labels.ndim != 1 or fault_labels.shape[0] != test_fault_windows.shape[0]:
        raise RuntimeError(
            "fault_labels.npy must be a 1D array with one entry per fault test window."
        )

    train_loader = make_loader(train_windows, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_windows, batch_size=args.batch_size, shuffle=False)
    test_healthy_loader = make_loader(test_healthy_windows, batch_size=args.batch_size, shuffle=False)
    test_fault_loader = make_loader(test_fault_windows, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactConvAutoencoder().to(device)
    parameter_total = parameter_count(model)

    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_output = model(sample_batch.to(device)).cpu()
    batch_shape_check = {
        "input": list(sample_batch.shape),
        "output": list(sample_output.shape),
    }

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    val_errors = compute_reconstruction_errors(model, val_loader, device)
    test_healthy_errors = compute_reconstruction_errors(model, test_healthy_loader, device)
    test_fault_errors = compute_reconstruction_errors(model, test_fault_loader, device)
    threshold_meta = select_threshold(val_errors, args.threshold_rule)
    threshold = float(threshold_meta["threshold"])
    metrics = evaluate_scores(
        threshold=threshold,
        test_healthy_errors=test_healthy_errors,
        test_fault_errors=test_fault_errors,
    )

    artifact_paths = {
        "model": model_dir / "cwru_ae_baseline.pt",
        "history": metrics_dir / "cwru_ae_history.json",
        "threshold": metrics_dir / "cwru_ae_threshold.json",
        "metrics": metrics_dir / "cwru_ae_metrics.json",
        "report": metrics_dir / "cwru_ae_report.md",
        "scores": metrics_dir / "cwru_ae_scores.csv",
        "loss_plot": plots_dir / "cwru_ae_loss_curve.png",
        "val_hist": plots_dir / "cwru_ae_val_error_hist.png",
        "test_hist": plots_dir / "cwru_ae_test_error_hist.png",
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "CompactConvAutoencoder",
        "input_shape": [1, 2048],
        "parameter_count": parameter_total,
        "history": history,
        "threshold": threshold_meta,
        "metrics": metrics,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    torch.save(checkpoint, artifact_paths["model"])

    history_payload = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
    }
    write_json(artifact_paths["history"], history_payload)

    threshold_payload = {
        **threshold_meta,
        "num_val_windows": int(val_errors.shape[0]),
    }
    write_json(artifact_paths["threshold"], threshold_payload)

    metrics_payload = {
        **metrics,
        "parameter_count": parameter_total,
        "batch_shape_check": batch_shape_check,
        "threshold": threshold,
        "threshold_rule": threshold_meta["rule"],
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "device": str(device),
    }
    write_json(artifact_paths["metrics"], metrics_payload)

    score_rows: list[dict[str, Any]] = []
    score_rows.extend(
        build_score_rows(
            split="val_healthy",
            true_label=0,
            errors=val_errors,
            threshold=threshold,
        )
    )
    score_rows.extend(
        build_score_rows(
            split="test_healthy",
            true_label=0,
            errors=test_healthy_errors,
            threshold=threshold,
        )
    )
    score_rows.extend(
        build_score_rows(
            split="test_fault",
            true_label=1,
            errors=test_fault_errors,
            threshold=threshold,
        )
    )
    write_scores_csv(artifact_paths["scores"], score_rows)

    plot_loss_curve(history, artifact_paths["loss_plot"])
    plot_histogram(
        series=[("Val healthy", val_errors)],
        title="Validation Healthy Reconstruction Errors",
        path=artifact_paths["val_hist"],
    )
    plot_histogram(
        series=[
            ("Test healthy", test_healthy_errors),
            ("Test fault", test_fault_errors),
        ],
        title="Test Reconstruction Errors",
        path=artifact_paths["test_hist"],
    )

    report_text = build_report(
        history=history,
        threshold_meta=threshold_meta,
        metrics=metrics,
        parameter_total=parameter_total,
        batch_shape_check=batch_shape_check,
        paths=artifact_paths,
        device=device,
    )
    artifact_paths["report"].write_text(report_text, encoding="utf-8")

    print("\nAE Baseline Summary")
    print(f"  final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  threshold ({threshold_meta['rule']}): {threshold:.6f}")
    print(f"  auroc: {metrics['auroc']:.6f}")
    print(f"  auprc: {metrics['auprc']:.6f}")
    print(f"  f1: {metrics['f1']:.6f}")
    print(f"  recall_fault: {metrics['recall_fault']:.6f}")
    print(f"  false_alarm_rate: {metrics['false_alarm_rate']:.6f}")
    print(f"  parameter_count: {parameter_total}")
    print(f"  input batch shape: {tuple(batch_shape_check['input'])}")
    print(f"  output batch shape: {tuple(batch_shape_check['output'])}")
    print(f"  saved model: {artifact_paths['model'].as_posix()}")
    print(f"  saved metrics: {artifact_paths['metrics'].as_posix()}")
    print(f"  saved report: {artifact_paths['report'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
