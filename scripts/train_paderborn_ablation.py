from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("PyTorch is required to run the Paderborn ablation harness.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "paderborn_ablation"

VARIANT_ORDER = (
    "compact_ae",
    "dilated_ae",
    "res_ae",
    "resdilated_time",
    "resdilated_full",
)

DEFAULT_BASE_CHANNELS = 16
DEFAULT_DROPOUT = 0.05
DEFAULT_FREQ_LOSS_WEIGHT = 0.10


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


@dataclass(frozen=True)
class VariantConfig:
    name: str
    model_family: str
    use_residual: bool
    use_dilation: bool
    frequency_loss_weight: float
    description: str


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    history_json: Path
    run_config_json: Path
    val_healthy_scores_npy: Path
    test_healthy_scores_npy: Path
    test_fault_scores_npy: Path
    sanity_metrics_json: Path


VARIANT_CONFIGS = {
    "compact_ae": VariantConfig(
        name="compact_ae",
        model_family="compact",
        use_residual=False,
        use_dilation=False,
        frequency_loss_weight=0.0,
        description="Compact convolutional AE baseline; no local residuals, no dilation, no frequency loss.",
    ),
    "dilated_ae": VariantConfig(
        name="dilated_ae",
        model_family="configurable_resdilated",
        use_residual=False,
        use_dilation=True,
        frequency_loss_weight=DEFAULT_FREQ_LOSS_WEIGHT,
        description="ResDilatedAE topology with dilated blocks but no local residual additions.",
    ),
    "res_ae": VariantConfig(
        name="res_ae",
        model_family="configurable_resdilated",
        use_residual=True,
        use_dilation=False,
        frequency_loss_weight=DEFAULT_FREQ_LOSS_WEIGHT,
        description="ResDilatedAE topology with local residual additions but all dilations fixed to 1.",
    ),
    "resdilated_time": VariantConfig(
        name="resdilated_time",
        model_family="configurable_resdilated",
        use_residual=True,
        use_dilation=True,
        frequency_loss_weight=0.0,
        description="Final ResDilatedAE topology trained with time-domain reconstruction loss only.",
    ),
    "resdilated_full": VariantConfig(
        name="resdilated_full",
        model_family="configurable_resdilated",
        use_residual=True,
        use_dilation=True,
        frequency_loss_weight=DEFAULT_FREQ_LOSS_WEIGHT,
        description="Final ResDilatedAE topology trained with time-domain plus frequency-aware loss.",
    ),
}


class MemmapWindowDataset(Dataset):
    def __init__(self, path: Path, *, limit: int = 0) -> None:
        self.path = path
        self.windows = np.load(path, mmap_mode="r")
        if self.windows.ndim != 2:
            raise ValueError(f"Expected a 2D window array at {path.as_posix()}, got {self.windows.shape}.")
        self.full_size = int(self.windows.shape[0])
        self.window_size = int(self.windows.shape[1])
        self.effective_size = min(int(limit), self.full_size) if int(limit) > 0 else self.full_size

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.effective_size:
            raise IndexError(index)
        window = np.array(self.windows[index], dtype=np.float32, copy=True)
        return torch.from_numpy(window).unsqueeze(0)


class CompactConvAutoencoder(nn.Module):
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
        return self.decoder(self.encoder(inputs))


def choose_group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConfigurableResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        use_residual: bool,
        kernel_size: int = 5,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.use_residual = use_residual
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(choose_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(choose_group_count(out_channels), out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.norm1(outputs)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)
        if self.use_residual:
            outputs = outputs + self.residual(inputs)
        return self.act(outputs)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConfigurableResDilatedAE(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        dropout: float = DEFAULT_DROPOUT,
        use_residual: bool,
        use_dilation: bool,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4
        dilations = {
            "enc1": 1,
            "enc2": 2 if use_dilation else 1,
            "enc3": 4 if use_dilation else 1,
            "bottleneck1": 8 if use_dilation else 1,
            "bottleneck2": 16 if use_dilation else 1,
            "dec3": 4 if use_dilation else 1,
            "dec2": 2 if use_dilation else 1,
            "dec1": 1,
        }

        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.GroupNorm(choose_group_count(c1), c1),
            nn.SiLU(),
        )
        self.enc1 = ConfigurableResidualBlock(
            c1,
            c1,
            dilation=dilations["enc1"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.down1 = DownsampleBlock(c1, c2)
        self.enc2 = ConfigurableResidualBlock(
            c2,
            c2,
            dilation=dilations["enc2"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.down2 = DownsampleBlock(c2, c3)
        self.enc3 = ConfigurableResidualBlock(
            c3,
            c3,
            dilation=dilations["enc3"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.down3 = DownsampleBlock(c3, c4)
        self.bottleneck = nn.Sequential(
            ConfigurableResidualBlock(
                c4,
                c4,
                dilation=dilations["bottleneck1"],
                use_residual=use_residual,
                dropout=dropout,
            ),
            ConfigurableResidualBlock(
                c4,
                c4,
                dilation=dilations["bottleneck2"],
                use_residual=use_residual,
                dropout=dropout,
            ),
        )
        self.up3 = UpsampleBlock(c4, c3)
        self.dec3 = ConfigurableResidualBlock(
            c3 + c3,
            c3,
            dilation=dilations["dec3"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.up2 = UpsampleBlock(c3, c2)
        self.dec2 = ConfigurableResidualBlock(
            c2 + c2,
            c2,
            dilation=dilations["dec2"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.up1 = UpsampleBlock(c2, c1)
        self.dec1 = ConfigurableResidualBlock(
            c1 + c1,
            c1,
            dilation=dilations["dec1"],
            use_residual=use_residual,
            dropout=dropout,
        )
        self.head = nn.Conv1d(c1, 1, kernel_size=7, padding=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(inputs)
        x1 = self.enc1(x0)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        latent = self.bottleneck(self.down3(x3))
        y3 = self.up3(latent)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))
        y2 = self.up2(y3)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.head(y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train controlled Paderborn architectural ablation variants without changing the data protocol.",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size-cuda", type=int, default=256)
    parser.add_argument("--batch-size-cpu", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size_cuda <= 0 or args.batch_size_cpu <= 0:
        parser.error("--batch-size-cuda and --batch-size-cpu must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    for name in ("train_subset", "val_subset", "test_subset"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    return args


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_artifacts_root(path: Path) -> Path:
    resolved = path.resolve()
    artifacts_root = ARTIFACTS_ROOT.resolve()
    if not path_is_relative_to(resolved, artifacts_root):
        raise ValueError(
            "Ablation artifacts must stay under the repository artifacts directory: "
            f"{artifacts_root.as_posix()}"
        )
    if not resolved.name.startswith("paderborn_ablation"):
        raise ValueError(
            "Ablation artifact root must be named paderborn_ablation* so it cannot overlap existing final outputs: "
            f"{resolved.as_posix()}"
        )
    return resolved


def resolve_array_paths(processed_root: Path) -> ArrayPaths:
    return ArrayPaths(
        train_healthy=processed_root / "train" / "healthy_windows.npy",
        val_healthy=processed_root / "val" / "healthy_windows.npy",
        test_healthy=processed_root / "test" / "healthy_windows.npy",
        test_fault=processed_root / "test" / "fault_windows.npy",
        fault_labels=processed_root / "test" / "fault_labels.npy",
    )


def ensure_required_arrays(paths: ArrayPaths) -> None:
    missing = [path for path in asdict(paths).values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required Paderborn arrays: " + ", ".join(path.as_posix() for path in missing))


def read_array_shape(path: Path) -> tuple[int, ...]:
    array = np.load(path, mmap_mode="r")
    return tuple(int(value) for value in array.shape)


def build_dataset_sizes(paths: ArrayPaths, train_subset: int, val_subset: int, test_subset: int) -> dict[str, Any]:
    shapes = {
        "train_healthy": read_array_shape(paths.train_healthy),
        "val_healthy": read_array_shape(paths.val_healthy),
        "test_healthy": read_array_shape(paths.test_healthy),
        "test_fault": read_array_shape(paths.test_fault),
        "fault_labels": read_array_shape(paths.fault_labels),
    }
    if shapes["train_healthy"][1] != shapes["val_healthy"][1]:
        raise RuntimeError("Train and validation window widths do not match.")
    if shapes["test_fault"][0] != shapes["fault_labels"][0]:
        raise RuntimeError("fault_labels.npy must contain one label per fault test window.")
    return {
        "full": {name: list(shape) for name, shape in shapes.items()},
        "effective": {
            "train_healthy": min(train_subset, shapes["train_healthy"][0]) if train_subset > 0 else shapes["train_healthy"][0],
            "val_healthy": min(val_subset, shapes["val_healthy"][0]) if val_subset > 0 else shapes["val_healthy"][0],
            "test_healthy": min(test_subset, shapes["test_healthy"][0]) if test_subset > 0 else shapes["test_healthy"][0],
            "test_fault": min(test_subset, shapes["test_fault"][0]) if test_subset > 0 else shapes["test_fault"][0],
        },
        "window_size": int(shapes["train_healthy"][1]),
    }


def build_run_paths(artifacts_root: Path, variant: str, seed: int) -> RunPaths:
    run_dir = artifacts_root / variant / f"seed_{seed}"
    return RunPaths(
        run_dir=run_dir,
        best_checkpoint=run_dir / "best.pt",
        latest_checkpoint=run_dir / "latest.pt",
        history_json=run_dir / "history.json",
        run_config_json=run_dir / "run_config.json",
        val_healthy_scores_npy=run_dir / "val_healthy_scores.npy",
        test_healthy_scores_npy=run_dir / "test_healthy_scores.npy",
        test_fault_scores_npy=run_dir / "test_fault_scores.npy",
        sanity_metrics_json=run_dir / "sanity_metrics.json",
    )


def prepare_run_dir(run_dir: Path, artifacts_root: Path, *, overwrite: bool) -> None:
    resolved_root = artifacts_root.resolve()
    resolved_run = run_dir.resolve()
    if not path_is_relative_to(resolved_run, resolved_root):
        raise ValueError(f"Refusing to prepare run directory outside artifact root: {resolved_run.as_posix()}")
    relative_parts = resolved_run.relative_to(resolved_root).parts
    if len(relative_parts) != 2 or not relative_parts[1].startswith("seed_"):
        raise ValueError(f"Unexpected ablation run layout: {resolved_run.as_posix()}")
    if resolved_run.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {resolved_run.as_posix()}. Use --overwrite to replace it."
            )
        shutil.rmtree(resolved_run)
    resolved_run.mkdir(parents=True, exist_ok=False)


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


def save_torch_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_frequency_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction_fft = torch.fft.rfft(reconstruction.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    reconstruction_mag = torch.log1p(torch.abs(reconstruction_fft))
    target_mag = torch.log1p(torch.abs(target_fft))
    return F.mse_loss(reconstruction_mag, target_mag, reduction="mean")


def build_model(variant_config: VariantConfig) -> nn.Module:
    if variant_config.model_family == "compact":
        return CompactConvAutoencoder()
    return ConfigurableResDilatedAE(
        base_channels=DEFAULT_BASE_CHANNELS,
        dropout=DEFAULT_DROPOUT,
        use_residual=variant_config.use_residual,
        use_dilation=variant_config.use_dilation,
    )


def validate_model_shape(model: nn.Module, window_size: int) -> list[int]:
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        probe = torch.zeros(2, 1, window_size)
        output = model(probe)
    if tuple(output.shape) != tuple(probe.shape):
        raise RuntimeError(f"Model output shape {tuple(output.shape)} does not match input shape {tuple(probe.shape)}.")
    if model_was_training:
        model.train()
    return list(output.shape)


def make_loader(path: Path, *, batch_size: int, shuffle: bool, limit: int, seed: int) -> DataLoader:
    dataset = MemmapWindowDataset(path, limit=limit)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def build_loaders(paths: ArrayPaths, *, batch_size: int, train_subset: int, val_subset: int, test_subset: int, seed: int) -> dict[str, DataLoader]:
    return {
        "train": make_loader(paths.train_healthy, batch_size=batch_size, shuffle=True, limit=train_subset, seed=seed),
        "val": make_loader(paths.val_healthy, batch_size=batch_size, shuffle=False, limit=val_subset, seed=seed),
        "test_healthy": make_loader(
            paths.test_healthy,
            batch_size=batch_size,
            shuffle=False,
            limit=test_subset,
            seed=seed,
        ),
        "test_fault": make_loader(paths.test_fault, batch_size=batch_size, shuffle=False, limit=test_subset, seed=seed),
    }


def empty_history() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_total_loss": [],
        "train_time_loss": [],
        "train_freq_loss": [],
        "val_total_loss": [],
        "val_time_loss": [],
        "val_freq_loss": [],
        "lr": [],
    }


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    frequency_loss_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_time = 0.0
    total_freq = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device):
            reconstruction = model(batch)
            time_loss = F.mse_loss(reconstruction.float(), batch.float(), reduction="mean")
            freq_loss = (
                compute_frequency_loss(reconstruction, batch)
                if frequency_loss_weight > 0
                else torch.zeros((), device=device)
            )
            loss = time_loss + (frequency_loss_weight * freq_loss)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_time += float(time_loss.item()) * batch_size
        total_freq += float(freq_loss.item()) * batch_size
        total_samples += batch_size

    divisor = max(total_samples, 1)
    return {
        "total_loss": total_loss / divisor,
        "time_loss": total_time / divisor,
        "freq_loss": total_freq / divisor,
    }


def evaluate_loss(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    frequency_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    total_freq = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            reconstruction = model(batch)
            time_loss = F.mse_loss(reconstruction.float(), batch.float(), reduction="mean")
            freq_loss = (
                compute_frequency_loss(reconstruction, batch)
                if frequency_loss_weight > 0
                else torch.zeros((), device=device)
            )
            loss = time_loss + (frequency_loss_weight * freq_loss)
            batch_size = int(batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_time += float(time_loss.item()) * batch_size
            total_freq += float(freq_loss.item()) * batch_size
            total_samples += batch_size
    divisor = max(total_samples, 1)
    return {
        "total_loss": total_loss / divisor,
        "time_loss": total_time / divisor,
        "freq_loss": total_freq / divisor,
    }


def compute_reconstruction_scores(*, model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            reconstruction = model(batch)
            per_window_mse = torch.mean((reconstruction.float() - batch.float()) ** 2, dim=(1, 2))
            scores.append(per_window_mse.cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def summarize_scores(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_sanity_metrics(
    *,
    variant_config: VariantConfig,
    seed: int,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_loss: float,
    val_scores: np.ndarray,
    test_healthy_scores: np.ndarray,
    test_fault_scores: np.ndarray,
    run_paths: RunPaths,
) -> dict[str, Any]:
    y_true = np.concatenate(
        [
            np.zeros(test_healthy_scores.shape[0], dtype=np.int64),
            np.ones(test_fault_scores.shape[0], dtype=np.int64),
        ]
    )
    scores = np.concatenate([test_healthy_scores, test_fault_scores]).astype(np.float64, copy=False)
    return {
        "study": "paderborn_architectural_ablation_chunk2_sanity",
        "variant": variant_config.name,
        "seed": int(seed),
        "ranking_metrics": {
            "auroc": float(roc_auc_score(y_true, scores)),
            "auprc": float(average_precision_score(y_true, scores)),
        },
        "score_summary": {
            "val_healthy": summarize_scores(val_scores),
            "test_healthy": summarize_scores(test_healthy_scores),
            "test_fault": summarize_scores(test_fault_scores),
        },
        "training_summary": {
            "epochs_ran": len(history["epoch"]),
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "final_train_total_loss": float(history["train_total_loss"][-1]),
            "final_val_total_loss": float(history["val_total_loss"][-1]),
        },
        "artifacts": {
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
            "history_json": run_paths.history_json.as_posix(),
            "run_config_json": run_paths.run_config_json.as_posix(),
            "val_healthy_scores_npy": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores_npy": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores_npy": run_paths.test_fault_scores_npy.as_posix(),
        },
    }


def build_run_config(
    *,
    variant_config: VariantConfig,
    seed: int,
    model: nn.Module,
    sample_output_shape: list[int],
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    array_paths: ArrayPaths,
    dataset_sizes: dict[str, Any],
    artifacts_root: Path,
    run_paths: RunPaths,
) -> dict[str, Any]:
    cuda_info = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return {
        "study": "paderborn_architectural_ablation_chunk2",
        "variant": variant_config.name,
        "seed": int(seed),
        "description": variant_config.description,
        "model_hyperparameters": {
            "model_family": variant_config.model_family,
            "use_residual": bool(variant_config.use_residual),
            "use_dilation": bool(variant_config.use_dilation),
            "base_channels": DEFAULT_BASE_CHANNELS if variant_config.model_family != "compact" else None,
            "dropout": DEFAULT_DROPOUT if variant_config.model_family != "compact" else 0.0,
            "frequency_loss_weight": float(variant_config.frequency_loss_weight),
            "parameter_count": int(parameter_count(model)),
            "input_shape": [1, int(dataset_sizes["window_size"])],
            "sample_output_shape": sample_output_shape,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "scheduler": "CosineAnnealingLR",
        },
        "training": {
            "batch_size": int(batch_size),
            "epochs_requested": int(args.epochs),
            "early_stopping": {
                "enabled": True,
                "monitor": "val_total_loss",
                "patience": int(args.patience),
                "min_delta": 1e-6,
            },
            "frequency_loss_weight": float(variant_config.frequency_loss_weight),
            "train_subset": int(args.train_subset),
            "val_subset": int(args.val_subset),
            "test_subset": int(args.test_subset),
            "amp_enabled": bool(device.type == "cuda"),
        },
        "reproducibility": {
            "python_random_seed": int(seed),
            "numpy_seed": int(seed),
            "torch_seed": int(seed),
            "torch_cuda_manual_seed_all": bool(torch.cuda.is_available()),
            "torch_backends_cudnn_benchmark": False,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda": cuda_info,
        },
        "data": {
            "protocol": "existing_processed_paderborn_healthy_only",
            "paths": {key: Path(value).as_posix() for key, value in asdict(array_paths).items()},
            "dataset_sizes": dataset_sizes,
        },
        "artifacts": {
            "artifacts_root": artifacts_root.as_posix(),
            "run_dir": run_paths.run_dir.as_posix(),
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
            "history_json": run_paths.history_json.as_posix(),
            "run_config_json": run_paths.run_config_json.as_posix(),
            "val_healthy_scores_npy": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores_npy": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores_npy": run_paths.test_fault_scores_npy.as_posix(),
            "sanity_metrics_json": run_paths.sanity_metrics_json.as_posix(),
        },
    }


def save_latest_checkpoint(
    *,
    run_paths: RunPaths,
    variant_config: VariantConfig,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_loss: float,
    stale_epochs: int,
    epoch: int,
    run_config: dict[str, Any],
) -> None:
    save_torch_payload(
        run_paths.latest_checkpoint,
        {
            "format_version": 1,
            "study": "paderborn_architectural_ablation_chunk2",
            "variant": variant_config.name,
            "seed": int(seed),
            "epoch": int(epoch),
            "model_state_dict": clone_model_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "stale_epochs": int(stale_epochs),
            "run_config": run_config,
            "saved_at_unix": time.time(),
        },
    )


def save_best_checkpoint(
    *,
    run_paths: RunPaths,
    variant_config: VariantConfig,
    seed: int,
    model: nn.Module,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_loss: float,
    run_config: dict[str, Any],
) -> None:
    save_torch_payload(
        run_paths.best_checkpoint,
        {
            "format_version": 1,
            "study": "paderborn_architectural_ablation_chunk2",
            "variant": variant_config.name,
            "seed": int(seed),
            "state_dict": clone_model_state(model),
            "history": history,
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "run_config": run_config,
            "saved_at_unix": time.time(),
        },
    )


def run_variant_seed(
    *,
    variant_config: VariantConfig,
    seed: int,
    args: argparse.Namespace,
    artifacts_root: Path,
    array_paths: ArrayPaths,
    dataset_sizes: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> None:
    set_global_seed(seed)
    run_paths = build_run_paths(artifacts_root, variant_config.name, seed)
    prepare_run_dir(run_paths.run_dir, artifacts_root, overwrite=args.overwrite)
    loaders = build_loaders(
        array_paths,
        batch_size=batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        seed=seed,
    )

    model = build_model(variant_config)
    sample_output_shape = validate_model_shape(model, int(dataset_sizes["window_size"]))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_grad_scaler(device)
    history = empty_history()
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()

    run_config = build_run_config(
        variant_config=variant_config,
        seed=seed,
        model=model,
        sample_output_shape=sample_output_shape,
        args=args,
        device=device,
        batch_size=batch_size,
        array_paths=array_paths,
        dataset_sizes=dataset_sizes,
        artifacts_root=artifacts_root,
        run_paths=run_paths,
    )
    write_json(run_paths.run_config_json, run_config)

    print(f"\n[{variant_config.name} seed {seed}] training in {run_paths.run_dir.as_posix()}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_summary = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            frequency_loss_weight=variant_config.frequency_loss_weight,
        )
        val_summary = evaluate_loss(
            model=model,
            loader=loaders["val"],
            device=device,
            frequency_loss_weight=variant_config.frequency_loss_weight,
        )
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_total_loss"].append(train_summary["total_loss"])
        history["train_time_loss"].append(train_summary["time_loss"])
        history["train_freq_loss"].append(train_summary["freq_loss"])
        history["val_total_loss"].append(val_summary["total_loss"])
        history["val_time_loss"].append(val_summary["time_loss"])
        history["val_freq_loss"].append(val_summary["freq_loss"])
        history["lr"].append(epoch_lr)

        improved = val_summary["total_loss"] < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_summary["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
            save_best_checkpoint(
                run_paths=run_paths,
                variant_config=variant_config,
                seed=seed,
                model=model,
                history=history,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                run_config=run_config,
            )
        else:
            stale_epochs += 1

        write_json(
            run_paths.history_json,
            {
                "variant": variant_config.name,
                "seed": int(seed),
                "status": "running",
                "best_epoch": int(best_epoch),
                "best_val_total_loss": float(best_val_loss),
                "stale_epochs": int(stale_epochs),
                "history": history,
            },
        )
        save_latest_checkpoint(
            run_paths=run_paths,
            variant_config=variant_config,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            stale_epochs=stale_epochs,
            epoch=epoch,
            run_config=run_config,
        )

        print(
            f"[{variant_config.name} seed {seed}] epoch {epoch:02d}/{args.epochs:02d} "
            f"train_total={train_summary['total_loss']:.6f} "
            f"val_total={val_summary['total_loss']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(
                f"[{variant_config.name} seed {seed}] early stopping after {stale_epochs} stale epochs.",
                flush=True,
            )
            break

    if best_epoch <= 0 or not run_paths.best_checkpoint.exists():
        raise RuntimeError(f"{variant_config.name} seed {seed} did not produce a best checkpoint.")

    best_payload = load_torch_payload(run_paths.best_checkpoint)
    model.load_state_dict(best_payload["state_dict"])
    model = model.to(device)
    val_scores = compute_reconstruction_scores(model=model, loader=loaders["val"], device=device)
    test_healthy_scores = compute_reconstruction_scores(model=model, loader=loaders["test_healthy"], device=device)
    test_fault_scores = compute_reconstruction_scores(model=model, loader=loaders["test_fault"], device=device)

    save_numpy_array(run_paths.val_healthy_scores_npy, val_scores)
    save_numpy_array(run_paths.test_healthy_scores_npy, test_healthy_scores)
    save_numpy_array(run_paths.test_fault_scores_npy, test_fault_scores)
    sanity_metrics = build_sanity_metrics(
        variant_config=variant_config,
        seed=seed,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        val_scores=val_scores,
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        run_paths=run_paths,
    )
    write_json(run_paths.sanity_metrics_json, sanity_metrics)

    elapsed_seconds = time.perf_counter() - start_time
    run_config["training_result"] = {
        "status": "completed",
        "epochs_ran": len(history["epoch"]),
        "best_epoch": int(best_epoch),
        "best_val_total_loss": float(best_val_loss),
        "elapsed_seconds": float(elapsed_seconds),
        "sanity_metrics_json": run_paths.sanity_metrics_json.as_posix(),
    }
    write_json(run_paths.run_config_json, run_config)
    write_json(
        run_paths.history_json,
        {
            "variant": variant_config.name,
            "seed": int(seed),
            "status": "completed",
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "elapsed_seconds": float(elapsed_seconds),
            "history": history,
        },
    )
    print(
        f"[{variant_config.name} seed {seed}] complete: "
        f"AUROC={sanity_metrics['ranking_metrics']['auroc']:.6f}, "
        f"AUPRC={sanity_metrics['ranking_metrics']['auprc']:.6f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    artifacts_root = validate_artifacts_root(args.artifacts_root)
    array_paths = resolve_array_paths(PROCESSED_ROOT)
    ensure_required_arrays(array_paths)
    dataset_sizes = build_dataset_sizes(array_paths, args.train_subset, args.val_subset, args.test_subset)
    device = get_device()
    batch_size = args.batch_size_cuda if device.type == "cuda" else args.batch_size_cpu

    print("Paderborn ablation Chunk 2 training harness", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Artifacts root: {artifacts_root.as_posix()}", flush=True)
    print(f"Variants: {', '.join(args.variants)}", flush=True)
    print(f"Seeds: {', '.join(str(seed) for seed in args.seeds)}", flush=True)

    for variant in args.variants:
        variant_config = VARIANT_CONFIGS[variant]
        for seed in args.seeds:
            run_variant_seed(
                variant_config=variant_config,
                seed=int(seed),
                args=args,
                artifacts_root=artifacts_root,
                array_paths=array_paths,
                dataset_sizes=dataset_sizes,
                device=device,
                batch_size=batch_size,
            )
    print("\nChunk 2 ablation training complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
