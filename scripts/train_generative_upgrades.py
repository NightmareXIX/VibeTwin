from __future__ import annotations

import argparse
import json
import math
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from train_ae_baseline import (
        DataLoader,
        evaluate_scores as evaluate_binary_scores,
        parameter_count,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from train_paderborn_baselines import (
        MemmapWindowDataset,
        ensure_required_files,
        load_label_array,
        read_json,
        resolve_paths,
        subgroup_metrics_from_manifest,
    )
except ModuleNotFoundError:
    from scripts.train_ae_baseline import (
        DataLoader,
        evaluate_scores as evaluate_binary_scores,
        parameter_count,
        require_torch,
        select_threshold,
        set_seed,
        torch,
    )
    from scripts.train_paderborn_baselines import (
        MemmapWindowDataset,
        ensure_required_files,
        load_label_array,
        read_json,
        resolve_paths,
        subgroup_metrics_from_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
BASELINE_AE_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "paderborn_ae_metrics.json"
BASELINE_IFOREST_METRICS_PATH = ARTIFACTS_ROOT / "metrics" / "paderborn_iforest_metrics.json"

nn = torch.nn if torch is not None else None
F = torch.nn.functional if torch is not None else None
ModuleBase = nn.Module if nn is not None else object
MODEL_CHOICES = ("resdilated_ae", "conv_vae", "denoising_resdilated_ae", "memae")
MEMAE_MEMORY_SIZE = 500
MEMAE_ENTROPY_WEIGHT = 2e-4
MEMAE_EPSILON = 1e-12
MEMAE_ADDRESSING_CHOICES = ("dot", "cosine")
MEMAE_ADDRESSING = "dot"


def choose_group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


@dataclass(frozen=True)
class ModelRunConfig:
    name: str
    cli_name: str
    output_stem: str
    model_kind: str
    denoising: bool = False


@dataclass(frozen=True)
class RunPaths:
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


class RunInterrupted(RuntimeError):
    pass


class InterruptTracker:
    def __init__(self) -> None:
        self.signal_name: str | None = None
        self._previous_handlers: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self.signal_name is not None

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self.signal_name is None:
            self.signal_name = getattr(signal.Signals(signum), "name", str(signum))
            print(f"\nReceived {self.signal_name}; saving progress before exit...", flush=True)
            return
        raise KeyboardInterrupt

    def __enter__(self) -> "InterruptTracker":
        for signal_name in ("SIGINT", "SIGTERM"):
            if not hasattr(signal, signal_name):
                continue
            signal_value = getattr(signal, signal_name)
            self._previous_handlers[signal_value] = signal.getsignal(signal_value)
            signal.signal(signal_value, self._handle_signal)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for signal_value, previous_handler in self._previous_handlers.items():
            signal.signal(signal_value, previous_handler)


class ResidualDilatedBlock(ModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        groups = choose_group_count(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        outputs = self.conv1(inputs)
        outputs = self.norm1(outputs)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)
        outputs = outputs + residual
        return self.act(outputs)


class DownsampleBlock(ModuleBase):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpsampleBlock(ModuleBase):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ResDilatedAE(ModuleBase):
    def __init__(self, base_channels: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.GroupNorm(choose_group_count(c1), c1),
            nn.SiLU(),
        )
        self.enc1 = ResidualDilatedBlock(c1, c1, dilation=1, dropout=dropout)
        self.down1 = DownsampleBlock(c1, c2)
        self.enc2 = ResidualDilatedBlock(c2, c2, dilation=2, dropout=dropout)
        self.down2 = DownsampleBlock(c2, c3)
        self.enc3 = ResidualDilatedBlock(c3, c3, dilation=4, dropout=dropout)
        self.down3 = DownsampleBlock(c3, c4)
        self.bottleneck = nn.Sequential(
            ResidualDilatedBlock(c4, c4, dilation=8, dropout=dropout),
            ResidualDilatedBlock(c4, c4, dilation=16, dropout=dropout),
        )
        self.up3 = UpsampleBlock(c4, c3)
        self.dec3 = ResidualDilatedBlock(c3 + c3, c3, dilation=4, dropout=dropout)
        self.up2 = UpsampleBlock(c3, c2)
        self.dec2 = ResidualDilatedBlock(c2 + c2, c2, dilation=2, dropout=dropout)
        self.up1 = UpsampleBlock(c2, c1)
        self.dec1 = ResidualDilatedBlock(c1 + c1, c1, dilation=1, dropout=dropout)
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


class ConvVAE(ModuleBase):
    def __init__(
        self,
        *,
        input_length: int = 2048,
        base_channels: int = 24,
        latent_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(choose_group_count(c1), c1),
            nn.SiLU(),
            ResidualDilatedBlock(c1, c1, dilation=1, dropout=dropout),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(choose_group_count(c2), c2),
            nn.SiLU(),
            ResidualDilatedBlock(c2, c2, dilation=2, dropout=dropout),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(choose_group_count(c3), c3),
            nn.SiLU(),
            ResidualDilatedBlock(c3, c3, dilation=4, dropout=dropout),
            nn.Conv1d(c3, c4, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(choose_group_count(c4), c4),
            nn.SiLU(),
            ResidualDilatedBlock(c4, c4, dilation=8, dropout=dropout),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 1, input_length)
            encoded = self.encoder(probe)
        self.encoded_shape = tuple(int(value) for value in encoded.shape[1:])
        flattened_dim = int(np.prod(self.encoded_shape))
        self.to_mu = nn.Linear(flattened_dim, latent_dim)
        self.to_logvar = nn.Linear(flattened_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(c4, c3, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(c3), c3),
            nn.SiLU(),
            ResidualDilatedBlock(c3, c3, dilation=4, dropout=dropout),
            nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(c2), c2),
            nn.SiLU(),
            ResidualDilatedBlock(c2, c2, dilation=2, dropout=dropout),
            nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(c1), c1),
            nn.SiLU(),
            ResidualDilatedBlock(c1, c1, dilation=1, dropout=dropout),
            nn.ConvTranspose1d(c1, max(8, c1 // 2), kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(choose_group_count(max(8, c1 // 2)), max(8, c1 // 2)),
            nn.SiLU(),
            nn.Conv1d(max(8, c1 // 2), 1, kernel_size=7, padding=3),
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs)
        flattened = features.flatten(start_dim=1)
        return self.to_mu(flattened), self.to_logvar(flattened)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + (torch.randn_like(std) * std)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent)
        decoded = decoded.view(latent.shape[0], *self.encoded_shape)
        return self.decoder(decoded)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar


class MemoryModule(ModuleBase):
    """Memory-addressing module of MemAE (Gong et al., ICCV 2019).

    Each latent position is used as a query against a learned memory bank ``M``
    of ``memory_size`` prototypes. The query is replaced by a sparse convex
    combination of those prototypes, so the decoder can only reconstruct from
    memorized normal patterns.

    ``addressing`` selects the similarity used for the softmax logits. ``"dot"``
    is the unnormalized inner product of the reference release
    (``donggong1/memae-anomaly-detection``) and is the default; ``"cosine"`` is
    the similarity written in the paper. Cosine bounds every logit to [-1, 1],
    which caps the spread of the softmax at a factor of e^2 and leaves every
    weight too close to 1/N for lambda in [1/N, 3/N] to prune anything: measured
    over ten epochs on Paderborn healthy windows, addressing entropy stays at
    log(500) = 6.21 and all 500 slots stay active, so the memory degenerates into
    a low-rank linear layer. Under ``"dot"`` the same run reaches entropy 1.91
    with 12 active slots per position. See
    ``implementation_docs/memae_phase3_notes.md``.
    """

    def __init__(
        self,
        *,
        memory_size: int = MEMAE_MEMORY_SIZE,
        feature_dim: int,
        shrink_threshold: float | None = None,
        addressing: str = MEMAE_ADDRESSING,
        epsilon: float = MEMAE_EPSILON,
    ) -> None:
        super().__init__()
        if memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.memory_size = int(memory_size)
        self.feature_dim = int(feature_dim)
        self.shrink_threshold = (
            default_shrink_threshold(memory_size) if shrink_threshold is None else float(shrink_threshold)
        )
        if self.shrink_threshold < 0.0:
            raise ValueError("shrink_threshold must be non-negative.")
        if addressing not in MEMAE_ADDRESSING_CHOICES:
            raise ValueError(f"addressing must be one of {MEMAE_ADDRESSING_CHOICES}, got {addressing!r}.")
        self.addressing = addressing
        self.epsilon = float(epsilon)
        self.memory = nn.Parameter(torch.empty(self.memory_size, self.feature_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.feature_dim)
        with torch.no_grad():
            self.memory.uniform_(-bound, bound)

    def hard_shrink(self, attention: torch.Tensor) -> torch.Tensor:
        """Differentiable hard shrinkage: max(w - lambda, 0) * w / (|w - lambda| + eps).

        The ``if w > lambda`` form has zero gradient almost everywhere and would
        leave the memory untrained, so the ReLU form is used deliberately.
        """
        offset = attention - self.shrink_threshold
        return (F.relu(offset) * attention) / (offset.abs() + self.epsilon)

    def shrink_and_renormalize(self, attention: torch.Tensor) -> torch.Tensor:
        """Apply hard shrinkage, then renormalize to unit L1 mass (Eq. 7, in that order).

        Equation 7 divides by ``||shrunk||_1``, which is undefined when shrinkage
        zeroes an entire addressing row. That happens whenever addressing is close
        to uniform, which is where training starts and where cosine addressing
        stays: every softmax weight then sits near 1/N and none survives lambda.
        Left undefined the readout is zero, the memory receives no gradient, and
        the bank never trains. Such rows therefore fall back to the unshrunk
        weights, which is the minimal well-defined completion; shrinkage takes
        effect as addressing sharpens.
        """
        shrunk = self.hard_shrink(attention)
        mass = shrunk.sum(dim=-1, keepdim=True)
        collapsed = mass <= self.epsilon
        shrunk = torch.where(collapsed, attention, shrunk)
        mass = torch.where(collapsed, attention.sum(dim=-1, keepdim=True), mass)
        return shrunk / mass.clamp_min(self.epsilon)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: (B, C, L_z) -> queries of shape (B, L_z, C), one per latent position.
        queries = latent.transpose(1, 2)
        if self.addressing == "cosine":
            queries = F.normalize(queries, p=2.0, dim=-1, eps=self.epsilon)
            memory = F.normalize(self.memory, p=2.0, dim=-1, eps=self.epsilon)
        else:
            memory = self.memory
        similarity = torch.matmul(queries, memory.t())
        attention = self.shrink_and_renormalize(F.softmax(similarity, dim=-1))
        readout = torch.matmul(attention, self.memory)
        return readout.transpose(1, 2), attention


def default_shrink_threshold(memory_size: int) -> float:
    """Low end of the paper's recommended range lambda in [1/N, 3/N].

    The midpoint 2/N collapses the memory on this data: in a ten-epoch probe on
    Paderborn healthy windows a single slot takes 65% of the addressing mass and
    validation reconstruction stalls at 0.246, against 0.6% and 0.056 at 1/N.
    3/N behaves the same as 2/N. See ``implementation_docs/memae_phase3_notes.md``.
    """
    return 1.0 / float(memory_size)


def memory_entropy_loss(attention: torch.Tensor, epsilon: float = MEMAE_EPSILON) -> torch.Tensor:
    """Entropy of the shrunk, renormalized addressing weights, averaged over positions and batch."""
    entropy = -(attention * torch.log(attention + epsilon)).sum(dim=-1)
    return entropy.mean()


class MemAE(ModuleBase):
    """Memory-augmented autoencoder comparator.

    Deliberately a plain strided conv encoder-decoder with no skip connections:
    every reconstruction path runs through the memory bottleneck. Channel widths
    are chosen to match the ResDilatedAE parameter budget so that any difference
    is attributable to the memory mechanism rather than to capacity.
    """

    def __init__(
        self,
        *,
        base_channels: int = 24,
        memory_size: int = MEMAE_MEMORY_SIZE,
        shrink_threshold: float | None = None,
        addressing: str = MEMAE_ADDRESSING,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4
        self.latent_channels = c4

        self.encoder = nn.Sequential(
            *encoder_stage(1, c1, dropout=dropout),
            *encoder_stage(c1, c2, dropout=dropout),
            *encoder_stage(c2, c3, dropout=dropout),
            *encoder_stage(c3, c4, dropout=dropout),
        )
        self.memory = MemoryModule(
            memory_size=memory_size,
            feature_dim=c4,
            shrink_threshold=shrink_threshold,
            addressing=addressing,
        )
        self.decoder = nn.Sequential(
            *decoder_stage(c4, c3, dropout=dropout),
            *decoder_stage(c3, c2, dropout=dropout),
            *decoder_stage(c2, c1, dropout=dropout),
            *decoder_stage(c1, c1, dropout=dropout),
            nn.Conv1d(c1, 1, kernel_size=7, padding=3),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        readout, attention = self.memory(latent)
        return self.decoder(readout), attention


def encoder_stage(in_channels: int, out_channels: int, *, dropout: float = 0.0) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
        nn.GroupNorm(choose_group_count(out_channels), out_channels),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def decoder_stage(in_channels: int, out_channels: int, *, dropout: float = 0.0) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size=8, stride=2, padding=3),
        nn.GroupNorm(choose_group_count(out_channels), out_channels),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stronger generative anomaly-detection upgrades on processed Paderborn windows.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument(
        "--model",
        choices=("all",) + MODEL_CHOICES,
        default="all",
        help="Run a single model by name, or use 'all' to run every model sequentially.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size-cuda", type=int, default=256)
    parser.add_argument("--batch-size-cpu", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-rule", choices=("mean_plus_3std",), default="mean_plus_3std")
    parser.add_argument("--freq-loss-weight", type=float, default=0.10)
    parser.add_argument("--vae-beta-max", type=float, default=1e-3)
    parser.add_argument("--vae-kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--memae-memory-size", type=int, default=MEMAE_MEMORY_SIZE)
    parser.add_argument(
        "--memae-shrink-threshold",
        type=float,
        default=None,
        help="Hard shrinkage lambda. Defaults to 1 / memory_size, the low end of the paper's range.",
    )
    parser.add_argument("--memae-entropy-weight", type=float, default=MEMAE_ENTROPY_WEIGHT)
    parser.add_argument(
        "--memae-addressing",
        choices=MEMAE_ADDRESSING_CHOICES,
        default=MEMAE_ADDRESSING,
        help="Memory addressing logits: unnormalized dot product (reference release) or cosine similarity (paper text).",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=1,
        help="Persist the resumable latest checkpoint, history JSON, and status JSON every N epochs.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a previously interrupted or incomplete run.")
    parser.add_argument("--best-candidate-extra-seeds", type=int, default=0)
    args = parser.parse_args()
    if args.save_every_epochs <= 0:
        parser.error("--save-every-epochs must be positive.")
    return args


def make_loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MemmapWindowDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def build_loader_bundle(
    *,
    array_paths: Any,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    test_subset: int,
) -> dict[str, DataLoader]:
    loaders = {
        "train": make_loader(array_paths.train_healthy, batch_size=batch_size, shuffle=True),
        "val": make_loader(array_paths.val_healthy, batch_size=batch_size, shuffle=False),
        "test_healthy": make_loader(array_paths.test_healthy, batch_size=batch_size, shuffle=False),
        "test_fault": make_loader(array_paths.test_fault, batch_size=batch_size, shuffle=False),
    }
    if train_subset > 0:
        subset = torch.utils.data.Subset(loaders["train"].dataset, list(range(min(train_subset, len(loaders["train"].dataset)))))
        loaders["train"] = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    if val_subset > 0:
        subset = torch.utils.data.Subset(loaders["val"].dataset, list(range(min(val_subset, len(loaders["val"].dataset)))))
        loaders["val"] = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    if test_subset > 0:
        subset_healthy = torch.utils.data.Subset(
            loaders["test_healthy"].dataset,
            list(range(min(test_subset, len(loaders["test_healthy"].dataset)))),
        )
        subset_fault = torch.utils.data.Subset(
            loaders["test_fault"].dataset,
            list(range(min(test_subset, len(loaders["test_fault"].dataset)))),
        )
        loaders["test_healthy"] = DataLoader(subset_healthy, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        loaders["test_fault"] = DataLoader(subset_fault, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return loaders


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def compute_frequency_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction_fft = torch.fft.rfft(reconstruction.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    reconstruction_mag = torch.log1p(torch.abs(reconstruction_fft))
    target_mag = torch.log1p(torch.abs(target_fft))
    return F.mse_loss(reconstruction_mag, target_mag, reduction="mean")


def compute_vae_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def apply_denoising_corruption(
    inputs: torch.Tensor,
    *,
    gaussian_std: float = 0.015,
    amplitude_scale: float = 0.05,
    max_roll: int = 8,
) -> torch.Tensor:
    outputs = inputs
    if gaussian_std > 0:
        outputs = outputs + (torch.randn_like(outputs) * gaussian_std)
    if amplitude_scale > 0:
        scale = 1.0 + ((torch.rand(outputs.shape[0], 1, 1, device=outputs.device) * 2.0) - 1.0) * amplitude_scale
        outputs = outputs * scale
    if max_roll > 0:
        shift = int(torch.randint(-max_roll, max_roll + 1, (1,), device=outputs.device).item())
        if shift != 0:
            outputs = torch.roll(outputs, shifts=shift, dims=-1)
    return outputs


def build_models(
    window_size: int,
    dropout: float,
    *,
    memae_memory_size: int = MEMAE_MEMORY_SIZE,
    memae_shrink_threshold: float | None = None,
    memae_addressing: str = MEMAE_ADDRESSING,
) -> list[tuple[ModelRunConfig, nn.Module]]:
    return [
        (
            ModelRunConfig(
                name="ResDilatedAE",
                cli_name="resdilated_ae",
                output_stem="resdilated_ae",
                model_kind="ae",
            ),
            ResDilatedAE(base_channels=16, dropout=dropout),
        ),
        (
            ModelRunConfig(
                name="ConvVAE",
                cli_name="conv_vae",
                output_stem="conv_vae",
                model_kind="vae",
            ),
            ConvVAE(input_length=window_size, base_channels=16, latent_dim=48, dropout=dropout),
        ),
        (
            ModelRunConfig(
                name="DenoisingResDilatedAE",
                cli_name="denoising_resdilated_ae",
                output_stem="denoising_resdilated_ae",
                model_kind="ae",
                denoising=True,
            ),
            ResDilatedAE(base_channels=16, dropout=dropout),
        ),
        (
            ModelRunConfig(
                name="MemAE",
                cli_name="memae",
                output_stem="memae",
                model_kind="memae",
            ),
            MemAE(
                base_channels=24,
                memory_size=memae_memory_size,
                shrink_threshold=memae_shrink_threshold,
                addressing=memae_addressing,
                dropout=dropout,
            ),
        ),
    ]


def select_models(
    models: list[tuple[ModelRunConfig, nn.Module]],
    requested_model: str,
) -> list[tuple[ModelRunConfig, nn.Module]]:
    if requested_model == "all":
        return models
    return [(run_config, model) for run_config, model in models if run_config.cli_name == requested_model]


def build_run_paths(*, artifacts_root: Path, run_config: ModelRunConfig, seed: int) -> RunPaths:
    run_dir = artifacts_root / "generative_upgrades" / run_config.output_stem / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        latest_checkpoint=checkpoints_dir / "latest.pt",
        best_checkpoint=checkpoints_dir / "best.pt",
        val_healthy_scores_npy=run_dir / "val_healthy_scores.npy",
        test_healthy_scores_npy=run_dir / "test_healthy_scores.npy",
        test_fault_scores_npy=run_dir / "test_fault_scores.npy",
        history_json=run_dir / "history.json",
        status_json=run_dir / "status.json",
        metrics_json=run_dir / "metrics.json",
        report_md=run_dir / "report.md",
        plot_png=run_dir / "summary.png",
        train_log=run_dir / "train.log",
    )


def format_command(parts: list[str]) -> str:
    formatted: list[str] = []
    for part in parts:
        if any(character.isspace() for character in part):
            formatted.append(f'"{part}"')
        else:
            formatted.append(part)
    return " ".join(formatted)


def build_manual_command(args: argparse.Namespace, model_name: str, *, resume: bool = False) -> str:
    command = [
        "python",
        "scripts/train_generative_upgrades.py",
        "--model",
        model_name,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--save-every-epochs",
        str(args.save_every_epochs),
    ]
    if args.artifacts_root != ARTIFACTS_ROOT:
        command.extend(["--artifacts-root", args.artifacts_root.as_posix()])
    if args.processed_root != PROCESSED_ROOT:
        command.extend(["--processed-root", args.processed_root.as_posix()])
    if args.metadata_root != METADATA_ROOT:
        command.extend(["--metadata-root", args.metadata_root.as_posix()])
    if args.memae_addressing != MEMAE_ADDRESSING:
        command.extend(["--memae-addressing", args.memae_addressing])
    if resume:
        command.append("--resume")
    return format_command(command)


def print_manual_run_commands(
    *,
    args: argparse.Namespace,
    model_configs: list[ModelRunConfig],
    artifacts_root: Path,
) -> None:
    print("\nManual GPU commands", flush=True)
    for run_config in model_configs:
        run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=run_config, seed=args.seed)
        print(f"  {run_config.name}:", flush=True)
        print(f"    train : {build_manual_command(args, run_config.cli_name)}", flush=True)
        print(f"    resume: {build_manual_command(args, run_config.cli_name, resume=True)}", flush=True)
        print(f"    output: {run_paths.run_dir.as_posix()}", flush=True)


def calculate_beta(epoch_index: int, warmup_epochs: int, beta_max: float) -> float:
    if warmup_epochs <= 0:
        return beta_max
    return float(beta_max * min(1.0, epoch_index / warmup_epochs))


def build_empty_history() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_total_loss": [],
        "train_time_loss": [],
        "train_freq_loss": [],
        "train_kl_loss": [],
        "train_mem_loss": [],
        "val_total_loss": [],
        "val_time_loss": [],
        "val_freq_loss": [],
        "val_kl_loss": [],
        "val_mem_loss": [],
        "beta": [],
        "lr": [],
    }


def set_epoch_seed(base_seed: int, epoch: int) -> None:
    set_seed(base_seed + epoch)


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


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
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


def save_numpy_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.float32))
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        handle.write("\n")


def log_message(message: str, log_path: Path | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        append_log(log_path, line)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=device.type == "cuda")


def save_best_checkpoint(
    *,
    run_config: ModelRunConfig,
    run_paths: RunPaths,
    model: nn.Module,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_loss: float,
    seed: int,
    training_settings: dict[str, Any],
) -> None:
    save_torch_payload(
        run_paths.best_checkpoint,
        {
            "format_version": 2,
            "model_name": run_config.name,
            "model_cli_name": run_config.cli_name,
            "model_kind": run_config.model_kind,
            "denoising": run_config.denoising,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "state_dict": clone_model_state(model),
            "history": history,
            "training_settings": training_settings,
            "saved_at_unix": time.time(),
        },
    )


def save_training_progress(
    *,
    run_config: ModelRunConfig,
    run_paths: RunPaths,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_loss: float,
    stale_epochs: int,
    next_epoch: int,
    epochs: int,
    elapsed_seconds: float,
    seed: int,
    training_settings: dict[str, Any],
    status: str,
    stage: str,
    message: str,
    epoch_in_progress: int | None = None,
) -> None:
    completed_epochs = len(history["train_total_loss"])
    latest_payload = {
        "format_version": 2,
        "model_name": run_config.name,
        "model_cli_name": run_config.cli_name,
        "model_kind": run_config.model_kind,
        "denoising": run_config.denoising,
        "seed": int(seed),
        "status": status,
        "stage": stage,
        "message": message,
        "requested_epochs": int(epochs),
        "completed_epochs": int(completed_epochs),
        "next_epoch": int(next_epoch),
        "epoch_in_progress": int(epoch_in_progress) if epoch_in_progress is not None else None,
        "best_epoch": int(best_epoch),
        "best_val_total_loss": float(best_val_loss),
        "stale_epochs": int(stale_epochs),
        "elapsed_seconds": float(elapsed_seconds),
        "model_state_dict": clone_model_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "training_settings": training_settings,
        "saved_at_unix": time.time(),
    }
    save_torch_payload(run_paths.latest_checkpoint, latest_payload)
    write_json(
        run_paths.history_json,
        {
            "model_name": run_config.name,
            "model_cli_name": run_config.cli_name,
            "seed": int(seed),
            "status": status,
            "stage": stage,
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "completed_epochs": int(completed_epochs),
            "requested_epochs": int(epochs),
            "elapsed_seconds": float(elapsed_seconds),
            "history": history,
            "training_settings": training_settings,
        },
    )
    write_json(
        run_paths.status_json,
        {
            "model_name": run_config.name,
            "model_cli_name": run_config.cli_name,
            "seed": int(seed),
            "status": status,
            "stage": stage,
            "message": message,
            "run_dir": run_paths.run_dir.as_posix(),
            "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "history_json": run_paths.history_json.as_posix(),
            "metrics_json": run_paths.metrics_json.as_posix(),
            "report_md": run_paths.report_md.as_posix(),
            "plot_png": run_paths.plot_png.as_posix(),
            "train_log": run_paths.train_log.as_posix(),
            "requested_epochs": int(epochs),
            "completed_epochs": int(completed_epochs),
            "next_epoch": int(next_epoch),
            "epoch_in_progress": int(epoch_in_progress) if epoch_in_progress is not None else None,
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "stale_epochs": int(stale_epochs),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_at_unix": time.time(),
        },
    )


def evaluate_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    freq_loss_weight: float,
    beta: float,
    model_kind: str,
    memae_entropy_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    total_freq = 0.0
    total_kl = 0.0
    total_mem = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if model_kind == "vae":
                reconstruction, mu, logvar = model(batch)
                time_loss = F.mse_loss(reconstruction.float(), batch.float(), reduction="mean")
                kl_loss = compute_vae_kl(mu.float(), logvar.float())
                mem_loss = torch.zeros((), device=device)
            elif model_kind == "memae":
                reconstruction, attention = model(batch)
                time_loss = F.mse_loss(reconstruction.float(), batch.float(), reduction="mean")
                kl_loss = torch.zeros((), device=device)
                mem_loss = memory_entropy_loss(attention.float())
            else:
                reconstruction = model(batch)
                time_loss = F.mse_loss(reconstruction.float(), batch.float(), reduction="mean")
                kl_loss = torch.zeros((), device=device)
                mem_loss = torch.zeros((), device=device)
            freq_loss = compute_frequency_loss(reconstruction, batch) if freq_loss_weight > 0 else torch.zeros((), device=device)
            total_batch_loss = (
                time_loss + (freq_loss_weight * freq_loss) + (beta * kl_loss) + (memae_entropy_weight * mem_loss)
            )
            batch_size = int(batch.shape[0])
            total_loss += float(total_batch_loss.item()) * batch_size
            total_time += float(time_loss.item()) * batch_size
            total_freq += float(freq_loss.item()) * batch_size
            total_kl += float(kl_loss.item()) * batch_size
            total_mem += float(mem_loss.item()) * batch_size
            total_samples += batch_size
    divisor = max(total_samples, 1)
    return {
        "total_loss": total_loss / divisor,
        "time_loss": total_time / divisor,
        "freq_loss": total_freq / divisor,
        "kl_loss": total_kl / divisor,
        "mem_loss": total_mem / divisor,
    }


def train_single_model(
    *,
    run_config: ModelRunConfig,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    freq_loss_weight: float,
    vae_beta_max: float,
    vae_warmup_epochs: int,
    memae_entropy_weight: float,
    run_paths: RunPaths,
    seed: int,
    save_every_epochs: int,
    resume: bool,
    training_settings: dict[str, Any],
) -> dict[str, Any]:
    # MemAE's decoder reads only from the memory bottleneck, so a frequency-domain
    # loss term (designed to sharpen skip-connected AE detail) does not apply to it.
    freq_loss_weight = 0.0 if run_config.model_kind == "memae" else freq_loss_weight
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = build_empty_history()
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    elapsed_before_resume = 0.0
    current_epoch = 1

    if resume:
        if not run_paths.latest_checkpoint.exists():
            raise FileNotFoundError(
                f"No resumable checkpoint was found at {run_paths.latest_checkpoint.as_posix()} for {run_config.name}."
            )
        resume_payload = load_torch_payload(run_paths.latest_checkpoint)
        if resume_payload.get("model_cli_name") != run_config.cli_name:
            raise RuntimeError(
                f"Resume checkpoint {run_paths.latest_checkpoint.as_posix()} belongs to "
                f"{resume_payload.get('model_cli_name')} instead of {run_config.cli_name}."
            )
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler_state = resume_payload.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            if hasattr(scheduler, "T_max"):
                scheduler.T_max = max(epochs, 1)
        scaler_state = resume_payload.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        history = resume_payload.get("history", build_empty_history())
        for key, default_values in build_empty_history().items():
            history.setdefault(key, default_values.copy())
        best_val_loss = float(resume_payload.get("best_val_total_loss", math.inf))
        best_epoch = int(resume_payload.get("best_epoch", 0))
        stale_epochs = int(resume_payload.get("stale_epochs", 0))
        start_epoch = int(resume_payload.get("next_epoch", len(history["train_total_loss"]) + 1))
        elapsed_before_resume = float(resume_payload.get("elapsed_seconds", 0.0))
        log_message(
            f"[{run_config.name}] Resuming from epoch {start_epoch} using {run_paths.latest_checkpoint.as_posix()}",
            run_paths.train_log,
        )
    else:
        log_message(f"[{run_config.name}] Starting fresh run in {run_paths.run_dir.as_posix()}", run_paths.train_log)

    if start_epoch > epochs:
        elapsed_seconds = elapsed_before_resume
        save_training_progress(
            run_config=run_config,
            run_paths=run_paths,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            stale_epochs=stale_epochs,
            next_epoch=start_epoch,
            epochs=epochs,
            elapsed_seconds=elapsed_seconds,
            seed=seed,
            training_settings=training_settings,
            status="completed",
            stage="training",
            message="Training had already reached or exceeded the requested epoch budget.",
        )
        model.eval()
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val_loss,
            "elapsed_seconds": elapsed_seconds,
            "status": "completed",
            "next_epoch": start_epoch,
        }

    save_training_progress(
        run_config=run_config,
        run_paths=run_paths,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stale_epochs=stale_epochs,
        next_epoch=start_epoch,
        epochs=epochs,
        elapsed_seconds=elapsed_before_resume,
        seed=seed,
        training_settings=training_settings,
        status="running",
        stage="training",
        message="Training initialized.",
    )

    run_start_time = time.perf_counter()
    with InterruptTracker() as interrupts:
        try:
            for epoch in range(start_epoch, epochs + 1):
                current_epoch = epoch
                set_epoch_seed(seed, epoch)
                beta = calculate_beta(epoch, vae_warmup_epochs, vae_beta_max) if run_config.model_kind == "vae" else 0.0
                epoch_lr = float(optimizer.param_groups[0]["lr"])
                model.train()
                epoch_total = 0.0
                epoch_time = 0.0
                epoch_freq = 0.0
                epoch_kl = 0.0
                epoch_mem = 0.0
                sample_count = 0

                for batch in loaders["train"]:
                    batch = batch.to(device, non_blocking=device.type == "cuda")
                    clean_target = batch
                    model_input = apply_denoising_corruption(batch) if run_config.denoising else batch
                    optimizer.zero_grad(set_to_none=True)
                    with get_autocast(device):
                        if run_config.model_kind == "vae":
                            reconstruction, mu, logvar = model(model_input)
                            time_loss = F.mse_loss(reconstruction.float(), clean_target.float(), reduction="mean")
                            kl_loss = compute_vae_kl(mu.float(), logvar.float())
                            mem_loss = torch.zeros((), device=device)
                        elif run_config.model_kind == "memae":
                            reconstruction, attention = model(model_input)
                            time_loss = F.mse_loss(reconstruction.float(), clean_target.float(), reduction="mean")
                            kl_loss = torch.zeros((), device=device)
                            mem_loss = memory_entropy_loss(attention.float())
                        else:
                            reconstruction = model(model_input)
                            time_loss = F.mse_loss(reconstruction.float(), clean_target.float(), reduction="mean")
                            kl_loss = torch.zeros((), device=device)
                            mem_loss = torch.zeros((), device=device)
                        freq_loss = (
                            compute_frequency_loss(reconstruction, clean_target)
                            if freq_loss_weight > 0
                            else torch.zeros((), device=device)
                        )
                        total_loss = (
                            time_loss
                            + (freq_loss_weight * freq_loss)
                            + (beta * kl_loss)
                            + (memae_entropy_weight * mem_loss)
                        )
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    batch_size = int(batch.shape[0])
                    epoch_total += float(total_loss.item()) * batch_size
                    epoch_time += float(time_loss.item()) * batch_size
                    epoch_freq += float(freq_loss.item()) * batch_size
                    epoch_kl += float(kl_loss.item()) * batch_size
                    epoch_mem += float(mem_loss.item()) * batch_size
                    sample_count += batch_size
                    if interrupts.requested:
                        raise RunInterrupted(f"{interrupts.signal_name} received during epoch {epoch}.")

                scheduler.step()
                divisor = max(sample_count, 1)
                train_summary = {
                    "total_loss": epoch_total / divisor,
                    "time_loss": epoch_time / divisor,
                    "freq_loss": epoch_freq / divisor,
                    "kl_loss": epoch_kl / divisor,
                    "mem_loss": epoch_mem / divisor,
                }
                val_summary = evaluate_epoch(
                    model=model,
                    loader=loaders["val"],
                    device=device,
                    freq_loss_weight=freq_loss_weight,
                    beta=beta,
                    model_kind=run_config.model_kind,
                    memae_entropy_weight=memae_entropy_weight,
                )
                history["epoch"].append(epoch)
                history["train_total_loss"].append(train_summary["total_loss"])
                history["train_time_loss"].append(train_summary["time_loss"])
                history["train_freq_loss"].append(train_summary["freq_loss"])
                history["train_kl_loss"].append(train_summary["kl_loss"])
                history["train_mem_loss"].append(train_summary["mem_loss"])
                history["val_total_loss"].append(val_summary["total_loss"])
                history["val_time_loss"].append(val_summary["time_loss"])
                history["val_freq_loss"].append(val_summary["freq_loss"])
                history["val_kl_loss"].append(val_summary["kl_loss"])
                history["val_mem_loss"].append(val_summary["mem_loss"])
                history["beta"].append(beta)
                history["lr"].append(epoch_lr)
                epoch_message = (
                    f"[{run_config.name}] Epoch {epoch:02d}/{epochs:02d} "
                    f"- train_total={train_summary['total_loss']:.6f} "
                    f"- train_time={train_summary['time_loss']:.6f} "
                    f"- val_total={val_summary['total_loss']:.6f} "
                    f"- val_time={val_summary['time_loss']:.6f} "
                    f"- beta={beta:.6f} "
                    f"- lr={epoch_lr:.6g}"
                )
                log_message(epoch_message, run_paths.train_log)

                if val_summary["total_loss"] < best_val_loss - 1e-6:
                    best_val_loss = val_summary["total_loss"]
                    best_epoch = epoch
                    stale_epochs = 0
                    save_best_checkpoint(
                        run_config=run_config,
                        run_paths=run_paths,
                        model=model,
                        history=history,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        seed=seed,
                        training_settings=training_settings,
                    )
                else:
                    stale_epochs += 1

                elapsed_seconds = elapsed_before_resume + (time.perf_counter() - run_start_time)
                should_save = (epoch % save_every_epochs == 0) or epoch == epochs or stale_epochs >= patience
                if should_save:
                    save_training_progress(
                        run_config=run_config,
                        run_paths=run_paths,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        history=history,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        stale_epochs=stale_epochs,
                        next_epoch=epoch + 1,
                        epochs=epochs,
                        elapsed_seconds=elapsed_seconds,
                        seed=seed,
                        training_settings=training_settings,
                        status="running",
                        stage="training",
                        message=epoch_message,
                    )
                if stale_epochs >= patience:
                    log_message(
                        f"[{run_config.name}] Early stopping at epoch {epoch} after {stale_epochs} stale epochs.",
                        run_paths.train_log,
                    )
                    break
        except (KeyboardInterrupt, RunInterrupted) as exc:
            elapsed_seconds = elapsed_before_resume + (time.perf_counter() - run_start_time)
            interrupt_message = f"[{run_config.name}] Interrupted. Latest progress saved to {run_paths.latest_checkpoint.as_posix()}"
            save_training_progress(
                run_config=run_config,
                run_paths=run_paths,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                history=history,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                stale_epochs=stale_epochs,
                next_epoch=current_epoch,
                epochs=epochs,
                elapsed_seconds=elapsed_seconds,
                seed=seed,
                training_settings=training_settings,
                status="interrupted",
                stage="training",
                message=str(exc),
                epoch_in_progress=current_epoch,
            )
            log_message(interrupt_message, run_paths.train_log)
            model.eval()
            return {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_total_loss": best_val_loss,
                "elapsed_seconds": elapsed_seconds,
                "status": "interrupted",
                "next_epoch": current_epoch,
            }

    if best_epoch <= 0 or not run_paths.best_checkpoint.exists():
        raise RuntimeError(f"{run_config.name} did not produce a usable best checkpoint.")

    elapsed_seconds = elapsed_before_resume + (time.perf_counter() - run_start_time)
    save_training_progress(
        run_config=run_config,
        run_paths=run_paths,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stale_epochs=stale_epochs,
        next_epoch=min(current_epoch + 1, epochs + 1),
        epochs=epochs,
        elapsed_seconds=elapsed_seconds,
        seed=seed,
        training_settings=training_settings,
        status="completed",
        stage="training",
        message="Training completed.",
    )
    model.eval()
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val_loss,
        "elapsed_seconds": elapsed_seconds,
        "status": "completed",
        "next_epoch": min(current_epoch + 1, epochs + 1),
    }


def compute_reconstruction_scores(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_kind: str,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if model_kind == "vae":
                reconstruction, _, _ = model(batch)
            elif model_kind == "memae":
                reconstruction, _ = model(batch)
            else:
                reconstruction = model(batch)
            per_window_mse = torch.mean((reconstruction.float() - batch.float()) ** 2, dim=(1, 2))
            scores.append(per_window_mse.cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def summarize_model_result(
    *,
    run_config: ModelRunConfig,
    training_summary: dict[str, Any],
    metrics: dict[str, Any],
    threshold_meta: dict[str, Any],
    by_damage_group: dict[str, Any],
    by_condition: dict[str, Any],
    run_paths: RunPaths,
    device: torch.device,
    model: nn.Module,
    sample_batch_shape: list[int],
    sample_output_shape: list[int],
    freq_loss_weight: float,
) -> dict[str, Any]:
    checkpoint_size_bytes = run_paths.best_checkpoint.stat().st_size if run_paths.best_checkpoint.exists() else 0
    history = training_summary["history"]
    return {
        "model_name": run_config.name,
        "model_cli_name": run_config.cli_name,
        "model_kind": run_config.model_kind,
        "denoising": run_config.denoising,
        "metrics": metrics,
        "threshold": float(threshold_meta["threshold"]),
        "threshold_rule": threshold_meta["rule"],
        "threshold_meta": threshold_meta,
        "training": {
            "epochs_ran": len(history["train_total_loss"]),
            "best_epoch": int(training_summary["best_epoch"]),
            "elapsed_seconds": float(training_summary["elapsed_seconds"]),
            "final_train_loss": float(history["train_total_loss"][-1]),
            "final_val_loss": float(history["val_total_loss"][-1]),
            "final_train_time_loss": float(history["train_time_loss"][-1]),
            "final_val_time_loss": float(history["val_time_loss"][-1]),
            "final_train_freq_loss": float(history["train_freq_loss"][-1]),
            "final_val_freq_loss": float(history["val_freq_loss"][-1]),
            "final_train_kl_loss": float(history["train_kl_loss"][-1]),
            "final_val_kl_loss": float(history["val_kl_loss"][-1]),
            "final_train_mem_loss": float(history["train_mem_loss"][-1]),
            "final_val_mem_loss": float(history["val_mem_loss"][-1]),
            "history": history,
            "freq_loss_weight": float(freq_loss_weight),
            "parameter_count": int(parameter_count(model)),
            "checkpoint_size_bytes": int(checkpoint_size_bytes),
            "checkpoint_size_mb": float(checkpoint_size_bytes / (1024 * 1024)),
            "device": str(device),
            "input_batch_shape": sample_batch_shape,
            "output_batch_shape": sample_output_shape,
        },
        "by_damage_group": by_damage_group,
        "by_condition": by_condition,
        "artifacts": {
            "run_dir": run_paths.run_dir.as_posix(),
            "best_checkpoint": run_paths.best_checkpoint.as_posix(),
            "latest_checkpoint": run_paths.latest_checkpoint.as_posix(),
            "val_healthy_scores_npy": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores_npy": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores_npy": run_paths.test_fault_scores_npy.as_posix(),
            "history_json": run_paths.history_json.as_posix(),
            "status_json": run_paths.status_json.as_posix(),
            "metrics_json": run_paths.metrics_json.as_posix(),
            "report_md": run_paths.report_md.as_posix(),
            "plot_png": run_paths.plot_png.as_posix(),
            "train_log": run_paths.train_log.as_posix(),
        },
    }


def build_baseline_reference() -> dict[str, Any]:
    references: dict[str, Any] = {}
    if BASELINE_AE_METRICS_PATH.exists():
        references["CompactAE"] = read_json(BASELINE_AE_METRICS_PATH)
    if BASELINE_IFOREST_METRICS_PATH.exists():
        references["IsolationForest"] = read_json(BASELINE_IFOREST_METRICS_PATH)
    return references


def plot_summary(
    *,
    results: dict[str, dict[str, Any]],
    baseline_reference: dict[str, Any],
    path: Path,
) -> None:
    model_names = list(baseline_reference.keys()) + list(results.keys())
    metrics_for_plot = {"AUROC": [], "AUPRC": [], "F1": [], "Recall Fault": [], "False Alarm Rate": []}
    for model_name in model_names:
        payload = results[model_name]["metrics"] if model_name in results else baseline_reference[model_name]
        metrics_for_plot["AUROC"].append(float(payload["auroc"]))
        metrics_for_plot["AUPRC"].append(float(payload["auprc"]))
        metrics_for_plot["F1"].append(float(payload["f1"]))
        metrics_for_plot["Recall Fault"].append(float(payload["recall_fault"]))
        metrics_for_plot["False Alarm Rate"].append(float(payload["false_alarm_rate"]))

    figure, axes = plt.subplots(1, len(metrics_for_plot), figsize=(18, 4.5), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    x_positions = np.arange(len(model_names))
    colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262FD"][: len(model_names)]
    for axis, (metric_name, values) in zip(axes, metrics_for_plot.items(), strict=True):
        axis.bar(x_positions, values, color=colors)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(model_names, rotation=25, ha="right")
        axis.set_title(metric_name)
        axis.grid(axis="y", alpha=0.3)
        if metric_name != "False Alarm Rate":
            axis.set_ylim(0.0, 1.05)
    figure.suptitle("Paderborn Generative Upgrade Comparison", fontsize=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def format_metric_row(label: str, payload: dict[str, Any]) -> str:
    return (
        "| "
        f"{label} | "
        f"{payload['auroc']:.6f} | "
        f"{payload['auprc']:.6f} | "
        f"{payload['f1']:.6f} | "
        f"{payload['precision']:.6f} | "
        f"{payload['recall_fault']:.6f} | "
        f"{payload['false_alarm_rate']:.6f} |"
    )


def format_overall_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Model | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(format_metric_row(label, payload) for label, payload in rows)
    return "\n".join(lines)


def format_subgroup_table(title: str, payload: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"| {title} | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate | Fault Windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, metrics in payload.items():
        lines.append(
            "| "
            f"{key} | "
            f"{metrics['auroc']:.6f} | "
            f"{metrics['auprc']:.6f} | "
            f"{metrics['f1']:.6f} | "
            f"{metrics['precision']:.6f} | "
            f"{metrics['recall_fault']:.6f} | "
            f"{metrics['false_alarm_rate']:.6f} | "
            f"{metrics['num_test_fault']} |"
        )
    return "\n".join(lines)


def determine_best_model(results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    scored = sorted(
        results.items(),
        key=lambda item: (
            item[1]["metrics"]["f1"],
            item[1]["metrics"]["auroc"],
            -item[1]["metrics"]["false_alarm_rate"],
        ),
        reverse=True,
    )
    return scored[0]


def build_interpretation(
    *,
    results: dict[str, dict[str, Any]],
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    best_name, best_payload = determine_best_model(results)
    compact_ae = baseline_reference.get("CompactAE")
    iforest = baseline_reference.get("IsolationForest")
    if compact_ae is None or iforest is None:
        return {
            "best_model": best_name,
            "best_model_is_competitive_with_iforest": None,
            "loss_vs_capacity_story": "Baseline comparison artifacts were incomplete, so the upgrade interpretation is partial.",
            "summary_note": "Baseline comparison artifacts were incomplete.",
        }

    best_metrics = best_payload["metrics"]
    competitive = best_metrics["f1"] >= (iforest["f1"] - 0.05) and best_metrics["auroc"] >= (iforest["auroc"] - 0.03)
    ae_f1_gain = best_metrics["f1"] - compact_ae["f1"]
    ae_auroc_gain = best_metrics["auroc"] - compact_ae["auroc"]
    ae_far_delta = best_metrics["false_alarm_rate"] - compact_ae["false_alarm_rate"]
    if ae_f1_gain > 0.15 and ae_auroc_gain > 0.15:
        story = "The Paderborn gap looked like both under-capacity and loss-design weakness: deeper residual context and spectral guidance materially improved ranking and detection."
    elif ae_f1_gain > 0.10:
        story = "The upgrade mostly points to an under-capacity issue, with the stronger generative backbone recovering much more recall than the compact AE."
    else:
        story = "The upgrade suggests only a partial capacity/loss gain so far; Paderborn likely needs both stronger modeling and more careful scoring/calibration."
    summary_note = (
        f"{best_name} closes most of the gap to Isolation Forest and puts the generative path back into the race on Paderborn."
        if competitive
        else f"{best_name} improves over the compact AE but still trails Isolation Forest enough that the generative path is promising rather than fully competitive."
    )
    return {
        "best_model": best_name,
        "best_model_is_competitive_with_iforest": competitive,
        "loss_vs_capacity_story": story,
        "summary_note": summary_note,
        "compact_ae_delta": {"f1": ae_f1_gain, "auroc": ae_auroc_gain, "false_alarm_rate": ae_far_delta},
    }


def build_report(
    *,
    preprocessing_config: dict[str, Any],
    label_map: dict[str, Any],
    results: dict[str, dict[str, Any]],
    baseline_reference: dict[str, Any],
    interpretation: dict[str, Any],
    run_paths: RunPaths,
    device: torch.device,
    batch_size: int,
    freq_loss_weight: float,
) -> str:
    overview_rows: list[tuple[str, dict[str, Any]]] = []
    if "CompactAE" in baseline_reference:
        overview_rows.append(("CompactAE", baseline_reference["CompactAE"]))
    if "IsolationForest" in baseline_reference:
        overview_rows.append(("IsolationForest", baseline_reference["IsolationForest"]))
    overview_rows.extend((name, payload["metrics"]) for name, payload in results.items())

    lines = [
        "# Paderborn Generative Upgrade Report",
        "",
        "## Label Provenance",
        f"- Total bearings: `{label_map['summary']['total_bearings']}`",
        f"- Verified bearings: `{label_map['summary']['verified_pdf_count']}`",
        f"- Inferred bearings: `{label_map['summary']['inferred_family_rule_count']}`",
        "- All current Paderborn evaluation labels remain bearing-family inferences; local support PDFs exist but were not parsed automatically in this pass.",
        "",
        "## Setup",
        f"- Device used: `{device}`",
        f"- Selected signal channel: `{preprocessing_config['channel']}`",
        f"- Window size: `{preprocessing_config['window_size']}`",
        f"- Stride: `{preprocessing_config['stride']}`",
        f"- Threshold rule: `mean_plus_3std`",
        f"- Frequency-loss weight: `{freq_loss_weight:.3f}`",
        f"- Effective batch size: `{batch_size}`",
        f"- CUDA available at runtime: `{torch.cuda.is_available()}`",
        "",
        "## Overall Comparison",
        format_overall_table(overview_rows),
        "",
        "## Interpretation",
        f"- Best upgraded generative model: `{interpretation['best_model']}`",
        f"- Competitive with Isolation Forest: `{interpretation['best_model_is_competitive_with_iforest']}`",
        f"- Capacity/loss read: {interpretation['loss_vs_capacity_story']}",
        f"- Summary note: {interpretation['summary_note']}",
        "",
    ]

    for model_name, payload in results.items():
        training = payload["training"]
        lines.extend(
            [
                f"## {model_name}",
                f"- Threshold: `{payload['threshold']:.6f}`",
                f"- AUROC: `{payload['metrics']['auroc']:.6f}`",
                f"- AUPRC: `{payload['metrics']['auprc']:.6f}`",
                f"- F1: `{payload['metrics']['f1']:.6f}`",
                f"- Precision: `{payload['metrics']['precision']:.6f}`",
                f"- Recall fault: `{payload['metrics']['recall_fault']:.6f}`",
                f"- False alarm rate: `{payload['metrics']['false_alarm_rate']:.6f}`",
                f"- Final train loss: `{training['final_train_loss']:.6f}`",
                f"- Final val loss: `{training['final_val_loss']:.6f}`",
                f"- Final train time loss: `{training['final_train_time_loss']:.6f}`",
                f"- Final val time loss: `{training['final_val_time_loss']:.6f}`",
                f"- Final train freq loss: `{training['final_train_freq_loss']:.6f}`",
                f"- Final val freq loss: `{training['final_val_freq_loss']:.6f}`",
                f"- Final train KL loss: `{training['final_train_kl_loss']:.6f}`",
                f"- Final val KL loss: `{training['final_val_kl_loss']:.6f}`",
                f"- Final train memory-entropy loss: `{training['final_train_mem_loss']:.6f}`",
                f"- Final val memory-entropy loss: `{training['final_val_mem_loss']:.6f}`",
                f"- Parameter count: `{training['parameter_count']}`",
                f"- Model size on disk: `{training['checkpoint_size_mb']:.3f}` MB",
                f"- Training time: `{training['elapsed_seconds']:.2f}` seconds",
                "",
                "### By Damage Group",
                format_subgroup_table("Damage Group", payload["by_damage_group"]),
                "",
                "### By Operating Condition",
                format_subgroup_table("Condition", payload["by_condition"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Saved Artifacts",
            f"- Run directory: `{run_paths.run_dir.as_posix()}`",
            f"- Best checkpoint: `{run_paths.best_checkpoint.as_posix()}`",
            f"- Latest checkpoint: `{run_paths.latest_checkpoint.as_posix()}`",
            f"- Validation healthy scores: `{run_paths.val_healthy_scores_npy.as_posix()}`",
            f"- Test healthy scores: `{run_paths.test_healthy_scores_npy.as_posix()}`",
            f"- Test fault scores: `{run_paths.test_fault_scores_npy.as_posix()}`",
            f"- Training history JSON: `{run_paths.history_json.as_posix()}`",
            f"- Run status JSON: `{run_paths.status_json.as_posix()}`",
            f"- Training log: `{run_paths.train_log.as_posix()}`",
            f"- Metrics JSON: `{run_paths.metrics_json.as_posix()}`",
            f"- Markdown report: `{run_paths.report_md.as_posix()}`",
            f"- Summary plot: `{run_paths.plot_png.as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def maybe_run_extra_seeds(interpretation: dict[str, Any], extra_seed_count: int) -> dict[str, Any]:
    if extra_seed_count <= 0:
        return {"executed": False, "reason": "extra seed sweep disabled"}
    return {
        "executed": False,
        "reason": "extra seed sweep is implemented as a future extension but was skipped in this first CPU-only run to keep the study tractable.",
        "requested_extra_seed_count": int(extra_seed_count),
        "best_model": interpretation["best_model"],
    }


def ensure_run_is_ready(run_paths: RunPaths, *, resume: bool) -> None:
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        path
        for path in (
            run_paths.latest_checkpoint,
            run_paths.best_checkpoint,
            run_paths.val_healthy_scores_npy,
            run_paths.test_healthy_scores_npy,
            run_paths.test_fault_scores_npy,
            run_paths.history_json,
            run_paths.status_json,
            run_paths.metrics_json,
            run_paths.report_md,
            run_paths.plot_png,
            run_paths.train_log,
        )
        if path.exists()
    ]
    if resume:
        if not run_paths.latest_checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume because {run_paths.latest_checkpoint.as_posix()} does not exist yet."
            )
        return
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite an existing run directory without --resume. "
            f"Existing artifacts: {', '.join(path.as_posix() for path in existing_outputs)}"
        )


def evaluate_and_write_run_outputs(
    *,
    run_config: ModelRunConfig,
    model: nn.Module,
    training_summary: dict[str, Any],
    loaders: dict[str, DataLoader],
    device: torch.device,
    metadata_root: Path,
    processed_root: Path,
    run_paths: RunPaths,
    preprocessing_config: dict[str, Any],
    label_map: dict[str, Any],
    baseline_reference: dict[str, Any],
    fault_labels: np.ndarray,
    fault_label_map: dict[str, int],
    sample_batch: torch.Tensor,
    batch_size: int,
    seed: int,
    freq_loss_weight: float,
    vae_beta_max: float,
    vae_kl_warmup_epochs: int,
    memae_memory_size: int,
    memae_shrink_threshold: float,
    memae_entropy_weight: float,
    memae_addressing: str,
    best_candidate_extra_seeds: int,
    threshold_rule: str,
    resume_used: bool,
) -> dict[str, Any]:
    best_payload = load_torch_payload(run_paths.best_checkpoint)
    model.load_state_dict(best_payload["state_dict"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        sample_output = model(sample_batch.to(device))
        sample_output_tensor = sample_output[0].cpu() if isinstance(sample_output, tuple) else sample_output.cpu()

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
    threshold_meta = select_threshold(val_scores, threshold_rule)
    metrics = evaluate_binary_scores(
        threshold=float(threshold_meta["threshold"]),
        test_healthy_errors=test_healthy_scores,
        test_fault_errors=test_fault_scores,
    )
    subgroups = subgroup_metrics_from_manifest(
        window_manifest_path=metadata_root / "window_manifest.csv",
        test_healthy_scores=test_healthy_scores,
        test_fault_scores=test_fault_scores,
        fault_labels=fault_labels[: test_fault_scores.shape[0]],
        fault_label_map=fault_label_map,
        threshold=float(threshold_meta["threshold"]),
    )
    result = summarize_model_result(
        run_config=run_config,
        training_summary=training_summary,
        metrics=metrics,
        threshold_meta=threshold_meta,
        by_damage_group=subgroups["by_damage_group"],
        by_condition=subgroups["by_condition"],
        run_paths=run_paths,
        device=device,
        model=model,
        sample_batch_shape=list(sample_batch.shape),
        sample_output_shape=list(sample_output_tensor.shape),
        freq_loss_weight=freq_loss_weight,
    )

    results = {run_config.name: result}
    interpretation = build_interpretation(results=results, baseline_reference=baseline_reference)
    metrics_payload = {
        "dataset": "paderborn",
        "selected_model": run_config.name,
        "selected_model_cli_name": run_config.cli_name,
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "run_dir": run_paths.run_dir.as_posix(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "resume_used": bool(resume_used),
        "threshold_rule": threshold_rule,
        "freq_loss_weight": float(freq_loss_weight),
        "vae_beta_max": float(vae_beta_max),
        "vae_kl_warmup_epochs": int(vae_kl_warmup_epochs),
        "memae_memory_size": int(memae_memory_size),
        "memae_shrink_threshold": float(memae_shrink_threshold),
        "memae_entropy_weight": float(memae_entropy_weight),
        "memae_addressing": memae_addressing,
        "label_provenance": label_map["summary"],
        "saved_score_arrays": {
            "val_healthy_scores": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores": run_paths.test_fault_scores_npy.as_posix(),
        },
        "baseline_reference": baseline_reference,
        "models": results,
        "interpretation": interpretation,
        "extra_seed_summary": maybe_run_extra_seeds(interpretation, best_candidate_extra_seeds),
    }
    write_json(run_paths.metrics_json, metrics_payload)
    write_text(
        run_paths.report_md,
        build_report(
            preprocessing_config=preprocessing_config,
            label_map=label_map,
            results=results,
            baseline_reference=baseline_reference,
            interpretation=interpretation,
            run_paths=run_paths,
            device=device,
            batch_size=batch_size,
            freq_loss_weight=freq_loss_weight,
        ),
    )
    plot_summary(results=results, baseline_reference=baseline_reference, path=run_paths.plot_png)

    status_payload = read_json(run_paths.status_json) if run_paths.status_json.exists() else {}
    status_payload.update(
        {
            "status": "completed",
            "stage": "complete",
            "message": "Training and evaluation completed.",
            "metrics_json": run_paths.metrics_json.as_posix(),
            "report_md": run_paths.report_md.as_posix(),
            "plot_png": run_paths.plot_png.as_posix(),
            "val_healthy_scores_npy": run_paths.val_healthy_scores_npy.as_posix(),
            "test_healthy_scores_npy": run_paths.test_healthy_scores_npy.as_posix(),
            "test_fault_scores_npy": run_paths.test_fault_scores_npy.as_posix(),
            "summary_metrics": {
                "auroc": float(result["metrics"]["auroc"]),
                "auprc": float(result["metrics"]["auprc"]),
                "f1": float(result["metrics"]["f1"]),
                "recall_fault": float(result["metrics"]["recall_fault"]),
                "false_alarm_rate": float(result["metrics"]["false_alarm_rate"]),
                "threshold": float(result["threshold"]),
            },
        }
    )
    write_json(run_paths.status_json, status_payload)
    log_message(
        f"[{run_config.name}] Final artifacts saved in {run_paths.run_dir.as_posix()}",
        run_paths.train_log,
    )
    return result


def main() -> int:
    require_torch()
    args = parse_args()
    set_seed(args.seed)

    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    array_paths = resolve_paths(processed_root)
    ensure_required_files(array_paths, metadata_root)
    required_extra = [
        metadata_root / "bearing_label_map.json",
        metadata_root / "preprocessing_config.json",
        metadata_root / "window_manifest.csv",
    ]
    missing_extra = [path for path in required_extra if not path.exists()]
    if missing_extra:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(path.as_posix() for path in missing_extra))

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    label_map = read_json(metadata_root / "bearing_label_map.json")
    fault_label_map = {key: int(value) for key, value in preprocessing_config["fault_label_map"].items()}
    fault_labels = load_label_array(array_paths.fault_labels)
    expected_width = int(preprocessing_config["window_size"])

    device = get_device()
    batch_size = args.batch_size_cuda if device.type == "cuda" else args.batch_size_cpu
    print(f"Device selected: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Effective batch size: {batch_size}")

    loaders = build_loader_bundle(
        array_paths=array_paths,
        batch_size=batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )
    sample_batch = next(iter(loaders["train"]))
    baseline_reference = build_baseline_reference()
    memae_shrink_threshold = (
        default_shrink_threshold(args.memae_memory_size)
        if args.memae_shrink_threshold is None
        else float(args.memae_shrink_threshold)
    )
    all_models = build_models(
        expected_width,
        args.dropout,
        memae_memory_size=args.memae_memory_size,
        memae_shrink_threshold=memae_shrink_threshold,
        memae_addressing=args.memae_addressing,
    )
    selected_models = select_models(all_models, args.model)
    if not selected_models:
        raise RuntimeError(f"No model matched --model {args.model}.")

    print_manual_run_commands(
        args=args,
        model_configs=[run_config for run_config, _model in all_models],
        artifacts_root=artifacts_root,
    )

    training_settings = {
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "freq_loss_weight": float(args.freq_loss_weight),
        "vae_beta_max": float(args.vae_beta_max),
        "vae_kl_warmup_epochs": int(args.vae_kl_warmup_epochs),
        "memae_memory_size": int(args.memae_memory_size),
        "memae_shrink_threshold": float(memae_shrink_threshold),
        "memae_entropy_weight": float(args.memae_entropy_weight),
        "memae_addressing": args.memae_addressing,
        "save_every_epochs": int(args.save_every_epochs),
        "batch_size": int(batch_size),
        "threshold_rule": args.threshold_rule,
        "dropout": float(args.dropout),
    }
    completed_results: dict[str, dict[str, Any]] = {}

    for run_config, model in selected_models:
        run_paths = build_run_paths(artifacts_root=artifacts_root, run_config=run_config, seed=args.seed)
        ensure_run_is_ready(run_paths, resume=args.resume)
        log_message(f"[{run_config.name}] Run directory: {run_paths.run_dir.as_posix()}", run_paths.train_log)

        training_summary = train_single_model(
            run_config=run_config,
            model=model,
            loaders=loaders,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            freq_loss_weight=args.freq_loss_weight,
            vae_beta_max=args.vae_beta_max,
            vae_warmup_epochs=args.vae_kl_warmup_epochs,
            memae_entropy_weight=args.memae_entropy_weight,
            run_paths=run_paths,
            seed=args.seed,
            save_every_epochs=args.save_every_epochs,
            resume=args.resume,
            training_settings=training_settings,
        )
        if training_summary["status"] != "completed":
            print(f"\nPartial progress saved for {run_config.name} in {run_paths.run_dir.as_posix()}", flush=True)
            return 130

        try:
            completed_results[run_config.name] = evaluate_and_write_run_outputs(
                run_config=run_config,
                model=model,
                training_summary=training_summary,
                loaders=loaders,
                device=device,
                metadata_root=metadata_root,
                processed_root=processed_root,
                run_paths=run_paths,
                preprocessing_config=preprocessing_config,
                label_map=label_map,
                baseline_reference=baseline_reference,
                fault_labels=fault_labels,
                fault_label_map=fault_label_map,
                sample_batch=sample_batch,
                batch_size=batch_size,
                seed=args.seed,
                freq_loss_weight=args.freq_loss_weight,
                vae_beta_max=args.vae_beta_max,
                vae_kl_warmup_epochs=args.vae_kl_warmup_epochs,
                memae_memory_size=args.memae_memory_size,
                memae_shrink_threshold=memae_shrink_threshold,
                memae_entropy_weight=args.memae_entropy_weight,
                memae_addressing=args.memae_addressing,
                best_candidate_extra_seeds=args.best_candidate_extra_seeds,
                threshold_rule=args.threshold_rule,
                resume_used=args.resume,
            )
        except KeyboardInterrupt:
            status_payload = read_json(run_paths.status_json) if run_paths.status_json.exists() else {}
            status_payload.update(
                {
                    "status": "interrupted",
                    "stage": "evaluation",
                    "message": "Evaluation was interrupted. Training checkpoints remain intact.",
                }
            )
            write_json(run_paths.status_json, status_payload)
            log_message(
                f"[{run_config.name}] Evaluation interrupted. Training checkpoints are still available in {run_paths.run_dir.as_posix()}",
                run_paths.train_log,
            )
            return 130

    print("\nPaderborn Generative Upgrade Summary")
    for model_name, payload in completed_results.items():
        metrics = payload["metrics"]
        print(
            f"  {model_name}: "
            f"AUROC={metrics['auroc']:.6f}, "
            f"AUPRC={metrics['auprc']:.6f}, "
            f"F1={metrics['f1']:.6f}, "
            f"Recall={metrics['recall_fault']:.6f}, "
            f"FAR={metrics['false_alarm_rate']:.6f}, "
            f"Threshold={payload['threshold']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
