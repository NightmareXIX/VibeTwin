from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from train_ae_baseline import require_torch, torch
    from train_generative_upgrades import (
        MEMAE_ADDRESSING,
        MEMAE_MEMORY_SIZE,
        build_models,
        load_torch_payload,
        make_loader,
        select_models,
        write_json,
    )
    from train_paderborn_baselines import resolve_paths
except ModuleNotFoundError:
    from scripts.train_ae_baseline import require_torch, torch
    from scripts.train_generative_upgrades import (
        MEMAE_ADDRESSING,
        MEMAE_MEMORY_SIZE,
        build_models,
        load_torch_payload,
        make_loader,
        select_models,
        write_json,
    )
    from scripts.train_paderborn_baselines import resolve_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"

SPLIT_CHOICES = ("val", "test_healthy", "test_fault")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MemAE memory-addressing diagnostics (surviving slot count, top-slot mass "
            "share, addressing entropy) for a trained checkpoint. train_generative_upgrades.py "
            "does not record these itself; verify_memae.py computes them only for its own probe "
            "training, not for a full run's saved checkpoint."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing checkpoints/best.pt")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="val")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <run-dir>/memory_diagnostics.json")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_memae_from_checkpoint(checkpoint_path: Path, *, window_size: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_torch_payload(checkpoint_path)
    checkpoint_model = payload.get("model_cli_name")
    if checkpoint_model is not None and checkpoint_model != "memae":
        raise RuntimeError(f"Expected a memae checkpoint, found model_cli_name={checkpoint_model!r}")

    settings = payload.get("training_settings", {})
    dropout = float(settings.get("dropout", 0.05))
    memae_memory_size = int(settings.get("memae_memory_size", MEMAE_MEMORY_SIZE))
    shrink_setting = settings.get("memae_shrink_threshold")
    memae_shrink_threshold = None if shrink_setting is None else float(shrink_setting)
    memae_addressing = str(settings.get("memae_addressing", MEMAE_ADDRESSING))

    candidates = select_models(
        build_models(
            window_size,
            dropout,
            memae_memory_size=memae_memory_size,
            memae_shrink_threshold=memae_shrink_threshold,
            memae_addressing=memae_addressing,
        ),
        "memae",
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one MemAE model definition, found {len(candidates)}")
    _run_config, model = candidates[0]
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"Checkpoint has no state dict: {checkpoint_path.as_posix()}")
    model.load_state_dict(state_dict)
    return model, settings


def collect_memory_usage(*, model: torch.nn.Module, loader, device: torch.device) -> dict[str, Any]:
    """Average addressing weights over every latent position scored by ``loader``."""
    model.eval()
    memory_size = int(model.memory.memory_size)
    usage = torch.zeros(memory_size, dtype=torch.float64)
    nonzero_total = 0.0
    entropy_total = 0.0
    positions = 0
    recon_sq_error_total = 0.0
    windows_scored = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction, attention = model(batch)
            per_window = torch.mean((reconstruction.float() - batch.float()) ** 2, dim=(1, 2))
            recon_sq_error_total += float(per_window.sum().item())
            windows_scored += int(batch.shape[0])

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
        "windows_scored": windows_scored,
        "positions_scored": positions,
        "recon_mse": recon_sq_error_total / max(windows_scored, 1),
        "utilized_slots": utilized,
        "utilized_fraction": utilized / float(memory_size),
        "top_slot_share": float(mean_usage[order[0]]),
        "top10_slot_share": float(mean_usage[order[:10]].sum()),
        "mean_surviving_slots_per_position": nonzero_total / divisor,
        "mean_addressing_entropy": entropy_total / divisor,
        "uniform_entropy": math.log(memory_size),
    }


def main() -> int:
    args = parse_args()
    require_torch()
    device = resolve_device(args.device)

    checkpoint_path = args.run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path.as_posix()}")

    model, settings = load_memae_from_checkpoint(checkpoint_path, window_size=2048)
    model = model.to(device)
    model.eval()

    array_paths = resolve_paths(args.processed_root)
    split_path = {
        "val": array_paths.val_healthy,
        "test_healthy": array_paths.test_healthy,
        "test_fault": array_paths.test_fault,
    }[args.split]
    loader = make_loader(split_path, batch_size=args.batch_size, shuffle=False)

    usage = collect_memory_usage(model=model, loader=loader, device=device)

    payload = {
        "run_dir": args.run_dir.resolve().as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "split": args.split,
        "device": device.type,
        "training_settings": settings,
        "memory_usage": usage,
    }

    output_path = args.output if args.output is not None else args.run_dir / "memory_diagnostics.json"
    write_json(output_path, payload)

    print(
        f"[{args.run_dir.name}] split={args.split} recon_mse={usage['recon_mse']:.6f} "
        f"surviving_slots={usage['mean_surviving_slots_per_position']:.1f}/{usage['memory_size']} "
        f"utilized={usage['utilized_slots']}/{usage['memory_size']} "
        f"top_slot_share={usage['top_slot_share']:.3%} "
        f"entropy={usage['mean_addressing_entropy']:.3f}/{usage['uniform_entropy']:.3f} "
        f"-> {output_path.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
