"""CPU latency and stream-replay profile of the camera-ready ResDilatedAE-T (one session, one thread).

1. Per-window latency: batch size 1, single CPU thread, float32, on healthy test windows.
2. Stream replay: complete raw Paderborn recordings (64 kHz vibration_1) are replayed hop by hop
   (window 2048, hop 1024) through the full online pipeline: slice the newest window, z-score it with
   the training statistics, reconstruct, compute the MSE score, compare it with the calibrated tau.
   Real-time factor = recording duration / processing time (file loading is excluded).

Usage: .venv/bin/python scripts/profile_v3_cpu.py [--seed 42] [--output artifacts/v3/analysis/cpu_profile.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from preprocess_paderborn import extract_signal_channels
from train_paderborn_ablation import VARIANT_CONFIGS, build_model, parameter_count

ROOT = Path(__file__).resolve().parents[1]
FS, WIN, HOP = 64000, 2048, 1024
REPLAY = [f"{c}_{b}_1" for c in ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
          for b in ("K001", "KA04", "KI04", "KB23")]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latency-windows", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/v3/analysis/cpu_profile.json")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = ROOT / f"artifacts/v3/paderborn_ablation/resdilated_time/seed_{args.seed}"
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(VARIANT_CONFIGS["resdilated_time"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tau = float(np.percentile(np.load(run / "val_healthy_scores.npy").astype(np.float64), 99.5))
    norm = json.loads((ROOT / "data/metadata/paderborn/normalization_stats.json").read_text())
    mean, std = float(norm["mean"]), float(norm["std"])

    # 1. per-window latency on stored (already normalized) healthy test windows
    windows = np.load(ROOT / "data/processed/paderborn/test/healthy_windows.npy", mmap_mode="r")
    pool = np.ascontiguousarray(windows[: args.latency_windows + args.warmup], dtype=np.float32)
    times = []
    with torch.inference_mode():
        for i, w in enumerate(pool):
            x = torch.from_numpy(w).view(1, 1, WIN)
            t0 = time.perf_counter()
            r = model(x)
            float(torch.mean((r - x) ** 2))
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(dt * 1e3)
    lat = np.array(times)

    # 2. stream replay through the full online pipeline
    replay = []
    with torch.inference_mode():
        for mid in REPLAY:
            bearing = mid.split("_")[3]
            path = ROOT / "data/raw/paderborn" / bearing / f"{mid}.mat"
            sig = np.asarray(extract_signal_channels(loadmat(path, simplify_cells=True)[mid])["vibration_1"],
                             dtype=np.float32).reshape(-1)
            n_win = 1 + (sig.size - WIN) // HOP
            alarms = 0
            t0 = time.perf_counter()
            for k in range(n_win):
                w = (sig[k * HOP: k * HOP + WIN] - mean) / std
                x = torch.from_numpy(w).view(1, 1, WIN)
                s = float(torch.mean((model(x) - x) ** 2))
                alarms += s > tau
            elapsed = time.perf_counter() - t0
            duration = sig.size / FS
            replay.append({"recording": mid, "duration_s": duration, "windows": n_win, "elapsed_s": elapsed,
                           "realtime_factor": duration / elapsed, "ms_per_hop": 1e3 * elapsed / n_win,
                           "alarm_fraction": alarms / n_win})
            print(f"{mid:24s} {duration:5.2f} s signal in {elapsed:5.2f} s -> {duration / elapsed:5.1f}x real time, "
                  f"{1e3 * elapsed / n_win:5.2f} ms/hop, alarms {alarms / n_win:.3f}")

    total_dur = sum(r["duration_s"] for r in replay)
    total_el = sum(r["elapsed_s"] for r in replay)
    out = {
        "cpu": cpu_name(), "torch": torch.__version__, "threads": torch.get_num_threads(), "batch_size": 1,
        "seed": args.seed, "parameters": parameter_count(model),
        "checkpoint_mb": (run / "best.pt").stat().st_size / 2**20,
        "state_dict_mb": sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 2**20,
        "latency_ms": {"n": int(lat.size), "mean": float(lat.mean()), "median": float(np.median(lat)),
                       "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))},
        "budget_ms": 1e3 * HOP / FS,
        "replay": {"recordings": replay, "total_duration_s": total_dur, "total_elapsed_s": total_el,
                   "realtime_factor": total_dur / total_el, "ms_per_hop": 1e3 * total_el / sum(r["windows"] for r in replay)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "replay"}, indent=2))
    print(f"replay: {total_dur:.1f} s of signal in {total_el:.1f} s -> {total_dur / total_el:.1f}x real time")


if __name__ == "__main__":
    main()
