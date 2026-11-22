"""Fig. 2 of the camera-ready paper: post-alert inspection of one Paderborn window by ResDilatedAE-T.

The case is chosen by rule, not by eye: among the true-positive windows of the KA group in the hardest
condition (N09_M07_F10) at the seed's validation-p99.5 threshold, take the one with the median score.
The figure is drawn at IEEE single-column width so the fonts print at 7-8 pt.

Usage: .venv/bin/python scripts/make_fig2_v3.py [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_paderborn_ablation import VARIANT_CONFIGS, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 64000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--condition", default="N09_M07_F10")
    ap.add_argument("--group", default="KA")
    ap.add_argument("--output", type=Path, default=ROOT / "paper_latex/figures/explanation_case_v3.png")
    args = ap.parse_args()

    run = ROOT / f"artifacts/v3/paderborn_ablation/resdilated_time/seed_{args.seed}"
    val = np.load(run / "val_healthy_scores.npy").astype(np.float64)
    fault = np.load(run / "test_fault_scores.npy").astype(np.float64)
    tau = float(np.percentile(val, 99.5))

    cand = []
    with open(ROOT / "data/metadata/paderborn/window_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["subset"] == "test_fault" and r["condition_code"] == args.condition and r["damage_group"] == args.group:
                i = int(r["window_index"])
                if fault[i] > tau:
                    cand.append((fault[i], i, r["measurement_id"]))
    cand.sort()
    score, idx, mid = cand[len(cand) // 2]

    windows = np.load(ROOT / "data/processed/paderborn/test/fault_windows.npy", mmap_mode="r")
    x = np.array(windows[idx], dtype=np.float32)
    model = build_model(VARIANT_CONFIGS["resdilated_time"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    with torch.inference_mode():
        xh = model(torch.from_numpy(x).view(1, 1, -1)).view(-1).numpy().astype(np.float64)
    x = x.astype(np.float64)
    check = float(np.mean((x - xh) ** 2))
    assert abs(check - score) / score < 1e-3, (check, score)
    res = x - xh

    t = np.arange(x.size) / FS * 1e3
    fr = np.fft.rfftfreq(x.size, d=1.0 / FS)[1:] / 1e3
    spec = lambda v: np.log1p(np.abs(np.fft.rfft(v)))[1:]  # noqa: E731  (DC bin dropped)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    blue, green, red = "#1f3c88", "#1b998b", "#c0392b"
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 2.2), dpi=600, layout="constrained")
    (a, b), (c, d) = ax
    a.plot(t, x, color=blue, lw=0.45, label="Observed")
    a.plot(t, xh, color=green, lw=0.45, label="Healthy reconstruction")
    a.set(title="(a) Waveform", xlabel="Time (ms)", ylabel="Amplitude")
    b.plot(t, res, color=red, lw=0.45)
    b.set(title="(b) Residual $r=x-\\hat{x}$", xlabel="Time (ms)")
    c.plot(fr, spec(x), color=blue, lw=0.45)
    c.plot(fr, spec(xh), color=green, lw=0.45)
    c.set(title="(c) Spectrum", xlabel="Frequency (kHz)", ylabel="log(1+|FFT|)")
    d.plot(fr, spec(res), color=red, lw=0.45)
    d.set(title="(d) Residual spectrum", xlabel="Frequency (kHz)")
    for axis in (a, b):
        axis.set_xlim(0, t[-1])
    for axis in (c, d):
        axis.set_xlim(0, 32)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
               handlelength=1.5, columnspacing=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor="white")
    meta = {"seed": args.seed, "condition": args.condition, "group": args.group, "measurement_id": mid,
            "test_fault_index": idx, "score": score, "threshold": tau, "score_over_threshold": score / tau,
            "candidates": len(cand), "rule": "median-score true positive"}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
