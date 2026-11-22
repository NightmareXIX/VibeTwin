"""How much of the ResDilatedAE-T Paderborn score comes from a constant (DC) offset of the residual?

The per-window score s = mean(r^2) splits exactly into mean(r)^2 (offset part) + var(r) (waveform part),
with r = x - x_hat. For every seed this script recomputes reconstructions for all val/test windows and
reports the offset share of the score, and AUROC/recall/FAR of the waveform part alone, thresholded at
its own validation p99.5 (alarm: score > tau). It also reports how well the raw window mean alone
separates fault from healthy windows.

Usage: .venv/bin/python scripts/check_dc_offset_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_paderborn_ablation import VARIANT_CONFIGS, build_model

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 7, 123, 2024, 2026)
ARRAYS = {"val_healthy": "val/healthy_windows.npy", "test_healthy": "test/healthy_windows.npy",
          "test_fault": "test/fault_windows.npy"}


def decompose(model, path: Path, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.load(path, mmap_mode="r")
    dc, ac, xm = [], [], []
    with torch.inference_mode():
        for i in range(0, a.shape[0], 4096):
            x = torch.from_numpy(np.ascontiguousarray(a[i:i + 4096])).float().to(device).unsqueeze(1)
            r = (x - model(x)).squeeze(1).double()
            dc.append((r.mean(1) ** 2).cpu().numpy())
            ac.append(r.var(1, unbiased=False).cpu().numpy())
            xm.append(x.squeeze(1).double().mean(1).cpu().numpy())
    return np.concatenate(dc), np.concatenate(ac), np.concatenate(xm)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = ROOT / "data/processed/paderborn"
    seed_dir = ROOT / "artifacts/v3/analysis/dc_offset"  # one JSON per seed, so an interrupted run resumes
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = {"seeds": {}}
    for seed in args.seeds:
        if (seed_dir / f"seed_{seed}.json").exists():
            continue
        run = ROOT / f"artifacts/v3/paderborn_ablation/resdilated_time/seed_{seed}"
        model = build_model(VARIANT_CONFIGS["resdilated_time"])
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["state_dict"])
        model.to(device).eval()
        d = {k: decompose(model, proc / p, device) for k, p in ARRAYS.items()}
        saved = np.load(run / "test_fault_scores.npy").astype(np.float64)
        full_f = d["test_fault"][0] + d["test_fault"][1]
        rel = np.abs(full_f - saved) / np.maximum(saved, 1e-12)
        assert np.corrcoef(full_f, saved)[0, 1] > 0.99999 and np.median(rel) < 1e-3, "recomputed scores differ from saved scores"
        y = np.r_[np.zeros(d["test_healthy"][0].size), np.ones(d["test_fault"][0].size)]
        res = {}
        for part, f in (("full", lambda t: t[0] + t[1]), ("waveform", lambda t: t[1]), ("offset", lambda t: t[0])):
            tau = np.percentile(f(d["val_healthy"]), 99.5)
            h, fa = f(d["test_healthy"]), f(d["test_fault"])
            res[part] = {"auroc": float(roc_auc_score(y, np.r_[h, fa])), "recall": float(np.mean(fa > tau)),
                         "far": float(np.mean(h > tau))}
        for k in ("test_healthy", "test_fault"):
            dc, ac, _ = d[k]
            res[f"offset_share_{k}"] = {"mean": float(np.mean(dc / (dc + ac))), "median": float(np.median(dc / (dc + ac)))}
        vm = d["val_healthy"][2].mean()
        res["auroc_raw_window_mean"] = float(roc_auc_score(y, np.abs(np.r_[d["test_healthy"][2], d["test_fault"][2]] - vm)))
        (seed_dir / f"seed_{seed}.json").write_text(json.dumps(res, indent=2))
        print(seed, json.dumps(res), flush=True)
    for seed in SEEDS:
        f = seed_dir / f"seed_{seed}.json"
        if not f.exists():
            print(f"seed {seed} not done yet; aggregate not written")
            return
        out["seeds"][seed] = json.loads(f.read_text())
    keys = ("full", "waveform", "offset")
    out["mean"] = {p: {m: float(np.mean([out["seeds"][s][p][m] for s in SEEDS])) for m in ("auroc", "recall", "far")}
                   for p in keys}
    for k in ("offset_share_test_healthy", "offset_share_test_fault"):
        out["mean"][k] = float(np.mean([out["seeds"][s][k]["mean"] for s in SEEDS]))
    out["mean"]["auroc_raw_window_mean"] = out["seeds"][SEEDS[0]]["auroc_raw_window_mean"]
    path = ROOT / "artifacts/v3/analysis/dc_offset_check.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["mean"], indent=2))


if __name__ == "__main__":
    main()
