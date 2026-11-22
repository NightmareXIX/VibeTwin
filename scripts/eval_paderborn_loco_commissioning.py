"""Commissioning rule on the Paderborn leave-one-condition-out folds (no retraining).

For each fold, the held-out condition's healthy test windows are ordered in time within each
recording. Commissioning refits tau = p99.5 on the first n windows of the first recording
(lowest repeat_index) of ONE healthy bearing; those n windows are then dropped from the healthy
test set. The result is averaged over the six choices of commissioning bearing and the seeds.
The "global" row is the unchanged validation-p99.5 threshold for comparison.

Needs the LOCO fold window manifests (scripts/build_paderborn_transfer_splits.py) and the saved
per-window scores of the transfer runs under artifacts/v3/paderborn_transfer/loco/.

Usage: .venv/bin/python scripts/eval_paderborn_loco_commissioning.py
       [--output artifacts/v3/analysis/paderborn_loco_commissioning.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_paderborn_v3 import load_run_scores  # noqa: E402

ROOT = REPO / "artifacts/v3/paderborn_transfer/loco"
MAN = REPO / "data/metadata/paderborn_v3/loco"
SEEDS = (42, 7, 123)
MODELS = {
    "ResDilatedAE-T": "paderborn_ablation/resdilated_time/seed_{s}",
    "Isolation Forest": "baselines/isolation_forest/seed_{s}",
    "Classical": "baselines/classical/seed_{s}",
}
NS = (16, 64, 128)


def healthy_order(fold: str):
    rows = []
    with open(MAN / fold / "window_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["subset"] == "test_healthy":
                rows.append((int(r["window_index"]), r["bearing_code"], int(r["repeat_index"]),
                             int(r["window_start"]), r["condition_code"]))
    rows.sort()
    idx = np.array([r[0] for r in rows])
    assert np.array_equal(idx, np.arange(len(rows))), "window_index is not a running offset"
    by_bearing = defaultdict(list)
    for i, b, rep, start, _cond in rows:
        by_bearing[b].append((rep, start, i))
    first = {}
    for b, lst in by_bearing.items():
        lst.sort()
        rep0 = lst[0][0]
        first[b] = [i for rep, start, i in lst if rep == rep0]
    return len(rows), first, rows[0][4]


def metrics(h, fa, tau):
    far = float(np.mean(h > tau))
    rec = float(np.mean(fa > tau))
    return rec, far, (rec + 1 - far) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=REPO / "artifacts/v3/analysis/paderborn_loco_commissioning.json")
    args = ap.parse_args()
    out = {"seeds": list(SEEDS), "commissioning_windows": list(NS), "threshold": "p99.5 (alarm: score > tau)",
           "folds": {}}
    for fold in sorted(p.name for p in ROOT.iterdir()):
        n_h, first, cond = healthy_order(fold)
        print(f"== {fold} held-out {cond}: {n_h} healthy test windows, bearings {sorted(first)}")
        fold_out = {"held_out_condition": cond, "healthy_test_windows": n_h, "models": {}}
        for name, tmpl in MODELS.items():
            res = defaultdict(list)
            for s in SEEDS:
                got = load_run_scores(ROOT / fold / tmpl.format(s=s), "percentile_99_5")
                if got is None:
                    print("  missing", name, s)
                    continue
                sc = got[0]
                val, h, fa = (np.asarray(sc[k], dtype=np.float64) for k in ("val_healthy", "test_healthy", "test_fault"))
                assert h.size == n_h, (h.size, n_h)
                tau = np.percentile(val, 99.5)
                res["global"].append(metrics(h, fa, tau))
                for n in NS:
                    for b, ids in first.items():
                        c = np.array(ids[:n])
                        keep = np.ones(h.size, bool)
                        keep[c] = False
                        res[str(n)].append(metrics(h[keep], fa, np.percentile(h[c], 99.5)))
            model_out = {}
            for k, v in res.items():
                a = np.array(v)
                model_out[k] = {"recall": float(a[:, 0].mean()), "far": float(a[:, 1].mean()),
                                "far_max": float(a[:, 1].max()), "ba": float(a[:, 2].mean()), "runs": int(len(a))}
                print(f"  {name:17s} {k:>6s}: recall {a[:, 0].mean():.3f}  FAR {a[:, 1].mean():.4f} "
                      f"(max {a[:, 1].max():.4f})  BA {a[:, 2].mean():.3f}")
            fold_out["models"][name] = model_out
        out["folds"][fold] = fold_out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
