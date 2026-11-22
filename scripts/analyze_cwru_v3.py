"""Summarize the camera-ready CWRU leave-one-load-out runs (full held-out healthy recording).

Reads the five-seed runs written by eval_cwru_resdilated_load_shift.py and eval_cwru_load_shift.py
with --healthy-test-scope full_recording, and reports:
  * mean rows per model (AUROC, F1, balanced accuracy, FAR),
  * ResDilatedAE per held-out load,
  * commissioning recalibration: refit the mean_plus_3std threshold on the first n healthy windows
    of the new load and measure FAR on the remaining healthy windows of that load.
Nothing is retrained; ResDilatedAE metrics are recomputed from the saved per-window score arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts" / "v3" / "cwru_full_recording"
DEFAULT_CLASSICAL_ROOT = PROJECT_ROOT / "artifacts" / "v3" / "cwru_full_recording_classical"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "v3" / "analysis" / "cwru_summary.json"
SEEDS = (42, 7, 123, 2024, 2026)
LOADS = (0, 1, 2, 3)
COMMISSIONING_N = (16, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--classical-root", type=Path, default=DEFAULT_CLASSICAL_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def alarm_metrics(healthy: np.ndarray, fault: np.ndarray, threshold: float) -> dict[str, float]:
    fp = int((healthy > threshold).sum())
    tp = int((fault > threshold).sum())
    fn = int(fault.size - tp)
    tn = int(healthy.size - fp)
    recall = tp / max(fault.size, 1)
    far = fp / max(healthy.size, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"recall": recall, "far": far, "precision": precision, "f1": f1,
            "ba": 0.5 * (recall + 1.0 - far), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def resdilated_fold(root: Path, seed: int, load: int) -> dict[str, np.ndarray]:
    fold = root / "generative_upgrades" / "cwru_load_shift" / "resdilated_ae" / f"seed_{seed}" / f"load_{load}"
    return {name: np.load(fold / f"{name}_scores.npy").astype(np.float64)
            for name in ("val_healthy", "test_healthy", "test_fault")}


def resdilated_metrics(scores: dict[str, np.ndarray], rule: str) -> dict[str, float]:
    val = scores["val_healthy"]
    threshold = float(val.mean() + 3.0 * val.std()) if rule == "mean_plus_3std" else float(np.percentile(val, 99))
    out = alarm_metrics(scores["test_healthy"], scores["test_fault"], threshold)
    labels = np.r_[np.zeros(scores["test_healthy"].size), np.ones(scores["test_fault"].size)]
    out["auroc"] = float(roc_auc_score(labels, np.r_[scores["test_healthy"], scores["test_fault"]]))
    return out


def baseline_fold_metrics(root: Path, seed: int, run_dir: str | None = None) -> dict[tuple[str, int], dict[str, float]]:
    run_dir = run_dir if run_dir is not None else f"baselines_seed_{seed}"
    payload = json.loads((root / run_dir / "metrics" / "cwru_load_shift_metrics.json").read_text())
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for fold in payload["folds"]:
        for model, entry in fold["models"].items():
            m = entry["metrics"]
            rows[(model, int(fold["held_out_load_hp"]))] = {
                "auroc": m["auroc"], "f1": m["f1"], "recall": m["recall_fault"], "far": m["false_alarm_rate"],
                "ba": 0.5 * (m["recall_fault"] + 1.0 - m["false_alarm_rate"]),
            }
    return rows


def summarize(per_seed_load: dict[int, dict[int, dict[str, float]]], keys=("auroc", "f1", "ba", "far", "recall")):
    """per_seed_load[seed][load] -> metrics. Returns overall (load means, std across loads) and per-load stats."""
    seeds = sorted(per_seed_load)
    per_load = {}
    for load in LOADS:
        per_load[load] = {k: (float(np.mean([per_seed_load[s][load][k] for s in seeds])),
                              float(np.std([per_seed_load[s][load][k] for s in seeds])))
                          for k in keys}
    overall = {k: (float(np.mean([per_load[l][k][0] for l in LOADS])), float(np.std([per_load[l][k][0] for l in LOADS])))
               for k in keys}
    return overall, per_load


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {"seeds": args.seeds, "protocol": "leave_one_load_out, full held-out healthy recording"}

    # ResDilatedAE from score arrays.
    for rule in ("mean_plus_3std", "percentile_99"):
        per = {s: {l: resdilated_metrics(resdilated_fold(args.root, s, l), rule) for l in LOADS} for s in args.seeds}
        overall, per_load = summarize(per)
        report[f"ResDilatedAE[{rule}]"] = {"overall": overall, "per_load": per_load}

    # Baselines from their metric JSON.
    base = {s: baseline_fold_metrics(args.root, s) for s in args.seeds}
    if args.classical_root.exists():
        for s in args.seeds:
            base[s].update(baseline_fold_metrics(args.classical_root, s, run_dir=f"seed_{s}"))
    for model in sorted({m for rows in base.values() for (m, _) in rows}):
        per = {s: {l: base[s][(model, l)] for l in LOADS} for s in args.seeds}
        overall, per_load = summarize(per)
        report[model] = {"overall": overall, "per_load": per_load}

    # Commissioning recalibration for ResDilatedAE (healthy test windows are in recording order).
    comm: dict[str, dict] = {}
    for n in COMMISSIONING_N:
        per = {}
        for s in args.seeds:
            per[s] = {}
            for l in LOADS:
                sc = resdilated_fold(args.root, s, l)
                head, tail = sc["test_healthy"][:n], sc["test_healthy"][n:]
                tau = float(head.mean() + 3.0 * head.std())
                m = alarm_metrics(tail, sc["test_fault"], tau)
                m["auroc"] = float("nan")
                per[s][l] = m
        overall, per_load = summarize(per, keys=("far", "recall", "ba", "f1"))
        comm[str(n)] = {"overall": overall, "per_load": per_load}
    report["commissioning_recalibration_ResDilatedAE"] = comm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    def fmt(ms):
        return f"{ms[0]:.3f}$\\pm${ms[1]:.3f}"

    print("Mean over held-out loads (each load averaged over seeds); +- = std across loads")
    for name, entry in report.items():
        if isinstance(entry, dict) and "overall" in entry:
            o = entry["overall"]
            print(f"{name:32s} AUROC {fmt(o['auroc'])}  F1 {fmt(o['f1'])}  BA {fmt(o['ba'])}  FAR {fmt(o['far'])}  recall {fmt(o['recall'])}")
    print("\nResDilatedAE per held-out load (mean+-std over seeds), mean_plus_3std")
    for l, m in report["ResDilatedAE[mean_plus_3std]"]["per_load"].items():
        print(f"  load {l}: AUROC {fmt(m['auroc'])}  F1 {fmt(m['f1'])}  BA {fmt(m['ba'])}  FAR {fmt(m['far'])}")
    print("\nCommissioning recalibration (ResDilatedAE): tau from first n healthy windows of the new load")
    for n, entry in comm.items():
        pl = entry["per_load"]
        print(f"  n={n:>3}: mean FAR {fmt(entry['overall']['far'])}  recall {fmt(entry['overall']['recall'])}  "
              + "  ".join(f"load{l} FAR {pl[l]['far'][0]:.3f}" for l in LOADS))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
