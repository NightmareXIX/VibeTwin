"""Summarize the camera-ready Paderborn runs from saved per-window score arrays (nothing is retrained).

Reads, for every model/seed run directory, val_healthy_scores.npy, test_healthy_scores.npy and
test_fault_scores.npy (higher = more anomalous). Unified-baseline runs that only kept the
threshold-rule layout (<run>/percentile_99_5/{val_healthy_scores,test_scores,test_labels}.npy)
are split back into healthy/fault using test_labels.npy.

Row alignment: scripts/preprocess_paderborn.py writes one manifest row per window, and within each
subset (val_healthy, test_healthy, test_fault) `window_index` is the running row offset into that
subset's array (start_index + local_index). So array row i <-> the manifest row of that subset with
window_index == i; recordings are `measurement_id`, ordered in time by window_index. Counts are
asserted for every run.

Threshold: tau = np.percentile(val_healthy_scores as float64, 99.5) (numpy's default linear
interpolation), exactly as `calibrate_threshold(..., "percentile_99_5")` in
eval_paderborn_baselines_unified.py and `threshold_value("val_p99_5")` in eval_paderborn_ablation.py.
Alarm = score >= tau by default (the unified evaluator's rule; the ablation evaluator uses >, pick
it with --alarm-rule gt). Precision with no alarms is 0 (sklearn zero_division=0).

Outputs: artifacts/v3/analysis/paderborn_summary.json and LaTeX row snippets under
artifacts/v3/analysis/latex/. Run `--self-test` for the synthetic known-answer tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = PROJECT_ROOT / "artifacts" / "v3"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "paderborn" / "window_manifest.csv"
DEFAULT_PREPROCESSING_CONFIG = PROJECT_ROOT / "data" / "metadata" / "paderborn" / "preprocessing_config.json"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
DEFAULT_ABLATION_ROOT = V3_ROOT / "paderborn_ablation"
DEFAULT_BASELINES_ROOT = V3_ROOT / "paderborn_baselines"
DEFAULT_TRANSFER_ROOT = V3_ROOT / "paderborn_transfer"  # scripts/run_v3_queue2.sh, run_v3_cpu.sh
DEFAULT_TRANSFER_MANIFEST_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn_v3"
DEFAULT_TRANSFER_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn_v3"
DEFAULT_OUTPUT = V3_ROOT / "analysis" / "paderborn_summary.json"
DEFAULT_LATEX_DIR = V3_ROOT / "analysis" / "latex"

SEEDS = (42, 7, 123, 2024, 2026)
TRANSFER_SEEDS = (42, 7, 123)
SUBSETS = ("val_healthy", "test_healthy", "test_fault")
PROCESSED_ARRAYS = {"val_healthy": "val/healthy_windows.npy", "test_healthy": "test/healthy_windows.npy",
                    "test_fault": "test/fault_windows.npy"}
FAR_TARGETS = (0.005, 0.0069, 0.01)
PERSISTENCE_K = (1, 2, 4, 8, 16)
RUN_METRICS = ("auroc", "auprc", "f1", "precision", "recall", "far", "ba", "tn", "fp", "fn", "tp", "threshold")

# key -> (LaTeX label, short label, directory templates). Templates are str.format'ed with
# {baselines_root} {ablation_root} {v3_root} {name} {seed}; the first that holds scores wins.
BASELINE_MODELS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "classical": ("Classical indicator", "Classical", ()),
    "ocsvm": ("OC-SVM (linear)", "OC-SVM (lin.)", ()),
    "ocsvm_rbf": ("OC-SVM (RBF)", "OC-SVM (RBF)", ()),
    "isolation_forest": ("Isolation Forest", "Isolation Forest", ()),
    "compact_ae": ("CompactAE", "CompactAE", ()),
    "conv_vae": ("ConvVAE", "ConvVAE", ("{baselines_root}/{name}/seed_{seed}",
                                        "{v3_root}/generative/generative_upgrades/{name}/seed_{seed}")),
    "deep_svdd": ("Deep SVDD", "Deep SVDD", ()),
    "memae": ("MemAE", "MemAE", ("{baselines_root}/{name}/seed_{seed}",
                                 "{v3_root}/generative/generative_upgrades/{name}/seed_{seed}")),
    # scripts/run_v3_queue.sh phase 3 writes the control to <baselines_root>_memae_control/memae/.
    "memae_control": ("MemAE (memory disabled)", "MemAE (mem. off)", (
        "{baselines_root}_memae_control/memae/seed_{seed}",
        "{v3_root}/generative_memae_control/generative_upgrades/memae/seed_{seed}",
        "{baselines_root}/memae_control/seed_{seed}",
        "{v3_root}/memae_control/generative_upgrades/memae/seed_{seed}",
    )),
    "resdilated_ae": ("ResDilatedAE (unified copy)", "ResDilatedAE (unif.)", ()),
}
# variant -> (label, residual, dilation, frequency loss)
ABLATION_VARIANTS: dict[str, tuple[str, bool, bool, bool]] = {
    "compact_ae": ("CompactAE", False, False, False),
    "compact_ae_freq": ("CompactAE-F", False, False, True),
    "dilated_time": ("DilatedAE-T", False, True, False),
    "dilated_ae": ("DilatedAE", False, True, True),
    "res_time": ("ResAE-T", True, False, False),
    "res_ae": ("ResAE", True, False, True),
    "resdilated_time": ("ResDilatedAE-T", True, True, False),
    "resdilated_full": ("ResDilatedAE", True, True, True),
}
FREQ_LOSS_PAIRS = (("compact_ae", "compact_ae_freq"), ("dilated_time", "dilated_ae"),
                   ("res_time", "res_ae"), ("resdilated_time", "resdilated_full"))
DEFAULT_MAIN_MODELS = ("baseline/classical", "baseline/ocsvm_rbf", "baseline/isolation_forest", "ablation/compact_ae",
                       "baseline/conv_vae", "baseline/deep_svdd", "baseline/memae", "baseline/memae_control",
                       "ablation/resdilated_time")
DEFAULT_REFERENCE = "ablation/resdilated_time"
DEFAULT_PERSISTENCE_MODELS = ("ablation/resdilated_time", "baseline/isolation_forest", "baseline/classical")
DEFAULT_TRANSFER_MODELS = ("baseline/classical", "baseline/isolation_forest", "ablation/resdilated_time")
# Transfer runs: <transfer_root>/<protocol>/fold_<k>/ or <transfer_root>/<protocol>_fold_<k>/, then an optional
# per-group subdirectory (train_paderborn_ablation / eval_paderborn_baselines_unified roots), then <name>/seed_<s>.
TRANSFER_FOLD_DIRS = ("{transfer_root}/{protocol}/fold_{fold}", "{transfer_root}/{protocol}_fold_{fold}")
TRANSFER_SUBDIRS = {"ablation": ("", "paderborn_ablation/", "ablation/"),
                    "baseline": ("", "paderborn_baselines/", "baselines/")}
PROTOCOL_LABELS = {"unseen_bearing": "Unseen healthy bearings", "loco": "Unseen operating condition"}


@dataclass
class ModelSpec:
    key: str
    group: str
    name: str
    label: str
    short_label: str
    templates: list[str]
    flags: tuple[bool, bool, bool] | None = None


@dataclass
class SubsetMeta:
    n: int
    window_index: np.ndarray
    recording: np.ndarray
    condition: np.ndarray
    damage: np.ndarray
    bearing: np.ndarray
    row_mapping: str
    _plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    path: Path
    subsets: dict[str, SubsetMeta]
    conditions: list[str]
    damage_groups: list[str]
    bearings: list[str]


class Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
                   help="Optional cross-check of manifest counts against the processed array shapes.")
    p.add_argument("--preprocessing-config", type=Path, default=DEFAULT_PREPROCESSING_CONFIG,
                   help="Source of the window stride for detection delays (falls back to --stride).")
    p.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    p.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES_ROOT)
    p.add_argument("--v3-root", type=Path, default=V3_ROOT)
    p.add_argument("--model-dir", action="append", default=[], metavar="KEY=TEMPLATE",
                   help="Extra run-dir template (tried first) for a model, e.g. "
                        "'memae_control=artifacts/v3/x/memae/seed_{seed}'. Unknown 'baseline/foo' keys add a model.")
    p.add_argument("--threshold-subdir", default="percentile_99_5",
                   help="Unified-evaluator subdir used when split score arrays are absent.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--main-models", nargs="+", default=list(DEFAULT_MAIN_MODELS),
                   help="Rows (in order) of paderborn_main_rows.tex.")
    p.add_argument("--reference-model", default=DEFAULT_REFERENCE)
    p.add_argument("--significance-exclude", nargs="*", default=["baseline/resdilated_ae"])
    p.add_argument("--significance-far-target", type=float, default=0.005)
    p.add_argument("--far-targets", type=float, nargs="+", default=list(FAR_TARGETS))
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--alarm-rule", choices=("ge", "gt"), default="ge",
                   help="ge: score >= tau (eval_paderborn_baselines_unified); gt: score > tau (eval_paderborn_ablation).")
    p.add_argument("--persistence-models", nargs="+", default=list(DEFAULT_PERSISTENCE_MODELS))
    p.add_argument("--persistence-k", type=int, nargs="+", default=list(PERSISTENCE_K))
    p.add_argument("--condition-models", nargs="+", default=[DEFAULT_REFERENCE])
    p.add_argument("--miss-models", nargs="+", default=[DEFAULT_REFERENCE])
    p.add_argument("--fs", type=float, default=64000.0, help="Sampling rate (Hz) for detection delays.")
    p.add_argument("--stride", type=int, default=None, help="Window hop in samples (default: preprocessing config, else 1024).")
    p.add_argument("--ablation-spread", choices=("std", "ci"), default="std",
                   help="Spread printed in ablation_rows.tex (std over seeds or 95%% CI half-width).")
    p.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER_ROOT)
    p.add_argument("--transfer-manifest-root", type=Path, default=DEFAULT_TRANSFER_MANIFEST_ROOT)
    p.add_argument("--transfer-processed-root", type=Path, default=DEFAULT_TRANSFER_PROCESSED_ROOT)
    p.add_argument("--transfer-protocols", nargs="+", default=["unseen_bearing", "loco"])
    p.add_argument("--transfer-models", nargs="+", default=list(DEFAULT_TRANSFER_MODELS))
    p.add_argument("--transfer-seeds", type=int, nargs="+", default=list(TRANSFER_SEEDS))
    p.add_argument("--transfer-model-dir", action="append", default=[], metavar="KEY=TEMPLATE",
                   help="Extra transfer run-dir template (tried first); fields {transfer_root} {fold_root} "
                        "{protocol} {fold} {name} {group} {seed}.")
    p.add_argument("--no-main", action="store_true", help="Skip the main-protocol sections.")
    p.add_argument("--no-transfer", action="store_true", help="Skip the transfer protocols.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    p.add_argument("--self-test", action="store_true", help="Run the synthetic known-answer tests and exit.")
    return p.parse_args(argv)


def build_specs(model_dir_overrides: list[str]) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for name, (label, short, extra) in BASELINE_MODELS.items():
        templates = list(extra) if extra else ["{baselines_root}/{name}/seed_{seed}"]
        specs[f"baseline/{name}"] = ModelSpec(f"baseline/{name}", "baseline", name, label, short, templates)
    for name, (label, res, dil, freq) in ABLATION_VARIANTS.items():
        specs[f"ablation/{name}"] = ModelSpec(f"ablation/{name}", "ablation", name, label, label,
                                              ["{ablation_root}/{name}/seed_{seed}"], (res, dil, freq))
    for item in model_dir_overrides:
        key, template = split_override(item)
        if "/" in key and key not in specs:
            group, name = key.split("/", 1)
            if group not in ("baseline", "ablation"):
                raise SystemExit(f"--model-dir key must start with baseline/ or ablation/: {key}")
            specs[key] = ModelSpec(key, group, name, name.replace("_", r"\_"), name.replace("_", r"\_"), [])
        key = resolve_key(key, specs)
        specs[key].templates.insert(0, template)
    return specs


def split_override(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise SystemExit(f"Expected KEY=TEMPLATE, got {item!r}")
    key, template = item.split("=", 1)
    return key.strip(), template.strip()


def resolve_key(name: str, specs: dict[str, ModelSpec]) -> str:
    if name in specs:
        return name
    matches = [key for key in specs if key.split("/", 1)[1] == name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"Unknown model {name!r}; known: {', '.join(specs)}")
    raise SystemExit(f"Ambiguous model {name!r}: use one of {matches}")


# ---------------------------------------------------------------------------------------- manifest


def load_manifest(path: Path, subsets: tuple[str, ...] = SUBSETS) -> Manifest:
    """Stream the few manifest columns needed and reorder each subset into array-row order."""
    columns = ("subset", "measurement_id", "window_index", "condition_code", "damage_group", "bearing_code")
    tables: dict[str, dict[str, int]] = {name: {} for name in ("recording", "condition", "damage", "bearing")}
    raw: dict[str, dict[str, list]] = {s: {c: [] for c in ("wi", "rec", "cond", "dmg", "brg")} for s in subsets}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(f"{path}: manifest lacks columns {missing}")
        i_sub, i_rec, i_wi, i_cond, i_dmg, i_brg = (header.index(c) for c in columns)
        rec_t, cond_t, dmg_t, brg_t = tables["recording"], tables["condition"], tables["damage"], tables["bearing"]
        for row in reader:
            bucket = raw.get(row[i_sub])
            if bucket is None:
                continue
            bucket["wi"].append(int(row[i_wi]))
            bucket["rec"].append(rec_t.setdefault(row[i_rec], len(rec_t)))
            bucket["cond"].append(cond_t.setdefault(row[i_cond], len(cond_t)))
            bucket["dmg"].append(dmg_t.setdefault(row[i_dmg], len(dmg_t)))
            bucket["brg"].append(brg_t.setdefault(row[i_brg], len(brg_t)))
    metas: dict[str, SubsetMeta] = {}
    for subset in subsets:
        b = raw[subset]
        wi = np.asarray(b["wi"], dtype=np.int64)
        n = int(wi.size)
        cols = {k: np.asarray(b[k], dtype=np.int32) for k in ("rec", "cond", "dmg", "brg")}
        if n and np.array_equal(wi, np.arange(n)):
            mapping = "array row = window_index = manifest file order"
        elif n and np.array_equal(np.sort(wi), np.arange(n)):
            order = np.argsort(wi, kind="stable")
            wi = wi[order]
            cols = {k: v[order] for k, v in cols.items()}
            mapping = "array row = window_index (manifest rows reordered)"
        else:
            mapping = "array row = manifest file order (window_index is not a per-subset row index)"
        metas[subset] = SubsetMeta(n, wi, cols["rec"], cols["cond"], cols["dmg"], cols["brg"], mapping)
    names = {k: [None] * len(t) for k, t in tables.items()}
    for k, t in tables.items():
        for value, code in t.items():
            names[k][code] = value
    return Manifest(path, metas, names["condition"], names["damage"], names["bearing"])


def count_manifest_subsets(path: Path) -> dict[str, int]:
    counts = {s: 0 for s in SUBSETS}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        i_sub = next(reader).index("subset")
        for row in reader:
            if row[i_sub] in counts:
                counts[row[i_sub]] += 1
    return counts


def processed_counts(root: Path) -> dict[str, int] | None:
    paths = {s: root / rel for s, rel in PROCESSED_ARRAYS.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    return {s: int(np.load(p, mmap_mode="r").shape[0]) for s, p in paths.items()}


# ------------------------------------------------------------------------------------------ scores


def load_npy(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float64).reshape(-1)


def load_run_scores(run_dir: Path, threshold_subdir: str) -> tuple[dict[str, np.ndarray], str] | None:
    """Split layout first, then the unified evaluator's <run>/<threshold_subdir>/test_scores.npy layout."""
    direct = {s: run_dir / f"{s}_scores.npy" for s in SUBSETS}
    if all(p.exists() for p in direct.values()):
        return {s: load_npy(p) for s, p in direct.items()}, run_dir.as_posix()
    candidates = [run_dir / threshold_subdir] + sorted(p for p in run_dir.glob("*") if p.is_dir())
    for sub in candidates:
        val, test, labels = sub / "val_healthy_scores.npy", sub / "test_scores.npy", sub / "test_labels.npy"
        if val.exists() and test.exists() and labels.exists():
            scores, y = load_npy(test), np.load(labels).reshape(-1)
            if y.size != scores.size:
                raise AssertionError(f"{sub}: test_labels ({y.size}) and test_scores ({scores.size}) differ in length")
            n_h = int((y == 0).sum())
            if not (np.all(y[:n_h] == 0) and np.all(y[n_h:] == 1)):
                raise AssertionError(f"{sub}: test_labels are not healthy-then-fault ordered")
            return {"val_healthy": load_npy(val), "test_healthy": scores[:n_h], "test_fault": scores[n_h:]}, sub.as_posix()
    return None


def find_run(spec: ModelSpec, seed: int, fields: dict[str, Any], threshold_subdir: str, templates: list[str]):
    for template in templates:
        run_dir = Path(template.format(seed=seed, name=spec.name, group=spec.group, **fields))
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if run_dir.is_dir():
            loaded = load_run_scores(run_dir, threshold_subdir)
            if loaded is not None:
                return loaded
    return None


def check_counts(scores: dict[str, np.ndarray], expected: dict[str, int] | None, where: str) -> None:
    for subset in SUBSETS:
        values = scores[subset]
        if values.size == 0:
            raise AssertionError(f"{where}: {subset} scores are empty")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{where}: {subset} scores contain non-finite values")
        if expected is not None and values.size != expected[subset]:
            raise AssertionError(f"{where}: {subset} has {values.size} scores but the manifest has "
                                 f"{expected[subset]} windows")


# ----------------------------------------------------------------------------------------- metrics


def percentile_threshold(values: np.ndarray, pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64).reshape(-1), pct))


def far_target_percentile(target: float) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError(f"FAR target must lie in (0, 1): {target}")
    return round(100.0 * (1.0 - target), 10)


def alarm(scores: np.ndarray, tau: float, rule: str) -> np.ndarray:
    return scores >= tau if rule == "ge" else scores > tau


def confusion(healthy: np.ndarray, fault: np.ndarray, tau: float, rule: str) -> dict[str, float]:
    fp = int(np.count_nonzero(alarm(healthy, tau, rule)))
    tp = int(np.count_nonzero(alarm(fault, tau, rule)))
    tn, fn = int(healthy.size - fp), int(fault.size - tp)
    recall = tp / fault.size if fault.size else math.nan
    far = fp / healthy.size if healthy.size else math.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "far": far, "precision": precision, "f1": f1, "ba": 0.5 * (recall + 1.0 - far),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": float(tau)}


def run_metrics(scores: dict[str, np.ndarray], args: argparse.Namespace, ranking: bool = True) -> dict[str, Any]:
    val, healthy, fault = scores["val_healthy"], scores["test_healthy"], scores["test_fault"]
    tau = percentile_threshold(val, args.percentile)
    out: dict[str, Any] = confusion(healthy, fault, tau, args.alarm_rule)
    out["ties_at_threshold"] = int(np.count_nonzero(healthy == tau) + np.count_nonzero(fault == tau))
    if ranking:
        y = np.r_[np.zeros(healthy.size, dtype=np.int8), np.ones(fault.size, dtype=np.int8)]
        s = np.r_[healthy, fault]
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    out["far_matched"] = {}
    for target in args.far_targets:
        tau_t = percentile_threshold(val, far_target_percentile(target))
        m = confusion(healthy, fault, tau_t, args.alarm_rule)
        out["far_matched"][fmt_target(target)] = {k: m[k] for k in ("threshold", "recall", "far", "ba", "f1", "precision")}
    return out


def fmt_target(target: float) -> str:
    return f"{target:g}"


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": None, "std": None, "ci95_halfwidth": None, "n": 0, "values": []}
    std = float(arr.std(ddof=1)) if n > 1 else None
    hw = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else None
    return {"mean": float(arr.mean()), "std": std, "ci95_halfwidth": hw, "n": n, "values": [float(v) for v in arr]}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values; NaN entries are left out of the family."""
    finite = [i for i, p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    m = len(finite)
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(finite, key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def welch(a: list[float], b: list[float]) -> dict[str, Any]:
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = {"n_a": int(a_arr.size), "n_b": int(b_arr.size), "mean_a": float(a_arr.mean()) if a_arr.size else None,
           "mean_b": float(b_arr.mean()) if b_arr.size else None, "t": None, "df": None, "p_raw": None, "note": ""}
    if a_arr.size:
        out["diff"] = float(a_arr.mean() - b_arr.mean()) if b_arr.size else None
    if a_arr.size < 2 or b_arr.size < 2:
        out["note"] = "fewer than two seeds in a group"
        return out
    if a_arr.std(ddof=1) == 0 and b_arr.std(ddof=1) == 0:
        out["note"] = "zero variance in both groups; Welch test undefined"
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    out.update(t=float(res.statistic), df=float(res.df), p_raw=float(res.pvalue))
    return out


# ------------------------------------------------------------------------------------- persistence


def persistence_plan(meta: SubsetMeta) -> dict[str, Any]:
    """Sort windows by (recording, window_index); consecutive window_index runs form segments."""
    if "order" not in meta._plan:
        order = np.lexsort((meta.window_index, meta.recording))
        rec, wi = meta.recording[order], meta.window_index[order]
        same_rec = np.r_[False, rec[1:] == rec[:-1]]
        consecutive = np.r_[False, wi[1:] == wi[:-1] + 1]
        new_seg = ~(same_rec & consecutive)
        seg_start = np.flatnonzero(new_seg)
        seg_id = np.cumsum(new_seg) - 1
        meta._plan.update(order=order, pos=np.arange(meta.n) - seg_start[seg_id],
                          seg_lengths=np.diff(np.r_[seg_start, meta.n]),
                          n_recordings=int(np.unique(rec).size), gap_breaks=int(np.count_nonzero(new_seg & same_rec)))
    return meta._plan


def persistence_counts(meta: SubsetMeta, k: int) -> dict[str, int]:
    plan = persistence_plan(meta)
    kept = int(np.count_nonzero(plan["pos"] >= k - 1))
    return {"total": meta.n, "kept": kept, "dropped": meta.n - kept, "recordings": plan["n_recordings"],
            "segments": int(plan["seg_lengths"].size), "segments_shorter_than_k": int(np.count_nonzero(plan["seg_lengths"] < k)),
            "gap_breaks": plan["gap_breaks"]}


def trailing_mean(scores: np.ndarray, meta: SubsetMeta, k: int) -> np.ndarray:
    """Mean of each window and its k-1 predecessors in the same recording; windows lacking them are dropped."""
    plan = persistence_plan(meta)
    s = scores[plan["order"]]
    if k == 1:
        return s
    ends = np.flatnonzero(plan["pos"] >= k - 1)
    if ends.size == 0:
        return np.empty(0, dtype=np.float64)
    return sliding_window_view(s, k).mean(axis=1)[ends - (k - 1)]


def persistence_for_run(scores: dict[str, np.ndarray], manifest: Manifest, args: argparse.Namespace) -> dict[int, dict]:
    out = {}
    for k in args.persistence_k:
        sm = {s: trailing_mean(scores[s], manifest.subsets[s], k) for s in SUBSETS}
        tau = percentile_threshold(sm["val_healthy"], args.percentile) if sm["val_healthy"].size else math.nan
        out[k] = confusion(sm["test_healthy"], sm["test_fault"], tau, args.alarm_rule)
    return out


# ------------------------------------------------------------------------- condition / miss analysis


def condition_for_run(scores: dict[str, np.ndarray], manifest: Manifest, tau_global: float,
                      args: argparse.Namespace) -> dict[str, Any]:
    val, th, tf = (scores[s] for s in SUBSETS)
    mv, mh, mf = (manifest.subsets[s].condition for s in SUBSETS)
    pred_h_cs, pred_f_cs = np.zeros(th.size, bool), np.zeros(tf.size, bool)
    per: dict[str, Any] = {}
    for c in sorted(set(np.unique(mh)) | set(np.unique(mf)), key=lambda c: manifest.conditions[c]):
        vm, hm, fm = mv == c, mh == c, mf == c
        fallback = not vm.any()
        tau_c = tau_global if fallback else percentile_threshold(val[vm], args.percentile)
        ph, pf = alarm(th[hm], tau_c, args.alarm_rule), alarm(tf[fm], tau_c, args.alarm_rule)
        pred_h_cs[hm], pred_f_cs[fm] = ph, pf
        gh, gf = alarm(th[hm], tau_global, args.alarm_rule), alarm(tf[fm], tau_global, args.alarm_rule)
        per[manifest.conditions[c]] = {
            "tau_condition": tau_c, "fallback_to_global": fallback, "n_val": int(vm.sum()),
            "n_test_healthy": int(hm.sum()), "n_test_fault": int(fm.sum()),
            "global": rates(gh, gf), "condition_specific": rates(ph, pf)}
    gh_all, gf_all = alarm(th, tau_global, args.alarm_rule), alarm(tf, tau_global, args.alarm_rule)
    return {"overall": {"global": rates(gh_all, gf_all), "condition_specific": rates(pred_h_cs, pred_f_cs)},
            "per_condition": per}


def rates(pred_healthy: np.ndarray, pred_fault: np.ndarray) -> dict[str, float]:
    recall = float(pred_fault.mean()) if pred_fault.size else math.nan
    far = float(pred_healthy.mean()) if pred_healthy.size else math.nan
    return {"recall": recall, "far": far, "ba": 0.5 * (recall + 1.0 - far)}


def misses_for_run(scores: dict[str, np.ndarray], manifest: Manifest, tau: float, args: argparse.Namespace) -> dict:
    miss = ~alarm(scores["test_fault"], tau, args.alarm_rule)
    meta = manifest.subsets["test_fault"]
    out = {}
    for name, codes, names in (("condition_code", meta.condition, manifest.conditions),
                               ("damage_group", meta.damage, manifest.damage_groups),
                               ("bearing_code", meta.bearing, manifest.bearings)):
        total = np.bincount(codes, minlength=len(names))
        missed = np.bincount(codes, weights=miss.astype(np.float64), minlength=len(names))
        out[name] = {names[c]: {"n_fault_windows": int(total[c]), "miss_rate": float(missed[c] / total[c]),
                                "share_of_misses": float(missed[c] / miss.sum()) if miss.any() else 0.0}
                     for c in np.flatnonzero(total)}
    return out


# ------------------------------------------------------------------------------------ aggregation


def aggregate_runs(runs: list[dict[str, Any]], keys=RUN_METRICS) -> dict[str, Any]:
    out = {k: summarize([r[k] for r in runs if k in r]) for k in keys}
    targets = runs[0]["far_matched"].keys() if runs and "far_matched" in runs[0] else []
    out["far_matched"] = {t: {m: summarize([r["far_matched"][t][m] for r in runs])
                              for m in ("threshold", "recall", "far", "ba", "f1", "precision")} for t in targets}
    return out


def aggregate_nested(per_seed: list[dict], leaf_keys=("recall", "far", "ba")) -> dict:
    """Average identically shaped nested per-seed dicts: floats -> summarize(), equal ints/bools/strs kept."""
    first = per_seed[0]
    out: dict[str, Any] = {}
    for k, v in first.items():
        values = [d[k] for d in per_seed]
        if isinstance(v, dict):
            out[k] = aggregate_nested(values, leaf_keys)
        elif isinstance(v, (bool, str, np.bool_)) or (
                isinstance(v, (int, np.integer)) and k not in leaf_keys and all(x == v for x in values)):
            out[k] = v
        else:
            out[k] = summarize(values)
    return out


def evaluate_model(spec: ModelSpec, seeds: list[int], fields: dict[str, Any], templates: list[str],
                   expected: dict[str, int] | None, args: argparse.Namespace, log: Log,
                   manifest: Manifest | None = None, extras: set[str] = frozenset(), where: str = "") -> dict | None:
    runs, persistence, condition, misses = [], {}, [], []
    for seed in seeds:
        try:
            loaded = find_run(spec, seed, fields, args.threshold_subdir, templates)
        except (OSError, ValueError, EOFError) as exc:  # half-written array from a run still in progress
            log.warn(f"{where}{spec.key} seed {seed}: unreadable scores ({exc}); skipped")
            continue
        if loaded is None:
            continue
        scores, source = loaded
        check_counts(scores, expected, f"{where}{spec.key} seed {seed} ({source})")
        record = run_metrics(scores, args)
        record.update(seed=seed, source=source)
        runs.append(record)
        if "persistence" in extras:
            for k, m in persistence_for_run(scores, manifest, args).items():
                persistence.setdefault(k, []).append(m)
        if "condition" in extras:
            condition.append(condition_for_run(scores, manifest, record["threshold"], args))
        if "misses" in extras:
            misses.append(misses_for_run(scores, manifest, record["threshold"], args))
    if not runs:
        tried = ", ".join(t.format(seed="{seed}", name=spec.name, group=spec.group, **fields) for t in templates)
        log.warn(f"{where}{spec.key}: no runs found (tried {tried}); skipped")
        return None
    found = [r["seed"] for r in runs]
    if len(found) < len(seeds):
        log.warn(f"{where}{spec.key}: only seeds {found} of {seeds} found")
    entry = {"label": spec.label, "group": spec.group, "seeds": found, "n_seeds": len(found),
             "runs": runs, "summary": aggregate_runs(runs)}
    if persistence:
        entry["persistence"] = persistence
    if condition:
        entry["condition_thresholds"] = aggregate_nested(condition)
    if misses:
        entry["miss_analysis"] = aggregate_nested(misses, ("miss_rate", "share_of_misses"))
    return entry


def persistence_section(entry: dict, manifest: Manifest, args: argparse.Namespace, stride: int) -> list[dict]:
    rows = []
    for k in args.persistence_k:
        per_seed = entry["persistence"][k]
        rows.append({"k": k, "detection_delay_s": (k - 1) * stride / args.fs,
                     "window_counts": {s: persistence_counts(manifest.subsets[s], k) for s in SUBSETS},
                     **{m: summarize([r[m] for r in per_seed]) for m in ("threshold", "recall", "far", "ba", "f1")}})
    return rows


def significance_section(models: dict[str, dict], reference: str, args: argparse.Namespace) -> dict[str, Any]:
    target = fmt_target(args.significance_far_target)
    if target not in models[reference]["runs"][0]["far_matched"]:
        raise SystemExit(f"--significance-far-target {target} must be one of --far-targets")
    others = [k for k in models if k != reference and k not in set(args.significance_exclude)]
    out: dict[str, Any] = {"reference": reference, "test": "Welch two-sample t-test (scipy.stats.ttest_ind, "
                           "equal_var=False) over per-seed values; Holm-Bonferroni within each metric family"}
    extract = {f"recall_at_far_{target}": lambda r: r["far_matched"][target]["recall"], "f1": lambda r: r["f1"]}
    for metric, get in extract.items():
        ref_vals = [get(r) for r in models[reference]["runs"]]
        rows = []
        for key in others:
            row = welch(ref_vals, [get(r) for r in models[key]["runs"]])
            rows.append({"model": key, "label": models[key]["label"], **row})
        for row, adj in zip(rows, holm([r["p_raw"] if r["p_raw"] is not None else math.nan for r in rows])):
            row["p_holm"] = adj
        out[metric] = sorted(rows, key=lambda r: (r["p_raw"] is None, r["p_raw"] or 0.0))
    return out


def ablation_section(models: dict[str, dict]) -> dict[str, Any]:
    variants = {}
    for name, (label, res, dil, freq) in ABLATION_VARIANTS.items():
        entry = models.get(f"ablation/{name}")
        if entry is None:
            continue
        variants[name] = {"label": label, "residual": res, "dilation": dil, "frequency_loss": freq,
                          "n_seeds": entry["n_seeds"],
                          **{m: entry["summary"][m] for m in ("auroc", "f1", "recall", "far", "ba", "auprc")}}
    pairs = []
    for without, with_freq in FREQ_LOSS_PAIRS:
        a, b = models.get(f"ablation/{with_freq}"), models.get(f"ablation/{without}")
        if a is None or b is None:
            continue
        row = {"topology_without_freq": without, "topology_with_freq": with_freq}
        for metric in ("f1", "auroc"):
            test = welch([r[metric] for r in a["runs"]], [r[metric] for r in b["runs"]])
            row[metric] = {"delta_mean_with_minus_without": test.get("diff"), **test}
        pairs.append(row)
    mean_delta = {m: (float(np.mean([p[m]["delta_mean_with_minus_without"] for p in pairs])) if pairs else None)
                  for m in ("f1", "auroc")}
    return {"variants": variants, "frequency_loss_pairs": pairs, "mean_delta_over_pairs": mean_delta}


# ---------------------------------------------------------------------------------------- transfer


def fold_ids(transfer_root: Path, manifest_root: Path, protocol: str) -> list[int]:
    ids = set()
    for d, prefix in ((transfer_root / protocol, "fold_"), (manifest_root / protocol, "fold_"),
                      (transfer_root, f"{protocol}_fold_")):
        if d.is_dir():
            for p in d.glob(f"{prefix}*"):
                if p.is_dir() and p.name[len(prefix):].isdigit():
                    ids.add(int(p.name[len(prefix):]))
    return sorted(ids)


def transfer_templates(group: str) -> list[str]:
    return [f"{root}/{sub}{{name}}/seed_{{seed}}" for root in TRANSFER_FOLD_DIRS for sub in TRANSFER_SUBDIRS[group]]


def transfer_section(specs: dict[str, ModelSpec], args: argparse.Namespace, log: Log) -> dict[str, Any]:
    out: dict[str, Any] = {}
    model_keys = [resolve_key(m, specs) for m in args.transfer_models]
    overrides: dict[str, list[str]] = {}
    for item in args.transfer_model_dir:
        key, template = split_override(item)
        overrides.setdefault(resolve_key(key, specs), []).append(template)
    for protocol in args.transfer_protocols:
        folds = fold_ids(args.transfer_root, args.transfer_manifest_root, protocol)
        if not folds:
            log.warn(f"transfer {protocol}: no fold directories under {args.transfer_root} "
                     f"or {args.transfer_manifest_root / protocol}; skipped")
            continue
        per_fold: dict[str, dict] = {}
        for fold in folds:
            fold_root = args.transfer_root / protocol / f"fold_{fold}"
            fold_manifest = args.transfer_manifest_root / protocol / f"fold_{fold}" / "window_manifest.csv"
            fields = {"fold_root": fold_root.as_posix(), "transfer_root": args.transfer_root.as_posix(),
                      "protocol": protocol, "fold": fold}
            expected, count_source = None, "none"
            fold_out: dict[str, Any] = {}
            for key in model_keys:
                spec = specs[key]
                templates = overrides.get(key, []) + transfer_templates(spec.group)
                if not any(find_dirs(templates, spec, fields, args.transfer_seeds)):
                    log.warn(f"transfer {protocol} fold_{fold} {key}: no runs found; skipped")
                    continue
                if count_source == "none":
                    expected, count_source = transfer_expected_counts(fold_manifest, args, protocol, fold, log)
                entry = evaluate_model(spec, args.transfer_seeds, fields, templates, expected, args, log,
                                       where=f"transfer {protocol} fold_{fold} ")
                if entry is None:
                    continue
                fold_out[key] = {"label": spec.short_label, "seeds": entry["seeds"], "runs": entry["runs"],
                                 **{m: entry["summary"][m] for m in ("auroc", "auprc", "f1", "ba", "recall", "far")}}
            if fold_out:
                fold_out["_count_check"] = count_source
                per_fold[f"fold_{fold}"] = fold_out
        summary = {}
        for key in model_keys:
            rows = [f[key] for f in per_fold.values() if key in f]
            if rows:
                summary[key] = {"label": specs[key].short_label, "n_folds": len(rows),
                                **{m: summarize([r[m]["mean"] for r in rows])
                                   for m in ("auroc", "auprc", "f1", "ba", "recall", "far")}}
        out[protocol] = {"folds": per_fold, "summary": summary}
    return out


def find_dirs(templates: list[str], spec: ModelSpec, fields: dict[str, Any], seeds: list[int]):
    for seed in seeds:
        for template in templates:
            path = Path(template.format(seed=seed, name=spec.name, group=spec.group, **fields))
            yield (path if path.is_absolute() else PROJECT_ROOT / path).is_dir()


def transfer_expected_counts(manifest: Path, args, protocol, fold, log: Log):
    on_disk = processed_counts(args.transfer_processed_root / protocol / f"fold_{fold}")
    if manifest.exists():
        counts = count_manifest_subsets(manifest)
        if on_disk is not None and on_disk != counts:
            raise AssertionError(f"transfer {protocol} fold_{fold}: manifest counts {counts} != processed shapes {on_disk}")
        return counts, manifest.as_posix() + (" (= processed array shapes)" if on_disk else "")
    if on_disk is not None:
        return on_disk, "processed array shapes"
    log.warn(f"transfer {protocol} fold_{fold}: no manifest or processed arrays; score counts NOT checked")
    return None, "unchecked"


# ----------------------------------------------------------------------------------------- output


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str) and value.startswith(PROJECT_ROOT.as_posix()):
        # store repo-relative paths so the summary carries no local directory names
        return value[len(PROJECT_ROOT.as_posix()):].lstrip("/") or "."
    return value


def cell(stat: dict | None, decimals: int, spread: str = "ci95_halfwidth") -> str:
    if not stat or stat.get("mean") is None:
        return "--"
    text = f"{stat['mean']:.{decimals}f}"
    if stat.get(spread) is not None:
        text += f"$\\pm${stat[spread]:.{decimals}f}"
    return text


def main_rows_tex(models: dict, main_keys: list[str], args) -> str:
    lines = [f"% Generated by scripts/analyze_paderborn_v3.py ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC).",
             f"% Paderborn, healthy-validation p{args.percentile:g} threshold, alarm score {'>=' if args.alarm_rule == 'ge' else '>'} tau.",
             "% Cells: mean$\\pm$95% CI half-width over seeds (t-distribution). FAR with 4 decimals.",
             "% Model & AUROC & AUPRC & F1 & BA & Recall & FAR \\\\"]
    for key in list(main_keys) + [k for k in models if k not in main_keys]:
        entry = models.get(key)
        prefix = "" if key in main_keys else "% extra: "
        if entry is None:
            lines.append(f"% {key}: no runs found")
            continue
        s = entry["summary"]
        cells = [cell(s[m], 3) for m in ("auroc", "auprc", "f1", "ba", "recall")] + [cell(s["far"], 4)]
        note = "" if entry["n_seeds"] == len(args.seeds) else f"  % n={entry['n_seeds']} seeds"
        lines.append(f"{prefix}{entry['label']} & " + " & ".join(cells) + " \\\\" + note)
    return "\n".join(lines) + "\n"


def ablation_rows_tex(ablation: dict, args) -> str:
    spread = "std" if args.ablation_spread == "std" else "ci95_halfwidth"
    lines = ["% Generated by scripts/analyze_paderborn_v3.py. Paderborn ablation, validation-p99.5 threshold.",
             f"% Cells: mean$\\pm${'sample std (ddof=1)' if spread == 'std' else '95% CI half-width'} over seeds.",
             "% Variant & Res. & Dil. & Freq. & AUROC & F1 & Recall & FAR \\\\"]
    for name, (label, *_flags) in ABLATION_VARIANTS.items():
        v = ablation["variants"].get(name)
        if v is None:
            lines.append(f"% {label}: no runs found")
            continue
        marks = ["\\checkmark" if v[f] else "--" for f in ("residual", "dilation", "frequency_loss")]
        cells = [cell(v[m], 3, spread) for m in ("auroc", "f1", "recall")] + [cell(v["far"], 4, spread)]
        lines.append(f"{label} & " + " & ".join(marks + cells) + " \\\\")
    for pair in ablation["frequency_loss_pairs"]:
        lines.append(f"% freq-loss delta {pair['topology_with_freq']} - {pair['topology_without_freq']}: "
                     + ", ".join(f"{m} {fnum(pair[m]['delta_mean_with_minus_without'], '+.3f')} "
                                 f"(Welch p={fnum(pair[m]['p_raw'], '.3g')})" for m in ("f1", "auroc")))
    return "\n".join(lines) + "\n"


def generalization_rows_tex(transfer: dict, keys: list[str]) -> str:
    lines = ["% Generated by scripts/analyze_paderborn_v3.py. Transfer protocols, validation-p99.5 threshold.",
             "% Cells: mean$\\pm$std over folds (each fold = mean over seeds). Per-fold rows are commented below.",
             "% Model & AUROC & BA & Recall & FAR \\\\"]
    fold_lines = []
    for i, (protocol, block) in enumerate(transfer.items()):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{PROTOCOL_LABELS.get(protocol, protocol.replace('_', ' '))}}}}}\\\\")
        for key in keys:
            s = block["summary"].get(key)
            if s is None:
                lines.append(f"% {key}: no runs found")
                continue
            cells = [cell(s[m], 3, "std") for m in ("auroc", "ba", "recall")] + [cell(s["far"], 4, "std")]
            lines.append(f"{s['label']} & " + " & ".join(cells) + f" \\\\  % {s['n_folds']} folds")
        for fold, fold_block in block["folds"].items():
            for key in keys:
                f = fold_block.get(key)
                if f is not None:
                    cells = [cell(f[m], 3, "std") for m in ("auroc", "ba", "recall")] + [cell(f["far"], 4, "std")]
                    fold_lines.append(f"% {protocol} {fold} ({len(f['seeds'])} seeds, mean$\\pm$std): "
                                      f"{f['label']} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + fold_lines) + "\n"


def fnum(value: Any, spec: str) -> str:
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else format(value, spec)


def ms(stat: dict | None, decimals: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.{decimals}f}" + (f"+-{stat['std']:.{decimals}f}" if stat.get("std") is not None else "")


def print_summary(report: dict, args) -> None:
    models = report.get("models", {})
    if models:
        print(f"\nPaderborn main protocol (p{args.percentile:g}, alarm {args.alarm_rule}); mean+-std over seeds")
        print(f"{'model':28s} {'n':>2s} {'AUROC':>13s} {'F1':>13s} {'BA':>13s} {'recall':>13s} {'FAR':>15s}")
        for key, e in models.items():
            s = e["summary"]
            print(f"{key:28s} {e['n_seeds']:>2d} {ms(s['auroc']):>13s} {ms(s['f1']):>13s} {ms(s['ba']):>13s} "
                  f"{ms(s['recall']):>13s} {ms(s['far'], 4):>15s}")
        print("\nFAR-matched recall (val-fitted thresholds); realized test FAR in brackets")
        for key, e in models.items():
            fm = e["summary"]["far_matched"]
            print(f"  {key:28s} " + "  ".join(f"@{t}: {ms(v['recall'])} [{ms(v['far'], 4)}]" for t, v in fm.items()))
    sig = report.get("significance")
    if sig:
        for metric in [k for k in sig if k not in ("reference", "test")]:
            print(f"\nWelch {sig['reference']} vs others, {metric} (Holm-adjusted)")
            for r in sig[metric]:
                print(f"  {r['model']:28s} diff {fnum(r.get('diff'), '+.3f'):>7s}  p {fnum(r['p_raw'], '.2e'):>9s}"
                      f"  p_holm {fnum(r['p_holm'], '.2e'):>9s} {r['note']}")
    for key, rows in report.get("persistence", {}).items():
        print(f"\nPersistence rule, {key}: k, delay[s], kept fault/healthy windows, recall, FAR, BA")
        for r in rows:
            c = r["window_counts"]
            print(f"  k={r['k']:>2d} {r['detection_delay_s']:.3f}s  {c['test_fault']['kept']:>6d}/{c['test_healthy']['kept']:>5d}"
                  f"  recall {ms(r['recall'])}  FAR {ms(r['far'], 4)}  BA {ms(r['ba'])}")
    for key, e in models.items():
        ct = e.get("condition_thresholds")
        if ct:
            print(f"\nCondition-specific thresholds, {key} (global -> condition-specific)")
            o = ct["overall"]
            print(f"  overall   recall {ms(o['global']['recall'])} -> {ms(o['condition_specific']['recall'])}  "
                  f"FAR {ms(o['global']['far'], 4)} -> {ms(o['condition_specific']['far'], 4)}")
            for c, v in ct["per_condition"].items():
                print(f"  {c:12s} recall {ms(v['global']['recall'])} -> {ms(v['condition_specific']['recall'])}  "
                      f"FAR {ms(v['global']['far'], 4)} -> {ms(v['condition_specific']['far'], 4)}")
        ma = e.get("miss_analysis")
        if ma:
            print(f"\nMiss rate at the global threshold, {key}")
            for group in ("condition_code", "damage_group"):
                print("  " + "  ".join(f"{g}: {ms(v['miss_rate'])}" for g, v in ma[group].items()))
    ab = report.get("ablation")
    if ab and ab["frequency_loss_pairs"]:
        print("\nFrequency-loss effect (with - without), Welch p")
        for p in ab["frequency_loss_pairs"]:
            print(f"  {p['topology_with_freq']:16s} dF1 {fnum(p['f1']['delta_mean_with_minus_without'], '+.3f')} "
                  f"(p {fnum(p['f1']['p_raw'], '.2g')})  dAUROC {fnum(p['auroc']['delta_mean_with_minus_without'], '+.3f')} "
                  f"(p {fnum(p['auroc']['p_raw'], '.2g')})")
    for protocol, block in report.get("transfer", {}).items():
        print(f"\nTransfer {protocol}: mean+-std over folds")
        for key, s in block["summary"].items():
            print(f"  {key:28s} folds {s['n_folds']}  AUROC {ms(s['auroc'])}  BA {ms(s['ba'])}  "
                  f"recall {ms(s['recall'])}  FAR {ms(s['far'], 4)}")


# ------------------------------------------------------------------------------------------- main


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    log = Log()
    specs = build_specs(args.model_dir)
    stride = args.stride
    if stride is None:
        stride = 1024
        if args.preprocessing_config.exists():
            stride = int(json.loads(args.preprocessing_config.read_text()).get("stride", 1024))
    report: dict[str, Any] = {"meta": {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": f"np.percentile(val_healthy_scores as float64, {args.percentile:g}) (linear interpolation)",
        "alarm_rule": "score >= tau" if args.alarm_rule == "ge" else "score > tau",
        "precision_when_no_alarms": 0.0, "ci": "95% t-interval over seeds (half-width), std is ddof=1",
        "far_targets": args.far_targets, "seeds": args.seeds, "transfer_seeds": args.transfer_seeds,
        "persistence": f"trailing mean of k windows within a recording (measurement_id, by window_index); windows "
                       f"without k-1 predecessors in the recording are dropped; tau_k = p{args.percentile:g} of "
                       f"equally smoothed val scores; delay = (k-1)*{stride}/{args.fs:g} s",
        "roots": {"ablation": args.ablation_root, "baselines": args.baselines_root, "transfer": args.transfer_root},
    }}
    if not args.no_main:
        t0 = time.time()
        manifest = load_manifest(args.manifest)
        expected = {s: manifest.subsets[s].n for s in SUBSETS}
        print(f"Loaded manifest in {time.time() - t0:.1f}s: {expected}", flush=True)
        on_disk = processed_counts(args.processed_root)
        if on_disk is not None and on_disk != expected:
            raise AssertionError(f"Manifest counts {expected} != processed array shapes {on_disk}")
        report["meta"].update(manifest=args.manifest, window_counts=expected, processed_arrays_match=on_disk is not None,
                              row_mapping={s: manifest.subsets[s].row_mapping for s in SUBSETS},
                              recording_gap_breaks={s: persistence_plan(manifest.subsets[s])["gap_breaks"] for s in SUBSETS})
        reference = resolve_key(args.reference_model, specs)
        extras_for: dict[str, set[str]] = {}
        for option, extra in ((args.persistence_models, "persistence"), (args.condition_models, "condition"),
                              (args.miss_models, "misses")):
            for name in option:
                extras_for.setdefault(resolve_key(name, specs), set()).add(extra)
        fields = {"baselines_root": args.baselines_root.as_posix(), "ablation_root": args.ablation_root.as_posix(),
                  "v3_root": args.v3_root.as_posix()}
        models: dict[str, dict] = {}
        for key, spec in specs.items():
            t0 = time.time()
            entry = evaluate_model(spec, args.seeds, fields, spec.templates, expected, args, log, manifest,
                                   extras_for.get(key, set()))
            if entry is not None:
                models[key] = entry
                print(f"  {key}: {entry['n_seeds']} seeds in {time.time() - t0:.1f}s", flush=True)
        report["models"] = models
        report["persistence"] = {k: persistence_section(e, manifest, args, stride)
                                 for k, e in models.items() if "persistence" in e}
        for e in models.values():
            e.pop("persistence", None)
        if reference in models:
            report["significance"] = significance_section(models, reference, args)
        else:
            log.warn(f"reference model {reference} has no runs; significance tests skipped")
        report["ablation"] = ablation_section(models)
        main_keys = [resolve_key(m, specs) for m in args.main_models]
        args.latex_dir.mkdir(parents=True, exist_ok=True)
        (args.latex_dir / "paderborn_main_rows.tex").write_text(main_rows_tex(models, main_keys, args))
        (args.latex_dir / "ablation_rows.tex").write_text(ablation_rows_tex(report["ablation"], args))
    if not args.no_transfer:
        report["transfer"] = transfer_section(specs, args, log)
        args.latex_dir.mkdir(parents=True, exist_ok=True)
        (args.latex_dir / "generalization_rows.tex").write_text(
            generalization_rows_tex(report["transfer"], [resolve_key(m, specs) for m in args.transfer_models]))
    report["warnings"] = log.warnings
    report["meta"]["elapsed_s"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    report = run_analysis(args)
    print_summary(report, args)
    if report["warnings"]:
        print(f"\n{len(report['warnings'])} warning(s); see 'warnings' in the JSON.")
    print(f"\nWrote {args.output} and LaTeX rows to {args.latex_dir} in {report['meta']['elapsed_s']}s")
    return 0


# -------------------------------------------------------------------------------------- self-test


def _write_manifest(path: Path, recordings: list[tuple[str, str, str, str, dict[str, int]]], shuffle: bool = False):
    """recordings: (measurement_id, condition, damage_group, bearing, {subset: n_windows}) in file order."""
    header = ["split", "subset", "health_status", "damage_group", "bearing_code", "condition_code",
              "measurement_id", "window_index", "window_start", "window_end"]
    offsets = {s: 0 for s in ("train_healthy", *SUBSETS)}
    rows = []
    for mid, cond, dmg, brg, counts in recordings:
        for subset, n in counts.items():
            for local in range(n):
                rows.append([subset.split("_")[0], subset, "damaged" if dmg else "healthy", dmg, brg, cond, mid,
                             offsets[subset] + local, local * 1024, local * 1024 + 2048])
            offsets[subset] += n
    if shuffle:
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return offsets


def _save_split(run_dir: Path, scores: dict[str, np.ndarray]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for subset, values in scores.items():
        np.save(run_dir / f"{subset}_scores.npy", np.asarray(values, dtype=np.float32))


def _save_unified(run_dir: Path, scores: dict[str, np.ndarray]) -> None:
    sub = run_dir / "percentile_99_5"
    sub.mkdir(parents=True, exist_ok=True)
    np.save(sub / "val_healthy_scores.npy", scores["val_healthy"].astype(np.float32))
    np.save(sub / "test_scores.npy", np.r_[scores["test_healthy"], scores["test_fault"]].astype(np.float32))
    np.save(sub / "test_labels.npy", np.r_[np.zeros(scores["test_healthy"].size, np.int64),
                                           np.ones(scores["test_fault"].size, np.int64)])


def self_test() -> int:
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    # Holm order: p = [.01, .04, .03, .005] -> [.03, .06, .06, .02]; NaN stays out of the family.
    check(np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]), "holm known answer")
    adj = holm([0.02, math.nan, 0.01])
    check(np.allclose([adj[0], adj[2]], [0.02, 0.02]) and math.isnan(adj[1]), "holm with NaN")
    check(np.allclose(holm([0.5, 0.6]), [1.0, 1.0]), "holm caps at 1")
    # t-interval: [1,2,3] -> mean 2, std 1, half-width t(.975,2)/sqrt(3) = 2.4841
    s = summarize([1.0, 2.0, 3.0])
    check(abs(s["ci95_halfwidth"] - 4.302652729749464 / math.sqrt(3)) < 1e-12 and s["std"] == 1.0, "t-interval")
    check(summarize([5.0])["ci95_halfwidth"] is None, "single-seed CI is undefined")
    # Threshold: percentile of arange(201) at 99.5 is exactly 199 -> ties separate >= from >.
    val = np.arange(201, dtype=np.float64)
    check(percentile_threshold(val, 99.5) == 199.0, "p99.5 of arange(201)")
    check(percentile_threshold(val, far_target_percentile(0.005)) == percentile_threshold(val, 99.5),
          "FAR target 0.005 reproduces the p99.5 threshold")
    healthy, fault = np.array([0.0, 199.0, 200.0, 5.0]), np.array([199.0, 300.0, 1.0])
    m = confusion(healthy, fault, 199.0, "ge")
    check((m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 2, 1, 2), "confusion >=")
    m = confusion(healthy, fault, 199.0, "gt")
    check((m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 2, 1), "confusion >")
    check(abs(m["ba"] - 0.5 * (1 / 3 + 1 - 0.25)) < 1e-12 and abs(m["precision"] - 0.5) < 1e-12, "BA/precision")
    check(confusion(np.zeros(3), np.zeros(2), 1.0, "ge")["precision"] == 0.0, "precision zero_division=0")
    # Welch matches scipy; degenerate variance is reported, not turned into p=0.
    w = welch([0.6, 0.62, 0.65], [0.4, 0.45, 0.41])
    ref = stats.ttest_ind([0.6, 0.62, 0.65], [0.4, 0.45, 0.41], equal_var=False)
    check(abs(w["p_raw"] - ref.pvalue) < 1e-15 and abs(w["diff"] - 0.61 / 3) < 1e-12, "welch")
    check(welch([1.0, 1.0], [2.0, 2.0])["p_raw"] is None, "welch degenerate")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Healthy recordings carry val/test windows; faults carry test_fault windows. Lengths vary (1..20).
        recordings = [
            ("H1", "C1", "", "K001", {"train_healthy": 4, "val_healthy": 5, "test_healthy": 4}),
            ("F1", "C1", "KA", "KA01", {"test_fault": 16}),
            ("H2", "C2", "", "K002", {"train_healthy": 9, "val_healthy": 20, "test_healthy": 7}),
            ("F2", "C2", "KI", "KI01", {"test_fault": 1}),
            ("H3", "C1", "", "K003", {"val_healthy": 3, "test_healthy": 1}),
            ("F3", "C2", "KB", "KB01", {"test_fault": 9}),
            ("F4", "C1", "KA", "KA02", {"test_fault": 2}),
        ]
        man_path = root / "meta" / "window_manifest.csv"
        counts = _write_manifest(man_path, recordings, shuffle=True)
        manifest = load_manifest(man_path)
        check({s: manifest.subsets[s].n for s in SUBSETS} == {"val_healthy": 28, "test_healthy": 12, "test_fault": 28},
              "manifest subset counts")
        check(all("reordered" in manifest.subsets[s].row_mapping for s in SUBSETS), "shuffled file order is re-sorted")
        check(count_manifest_subsets(man_path) == {s: counts[s] for s in SUBSETS}, "count-only manifest pass")
        lengths = {"val_healthy": [5, 20, 3], "test_healthy": [4, 7, 1], "test_fault": [16, 1, 9, 2]}
        for subset, ls in lengths.items():
            meta = manifest.subsets[subset]
            check(persistence_plan(meta)["gap_breaks"] == 0, "no gaps")
            for k in (1, 2, 4, 8, 16):
                c = persistence_counts(meta, k)
                kept = sum(max(0, L - k + 1) for L in ls)
                check(c["kept"] == kept and c["dropped"] == sum(ls) - kept, f"persistence counts {subset} k={k}")
                check(c["segments_shorter_than_k"] == sum(L < k for L in ls), f"short segments {subset} k={k}")
        c4 = persistence_counts(manifest.subsets["test_fault"], 4)
        check((c4["kept"], c4["dropped"]) == (19, 9), "fault k=4 kept 13+0+6+0")
        # Score = position within its recording -> the trailing mean ending at position p is p-(k-1)/2.
        meta = manifest.subsets["test_fault"]
        pos_score = np.empty(meta.n)
        for rec in np.unique(meta.recording):
            rows = np.flatnonzero(meta.recording == rec)
            pos_score[rows[np.argsort(meta.window_index[rows])]] = np.arange(rows.size)
        for k in (1, 2, 4, 8, 16):
            expected = np.concatenate([np.arange(k - 1, L) - (k - 1) / 2 for L in lengths["test_fault"] if L >= k])
            check(np.allclose(np.sort(trailing_mean(pos_score, meta, k)), np.sort(expected)), f"trailing mean k={k}")
        # A gap inside a recording splits it into two segments.
        gap_meta = SubsetMeta(6, np.array([0, 1, 2, 4, 5, 6]), np.zeros(6, np.int32), np.zeros(6, np.int32),
                              np.zeros(6, np.int32), np.zeros(6, np.int32), "test")
        check(persistence_counts(gap_meta, 3)["kept"] == 2 and persistence_plan(gap_meta)["gap_breaks"] == 1, "gap break")

        # End-to-end on synthetic run trees with known answers.
        rng = np.random.default_rng(1)
        seeds = [1, 2, 3]
        n = {s: manifest.subsets[s].n for s in SUBSETS}
        abl, base, v3 = root / "abl", root / "base", root / "v3"
        truth: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for seed in seeds:
            sc = {"val_healthy": rng.normal(0, 1, n["val_healthy"]), "test_healthy": rng.normal(0, 1, n["test_healthy"]),
                  "test_fault": rng.normal(2 + 0.1 * seed, 1, n["test_fault"])}
            sc = {k: v.astype(np.float32).astype(np.float64) for k, v in sc.items()}
            truth[("ablation/resdilated_time", seed)] = sc
            _save_split(abl / "resdilated_time" / f"seed_{seed}", sc)
            other = {k: v * (0.5 if k == "test_fault" else 1.0) for k, v in sc.items()}
            truth[("baseline/isolation_forest", seed)] = other
            _save_unified(base / "isolation_forest" / f"seed_{seed}", other)  # unified fallback layout
            perfect = {"val_healthy": np.arange(n["val_healthy"]) / n["val_healthy"],
                       "test_healthy": np.zeros(n["test_healthy"]), "test_fault": np.full(n["test_fault"], 10.0)}
            _save_split(base / "classical" / f"seed_{seed}", perfect)  # identical across seeds
            _save_split(v3 / "memae_control" / "generative_upgrades" / "memae" / f"seed_{seed}", sc)
        _save_split(abl / "resdilated_full" / "seed_1", truth[("ablation/resdilated_time", 1)])
        out = root / "out"
        argv = ["--manifest", str(man_path), "--processed-root", str(root / "none"), "--ablation-root", str(abl),
                "--baselines-root", str(base), "--v3-root", str(v3), "--seeds", *map(str, seeds),
                "--transfer-root", str(root / "transfer"), "--transfer-manifest-root", str(root / "tmeta"),
                "--transfer-processed-root", str(root / "none"),
                "--transfer-seeds", "1", "2", "--output", str(out / "summary.json"), "--latex-dir", str(out / "latex")]
        # Transfer: two folds; resdilated_time in the fold root, classical under paderborn_baselines/.
        fold_means = []
        for fold in (0, 1):
            fmeta = root / "tmeta" / "unseen_bearing" / f"fold_{fold}" / "window_manifest.csv"
            _write_manifest(fmeta, recordings)
            per_seed = []
            for seed in (1, 2):
                sc = {k: v + (0.5 * fold if k == "test_fault" else 0.0)
                      for k, v in truth[("ablation/resdilated_time", seed)].items()}
                _save_split(root / "transfer" / "unseen_bearing" / f"fold_{fold}" / "resdilated_time" / f"seed_{seed}", sc)
                _save_split(root / "transfer" / f"unseen_bearing_fold_{fold}" / "paderborn_baselines" / "classical"
                            / f"seed_{seed}", sc)
                tau = np.percentile(sc["val_healthy"], 99.5)
                per_seed.append(np.mean(sc["test_fault"] >= tau))
            fold_means.append(np.mean(per_seed))
        args = parse_args(argv)
        report = run_analysis(args)
        models = report["models"]
        check("ablation/res_time" not in models and any("ablation/res_time" in w for w in report["warnings"]),
              "missing model skipped with a warning")
        check(models["baseline/memae_control"]["n_seeds"] == 3, "memae_control found via fallback template")
        check(models["ablation/resdilated_full"]["n_seeds"] == 1, "partial seeds kept")
        for key in ("ablation/resdilated_time", "baseline/isolation_forest"):
            for run in models[key]["runs"]:
                sc = truth[(key, run["seed"])]
                tau = np.percentile(sc["val_healthy"], 99.5)
                tp, fp = int((sc["test_fault"] >= tau).sum()), int((sc["test_healthy"] >= tau).sum())
                check(run["threshold"] == tau and (run["tp"], run["fp"]) == (tp, fp), f"{key} counts")
                check(abs(run["auroc"] - roc_auc_score(np.r_[np.zeros(12), np.ones(28)],
                                                       np.r_[sc["test_healthy"], sc["test_fault"]])) < 1e-12, "auroc")
                check(run["far_matched"]["0.005"]["recall"] == run["recall"], "FAR-matched 0.005 == main")
                tau01 = np.percentile(sc["val_healthy"], 99.0)
                check(run["far_matched"]["0.01"]["recall"] == np.mean(sc["test_fault"] >= tau01), "FAR-matched 0.01")
        cl = models["baseline/classical"]["summary"]
        check(cl["auroc"]["mean"] == 1.0 and cl["recall"]["mean"] == 1.0 and cl["far"]["mean"] == 0.0, "perfect model")
        ref_runs = models["ablation/resdilated_time"]["runs"]
        rec = [r["recall"] for r in ref_runs]
        check(abs(models["ablation/resdilated_time"]["summary"]["recall"]["mean"] - np.mean(rec)) < 1e-12, "mean")
        # Significance: raw p equals scipy; adjusted p equals Holm over the family.
        sig = report["significance"]["f1"]
        for row in sig:
            other = [r["f1"] for r in models[row["model"]]["runs"]]
            p = welch([r["f1"] for r in ref_runs], other)["p_raw"]
            check(row["p_raw"] == p or (row["p_raw"] is None and p is None), "significance raw p")
        raw = [r["p_raw"] if r["p_raw"] is not None else math.nan for r in sig]
        check(all((math.isnan(a) and math.isnan(b if b is not None else math.nan)) or a == b
                  for a, b in zip(holm(raw), [r["p_holm"] if r["p_holm"] is not None else math.nan for r in sig])),
              "significance Holm")
        # Persistence: k=1 reproduces the raw metrics; counts are reported.
        pers = report["persistence"]["ablation/resdilated_time"]
        check(pers[0]["k"] == 1 and pers[0]["recall"]["values"] == rec, "k=1 equals raw recall")
        check(pers[2]["window_counts"]["test_fault"]["kept"] == 19 and abs(pers[2]["detection_delay_s"] - 3 * 1024 / 64000) < 1e-15,
              "persistence counts and delay in report")
        sc1 = truth[("ablation/resdilated_time", 1)]
        sm = {s: trailing_mean(sc1[s], manifest.subsets[s], 4) for s in SUBSETS}
        tau4 = np.percentile(sm["val_healthy"], 99.5)
        check(pers[2]["recall"]["values"][0] == np.mean(sm["test_fault"] >= tau4), "persistence recall k=4")
        # Condition-specific thresholds, seed 1, condition C2 (val windows of H2 only).
        cond = models["ablation/resdilated_time"]["condition_thresholds"]["per_condition"]["C2"]
        mv, mh, mf = (manifest.subsets[s].condition for s in SUBSETS)
        c2 = manifest.conditions.index("C2")
        tau_c2 = np.percentile(sc1["val_healthy"][mv == c2], 99.5)
        check(cond["condition_specific"]["recall"]["values"][0] == np.mean(sc1["test_fault"][mf == c2] >= tau_c2),
              "condition-specific recall")
        check(cond["n_val"] == 20 and cond["n_test_fault"] == 10 and cond["n_test_healthy"] == 7, "condition counts")
        check(cond["tau_condition"]["values"][0] == tau_c2 and cond["fallback_to_global"] is False, "tau_c reported")
        # Miss analysis, seed 1, damage group KA (F1 + F4 = 18 windows).
        miss = models["ablation/resdilated_time"]["miss_analysis"]["damage_group"]["KA"]
        tau1 = np.percentile(sc1["val_healthy"], 99.5)
        ka = manifest.damage_groups.index("KA")
        check(miss["n_fault_windows"] == 18 and miss["miss_rate"]["values"][0]
              == np.mean(sc1["test_fault"][manifest.subsets["test_fault"].damage == ka] < tau1), "miss rate KA")
        # Ablation pair delta and transfer fold averaging.
        pair = report["ablation"]["frequency_loss_pairs"]
        check(len(pair) == 1 and pair[0]["topology_with_freq"] == "resdilated_full", "only complete pairs reported")
        tr = report["transfer"]["unseen_bearing"]
        check(set(tr["folds"]) == {"fold_0", "fold_1"} and tr["summary"]["ablation/resdilated_time"]["n_folds"] == 2,
              "transfer folds")
        check(abs(tr["summary"]["ablation/resdilated_time"]["recall"]["mean"] - np.mean(fold_means)) < 1e-12,
              "transfer mean over folds of seed means")
        check(tr["summary"]["baseline/classical"]["n_folds"] == 2, "transfer template fallback")
        check(any("transfer loco" in w for w in report["warnings"]), "missing protocol warned")
        json.loads((out / "summary.json").read_text())
        tex = (out / "latex" / "paderborn_main_rows.tex").read_text()
        check("ResDilatedAE-T & " in tex and "$\\pm$" in tex and "% baseline/ocsvm_rbf: no runs found" in tex, "main tex")
        check("\\emph{Unseen healthy bearings}" in (out / "latex" / "generalization_rows.tex").read_text(), "gen tex")
        check("ResDilatedAE-T & \\checkmark & \\checkmark & --" in (out / "latex" / "ablation_rows.tex").read_text(),
              "ablation tex")
        # A score/manifest count mismatch must stop the analysis.
        _save_split(abl / "res_time" / "seed_1", {"val_healthy": np.zeros(27), "test_healthy": np.zeros(12),
                                                   "test_fault": np.zeros(28)})
        try:
            run_analysis(parse_args(argv + ["--no-transfer"]))
        except AssertionError as exc:
            check("27 scores but the manifest has 28" in str(exc), "count mismatch message")
        else:
            raise AssertionError("count mismatch was not detected")
    print(f"self-test passed: {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
