"""Build Paderborn transfer protocols (unseen-bearing and leave-one-condition-out) from the main processed arrays.

The main protocol (``scripts/preprocess_paderborn.py``) splits every healthy measurement contiguously in time
into train/val/test regions. This script re-partitions those already-windowed arrays into two harder protocols
without re-reading raw data:

* ``unseen_bearing`` (3 folds): healthy bearings are disjoint across train/val/test. Each bearing contributes
  all of its healthy windows (main train + val + test regions). The fault test set is every fault window.
* ``loco`` (4 folds, leave-one-condition-out): train/val are the main-protocol train/val healthy windows of the
  three other operating conditions; test-healthy is every healthy window of the held-out condition and
  test-fault is every fault window of that condition.

Every fold refits the global z-score on its own healthy-train windows: the main normalization is undone
(raw = x * std + mean, from the source ``normalization_stats.json``) and the fold mean/std is applied.
Outputs mirror the main layout so the training/evaluation scripts can be pointed at a fold with
``--processed-root``/``--metadata-root``:

    <output-processed-root>/<protocol>/fold_<k>/{train,val}/healthy_windows.npy
    <output-processed-root>/<protocol>/fold_<k>/test/{healthy_windows,fault_windows,fault_labels}.npy
    <output-metadata-root>/<protocol>/fold_<k>/{window_manifest.csv,normalization_stats.json,
        preprocessing_config.json,fold_summary.json,bearing_label_map.json,...}

Row order: every fold subset lists its windows in main-manifest order, i.e. by measurement (bearing, condition,
repeat as ordered by preprocessing) and then by window_start within the recording. For pooled healthy subsets
this keeps each recording's windows consecutive (train region, then val region, then test region; regions are
separated by the preprocessing guard gap, so window_start jumps at region boundaries).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
SOURCE_METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
OUTPUT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn_v3"
OUTPUT_METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn_v3"

SUBSETS = ("train_healthy", "val_healthy", "test_healthy", "test_fault")
SUBSET_CODE = {name: code for code, name in enumerate(SUBSETS)}
SUBSET_SPLIT = {"train_healthy": "train", "val_healthy": "val", "test_healthy": "test", "test_fault": "test"}
ARRAY_RELPATHS = {
    "train_healthy": Path("train") / "healthy_windows.npy",
    "val_healthy": Path("val") / "healthy_windows.npy",
    "test_healthy": Path("test") / "healthy_windows.npy",
    "test_fault": Path("test") / "fault_windows.npy",
}
FAULT_LABELS_RELPATH = Path("test") / "fault_labels.npy"

HEALTHY_BEARINGS = ("K001", "K002", "K003", "K004", "K005", "K006")
CONDITIONS = ("N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10")
PROTOCOLS = ("unseen_bearing", "loco")
UNSEEN_BEARING_FOLDS = (
    {"train": ("K004", "K005", "K006"), "val": ("K003",), "test": ("K001", "K002")},
    {"train": ("K006", "K001", "K002"), "val": ("K005",), "test": ("K003", "K004")},
    {"train": ("K002", "K003", "K004"), "val": ("K001",), "test": ("K005", "K006")},
)

REQUIRED_MANIFEST_COLUMNS = (
    "split",
    "subset",
    "health_status",
    "damage_group",
    "fault_label_int",
    "bearing_code",
    "condition_code",
    "measurement_id",
    "window_index",
    "window_start",
)
EXTRA_MANIFEST_COLUMNS = ("source_subset", "source_window_index")
# Small metadata files copied verbatim into every fold (label maps and raw-file inventory are fold-independent).
COPIED_METADATA_FILES = (
    "bearing_label_map.json",
    "bearing_label_map.md",
    "file_map.csv",
    "file_map.json",
    "prepare_report.md",
    "window_manifest_schema.md",
)
FLUSH_EVERY_BYTES = 512 * 1024 * 1024


@dataclass
class Manifest:
    path: Path
    header: list[str]
    columns: dict[str, int]
    lines: list[str]
    line_terminator: str
    subset: np.ndarray
    bearing: np.ndarray
    condition: np.ndarray
    damage_group: np.ndarray
    measurement: np.ndarray
    window_index: np.ndarray
    window_start: np.ndarray
    fault_label: np.ndarray


@dataclass
class FoldSpec:
    protocol: str
    fold: int
    description: str
    held_out: dict[str, Any]
    rows: dict[str, np.ndarray]

    @property
    def name(self) -> str:
        return f"fold_{self.fold}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unseen-bearing and leave-one-condition-out Paderborn folds from the main processed arrays.",
    )
    parser.add_argument("--source-processed-root", type=Path, default=SOURCE_PROCESSED_ROOT)
    parser.add_argument("--source-metadata-root", type=Path, default=SOURCE_METADATA_ROOT)
    parser.add_argument("--output-processed-root", type=Path, default=OUTPUT_PROCESSED_ROOT)
    parser.add_argument("--output-metadata-root", type=Path, default=OUTPUT_METADATA_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Fold indices to build (default: all).")
    parser.add_argument("--chunk-rows", type=int, default=8192, help="Windows per read/transform/write chunk.")
    parser.add_argument("--spot-check-rows", type=int, default=256, help="Random rows per array re-derived from source.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the spot-check row sample.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing fold directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print fold definitions and counts without writing.")
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive.")
    if args.spot_check_rows < 0:
        parser.error("--spot-check-rows must be non-negative.")
    return args


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        raise FileNotFoundError(f"Missing source window manifest: {path.as_posix()}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    if '"' in text:
        raise RuntimeError(f"{path.as_posix()} contains quoted CSV fields; this builder expects plain comma rows.")
    line_terminator = "\r\n" if "\r\n" in text[:4096] else "\n"
    lines = text.splitlines()
    del text
    header = lines[0].split(",")
    lines = lines[1:]
    columns = {name: index for index, name in enumerate(header)}
    missing = [name for name in REQUIRED_MANIFEST_COLUMNS if name not in columns]
    if missing:
        raise RuntimeError(f"Source manifest is missing columns: {missing}")
    clash = [name for name in EXTRA_MANIFEST_COLUMNS if name in columns]
    if clash:
        raise RuntimeError(f"Source manifest already has transfer columns {clash}; point --source-* at the main protocol.")

    count = len(lines)
    subset = np.empty(count, dtype=np.int8)
    window_index = np.empty(count, dtype=np.int64)
    window_start = np.empty(count, dtype=np.int64)
    fault_label = np.full(count, -1, dtype=np.int64)
    bearing: list[str] = []
    condition: list[str] = []
    damage_group: list[str] = []
    measurement_codes: dict[str, int] = {}
    measurement = np.empty(count, dtype=np.int32)
    width = len(header)
    c_subset = columns["subset"]
    c_bearing = columns["bearing_code"]
    c_condition = columns["condition_code"]
    c_damage = columns["damage_group"]
    c_measurement = columns["measurement_id"]
    c_window_index = columns["window_index"]
    c_window_start = columns["window_start"]
    c_fault = columns["fault_label_int"]
    for row_index, line in enumerate(lines):
        fields = line.split(",")
        if len(fields) != width:
            raise RuntimeError(f"Manifest row {row_index + 2} has {len(fields)} fields; expected {width}.")
        subset_name = fields[c_subset]
        if subset_name not in SUBSET_CODE:
            raise RuntimeError(f"Manifest row {row_index + 2} has unknown subset {subset_name!r}.")
        subset[row_index] = SUBSET_CODE[subset_name]
        window_index[row_index] = int(fields[c_window_index])
        window_start[row_index] = int(fields[c_window_start])
        if fields[c_fault]:
            fault_label[row_index] = int(fields[c_fault])
        bearing.append(fields[c_bearing])
        condition.append(fields[c_condition])
        damage_group.append(fields[c_damage])
        measurement[row_index] = measurement_codes.setdefault(fields[c_measurement], len(measurement_codes))

    return Manifest(
        path=path,
        header=header,
        columns=columns,
        lines=lines,
        line_terminator=line_terminator,
        subset=subset,
        bearing=np.asarray(bearing),
        condition=np.asarray(condition),
        damage_group=np.asarray(damage_group),
        measurement=measurement,
        window_index=window_index,
        window_start=window_start,
        fault_label=fault_label,
    )


def open_source_arrays(processed_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, relpath in ARRAY_RELPATHS.items():
        path = processed_root / relpath
        if not path.exists():
            raise FileNotFoundError(f"Missing source array: {path.as_posix()}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.dtype != np.float32:
            raise RuntimeError(f"Expected a 2D float32 array at {path.as_posix()}, got {array.shape} {array.dtype}.")
        arrays[name] = array
    labels_path = processed_root / FAULT_LABELS_RELPATH
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing source array: {labels_path.as_posix()}")
    fault_labels = np.load(labels_path, mmap_mode="r")
    widths = {array.shape[1] for array in arrays.values()}
    if len(widths) != 1:
        raise RuntimeError(f"Source arrays disagree on window width: {widths}")
    return arrays, fault_labels


def check_recording_order(manifest: Manifest, rows: np.ndarray, label: str) -> None:
    """Rows of one recording must be consecutive and strictly increasing in window_start."""
    if rows.size < 2:
        return
    measurement = manifest.measurement[rows]
    starts = manifest.window_start[rows]
    same = measurement[1:] == measurement[:-1]
    if np.any(starts[1:][same] <= starts[:-1][same]):
        raise RuntimeError(f"{label}: window_start is not increasing within a recording.")
    if int(np.count_nonzero(~same)) + 1 != int(np.unique(measurement).size):
        raise RuntimeError(f"{label}: a recording's windows are not consecutive.")


def validate_source(manifest: Manifest, arrays: dict[str, np.ndarray], fault_labels: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in SUBSETS:
        rows = np.flatnonzero(manifest.subset == SUBSET_CODE[name])
        if rows.size != arrays[name].shape[0]:
            raise RuntimeError(f"Manifest has {rows.size} {name} rows but the array has {arrays[name].shape[0]}.")
        if not np.array_equal(manifest.window_index[rows], np.arange(rows.size, dtype=np.int64)):
            raise RuntimeError(f"Manifest {name} window_index is not the 0..n-1 array row order.")
        counts[name] = int(rows.size)
    fault_rows = np.flatnonzero(manifest.subset == SUBSET_CODE["test_fault"])
    if fault_labels.shape != (fault_rows.size,):
        raise RuntimeError(f"fault_labels.npy shape {fault_labels.shape} != ({fault_rows.size},)")
    if not np.array_equal(manifest.fault_label[fault_rows], np.asarray(fault_labels)):
        raise RuntimeError("fault_labels.npy does not match the manifest fault_label_int column.")
    healthy_mask = manifest.subset != SUBSET_CODE["test_fault"]
    healthy_bearings = sorted(set(manifest.bearing[healthy_mask].tolist()))
    if healthy_bearings != list(HEALTHY_BEARINGS):
        raise RuntimeError(f"Unexpected healthy bearings in the source manifest: {healthy_bearings}")
    conditions = sorted(set(manifest.condition.tolist()))
    if conditions != list(CONDITIONS):
        raise RuntimeError(f"Unexpected operating conditions in the source manifest: {conditions}")
    if set(manifest.bearing[fault_rows].tolist()) & set(HEALTHY_BEARINGS):
        raise RuntimeError("A healthy bearing code appears among the fault rows.")
    check_recording_order(manifest, np.flatnonzero(healthy_mask), "source healthy rows")
    check_recording_order(manifest, fault_rows, "source fault rows")
    return counts


def build_unseen_bearing_specs(manifest: Manifest) -> list[FoldSpec]:
    healthy_mask = manifest.subset != SUBSET_CODE["test_fault"]
    fault_rows = np.flatnonzero(manifest.subset == SUBSET_CODE["test_fault"])
    specs: list[FoldSpec] = []
    for fold, definition in enumerate(UNSEEN_BEARING_FOLDS):
        rows = {
            f"{split}_healthy": np.flatnonzero(healthy_mask & np.isin(manifest.bearing, definition[split]))
            for split in ("train", "val", "test")
        }
        rows["test_fault"] = fault_rows
        specs.append(
            FoldSpec(
                protocol="unseen_bearing",
                fold=fold,
                description=(
                    "Unseen healthy bearing: train on "
                    + "/".join(definition["train"])
                    + ", validate on "
                    + "/".join(definition["val"])
                    + ", test healthy on "
                    + "/".join(definition["test"])
                    + " (each bearing contributes all of its main-protocol train/val/test healthy windows); "
                    "test fault = every fault window."
                ),
                held_out={
                    "train_bearings": sorted(definition["train"]),
                    "val_bearings": sorted(definition["val"]),
                    "test_bearings": sorted(definition["test"]),
                },
                rows=rows,
            )
        )
    return specs


def build_loco_specs(manifest: Manifest) -> list[FoldSpec]:
    healthy_mask = manifest.subset != SUBSET_CODE["test_fault"]
    specs: list[FoldSpec] = []
    for fold, condition in enumerate(CONDITIONS):
        held_out = manifest.condition == condition
        rows = {
            "train_healthy": np.flatnonzero((manifest.subset == SUBSET_CODE["train_healthy"]) & ~held_out),
            "val_healthy": np.flatnonzero((manifest.subset == SUBSET_CODE["val_healthy"]) & ~held_out),
            "test_healthy": np.flatnonzero(healthy_mask & held_out),
            "test_fault": np.flatnonzero((manifest.subset == SUBSET_CODE["test_fault"]) & held_out),
        }
        specs.append(
            FoldSpec(
                protocol="loco",
                fold=fold,
                description=(
                    f"Leave-one-condition-out: hold out {condition}. Train/val = main-protocol train/val healthy "
                    f"windows of the other three conditions; test healthy = every healthy window of {condition} "
                    f"(main train+val+test regions); test fault = every fault window of {condition}."
                ),
                held_out={
                    "held_out_condition": condition,
                    "train_val_conditions": [item for item in CONDITIONS if item != condition],
                },
                rows=rows,
            )
        )
    return specs


def validate_spec(manifest: Manifest, spec: FoldSpec) -> dict[str, bool]:
    label = f"{spec.protocol}/{spec.name}"
    for name in SUBSETS:
        rows = spec.rows[name]
        if rows.size == 0:
            raise RuntimeError(f"{label}: {name} is empty.")
        if np.any(np.diff(rows) <= 0):
            raise RuntimeError(f"{label}: {name} rows are not in manifest order.")
        is_fault = manifest.subset[rows] == SUBSET_CODE["test_fault"]
        if name == "test_fault" and not np.all(is_fault):
            raise RuntimeError(f"{label}: healthy rows leaked into test_fault.")
        if name != "test_fault" and np.any(is_fault):
            raise RuntimeError(f"{label}: fault rows leaked into {name}.")
        check_recording_order(manifest, rows, f"{label} {name}")

    bearings = {name: set(manifest.bearing[spec.rows[name]].tolist()) for name in SUBSETS}
    conditions = {name: set(manifest.condition[spec.rows[name]].tolist()) for name in SUBSETS}
    checks: dict[str, bool] = {}
    if spec.protocol == "unseen_bearing":
        disjoint = (
            not (bearings["train_healthy"] & bearings["val_healthy"])
            and not (bearings["train_healthy"] & bearings["test_healthy"])
            and not (bearings["val_healthy"] & bearings["test_healthy"])
        )
        if not disjoint:
            raise RuntimeError(f"{label}: healthy bearing sets overlap across train/val/test: {bearings}")
        expected = {
            "train_healthy": set(spec.held_out["train_bearings"]),
            "val_healthy": set(spec.held_out["val_bearings"]),
            "test_healthy": set(spec.held_out["test_bearings"]),
        }
        for name, wanted in expected.items():
            if bearings[name] != wanted:
                raise RuntimeError(f"{label}: {name} bearings {sorted(bearings[name])} != {sorted(wanted)}")
        healthy_total = sum(spec.rows[name].size for name in ("train_healthy", "val_healthy", "test_healthy"))
        if healthy_total != int(np.count_nonzero(manifest.subset != SUBSET_CODE["test_fault"])):
            raise RuntimeError(f"{label}: healthy windows were dropped or duplicated.")
        if spec.rows["test_fault"].size != int(np.count_nonzero(manifest.subset == SUBSET_CODE["test_fault"])):
            raise RuntimeError(f"{label}: test_fault must contain every fault window.")
        checks["healthy_bearings_disjoint_across_train_val_test"] = True
        checks["every_healthy_window_used_once"] = True
        checks["test_fault_is_all_fault_windows"] = True
    else:
        held_out = spec.held_out["held_out_condition"]
        if held_out in conditions["train_healthy"] or held_out in conditions["val_healthy"]:
            raise RuntimeError(f"{label}: held-out condition {held_out} appears in train/val.")
        if conditions["test_healthy"] != {held_out} or conditions["test_fault"] != {held_out}:
            raise RuntimeError(f"{label}: test subsets contain conditions other than {held_out}.")
        checks["held_out_condition_absent_from_train_val"] = True
        checks["test_subsets_only_held_out_condition"] = True
    checks["subsets_in_recording_order"] = True
    return checks


def iter_chunks(count: int, chunk_rows: int):
    for start in range(0, count, chunk_rows):
        yield start, min(start + chunk_rows, count)


def read_source_rows(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size and int(indices[-1]) - int(indices[0]) + 1 == indices.size:
        return np.asarray(array[int(indices[0]) : int(indices[-1]) + 1])
    return array[indices]


def gather_windows(
    manifest: Manifest,
    rows: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    width: int,
) -> np.ndarray:
    """Return the source (main-protocol normalized) float32 windows for manifest rows, in the given order."""
    output = np.empty((rows.size, width), dtype=np.float32)
    codes = manifest.subset[rows]
    source_index = manifest.window_index[rows]
    for code in np.unique(codes):
        selected = codes == code
        indices = source_index[selected]
        if np.all(np.diff(indices) > 0):
            output[selected] = read_source_rows(source_arrays[SUBSETS[int(code)]], indices)
        else:  # not expected (rows are in manifest order) but keep the mapping exact if it ever happens
            order = np.argsort(indices, kind="stable")
            values = read_source_rows(source_arrays[SUBSETS[int(code)]], indices[order])
            block = np.empty_like(values)
            block[order] = values
            output[selected] = block
    return output


def renormalize(windows: np.ndarray, source_stats: tuple[float, float], fold_stats: tuple[float, float]) -> np.ndarray:
    source_mean, source_std = source_stats
    fold_mean, fold_std = fold_stats
    raw = windows.astype(np.float64) * source_std + source_mean
    return ((raw - fold_mean) / fold_std).astype(np.float32)


def fit_fold_normalization(
    manifest: Manifest,
    rows: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    width: int,
    source_stats: tuple[float, float],
    chunk_rows: int,
) -> tuple[float, float, int]:
    source_mean, source_std = source_stats
    total = 0.0
    total_sq = 0.0
    count = 0
    for start, end in iter_chunks(rows.size, chunk_rows):
        raw = gather_windows(manifest, rows[start:end], source_arrays, width).astype(np.float64) * source_std + source_mean
        total += float(raw.sum(dtype=np.float64))
        total_sq += float(np.square(raw).sum(dtype=np.float64))
        count += int(raw.size)
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean * mean, 0.0))
    if std == 0.0:
        raise RuntimeError("Fold healthy-train windows have zero standard deviation.")
    return mean, std, count


def write_fold_array(
    path: Path,
    manifest: Manifest,
    rows: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    width: int,
    source_stats: tuple[float, float],
    fold_stats: tuple[float, float],
    chunk_rows: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(path, mode="w+", dtype=np.float32, shape=(int(rows.size), width))
    total = 0.0
    total_sq = 0.0
    pending_bytes = 0
    for start, end in iter_chunks(rows.size, chunk_rows):
        values = renormalize(gather_windows(manifest, rows[start:end], source_arrays, width), source_stats, fold_stats)
        output[start:end] = values
        total += float(values.sum(dtype=np.float64))
        total_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        pending_bytes += values.nbytes
        if pending_bytes >= FLUSH_EVERY_BYTES:
            output.flush()
            pending_bytes = 0
    output.flush()
    del output
    count = int(rows.size) * width
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean * mean, 0.0))
    return {
        "path": display_path(path),
        "shape": [int(rows.size), int(width)],
        "dtype": "float32",
        "bytes": int(path.stat().st_size),
        "written_mean": mean,
        "written_std": std,
    }


def array_moments(path: Path, chunk_rows: int) -> tuple[float, float]:
    array = np.load(path, mmap_mode="r")
    total = 0.0
    total_sq = 0.0
    for start, end in iter_chunks(array.shape[0], chunk_rows):
        block = np.asarray(array[start:end], dtype=np.float64)
        total += float(block.sum())
        total_sq += float(np.square(block).sum())
    count = int(array.size)
    mean = total / count
    return mean, math.sqrt(max(total_sq / count - mean * mean, 0.0))


def spot_check_array(
    path: Path,
    manifest: Manifest,
    rows: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    width: int,
    source_stats: tuple[float, float],
    fold_stats: tuple[float, float],
    sample_rows: int,
    rng: np.random.Generator,
) -> float:
    if sample_rows == 0:
        return 0.0
    written = np.load(path, mmap_mode="r")
    positions = np.sort(rng.choice(rows.size, size=min(sample_rows, rows.size), replace=False))
    expected = renormalize(gather_windows(manifest, rows[positions], source_arrays, width), source_stats, fold_stats)
    error = float(np.max(np.abs(np.asarray(written[positions]) - expected)))
    if error > 0.0:
        raise RuntimeError(f"Spot check failed for {path.as_posix()}: max abs error {error}")
    return error


def write_fold_manifest(path: Path, manifest: Manifest, spec: FoldSpec) -> None:
    columns = manifest.columns
    c_split = columns["split"]
    c_subset = columns["subset"]
    c_window_index = columns["window_index"]
    terminator = manifest.line_terminator
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join([*manifest.header, *EXTRA_MANIFEST_COLUMNS]) + terminator)
        for name in SUBSETS:
            buffer: list[str] = []
            for new_index, row in enumerate(spec.rows[name].tolist()):
                fields = manifest.lines[row].split(",")
                source_subset = fields[c_subset]
                source_window_index = fields[c_window_index]
                fields[c_split] = SUBSET_SPLIT[name]
                fields[c_subset] = name
                fields[c_window_index] = str(new_index)
                fields.append(source_subset)
                fields.append(source_window_index)
                buffer.append(",".join(fields))
                if len(buffer) >= 50_000:
                    handle.write(terminator.join(buffer) + terminator)
                    buffer.clear()
            if buffer:
                handle.write(terminator.join(buffer) + terminator)


def verify_written_manifest(path: Path, spec: FoldSpec, arrays: dict[str, Path]) -> None:
    expected_next = {name: 0 for name in SUBSETS}
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        c_subset = header.index("subset")
        c_window_index = header.index("window_index")
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            name = fields[c_subset]
            if int(fields[c_window_index]) != expected_next[name]:
                raise RuntimeError(f"{path.as_posix()}: {name} window_index out of sequence.")
            expected_next[name] += 1
    for name in SUBSETS:
        rows_in_array = int(np.load(arrays[name], mmap_mode="r").shape[0])
        if expected_next[name] != spec.rows[name].size or expected_next[name] != rows_in_array:
            raise RuntimeError(
                f"{path.as_posix()}: {name} manifest rows {expected_next[name]} != array rows {rows_in_array}."
            )


def counter_dict(values: np.ndarray) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(Counter(values.tolist()).items())}


def subset_summary(manifest: Manifest, rows: np.ndarray, name: str) -> dict[str, Any]:
    bearings = manifest.bearing[rows]
    conditions = manifest.condition[rows]
    summary: dict[str, Any] = {
        "n_windows": int(rows.size),
        "bearings": sorted(set(bearings.tolist())),
        "conditions": sorted(set(conditions.tolist())),
        "n_recordings": int(np.unique(manifest.measurement[rows]).size),
        "source_subset_counts": {
            SUBSETS[int(code)]: int(count)
            for code, count in zip(*np.unique(manifest.subset[rows], return_counts=True), strict=True)
        },
        "counts_by_condition": counter_dict(conditions),
    }
    if name == "test_fault":
        summary["n_bearings"] = len(summary["bearings"])
        summary["counts_by_damage_group"] = counter_dict(manifest.damage_group[rows])
        summary["counts_by_fault_label"] = counter_dict(manifest.fault_label[rows].astype(str))
    else:
        summary["counts_by_bearing"] = counter_dict(bearings)
        summary["counts_by_bearing_condition"] = counter_dict(
            np.char.add(np.char.add(bearings, "|"), conditions)
        )
    return summary


def prepare_output_dirs(processed_dir: Path, metadata_dir: Path, overwrite: bool) -> None:
    for directory in (processed_dir, metadata_dir):
        if directory.exists() and any(directory.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{directory.as_posix()} is not empty; pass --overwrite to rebuild it.")
            shutil.rmtree(directory)
    for directory in (processed_dir / "train", processed_dir / "val", processed_dir / "test", metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)


def build_fold(
    *,
    spec: FoldSpec,
    manifest: Manifest,
    source_arrays: dict[str, np.ndarray],
    source_fault_labels: np.ndarray,
    source_stats_payload: dict[str, Any],
    source_config: dict[str, Any],
    args: argparse.Namespace,
    checks: dict[str, bool],
) -> dict[str, Any]:
    started = time.perf_counter()
    width = int(source_arrays["train_healthy"].shape[1])
    processed_dir = args.output_processed_root / spec.protocol / spec.name
    metadata_dir = args.output_metadata_root / spec.protocol / spec.name
    prepare_output_dirs(processed_dir, metadata_dir, args.overwrite)
    source_stats = (float(source_stats_payload["mean"]), float(source_stats_payload["std"]))

    log(f"{spec.protocol}/{spec.name}: fitting fold z-score on {spec.rows['train_healthy'].size} healthy-train windows")
    fold_mean, fold_std, fit_samples = fit_fold_normalization(
        manifest, spec.rows["train_healthy"], source_arrays, width, source_stats, args.chunk_rows
    )
    fold_stats = (fold_mean, fold_std)
    log(f"{spec.protocol}/{spec.name}: fold mean={fold_mean:.6f} std={fold_std:.6f}")

    array_paths = {name: processed_dir / relpath for name, relpath in ARRAY_RELPATHS.items()}
    array_info: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed + 1000 * PROTOCOLS.index(spec.protocol) + spec.fold)
    spot_errors: dict[str, float] = {}
    for name in SUBSETS:
        log(f"{spec.protocol}/{spec.name}: writing {name} ({spec.rows[name].size} windows)")
        array_info[name] = write_fold_array(
            array_paths[name],
            manifest,
            spec.rows[name],
            source_arrays,
            width,
            source_stats,
            fold_stats,
            args.chunk_rows,
        )
        spot_errors[name] = spot_check_array(
            array_paths[name],
            manifest,
            spec.rows[name],
            source_arrays,
            width,
            source_stats,
            fold_stats,
            args.spot_check_rows,
            rng,
        )

    fault_labels_path = processed_dir / FAULT_LABELS_RELPATH
    fault_labels = np.asarray(source_fault_labels)[manifest.window_index[spec.rows["test_fault"]]].astype(np.int64)
    if not np.array_equal(fault_labels, manifest.fault_label[spec.rows["test_fault"]]):
        raise RuntimeError("Fold fault labels disagree with the manifest fault_label_int column.")
    np.save(fault_labels_path, fault_labels)
    array_info["fault_labels"] = {
        "path": display_path(fault_labels_path),
        "shape": [int(fault_labels.size)],
        "dtype": "int64",
        "bytes": int(fault_labels_path.stat().st_size),
    }

    readback_mean, readback_std = array_moments(array_paths["train_healthy"], args.chunk_rows)
    if abs(readback_mean) > 1e-3 or abs(readback_std - 1.0) > 1e-3:
        raise RuntimeError(f"Fold train array is not standardized: mean={readback_mean}, std={readback_std}")

    log(f"{spec.protocol}/{spec.name}: writing metadata")
    manifest_path = metadata_dir / "window_manifest.csv"
    write_fold_manifest(manifest_path, manifest, spec)
    verify_written_manifest(manifest_path, spec, array_paths)

    train_rows = spec.rows["train_healthy"]
    fit_bearings = sorted(set(manifest.bearing[train_rows].tolist()))
    fit_conditions = sorted(set(manifest.condition[train_rows].tolist()))
    normalization_stats = {
        "method": "zscore_global",
        "fit_split": "train",
        "fit_condition": "healthy",
        "mean": float(fold_mean),
        "std": float(fold_std),
        "num_train_windows": int(train_rows.size),
        "window_size": int(width),
        "num_train_samples": int(fit_samples),
        "output_dtype": "float32",
        "signal_channel": source_stats_payload.get("signal_channel", "vibration_1"),
        "protocol": spec.protocol,
        "fold": int(spec.fold),
        "fit_bearings": fit_bearings,
        "fit_conditions": fit_conditions,
        "refit_from": {
            "source_normalization_stats": display_path(args.source_metadata_root / "normalization_stats.json"),
            "source_mean": float(source_stats[0]),
            "source_std": float(source_stats[1]),
            "rule": "raw = x_source * source_std + source_mean; x_fold = (raw - mean) / std",
        },
    }
    write_json(metadata_dir / "normalization_stats.json", normalization_stats)

    preprocessing_config = dict(source_config)
    preprocessing_config["processed_root"] = display_path(processed_dir)
    preprocessing_config["metadata_root"] = display_path(metadata_dir)
    preprocessing_config["normalization"] = {
        "method": "zscore_global",
        "fit_on": "healthy_train_only",
        "fit_scope": f"{spec.protocol}/{spec.name} healthy-train windows (refit; source normalization undone)",
    }
    preprocessing_config["transfer_protocol"] = {
        "protocol": spec.protocol,
        "fold": int(spec.fold),
        "description": spec.description,
        **spec.held_out,
        "source_processed_root": display_path(args.source_processed_root),
        "source_metadata_root": display_path(args.source_metadata_root),
        "builder": "scripts/build_paderborn_transfer_splits.py",
        "note": "window_size/stride/guard_gap/split_ratios describe the main-protocol windowing these windows come from.",
    }
    write_json(metadata_dir / "preprocessing_config.json", preprocessing_config)

    copied: list[str] = []
    for filename in COPIED_METADATA_FILES:
        source = args.source_metadata_root / filename
        if source.exists():
            shutil.copy2(source, metadata_dir / filename)
            copied.append(filename)
        else:
            log(f"warning: {source.as_posix()} not found; not copied")

    splits = {name: subset_summary(manifest, spec.rows[name], name) for name in SUBSETS}
    fold_summary = {
        "protocol": spec.protocol,
        "fold": int(spec.fold),
        "fold_name": spec.name,
        "description": spec.description,
        "held_out": spec.held_out,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "source": {
            "processed_root": display_path(args.source_processed_root),
            "metadata_root": display_path(args.source_metadata_root),
            "window_manifest": display_path(manifest.path),
            "normalization": {"mean": float(source_stats[0]), "std": float(source_stats[1])},
        },
        "output": {
            "processed_root": display_path(processed_dir),
            "metadata_root": display_path(metadata_dir),
            "arrays": array_info,
            "window_manifest": display_path(manifest_path),
            "copied_metadata_files": copied,
        },
        "window_counts": {name: int(spec.rows[name].size) for name in SUBSETS},
        "splits": splits,
        "normalization": {
            "method": "zscore_global refit on fold healthy-train windows",
            "mean": float(fold_mean),
            "std": float(fold_std),
            "num_train_windows": int(train_rows.size),
            "fit_bearings": fit_bearings,
            "fit_conditions": fit_conditions,
            "written_train_readback_mean": float(readback_mean),
            "written_train_readback_std": float(readback_std),
            "written_moments_by_subset": {
                name: {"mean": array_info[name]["written_mean"], "std": array_info[name]["written_std"]}
                for name in SUBSETS
            },
        },
        "checks": {
            **checks,
            "manifest_rows_match_array_rows": True,
            "fault_labels_match_manifest": True,
            "train_array_standardized_within_1e-3": True,
            "spot_check_max_abs_error": spot_errors,
        },
        "ordering": (
            "Each subset follows main-manifest order: by recording (bearing, condition, repeat as ordered by "
            "preprocess_paderborn.py) and then window_start. window_index is the row in the fold array; "
            "source_subset/source_window_index give the main-protocol array and row."
        ),
        "notes": [
            "Pooled healthy recordings concatenate the main train, val and test regions; window_start jumps by the "
            "guard gap (plus any region remainder) at region boundaries, so check window_start before treating "
            "neighbouring rows as overlapping windows.",
            "Fault windows and labels are the main-protocol fault windows re-expressed in the fold normalization.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(metadata_dir / "fold_summary.json", fold_summary)
    log(
        f"{spec.protocol}/{spec.name}: done in {fold_summary['elapsed_seconds']} s; "
        + ", ".join(f"{name}={spec.rows[name].size}" for name in SUBSETS)
    )
    return fold_summary


def main() -> int:
    args = parse_args()
    args.source_processed_root = args.source_processed_root.resolve()
    args.source_metadata_root = args.source_metadata_root.resolve()
    args.output_processed_root = args.output_processed_root.resolve()
    args.output_metadata_root = args.output_metadata_root.resolve()
    for output_root in (args.output_processed_root, args.output_metadata_root):
        for source_root in (args.source_processed_root, args.source_metadata_root):
            if output_root == source_root or source_root.is_relative_to(output_root):
                raise ValueError(f"Output root {output_root.as_posix()} would overlap the source {source_root.as_posix()}.")

    log(f"Loading source manifest {display_path(args.source_metadata_root / 'window_manifest.csv')}")
    manifest = load_manifest(args.source_metadata_root / "window_manifest.csv")
    source_arrays, source_fault_labels = open_source_arrays(args.source_processed_root)
    source_counts = validate_source(manifest, source_arrays, source_fault_labels)
    log("Source validated: " + ", ".join(f"{name}={count}" for name, count in source_counts.items()))
    source_stats_payload = read_json(args.source_metadata_root / "normalization_stats.json")
    if source_stats_payload.get("method") != "zscore_global":
        raise RuntimeError("Source normalization_stats.json is not a global z-score; cannot undo it.")
    source_config = read_json(args.source_metadata_root / "preprocessing_config.json")

    specs: list[FoldSpec] = []
    if "unseen_bearing" in args.protocols:
        specs.extend(build_unseen_bearing_specs(manifest))
    if "loco" in args.protocols:
        specs.extend(build_loco_specs(manifest))
    if args.folds is not None:
        specs = [spec for spec in specs if spec.fold in set(args.folds)]
    if not specs:
        raise RuntimeError("No folds selected.")

    validated = [(spec, validate_spec(manifest, spec)) for spec in specs]
    for spec, _ in validated:
        counts = ", ".join(f"{name}={spec.rows[name].size}" for name in SUBSETS)
        bearings = {name: sorted(set(manifest.bearing[spec.rows[name]].tolist())) for name in SUBSETS[:3]}
        conditions = {name: sorted(set(manifest.condition[spec.rows[name]].tolist())) for name in SUBSETS[:3]}
        log(f"{spec.protocol}/{spec.name}: {counts}")
        log(f"    bearings {bearings}")
        log(f"    conditions {conditions}")
    if args.dry_run:
        log("Dry run: nothing written.")
        return 0

    for spec, checks in validated:
        build_fold(
            spec=spec,
            manifest=manifest,
            source_arrays=source_arrays,
            source_fault_labels=source_fault_labels,
            source_stats_payload=source_stats_payload,
            source_config=source_config,
            args=args,
            checks=checks,
        )
    log("All requested folds built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
