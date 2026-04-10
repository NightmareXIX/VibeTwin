from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - dependency guard
    loadmat = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cwru"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
FILE_MAP_PATH = METADATA_ROOT / "file_map.csv"

CHANNEL_SUFFIXES = {
    "drive_end": "_DE_time",
    "fan_end": "_FE_time",
    "base": "_BA_time",
}

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

WINDOW_MANIFEST_COLUMNS = [
    "split",
    "condition",
    "class",
    "filename",
    "source_id",
    "load_hp",
    "subset_group",
    "signal_key",
    "rpm",
    "window_index",
    "window_start",
    "window_end",
]


@dataclass(frozen=True)
class FileRecord:
    filename: str
    signal_class: str
    load_hp: int
    condition: str
    source_id: int
    subset_group: str


@dataclass(frozen=True)
class LoadedSignal:
    record: FileRecord
    signal_key: str
    signal: np.ndarray
    rpm: int | None


@dataclass(frozen=True)
class Region:
    split: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the local CWRU dataset into leakage-aware window splits.",
    )
    parser.add_argument("--file-map", type=Path, default=FILE_MAP_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--channel", choices=sorted(CHANNEL_SUFFIXES), default="drive_end")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument(
        "--guard-gap",
        type=int,
        default=None,
        help="Gap in samples between healthy split regions. Defaults to one window length.",
    )
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_file_records(path: Path) -> list[FileRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                FileRecord(
                    filename=row["filename"],
                    signal_class=row["class"],
                    load_hp=int(row["load_hp"]),
                    condition=row["condition"],
                    source_id=int(row["source_id"]),
                    subset_group=row["subset_group"],
                )
            )
    return rows


def select_signal_key(keys: list[str], record: FileRecord, channel: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    suffix = CHANNEL_SUFFIXES[channel]
    expected_key = f"X{record.source_id:03d}{suffix}"
    candidate_keys = sorted(key for key in keys if key.endswith(suffix))

    if expected_key in candidate_keys:
        if len(candidate_keys) > 1:
            warnings.append(
                f"{record.filename}: selected {expected_key} using source_id={record.source_id} "
                f"from multiple {channel} candidates {candidate_keys}"
            )
        return expected_key, warnings

    if len(candidate_keys) == 1:
        only_key = candidate_keys[0]
        warnings.append(
            f"{record.filename}: expected {expected_key} was missing, falling back to sole {channel} key {only_key}"
        )
        return only_key, warnings

    any_time_keys = sorted(key for key in keys if key.endswith(tuple(CHANNEL_SUFFIXES.values())))
    if expected_key in any_time_keys:
        warnings.append(
            f"{record.filename}: requested channel '{channel}' was unavailable; falling back to expected source key {expected_key}"
        )
        return expected_key, warnings

    if len(any_time_keys) == 1:
        fallback = any_time_keys[0]
        warnings.append(
            f"{record.filename}: requested channel '{channel}' was unavailable; falling back to sole time-series key {fallback}"
        )
        return fallback, warnings

    raise ValueError(
        f"Could not resolve a unique signal key for {record.filename}. "
        f"Expected {expected_key}; found channel candidates={candidate_keys} and time keys={any_time_keys}."
    )


def extract_rpm(data: dict[str, Any], record: FileRecord) -> tuple[int | None, list[str]]:
    warnings: list[str] = []
    exact_key = f"X{record.source_id:03d}RPM"
    rpm_keys = sorted(key for key in data if key.endswith("RPM"))

    if exact_key in data:
        rpm_value = int(np.asarray(data[exact_key]).squeeze())
        return rpm_value, warnings

    if len(rpm_keys) == 1:
        rpm_key = rpm_keys[0]
        warnings.append(
            f"{record.filename}: expected RPM key {exact_key} was missing, falling back to {rpm_key}"
        )
        rpm_value = int(np.asarray(data[rpm_key]).squeeze())
        return rpm_value, warnings

    warnings.append(f"{record.filename}: RPM metadata missing for source_id={record.source_id}")
    return None, warnings


def resolve_raw_path(record: FileRecord, raw_root: Path) -> Path:
    return raw_root / record.subset_group / record.filename


def load_signal(record: FileRecord, channel: str, raw_root: Path) -> tuple[LoadedSignal | None, list[str]]:
    warnings: list[str] = []
    raw_path = resolve_raw_path(record, raw_root)

    if not raw_path.exists():
        warnings.append(f"{record.filename}: raw file not found at {raw_path.as_posix()}")
        return None, warnings

    try:
        data = loadmat(raw_path)
    except Exception as exc:  # pragma: no cover - runtime safety
        warnings.append(f"{record.filename}: failed to load MAT file ({exc})")
        return None, warnings

    keys = sorted(key for key in data if not key.startswith("__"))

    signal_key, signal_key_warnings = select_signal_key(keys, record, channel)
    warnings.extend(signal_key_warnings)

    rpm, rpm_warnings = extract_rpm(data, record)
    warnings.extend(rpm_warnings)

    signal = np.asarray(data[signal_key]).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1)

    if signal.size == 0:
        warnings.append(f"{record.filename}: resolved signal key {signal_key} was empty")
        return None, warnings

    loaded = LoadedSignal(
        record=record,
        signal_key=signal_key,
        signal=np.asarray(signal, dtype=np.float64),
        rpm=rpm,
    )
    return loaded, warnings


def make_windows(signal: np.ndarray, start: int, end: int, window_size: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    region_length = end - start
    if region_length < window_size:
        return np.empty((0, window_size), dtype=np.float64), []

    starts = list(range(start, end - window_size + 1, stride))
    windows = np.stack([signal[idx : idx + window_size] for idx in starts], axis=0)
    spans = [(idx, idx + window_size) for idx in starts]
    return windows, spans


def build_regions(length: int, window_size: int, guard_gap: int, ratios: dict[str, float]) -> list[Region]:
    minimum_required = (3 * window_size) + (2 * guard_gap)
    if length < minimum_required:
        raise ValueError(
            f"Signal length {length} is too short for train/val/test windowing with "
            f"window_size={window_size} and guard_gap={guard_gap}."
        )

    usable_length = length - (2 * guard_gap)
    train_length = int(usable_length * ratios["train"])
    val_length = int(usable_length * ratios["val"])
    test_length = usable_length - train_length - val_length

    # Keep every split large enough for at least one full window.
    train_length = max(train_length, window_size)
    val_length = max(val_length, window_size)
    remaining = usable_length - train_length - val_length
    if remaining < window_size:
        deficit = window_size - remaining
        shrinkable_train = max(train_length - window_size, 0)
        take_from_train = min(deficit, shrinkable_train)
        train_length -= take_from_train
        deficit -= take_from_train
        if deficit > 0:
            shrinkable_val = max(val_length - window_size, 0)
            take_from_val = min(deficit, shrinkable_val)
            val_length -= take_from_val
            deficit -= take_from_val
        remaining = usable_length - train_length - val_length

    if remaining < window_size:
        raise ValueError(
            f"Could not allocate healthy regions with at least one window each from signal length {length}."
        )

    test_length = remaining

    train_region = Region(split="train", start=0, end=train_length)
    val_start = train_region.end + guard_gap
    val_region = Region(split="val", start=val_start, end=val_start + val_length)
    test_start = val_region.end + guard_gap
    test_region = Region(split="test", start=test_start, end=test_start + test_length)

    return [train_region, val_region, test_region]


def as_output_array(batches: list[np.ndarray], window_size: int, dtype: np.dtype) -> np.ndarray:
    if not batches:
        return np.empty((0, window_size), dtype=dtype)
    return np.concatenate(batches, axis=0).astype(dtype, copy=False)


def normalize_windows(windows: np.ndarray, mean: float, std: float, dtype: np.dtype) -> np.ndarray:
    if windows.size == 0:
        return windows.astype(dtype, copy=False)
    normalized = (windows.astype(np.float64, copy=False) - mean) / std
    return normalized.astype(dtype, copy=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_window_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def find_first_overlap(
    left_ranges: list[tuple[int, int]],
    right_ranges: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    left_index = 0
    right_index = 0

    while left_index < len(left_ranges) and right_index < len(right_ranges):
        left = left_ranges[left_index]
        right = right_ranges[right_index]
        if ranges_overlap(left, right):
            return left, right

        if left[0] <= right[0]:
            left_index += 1
        else:
            right_index += 1

    return None


def validate_healthy_window_non_overlap(
    healthy_window_ranges_by_file: dict[str, dict[str, list[tuple[int, int]]]],
) -> str:
    checked_files = 0
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))

    for filename, split_ranges in sorted(healthy_window_ranges_by_file.items()):
        checked_files += 1
        for left_split, right_split in split_pairs:
            overlap = find_first_overlap(
                split_ranges.get(left_split, []),
                split_ranges.get(right_split, []),
            )
            if overlap is None:
                continue

            left_range, right_range = overlap
            raise RuntimeError(
                f"{filename}: overlapping healthy windows detected between {left_split} "
                f"{left_range} and {right_split} {right_range}."
            )

    return f"Confirmed no cross-split healthy window overlap for {checked_files} healthy files."


def validate_saved_arrays(
    array_paths: dict[str, Path],
    expected_empty: set[str] | None = None,
) -> dict[str, tuple[int, ...]]:
    expected_empty = expected_empty or set()
    loaded_arrays = {name: np.load(path) for name, path in array_paths.items()}

    for name, array in loaded_arrays.items():
        if array.size == 0 and name not in expected_empty:
            raise RuntimeError(f"Saved array {name} is unexpectedly empty: {array_paths[name].as_posix()}")

    if loaded_arrays["fault_labels"].shape[0] != loaded_arrays["test_fault"].shape[0]:
        raise RuntimeError(
            "Saved fault_labels length does not match saved fault_windows row count: "
            f"{loaded_arrays['fault_labels'].shape[0]} != {loaded_arrays['test_fault'].shape[0]}"
        )

    return {name: summarize_shape(array) for name, array in loaded_arrays.items()}


def aggregate_window_counts(
    rows: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for row in rows:
        group = str(row[group_key])
        split = row["split"]
        counts[group][split] += 1
        counts[group]["total"] += 1
    return dict(counts)


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["(none)" for _ in headers]]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_preprocessing_audit(
    *,
    records: list[FileRecord],
    file_summaries: dict[str, dict[str, Any]],
    window_manifest_rows: list[dict[str, Any]],
    train_healthy: np.ndarray,
    val_healthy: np.ndarray,
    test_healthy: np.ndarray,
    test_fault: np.ndarray,
    healthy_overlap_confirmation: str,
    fault_label_inverse_map: dict[str, str],
    guard_gap: int,
) -> str:
    file_counts = aggregate_window_counts(window_manifest_rows, "filename")
    class_counts = aggregate_window_counts(window_manifest_rows, "class")
    load_counts = aggregate_window_counts(window_manifest_rows, "load_hp")

    totals_rows = [
        ["healthy train windows", int(train_healthy.shape[0])],
        ["healthy val windows", int(val_healthy.shape[0])],
        ["healthy test windows", int(test_healthy.shape[0])],
        ["fault test windows", int(test_fault.shape[0])],
    ]

    file_rows: list[list[Any]] = []
    for record in records:
        summary = file_summaries.get(record.filename, {})
        counts = file_counts.get(record.filename, {"train": 0, "val": 0, "test": 0, "total": 0})
        if summary:
            rpm_value = summary.get("rpm")
            rpm_status = "missing" if rpm_value is None else f"present ({rpm_value})"
        else:
            rpm_status = "load_failed"
        file_rows.append(
            [
                record.filename,
                record.condition,
                record.signal_class,
                record.load_hp,
                summary.get("signal_length", "load_failed"),
                summary.get("signal_key", "load_failed"),
                rpm_status,
                counts["train"],
                counts["val"],
                counts["test"],
                counts["total"],
            ]
        )

    class_rows: list[list[Any]] = []
    for class_name in ["normal", "ball", "inner_race", "outer_race_6"]:
        counts = class_counts.get(class_name, {"train": 0, "val": 0, "test": 0, "total": 0})
        class_rows.append([class_name, counts["train"], counts["val"], counts["test"], counts["total"]])

    load_rows: list[list[Any]] = []
    for load_hp in sorted({record.load_hp for record in records}):
        counts = load_counts.get(str(load_hp), {"train": 0, "val": 0, "test": 0, "total": 0})
        load_rows.append([load_hp, counts["train"], counts["val"], counts["test"], counts["total"]])

    fault_label_summary = ", ".join(
        f"{index}={label}" for index, label in sorted(fault_label_inverse_map.items(), key=lambda item: int(item[0]))
    )

    lines = [
        "# Preprocessing Audit",
        "",
        "## Totals",
        format_markdown_table(["metric", "count"], totals_rows),
        "",
        "## Per-file Audit",
        format_markdown_table(
            [
                "filename",
                "condition",
                "class",
                "load_hp",
                "signal_length",
                "signal_key",
                "rpm",
                "train",
                "val",
                "test",
                "total",
            ],
            file_rows,
        ),
        "",
        "## Counts by Class",
        format_markdown_table(["class", "train", "val", "test", "total"], class_rows),
        "",
        "## Counts by Load HP",
        format_markdown_table(["load_hp", "train", "val", "test", "total"], load_rows),
        "",
        "## Healthy Split Check",
        f"- {healthy_overlap_confirmation}",
        f"- Healthy splits were generated from contiguous regions with a {guard_gap}-sample guard gap.",
        "",
        "## Fault Label Map",
        f"- {fault_label_summary}",
        "",
    ]
    return "\n".join(lines)


def summarize_shape(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in array.shape)


def print_summary(
    *,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_healthy_windows: np.ndarray,
    test_fault_windows: np.ndarray,
    fault_labels: np.ndarray,
    output_paths: dict[str, Path],
    config: dict[str, Any],
    oddities: list[str],
    assumptions: list[str],
    validation_summary: str,
) -> None:
    print("\nPreprocessing Summary")
    print(f"  train healthy windows: {summarize_shape(train_windows)} -> {output_paths['train_healthy'].as_posix()}")
    print(f"  val healthy windows: {summarize_shape(val_windows)} -> {output_paths['val_healthy'].as_posix()}")
    print(
        f"  test healthy windows: {summarize_shape(test_healthy_windows)} -> {output_paths['test_healthy'].as_posix()}"
    )
    print(f"  test fault windows: {summarize_shape(test_fault_windows)} -> {output_paths['test_fault'].as_posix()}")
    print(f"  test fault labels: {summarize_shape(fault_labels)} -> {output_paths['fault_labels'].as_posix()}")
    print(f"  fault label map: {output_paths['fault_label_map'].as_posix()}")
    print(f"  preprocessing audit: {output_paths['preprocessing_audit'].as_posix()}")

    print("\nSelected Defaults")
    print(f"  channel: {config['channel']}")
    print(f"  window_size: {config['window_size']}")
    print(f"  stride: {config['stride']}")
    print(f"  guard_gap: {config['guard_gap']}")
    print(f"  split_ratios: {config['split_ratios']}")
    print(f"  normalization: {config['normalization']['method']}")
    print(f"  dtype: {config['dtype']}")
    print(f"  seed: {config['seed']}")
    print(f"  filtering: {config['filtering']}")

    print("\nMissing/Odd Files Encountered")
    if oddities:
        for item in oddities:
            print(f"  - {item}")
    else:
        print("  - none")

    print("\nAssumptions")
    for item in assumptions:
        print(f"  - {item}")

    print("\nValidation")
    print(f"  - {validation_summary}")


def main() -> int:
    if loadmat is None:
        print("scipy is not available; install scipy to preprocess CWRU MAT files.")
        return 1

    args = parse_args()
    np.random.seed(args.seed)

    if args.guard_gap is None:
        args.guard_gap = args.window_size

    if args.window_size <= 0 or args.stride <= 0 or args.guard_gap < 0:
        raise ValueError("window_size, stride, and guard_gap must be positive, with guard_gap >= 0.")

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}.")

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()
    file_map_path = args.file_map.resolve()

    train_dir = processed_root / "train"
    val_dir = processed_root / "val"
    test_dir = processed_root / "test"
    for directory in (train_dir, val_dir, test_dir, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    records = load_file_records(file_map_path)
    healthy_records = [record for record in records if record.condition == "healthy"]
    fault_records = [record for record in records if record.condition == "fault"]

    oddities: list[str] = []
    assumptions = [
        "Drive-end channel is used by default unless --channel overrides it.",
        "Healthy train/val/test split uses contiguous per-recording regions with ratios 0.70/0.15/0.15 because no repo-level split ratio was defined.",
        "A one-window guard gap is inserted between healthy split regions to reduce temporal leakage.",
        "No additional filtering is applied at this stage.",
        "Z-score normalization uses global mean and std fit on healthy training windows only.",
        "Fault labels are integer encoded as ball=0, inner_race=1, outer_race_6=2.",
    ]

    healthy_batches: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    fault_batches: list[np.ndarray] = []
    fault_label_batches: list[np.ndarray] = []
    window_manifest_rows: list[dict[str, Any]] = []
    file_summaries: dict[str, dict[str, Any]] = {}
    healthy_window_ranges_by_file: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )

    fault_label_map = {label: index for index, label in enumerate(sorted({record.signal_class for record in fault_records}))}
    fault_label_inverse_map = {str(index): label for label, index in sorted(fault_label_map.items(), key=lambda item: item[1])}

    for record in healthy_records:
        loaded, warnings = load_signal(record, args.channel, raw_root)
        oddities.extend(warnings)
        if loaded is None:
            continue

        file_summaries[record.filename] = {
            "signal_length": int(loaded.signal.size),
            "signal_key": loaded.signal_key,
            "rpm": loaded.rpm,
        }

        try:
            regions = build_regions(
                length=loaded.signal.size,
                window_size=args.window_size,
                guard_gap=args.guard_gap,
                ratios=ratios,
            )
        except Exception as exc:
            oddities.append(f"{record.filename}: {exc}")
            continue

        for region in regions:
            windows, spans = make_windows(
                signal=loaded.signal,
                start=region.start,
                end=region.end,
                window_size=args.window_size,
                stride=args.stride,
            )
            if windows.size == 0:
                oddities.append(
                    f"{record.filename}: no {region.split} windows produced from region "
                    f"[{region.start}, {region.end})"
                )
                continue

            healthy_batches[region.split].append(windows)
            healthy_window_ranges_by_file[record.filename][region.split].extend(spans)
            base_index = len(window_manifest_rows)
            for offset, (window_start, window_end) in enumerate(spans):
                window_manifest_rows.append(
                    {
                        "split": region.split,
                        "condition": record.condition,
                        "class": record.signal_class,
                        "filename": record.filename,
                        "source_id": record.source_id,
                        "load_hp": record.load_hp,
                        "subset_group": record.subset_group,
                        "signal_key": loaded.signal_key,
                        "rpm": "" if loaded.rpm is None else loaded.rpm,
                        "window_index": base_index + offset,
                        "window_start": window_start,
                        "window_end": window_end,
                    }
                )

    for record in fault_records:
        loaded, warnings = load_signal(record, args.channel, raw_root)
        oddities.extend(warnings)
        if loaded is None:
            continue

        file_summaries[record.filename] = {
            "signal_length": int(loaded.signal.size),
            "signal_key": loaded.signal_key,
            "rpm": loaded.rpm,
        }

        windows, spans = make_windows(
            signal=loaded.signal,
            start=0,
            end=loaded.signal.size,
            window_size=args.window_size,
            stride=args.stride,
        )
        if windows.size == 0:
            oddities.append(f"{record.filename}: no fault windows produced")
            continue

        fault_batches.append(windows)
        fault_label = fault_label_map[record.signal_class]
        fault_label_batches.append(np.full((windows.shape[0],), fault_label, dtype=np.int64))
        base_index = len(window_manifest_rows)
        for offset, (window_start, window_end) in enumerate(spans):
            window_manifest_rows.append(
                {
                    "split": "test",
                    "condition": record.condition,
                    "class": record.signal_class,
                    "filename": record.filename,
                    "source_id": record.source_id,
                    "load_hp": record.load_hp,
                    "subset_group": record.subset_group,
                    "signal_key": loaded.signal_key,
                    "rpm": "" if loaded.rpm is None else loaded.rpm,
                    "window_index": base_index + offset,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

    output_dtype = np.dtype(args.dtype)
    train_healthy = as_output_array(healthy_batches["train"], args.window_size, np.float64)
    if train_healthy.size == 0:
        raise RuntimeError("No healthy training windows were produced; aborting preprocessing.")

    val_healthy = as_output_array(healthy_batches["val"], args.window_size, np.float64)
    test_healthy = as_output_array(healthy_batches["test"], args.window_size, np.float64)
    test_fault = as_output_array(fault_batches, args.window_size, np.float64)
    fault_labels = (
        np.concatenate(fault_label_batches, axis=0).astype(np.int64, copy=False)
        if fault_label_batches
        else np.empty((0,), dtype=np.int64)
    )
    healthy_overlap_confirmation = validate_healthy_window_non_overlap(healthy_window_ranges_by_file)

    mean = float(train_healthy.mean())
    std = float(train_healthy.std())
    if std == 0.0:
        raise RuntimeError("Healthy training windows have zero standard deviation; cannot z-score normalize.")

    train_healthy = normalize_windows(train_healthy, mean, std, output_dtype)
    val_healthy = normalize_windows(val_healthy, mean, std, output_dtype)
    test_healthy = normalize_windows(test_healthy, mean, std, output_dtype)
    test_fault = normalize_windows(test_fault, mean, std, output_dtype)

    output_paths = {
        "train_healthy": train_dir / "healthy_windows.npy",
        "val_healthy": val_dir / "healthy_windows.npy",
        "test_healthy": test_dir / "healthy_windows.npy",
        "test_fault": test_dir / "fault_windows.npy",
        "fault_labels": test_dir / "fault_labels.npy",
        "normalization_stats": metadata_root / "normalization_stats.json",
        "preprocessing_config": metadata_root / "preprocessing_config.json",
        "fault_label_map": metadata_root / "fault_label_map.json",
        "preprocessing_audit": metadata_root / "preprocessing_audit.md",
        "window_manifest": metadata_root / "window_manifest.csv",
    }

    np.save(output_paths["train_healthy"], train_healthy)
    np.save(output_paths["val_healthy"], val_healthy)
    np.save(output_paths["test_healthy"], test_healthy)
    np.save(output_paths["test_fault"], test_fault)
    np.save(output_paths["fault_labels"], fault_labels)

    normalization_stats = {
        "method": "zscore_global",
        "fit_split": "train",
        "fit_condition": "healthy",
        "mean": mean,
        "std": std,
        "num_train_windows": int(train_healthy.shape[0]),
        "window_size": int(args.window_size),
        "num_train_samples": int(train_healthy.shape[0] * train_healthy.shape[1]),
        "output_dtype": args.dtype,
    }
    write_json(output_paths["normalization_stats"], normalization_stats)

    preprocessing_config = {
        "dataset": "cwru",
        "file_map_path": file_map_path.as_posix(),
        "raw_root": raw_root.as_posix(),
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "channel": args.channel,
        "channel_suffix": CHANNEL_SUFFIXES[args.channel],
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "guard_gap": int(args.guard_gap),
        "split_ratios": {name: float(value) for name, value in ratios.items()},
        "normalization": {
            "method": "zscore_global",
            "fit_on": "healthy_train_only",
        },
        "filtering": "none",
        "dtype": args.dtype,
        "seed": int(args.seed),
        "fault_label_map": fault_label_map,
    }
    write_json(output_paths["preprocessing_config"], preprocessing_config)
    write_json(
        output_paths["fault_label_map"],
        {
            "integer_to_class": fault_label_inverse_map,
            "class_to_integer": fault_label_map,
        },
    )
    write_window_manifest(output_paths["window_manifest"], window_manifest_rows)
    audit_text = build_preprocessing_audit(
        records=records,
        file_summaries=file_summaries,
        window_manifest_rows=window_manifest_rows,
        train_healthy=train_healthy,
        val_healthy=val_healthy,
        test_healthy=test_healthy,
        test_fault=test_fault,
        healthy_overlap_confirmation=healthy_overlap_confirmation,
        fault_label_inverse_map=fault_label_inverse_map,
        guard_gap=args.guard_gap,
    )
    output_paths["preprocessing_audit"].write_text(audit_text, encoding="utf-8")

    validated_shapes = validate_saved_arrays(
        {
            "train_healthy": output_paths["train_healthy"],
            "val_healthy": output_paths["val_healthy"],
            "test_healthy": output_paths["test_healthy"],
            "test_fault": output_paths["test_fault"],
            "fault_labels": output_paths["fault_labels"],
        }
    )
    validation_summary = (
        f"{healthy_overlap_confirmation} Saved arrays verified with shapes "
        f"{validated_shapes} and matching fault label length."
    )

    print_summary(
        train_windows=train_healthy,
        val_windows=val_healthy,
        test_healthy_windows=test_healthy,
        test_fault_windows=test_fault,
        fault_labels=fault_labels,
        output_paths=output_paths,
        config=preprocessing_config,
        oddities=oddities,
        assumptions=assumptions,
        validation_summary=validation_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
