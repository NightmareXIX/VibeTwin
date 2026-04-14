from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - dependency guard
    loadmat = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "paderborn"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
FILE_MAP_PATH = METADATA_ROOT / "file_map.csv"

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

WINDOW_MANIFEST_COLUMNS = [
    "split",
    "subset",
    "health_status",
    "damage_group",
    "fault_label_int",
    "fault_label_name",
    "label_verification",
    "label_support_files_present",
    "bearing_code",
    "condition_code",
    "speed_code",
    "torque_code",
    "radial_force_code",
    "repeat_index",
    "filename",
    "relative_path",
    "measurement_id",
    "selected_signal",
    "available_channels",
    "signal_length",
    "window_index",
    "window_start",
    "window_end",
]

HEALTHY_PATTERN = re.compile(r"^K0\d+$", re.IGNORECASE)
DAMAGE_PATTERNS = {
    "KA": re.compile(r"^KA\d+$", re.IGNORECASE),
    "KB": re.compile(r"^KB\d+$", re.IGNORECASE),
    "KI": re.compile(r"^KI\d+$", re.IGNORECASE),
}


@dataclass(frozen=True)
class Region:
    split: str
    start: int
    end: int


@dataclass(frozen=True)
class FileRecord:
    path: Path
    relative_path: str
    filename: str
    measurement_id: str
    bearing_code: str
    condition_code: str
    speed_code: str
    torque_code: str
    radial_force_code: str
    repeat_index: str
    health_status: str
    damage_group: str
    label_verification: str
    label_support_files_present: bool


@dataclass(frozen=True)
class PreparedRecord:
    record: FileRecord
    selected_signal: str
    available_channels: tuple[str, ...]
    signal_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the staged Paderborn bearing dataset into CWRU-like train/val/test windows.",
    )
    parser.add_argument("--file-map", type=Path, default=FILE_MAP_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--channel", type=str, default="vibration_1")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument(
        "--guard-gap",
        type=int,
        default=None,
        help="Gap in samples between healthy train/val/test regions. Defaults to one window length.",
    )
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


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


def classify_bearing_code(bearing_code: str) -> tuple[str, str, str]:
    upper_code = bearing_code.upper()
    if HEALTHY_PATTERN.fullmatch(upper_code):
        return "healthy", "", "inferred_bearing_code_family"

    for damage_group, pattern in DAMAGE_PATTERNS.items():
        if pattern.fullmatch(upper_code):
            return "damaged", damage_group, "inferred_bearing_code_family"

    return "", "", ""


def support_files_present(path: Path, bearing_code: str) -> bool:
    return (
        (path.parent / f"{bearing_code}.pdf").exists()
        and (path.parent / f"measuring_log_{bearing_code}.pdf").exists()
    )


def load_file_records(file_map_path: Path, raw_root: Path) -> tuple[list[FileRecord], list[str]]:
    skipped: list[str] = []
    records: list[FileRecord] = []

    with file_map_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["source_location"] != "canonical_raw_root":
                continue
            if row["extension"].lower() != ".mat":
                continue
            if row["readable"].strip().lower() != "true":
                skipped.append(f"{row['relative_path']}: marked unreadable in file_map.csv")
                continue

            relative_path = row["relative_path"]
            path = PROJECT_ROOT / Path(relative_path)
            if not path.exists():
                skipped.append(f"{relative_path}: file is listed in file_map.csv but missing on disk")
                continue
            if raw_root not in path.parents:
                skipped.append(f"{relative_path}: canonical file is outside the configured raw root")
                continue

            bearing_code = row["bearing_code"].strip().upper()
            health_status, damage_group, label_verification = classify_bearing_code(bearing_code)
            if not health_status:
                skipped.append(f"{relative_path}: bearing code {bearing_code} could not be classified")
                continue

            records.append(
                FileRecord(
                    path=path,
                    relative_path=relative_path,
                    filename=row["filename"],
                    measurement_id=Path(row["filename"]).stem,
                    bearing_code=bearing_code,
                    condition_code=row["condition_code"],
                    speed_code=row["speed_code"],
                    torque_code=row["torque_code"],
                    radial_force_code=row["radial_force_code"],
                    repeat_index=row["repeat_index"],
                    health_status=health_status,
                    damage_group=damage_group,
                    label_verification=label_verification,
                    label_support_files_present=support_files_present(path, bearing_code),
                )
            )

    records.sort(key=lambda item: (item.bearing_code, item.condition_code, item.repeat_index, item.filename))
    return records, skipped


def clean_channel_name(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else "unnamed"
    if isinstance(value, np.ndarray):
        squeezed = np.asarray(value).squeeze()
        if squeezed.size == 0:
            return "unnamed"
        if squeezed.size == 1:
            text = str(squeezed.item()).strip()
            return text if text else "unnamed"
    return "unnamed"


def normalize_channel_entry(entry: Any) -> tuple[str, np.ndarray] | None:
    if not isinstance(entry, dict):
        return None
    if "Data" not in entry:
        return None

    signal = np.asarray(entry["Data"]).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    if signal.size == 0:
        return None

    channel_name = clean_channel_name(entry.get("Name", "unnamed"))
    return channel_name, np.asarray(signal, dtype=np.float64)


def extract_signal_channels(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    raw_entries = payload.get("Y")
    if isinstance(raw_entries, dict):
        raw_entries = [raw_entries]
    if not isinstance(raw_entries, (list, tuple)):
        return channels

    for entry in raw_entries:
        normalized = normalize_channel_entry(entry)
        if normalized is None:
            continue
        channel_name, signal = normalized
        dedup_name = channel_name
        suffix = 2
        while dedup_name in channels:
            dedup_name = f"{channel_name}_{suffix}"
            suffix += 1
        channels[dedup_name] = signal
    return channels


def choose_signal_channel(channels: dict[str, np.ndarray], preferred_channel: str) -> str | None:
    if preferred_channel in channels:
        return preferred_channel

    vibration_candidates = sorted(name for name in channels if "vibration" in name.lower())
    if not vibration_candidates:
        return None
    return vibration_candidates[0]


def load_selected_signal(
    record: FileRecord,
    preferred_channel: str,
) -> tuple[np.ndarray | None, str | None, tuple[str, ...], list[str]]:
    warnings: list[str] = []

    try:
        data = loadmat(record.path, simplify_cells=True)
    except Exception as exc:  # pragma: no cover - runtime safety
        return None, None, tuple(), [f"{record.relative_path}: failed to load MAT file ({exc})"]

    visible_keys = [key for key in data if not key.startswith("__")]
    if record.measurement_id in data:
        payload = data[record.measurement_id]
    elif len(visible_keys) == 1:
        payload = data[visible_keys[0]]
        warnings.append(
            f"{record.relative_path}: expected top-level key {record.measurement_id} was missing; used {visible_keys[0]}"
        )
    else:
        return None, None, tuple(), [f"{record.relative_path}: could not resolve a unique top-level measurement struct"]

    if not isinstance(payload, dict):
        return None, None, tuple(), [f"{record.relative_path}: top-level measurement payload is not a dictionary"]

    channels = extract_signal_channels(payload)
    if not channels:
        return None, None, tuple(), [f"{record.relative_path}: no usable Y-channel signals were found"]

    selected_channel = choose_signal_channel(channels, preferred_channel)
    if selected_channel is None:
        return None, None, tuple(sorted(channels)), [
            f"{record.relative_path}: preferred channel '{preferred_channel}' was unavailable and no vibration channel fallback was found"
        ]

    signal = channels[selected_channel]
    return signal, selected_channel, tuple(sorted(channels)), warnings


def count_windows(start: int, end: int, window_size: int, stride: int) -> int:
    region_length = end - start
    if region_length < window_size:
        return 0
    return ((region_length - window_size) // stride) + 1


def make_windows(signal: np.ndarray, start: int, end: int, window_size: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    count = count_windows(start, end, window_size, stride)
    if count == 0:
        return np.empty((0, window_size), dtype=np.float64), []

    starts = [start + (index * stride) for index in range(count)]
    windows = np.stack([signal[idx : idx + window_size] for idx in starts], axis=0)
    spans = [(idx, idx + window_size) for idx in starts]
    return windows, spans


def build_regions(length: int, window_size: int, guard_gap: int, ratios: dict[str, float]) -> list[Region]:
    minimum_required = (3 * window_size) + (2 * guard_gap)
    if length < minimum_required:
        raise ValueError(
            f"Signal length {length} is too short for train/val/test windowing with window_size={window_size} "
            f"and guard_gap={guard_gap}."
        )

    usable_length = length - (2 * guard_gap)
    train_length = int(usable_length * ratios["train"])
    val_length = int(usable_length * ratios["val"])
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
        raise ValueError(f"Could not allocate healthy regions with at least one window each from signal length {length}.")

    test_length = remaining
    train_region = Region(split="train", start=0, end=train_length)
    val_start = train_region.end + guard_gap
    val_region = Region(split="val", start=val_start, end=val_start + val_length)
    test_start = val_region.end + guard_gap
    test_region = Region(split="test", start=test_start, end=test_start + test_length)
    return [train_region, val_region, test_region]


def normalize_windows(windows: np.ndarray, mean: float, std: float, dtype: np.dtype) -> np.ndarray:
    if windows.size == 0:
        return windows.astype(dtype, copy=False)
    normalized = (windows.astype(np.float64, copy=False) - mean) / std
    return normalized.astype(dtype, copy=False)


def create_output_file(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap | None:
    if shape[0] == 0:
        np.save(path, np.empty(shape, dtype=dtype))
        return None
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def validate_saved_arrays(array_paths: dict[str, Path]) -> dict[str, tuple[int, ...]]:
    loaded = {name: np.load(path, mmap_mode="r") for name, path in array_paths.items()}
    if loaded["test_fault"].shape[0] != loaded["fault_labels"].shape[0]:
        raise RuntimeError(
            f"fault_windows rows and fault_labels length differ: {loaded['test_fault'].shape[0]} != {loaded['fault_labels'].shape[0]}"
        )
    return {name: tuple(int(dim) for dim in array.shape) for name, array in loaded.items()}


def summarize_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def build_preprocessing_report(
    *,
    totals: dict[str, int],
    signal_summary: dict[str, Any],
    normalization_stats: dict[str, Any],
    split_config: dict[str, Any],
    label_summary: dict[str, Any],
    per_bearing_rows: list[list[Any]],
    per_condition_rows: list[list[Any]],
    skipped_files: list[str],
    condition_counts: Counter[str],
    validation_summary: str,
) -> str:
    totals_rows = [
        ["healthy train windows", totals["train_healthy"]],
        ["healthy val windows", totals["val_healthy"]],
        ["healthy test windows", totals["test_healthy"]],
        ["fault test windows", totals["test_fault"]],
        ["fault label count", totals["fault_labels"]],
    ]

    signal_rows = [
        ["selected signal channel", signal_summary["selected_channel"]],
        ["files using fallback vibration selection", signal_summary["fallback_channel_count"]],
        ["files skipped for missing signal", signal_summary["missing_channel_count"]],
        ["available channels example", ", ".join(signal_summary["available_channels_example"])],
    ]

    config_rows = [
        ["window_size", split_config["window_size"]],
        ["stride", split_config["stride"]],
        ["guard_gap", split_config["guard_gap"]],
        ["split_ratios", split_config["split_ratios"]],
        ["dtype", split_config["dtype"]],
    ]

    label_rows = [
        ["measurement files with verified labels", label_summary["verified_measurement_files"]],
        ["measurement files with inferred labels", label_summary["inferred_measurement_files"]],
        ["bearings with support PDFs present", label_summary["bearings_with_support_files"]],
        ["support PDF parsing status", label_summary["support_parsing_status"]],
        ["fault label map", label_summary["fault_label_map_text"]],
    ]

    lines = [
        "# Paderborn Preprocessing Report",
        "",
        "## Totals",
        format_markdown_table(["metric", "value"], totals_rows),
        "",
        "## Signal Selection",
        format_markdown_table(["setting", "value"], signal_rows),
        "",
        "## Split and Normalization",
        format_markdown_table(["setting", "value"], config_rows),
        "",
        f"- Normalization method: {normalization_stats['method']}",
        f"- Healthy-train fit mean: {normalization_stats['mean']:.6f}",
        f"- Healthy-train fit std: {normalization_stats['std']:.6f}",
        f"- Training samples used for normalization: {normalization_stats['num_train_samples']}",
        "",
        "## Label Provenance",
        format_markdown_table(["item", "value"], label_rows),
        "",
        "- Exact per-bearing damage verification from the local PDFs remains unresolved in this preprocessing pass because no PDF text extractor is available in the current environment.",
        "- The train/val/test split uses bearing-code-family inference: `K0xx` as healthy, `KA/KB/KI` as damaged families.",
        "",
        "## Counts by Bearing",
        format_markdown_table(
            [
                "bearing_code",
                "health_status",
                "damage_group",
                "measurement_files",
                "train",
                "val",
                "test_healthy",
                "test_fault",
                "total_windows",
                "label_verification",
                "support_files",
            ],
            per_bearing_rows,
        ),
        "",
        "## Counts by Operating Condition",
        format_markdown_table(
            [
                "condition_code",
                "train",
                "val",
                "test_healthy",
                "test_fault",
                "total_windows",
            ],
            per_condition_rows,
        ),
        "",
        "## Condition Inventory",
        f"- Measurement files per condition: {json.dumps(summarize_counts(condition_counts), ensure_ascii=True)}",
        "",
        "## Skipped Files",
    ]

    if skipped_files:
        for item in skipped_files:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Validation",
            f"- {validation_summary}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if loadmat is None:
        print("scipy is not available; install scipy to preprocess Paderborn MAT files.")
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

    records, metadata_skips = load_file_records(file_map_path, raw_root)
    if not records:
        raise RuntimeError("No canonical Paderborn MAT measurement files were available for preprocessing.")

    healthy_records = [record for record in records if record.health_status == "healthy"]
    fault_records = [record for record in records if record.health_status == "damaged"]
    if not healthy_records:
        raise RuntimeError("No healthy Paderborn measurement files were found.")
    if not fault_records:
        raise RuntimeError("No damaged Paderborn measurement files were found.")

    prepared_records: list[PreparedRecord] = []
    skipped_files = list(metadata_skips)
    selected_channel_counter: Counter[str] = Counter()
    condition_file_counts: Counter[str] = Counter()
    missing_channel_count = 0
    fallback_channel_count = 0

    total_counts = {
        "train_healthy": 0,
        "val_healthy": 0,
        "test_healthy": 0,
        "test_fault": 0,
        "fault_labels": 0,
    }
    per_bearing_counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "health_status": "",
            "damage_group": "",
            "measurement_files": 0,
            "train": 0,
            "val": 0,
            "test_healthy": 0,
            "test_fault": 0,
            "label_verification": "",
            "support_files": False,
        }
    )
    per_condition_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test_healthy": 0, "test_fault": 0}
    )

    train_sample_sum = 0.0
    train_sample_sum_sq = 0.0
    train_num_samples = 0

    for record in records:
        signal, selected_channel, available_channels, warnings = load_selected_signal(record, args.channel)
        skipped_files.extend(warnings)
        if signal is None or selected_channel is None:
            missing_channel_count += 1
            continue

        prepared_records.append(
            PreparedRecord(
                record=record,
                selected_signal=selected_channel,
                available_channels=available_channels,
                signal_length=int(signal.size),
            )
        )
        selected_channel_counter[selected_channel] += 1
        condition_file_counts[record.condition_code] += 1
        if selected_channel != args.channel:
            fallback_channel_count += 1

        bearing_entry = per_bearing_counts[record.bearing_code]
        bearing_entry["health_status"] = record.health_status
        bearing_entry["damage_group"] = record.damage_group or "-"
        bearing_entry["measurement_files"] += 1
        bearing_entry["label_verification"] = record.label_verification
        bearing_entry["support_files"] = bearing_entry["support_files"] or record.label_support_files_present

        if record.health_status == "healthy":
            regions = build_regions(
                length=int(signal.size),
                window_size=args.window_size,
                guard_gap=args.guard_gap,
                ratios=ratios,
            )
            for region in regions:
                windows, _ = make_windows(
                    signal=signal,
                    start=region.start,
                    end=region.end,
                    window_size=args.window_size,
                    stride=args.stride,
                )
                window_count = int(windows.shape[0])
                if window_count == 0:
                    skipped_files.append(
                        f"{record.relative_path}: no {region.split} windows were produced from region [{region.start}, {region.end})"
                    )
                    continue

                subset_name = f"{region.split}_healthy"
                total_counts[subset_name] += window_count
                bearing_split_key = "test_healthy" if region.split == "test" else region.split
                per_bearing_counts[record.bearing_code][bearing_split_key] += window_count
                per_condition_counts[record.condition_code][bearing_split_key] += window_count

                if region.split == "train":
                    train_sample_sum += float(windows.sum(dtype=np.float64))
                    train_sample_sum_sq += float(np.square(windows, dtype=np.float64).sum(dtype=np.float64))
                    train_num_samples += int(windows.size)
        else:
            windows, _ = make_windows(
                signal=signal,
                start=0,
                end=int(signal.size),
                window_size=args.window_size,
                stride=args.stride,
            )
            window_count = int(windows.shape[0])
            if window_count == 0:
                skipped_files.append(f"{record.relative_path}: no fault test windows were produced")
                continue

            total_counts["test_fault"] += window_count
            total_counts["fault_labels"] += window_count
            per_bearing_counts[record.bearing_code]["test_fault"] += window_count
            per_condition_counts[record.condition_code]["test_fault"] += window_count

    if train_num_samples == 0:
        raise RuntimeError("No healthy training samples were available for normalization.")

    mean = train_sample_sum / train_num_samples
    variance = max((train_sample_sum_sq / train_num_samples) - (mean * mean), 0.0)
    std = math.sqrt(variance)
    if std == 0.0:
        raise RuntimeError("Healthy training windows have zero standard deviation; cannot z-score normalize.")

    fault_label_map = {
        damage_group: index
        for index, damage_group in enumerate(sorted({record.damage_group for record in fault_records if record.damage_group}))
    }
    if not fault_label_map:
        raise RuntimeError("No damaged-bearing label groups were resolved for fault label encoding.")

    output_dtype = np.dtype(args.dtype)
    output_paths = {
        "train_healthy": train_dir / "healthy_windows.npy",
        "val_healthy": val_dir / "healthy_windows.npy",
        "test_healthy": test_dir / "healthy_windows.npy",
        "test_fault": test_dir / "fault_windows.npy",
        "fault_labels": test_dir / "fault_labels.npy",
        "normalization_stats": metadata_root / "normalization_stats.json",
        "preprocessing_config": metadata_root / "preprocessing_config.json",
        "window_manifest": metadata_root / "window_manifest.csv",
        "preprocessing_report": metadata_root / "preprocessing_report.md",
    }

    train_array = create_output_file(output_paths["train_healthy"], (total_counts["train_healthy"], args.window_size), output_dtype)
    val_array = create_output_file(output_paths["val_healthy"], (total_counts["val_healthy"], args.window_size), output_dtype)
    test_healthy_array = create_output_file(
        output_paths["test_healthy"],
        (total_counts["test_healthy"], args.window_size),
        output_dtype,
    )
    test_fault_array = create_output_file(output_paths["test_fault"], (total_counts["test_fault"], args.window_size), output_dtype)
    fault_labels_array = create_output_file(output_paths["fault_labels"], (total_counts["fault_labels"],), np.int64)

    split_arrays = {
        "train_healthy": train_array,
        "val_healthy": val_array,
        "test_healthy": test_healthy_array,
        "test_fault": test_fault_array,
    }
    split_offsets = {name: 0 for name in split_arrays}
    fault_label_offset = 0

    with output_paths["window_manifest"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_MANIFEST_COLUMNS)
        writer.writeheader()

        for prepared in prepared_records:
            record = prepared.record
            signal, selected_channel, available_channels, warnings = load_selected_signal(record, args.channel)
            if warnings:
                raise RuntimeError(
                    f"Unexpected instability while reloading {record.relative_path}: {'; '.join(warnings)}"
                )
            if signal is None or selected_channel is None:
                raise RuntimeError(f"Unexpected reload failure for {record.relative_path}")

            if tuple(sorted(available_channels)) != prepared.available_channels:
                raise RuntimeError(f"Channel inventory changed unexpectedly for {record.relative_path}")

            if record.health_status == "healthy":
                regions = build_regions(
                    length=int(signal.size),
                    window_size=args.window_size,
                    guard_gap=args.guard_gap,
                    ratios=ratios,
                )
                for region in regions:
                    subset_name = f"{region.split}_healthy"
                    windows, spans = make_windows(
                        signal=signal,
                        start=region.start,
                        end=region.end,
                        window_size=args.window_size,
                        stride=args.stride,
                    )
                    normalized = normalize_windows(windows, mean, std, output_dtype)
                    if normalized.shape[0] == 0:
                        continue

                    start_index = split_offsets[subset_name]
                    end_index = start_index + normalized.shape[0]
                    target = split_arrays[subset_name]
                    if target is None:
                        raise RuntimeError(f"Output target for {subset_name} was not initialized.")
                    target[start_index:end_index] = normalized
                    split_offsets[subset_name] = end_index

                    for local_index, (window_start, window_end) in enumerate(spans):
                        writer.writerow(
                            {
                                "split": region.split,
                                "subset": subset_name,
                                "health_status": record.health_status,
                                "damage_group": "",
                                "fault_label_int": "",
                                "fault_label_name": "",
                                "label_verification": record.label_verification,
                                "label_support_files_present": str(record.label_support_files_present).lower(),
                                "bearing_code": record.bearing_code,
                                "condition_code": record.condition_code,
                                "speed_code": record.speed_code,
                                "torque_code": record.torque_code,
                                "radial_force_code": record.radial_force_code,
                                "repeat_index": record.repeat_index,
                                "filename": record.filename,
                                "relative_path": record.relative_path,
                                "measurement_id": record.measurement_id,
                                "selected_signal": selected_channel,
                                "available_channels": "|".join(prepared.available_channels),
                                "signal_length": prepared.signal_length,
                                "window_index": start_index + local_index,
                                "window_start": window_start,
                                "window_end": window_end,
                            }
                        )
            else:
                windows, spans = make_windows(
                    signal=signal,
                    start=0,
                    end=int(signal.size),
                    window_size=args.window_size,
                    stride=args.stride,
                )
                normalized = normalize_windows(windows, mean, std, output_dtype)
                if normalized.shape[0] == 0:
                    continue

                start_index = split_offsets["test_fault"]
                end_index = start_index + normalized.shape[0]
                target = split_arrays["test_fault"]
                if target is None or fault_labels_array is None:
                    raise RuntimeError("Fault output targets were not initialized.")
                target[start_index:end_index] = normalized
                split_offsets["test_fault"] = end_index

                fault_label_int = fault_label_map[record.damage_group]
                fault_labels_array[fault_label_offset : fault_label_offset + normalized.shape[0]] = fault_label_int
                fault_label_offset += normalized.shape[0]

                for local_index, (window_start, window_end) in enumerate(spans):
                    writer.writerow(
                        {
                            "split": "test",
                            "subset": "test_fault",
                            "health_status": record.health_status,
                            "damage_group": record.damage_group,
                            "fault_label_int": fault_label_int,
                            "fault_label_name": record.damage_group,
                            "label_verification": record.label_verification,
                            "label_support_files_present": str(record.label_support_files_present).lower(),
                            "bearing_code": record.bearing_code,
                            "condition_code": record.condition_code,
                            "speed_code": record.speed_code,
                            "torque_code": record.torque_code,
                            "radial_force_code": record.radial_force_code,
                            "repeat_index": record.repeat_index,
                            "filename": record.filename,
                            "relative_path": record.relative_path,
                            "measurement_id": record.measurement_id,
                            "selected_signal": selected_channel,
                            "available_channels": "|".join(prepared.available_channels),
                            "signal_length": prepared.signal_length,
                            "window_index": start_index + local_index,
                            "window_start": window_start,
                            "window_end": window_end,
                        }
                    )

    for array in split_arrays.values():
        if array is not None:
            array.flush()
    if fault_labels_array is not None:
        fault_labels_array.flush()

    normalization_stats = {
        "method": "zscore_global",
        "fit_split": "train",
        "fit_condition": "healthy",
        "mean": float(mean),
        "std": float(std),
        "num_train_windows": int(total_counts["train_healthy"]),
        "window_size": int(args.window_size),
        "num_train_samples": int(train_num_samples),
        "output_dtype": args.dtype,
        "signal_channel": args.channel,
    }
    write_json(output_paths["normalization_stats"], normalization_stats)

    preprocessing_config = {
        "dataset": "paderborn",
        "file_map_path": file_map_path.as_posix(),
        "raw_root": raw_root.as_posix(),
        "processed_root": processed_root.as_posix(),
        "metadata_root": metadata_root.as_posix(),
        "channel": args.channel,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "guard_gap": int(args.guard_gap),
        "split_ratios": {name: float(value) for name, value in ratios.items()},
        "normalization": {
            "method": "zscore_global",
            "fit_on": "healthy_train_only",
        },
        "dtype": args.dtype,
        "seed": int(args.seed),
        "fault_label_map": fault_label_map,
        "label_verification_policy": {
            "verified_labels_used": 0,
            "inferred_labels_used": len(prepared_records),
            "inference_rule": "K0xx=healthy, KA/KB/KI=damaged families",
            "support_pdf_parsed": False,
        },
    }
    write_json(output_paths["preprocessing_config"], preprocessing_config)

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
        f"Saved arrays verified with shapes {validated_shapes}; fault label count matches fault window count."
    )

    per_bearing_rows: list[list[Any]] = []
    for bearing_code in sorted(per_bearing_counts):
        counts = per_bearing_counts[bearing_code]
        total_windows = counts["train"] + counts["val"] + counts["test_healthy"] + counts["test_fault"]
        per_bearing_rows.append(
            [
                bearing_code,
                counts["health_status"],
                counts["damage_group"],
                counts["measurement_files"],
                counts["train"],
                counts["val"],
                counts["test_healthy"],
                counts["test_fault"],
                total_windows,
                counts["label_verification"],
                str(counts["support_files"]).lower(),
            ]
        )

    per_condition_rows: list[list[Any]] = []
    for condition_code in sorted(per_condition_counts):
        counts = per_condition_counts[condition_code]
        total_windows = counts["train"] + counts["val"] + counts["test_healthy"] + counts["test_fault"]
        per_condition_rows.append(
            [
                condition_code,
                counts["train"],
                counts["val"],
                counts["test_healthy"],
                counts["test_fault"],
                total_windows,
            ]
        )

    signal_summary = {
        "selected_channel": args.channel,
        "fallback_channel_count": int(fallback_channel_count),
        "missing_channel_count": int(missing_channel_count),
        "available_channels_example": list(prepared_records[0].available_channels) if prepared_records else [],
        "selected_channel_counts": dict(selected_channel_counter),
    }
    label_summary = {
        "verified_measurement_files": 0,
        "inferred_measurement_files": len(prepared_records),
        "bearings_with_support_files": int(sum(1 for counts in per_bearing_counts.values() if counts["support_files"])),
        "support_parsing_status": "local PDFs present but not parsed automatically",
        "fault_label_map_text": ", ".join(
            f"{label}={index}" for label, index in sorted(fault_label_map.items(), key=lambda item: item[1])
        ),
    }

    report_text = build_preprocessing_report(
        totals=total_counts,
        signal_summary=signal_summary,
        normalization_stats=normalization_stats,
        split_config=preprocessing_config,
        label_summary=label_summary,
        per_bearing_rows=per_bearing_rows,
        per_condition_rows=per_condition_rows,
        skipped_files=skipped_files,
        condition_counts=condition_file_counts,
        validation_summary=validation_summary,
    )
    output_paths["preprocessing_report"].write_text(report_text, encoding="utf-8")

    print("\nPaderborn Preprocessing Summary")
    print(f"  selected signal channel: {args.channel}")
    print(f"  healthy train windows: {total_counts['train_healthy']} -> {output_paths['train_healthy'].as_posix()}")
    print(f"  healthy val windows: {total_counts['val_healthy']} -> {output_paths['val_healthy'].as_posix()}")
    print(f"  healthy test windows: {total_counts['test_healthy']} -> {output_paths['test_healthy'].as_posix()}")
    print(f"  fault test windows: {total_counts['test_fault']} -> {output_paths['test_fault'].as_posix()}")
    print(f"  fault labels: {total_counts['fault_labels']} -> {output_paths['fault_labels'].as_posix()}")
    print(f"  normalization stats: {output_paths['normalization_stats'].as_posix()}")
    print(f"  preprocessing config: {output_paths['preprocessing_config'].as_posix()}")
    print(f"  window manifest: {output_paths['window_manifest'].as_posix()}")
    print(f"  preprocessing report: {output_paths['preprocessing_report'].as_posix()}")
    print(f"  skipped files: {len(skipped_files)}")
    print(f"  fallback channel selections: {fallback_channel_count}")
    print(f"  validation: {validation_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
