from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat, whosmat
except ImportError:  # pragma: no cover - dependency guard
    loadmat = None
    whosmat = None

try:
    import h5py
except ImportError:  # pragma: no cover - dependency guard
    h5py = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "paderborn"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"

FILE_MAP_CSV_PATH = METADATA_ROOT / "file_map.csv"
FILE_MAP_JSON_PATH = METADATA_ROOT / "file_map.json"
PREPARE_REPORT_PATH = METADATA_ROOT / "prepare_report.md"
EXTRACTED_ROOT_CANDIDATES = [
    PROJECT_ROOT / "paperborn_extracted",
    PROJECT_ROOT / "paderborn_extracted",
]
ARCHIVE_EXTENSIONS = {".rar", ".zip", ".7z"}

FILENAME_PATTERN = re.compile(
    r"^(?P<speed_code>N\d{2})_(?P<torque_code>M\d{2})_(?P<radial_force_code>F\d{2})_"
    r"(?P<bearing_code>[A-Za-z0-9]+)_(?P<repeat_index>\d+)$"
)
BEARING_CODE_PATTERN = re.compile(r"^K[0-9A-Z]+$", re.IGNORECASE)

CSV_COLUMNS = [
    "source_location",
    "source_root",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "file_format",
    "readable",
    "condition_code",
    "speed_code",
    "torque_code",
    "radial_force_code",
    "bearing_code",
    "repeat_index",
    "health_status_guess",
    "damage_group_guess",
    "label_source",
    "top_level_keys",
    "notes",
]

MEASUREMENT_EXTENSIONS = {".mat", ".h5", ".hdf5", ".npy", ".npz", ".csv", ".txt"}
SIGNAL_DATA_EXTENSIONS = {".mat", ".h5", ".hdf5", ".npy", ".npz", ".csv"}
NUMERIC_DTYPES = {
    "float64",
    "float32",
    "float16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
}


@dataclass
class ManifestEntry:
    source_location: str
    source_root: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    file_format: str
    readable: bool
    condition_code: str
    speed_code: str
    torque_code: str
    radial_force_code: str
    bearing_code: str
    repeat_index: str
    health_status_guess: str
    damage_group_guess: str
    label_source: str
    top_level_keys: str
    notes: str


@dataclass
class SampleInspection:
    relative_path: str
    file_format: str
    readable: bool
    top_level_keys: list[str]
    signal_keys: list[str]
    signal_shapes: list[str]
    signal_dtypes: list[str]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Paderborn dataset scaffolding and metadata inventory.",
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--sample-count", type=int, default=3)
    return parser.parse_args()


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def discover_files(raw_root: Path) -> list[tuple[str, Path, Path]]:
    discovered: list[tuple[str, Path, Path]] = []
    seen: set[Path] = set()

    candidate_roots: list[tuple[str, Path]] = []
    candidate_roots.append(("canonical_raw_root", raw_root))
    for candidate in EXTRACTED_ROOT_CANDIDATES:
        resolved = candidate.resolve()
        if resolved == raw_root.resolve():
            continue
        if resolved.exists():
            candidate_roots.append(("staging_extracted_root", resolved))

    for source_location, source_root in candidate_roots:
        if not source_root.exists():
            continue
        for path in sorted(candidate for candidate in source_root.rglob("*") if candidate.is_file()):
            resolved_path = path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            discovered.append((source_location, source_root, resolved_path))

    for path in sorted(PROJECT_ROOT.iterdir()):
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        lower_name = path.name.lower()
        if path.suffix.lower() in ARCHIVE_EXTENSIONS and BEARING_CODE_PATTERN.fullmatch(path.stem):
            discovered.append(("project_root_archive", PROJECT_ROOT, resolved_path))
            seen.add(resolved_path)
            continue
        if lower_name == "readme_versions.txt":
            discovered.append(("project_root_support", PROJECT_ROOT, resolved_path))
            seen.add(resolved_path)

    return discovered


def detect_file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".mat": "matlab",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".npy": "npy",
        ".npz": "npz",
        ".csv": "csv",
        ".txt": "text",
    }.get(suffix, suffix.lstrip(".") or "unknown")


def parse_filename_metadata(path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.match(path.stem)
    if not match:
        return {
            "condition_code": "",
            "speed_code": "",
            "torque_code": "",
            "radial_force_code": "",
            "bearing_code": "",
            "repeat_index": "",
        }

    groups = match.groupdict()
    return {
        "condition_code": f"{groups['speed_code']}_{groups['torque_code']}_{groups['radial_force_code']}",
        "speed_code": groups["speed_code"],
        "torque_code": groups["torque_code"],
        "radial_force_code": groups["radial_force_code"],
        "bearing_code": groups["bearing_code"],
        "repeat_index": groups["repeat_index"],
    }


def infer_bearing_code_from_path(path: Path, current_value: str) -> str:
    if current_value:
        return current_value

    candidates = [path.stem, path.parent.name]
    for candidate in candidates:
        if BEARING_CODE_PATTERN.fullmatch(candidate):
            return candidate.upper()
    return ""


def infer_label(path: Path, bearing_code: str) -> tuple[str, str, str]:
    joined = " ".join(part.lower() for part in path.parts)

    healthy_keywords = ("healthy", "normal", "undamaged", "good")
    damaged_keywords = ("fault", "faulty", "damage", "damaged", "defect")

    if any(keyword in joined for keyword in healthy_keywords):
        return "healthy", "", "path_keyword"
    if any(keyword in joined for keyword in damaged_keywords):
        return "damaged", "", "path_keyword"

    upper_code = bearing_code.upper()
    if re.fullmatch(r"K0\d+", upper_code):
        return "healthy", "", "bearing_code_prefix"
    if re.fullmatch(r"K[A-Z]\d+", upper_code):
        return "damaged", upper_code[:2], "bearing_code_prefix"

    return "", "", ""


def format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(dim)) for dim in shape) if shape else "scalar"


def safe_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): safe_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [safe_json(value) for value in data]
    if isinstance(data, Path):
        return data.as_posix()
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    return data


def collect_numeric_arrays(
    value: Any,
    prefix: str,
    arrays: list[tuple[str, tuple[int, ...], str]],
    *,
    depth: int = 0,
    max_items: int = 12,
) -> None:
    if len(arrays) >= max_items or depth > 3:
        return

    if isinstance(value, dict):
        if "Data" in value:
            channel_name = extract_channel_name(value.get("Name"))
            next_prefix = f"{prefix}.{channel_name}" if prefix else channel_name
            collect_numeric_arrays(value["Data"], next_prefix, arrays, depth=depth + 1, max_items=max_items)
            return
        for key, nested in list(value.items())[:max_items]:
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            collect_numeric_arrays(nested, next_prefix, arrays, depth=depth + 1, max_items=max_items)
        return

    if isinstance(value, (list, tuple)) and len(value) <= max_items:
        for index, nested in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            collect_numeric_arrays(nested, next_prefix, arrays, depth=depth + 1, max_items=max_items)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field_name in value.dtype.names[:max_items]:
                collect_numeric_arrays(
                    value[field_name],
                    f"{prefix}.{field_name}" if prefix else field_name,
                    arrays,
                    depth=depth + 1,
                    max_items=max_items,
                )
            return
        if value.dtype == object and value.size == 1:
            collect_numeric_arrays(value.flat[0], prefix, arrays, depth=depth + 1, max_items=max_items)
            return
        dtype_name = str(value.dtype)
        if dtype_name in NUMERIC_DTYPES or np.issubdtype(value.dtype, np.number):
            arrays.append((prefix or "array", tuple(int(dim) for dim in value.shape), dtype_name))
        return

    if hasattr(value, "__dict__"):
        for key, nested in list(vars(value).items())[:max_items]:
            if key.startswith("_"):
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            collect_numeric_arrays(nested, next_prefix, arrays, depth=depth + 1, max_items=max_items)


def extract_channel_name(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "unnamed"
    if isinstance(value, np.ndarray):
        squeezed = np.asarray(value).squeeze()
        if squeezed.size == 0:
            return "unnamed"
        if squeezed.size == 1:
            text = str(squeezed.item()).strip()
            return text if text else "unnamed"
    return "unnamed"


def inspect_mat_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    if whosmat is None:
        return False, [], ["scipy.io.whosmat is unavailable in the active environment"]

    try:
        variables = whosmat(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, [], [f"Failed to inspect MATLAB header: {exc}"]

    top_level = [
        f"{name}:{format_shape(tuple(shape))}:{class_name}"
        for name, shape, class_name in variables
    ]
    return True, top_level, []


def inspect_hdf5_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    if h5py is None:
        return False, [], ["h5py is unavailable in the active environment"]

    try:
        with h5py.File(path, "r") as handle:
            top_level = sorted(handle.keys())
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, [], [f"Failed to inspect HDF5 file: {exc}"]

    return True, top_level, []


def inspect_npy_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, [], [f"Failed to inspect NPY file: {exc}"]

    top_level = [f"array:{format_shape(tuple(array.shape))}:{array.dtype}"]
    return True, top_level, []


def inspect_npz_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            top_level = [
                f"{key}:{format_shape(tuple(archive[key].shape))}:{archive[key].dtype}"
                for key in archive.files
            ]
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, [], [f"Failed to inspect NPZ file: {exc}"]

    return True, top_level, []


def inspect_text_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().strip()
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, [], [f"Failed to inspect text file: {exc}"]

    preview = first_line[:120] if first_line else ""
    return True, [preview] if preview else [], []


def inspect_manifest_metadata(path: Path) -> tuple[bool, list[str], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return inspect_mat_metadata(path)
    if suffix in {".h5", ".hdf5"}:
        return inspect_hdf5_metadata(path)
    if suffix == ".npy":
        return inspect_npy_metadata(path)
    if suffix == ".npz":
        return inspect_npz_metadata(path)
    if suffix in {".csv", ".txt"}:
        return inspect_text_metadata(path)
    return False, [], ["No metadata inspector is implemented for this extension"]


def inspect_mat_sample(path: Path) -> SampleInspection:
    notes: list[str] = []
    readable, top_level, metadata_notes = inspect_mat_metadata(path)
    notes.extend(metadata_notes)
    signal_arrays: list[tuple[str, tuple[int, ...], str]] = []

    if loadmat is None:
        notes.append("scipy.io.loadmat is unavailable; skipping deep MATLAB inspection")
    else:
        try:
            data = loadmat(path, simplify_cells=True)
            visible_keys = sorted(key for key in data if not key.startswith("__"))
            for key in visible_keys:
                collect_numeric_arrays(data[key], key, signal_arrays)
            if not top_level:
                top_level = visible_keys
        except Exception as exc:  # pragma: no cover - runtime safety
            notes.append(f"Deep MATLAB inspection failed: {exc}")

    signal_keys = [name for name, _, _ in signal_arrays]
    signal_shapes = [f"{name}:{format_shape(shape)}" for name, shape, _ in signal_arrays]
    signal_dtypes = [f"{name}:{dtype}" for name, _, dtype in signal_arrays]

    return SampleInspection(
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="matlab",
        readable=readable,
        top_level_keys=top_level,
        signal_keys=signal_keys,
        signal_shapes=signal_shapes,
        signal_dtypes=signal_dtypes,
        notes=notes,
    )


def inspect_hdf5_sample(path: Path) -> SampleInspection:
    notes: list[str] = []
    top_level: list[str] = []
    signal_keys: list[str] = []
    signal_shapes: list[str] = []
    signal_dtypes: list[str] = []

    if h5py is None:
        return SampleInspection(
            relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
            file_format="hdf5",
            readable=False,
            top_level_keys=[],
            signal_keys=[],
            signal_shapes=[],
            signal_dtypes=[],
            notes=["h5py is unavailable in the active environment"],
        )

    try:
        with h5py.File(path, "r") as handle:
            top_level = sorted(handle.keys())

            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                if len(signal_keys) >= 12:
                    return
                signal_keys.append(name)
                signal_shapes.append(f"{name}:{format_shape(tuple(obj.shape))}")
                signal_dtypes.append(f"{name}:{obj.dtype}")

            handle.visititems(visitor)
        readable = True
    except Exception as exc:  # pragma: no cover - runtime safety
        readable = False
        notes.append(f"Deep HDF5 inspection failed: {exc}")

    return SampleInspection(
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="hdf5",
        readable=readable,
        top_level_keys=top_level,
        signal_keys=signal_keys,
        signal_shapes=signal_shapes,
        signal_dtypes=signal_dtypes,
        notes=notes,
    )


def inspect_npy_sample(path: Path) -> SampleInspection:
    notes: list[str] = []
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        readable = True
        top_level_keys = ["array"]
        signal_keys = ["array"]
        signal_shapes = [f"array:{format_shape(tuple(array.shape))}"]
        signal_dtypes = [f"array:{array.dtype}"]
    except Exception as exc:  # pragma: no cover - runtime safety
        readable = False
        top_level_keys = []
        signal_keys = []
        signal_shapes = []
        signal_dtypes = []
        notes.append(f"Deep NPY inspection failed: {exc}")

    return SampleInspection(
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="npy",
        readable=readable,
        top_level_keys=top_level_keys,
        signal_keys=signal_keys,
        signal_shapes=signal_shapes,
        signal_dtypes=signal_dtypes,
        notes=notes,
    )


def inspect_npz_sample(path: Path) -> SampleInspection:
    notes: list[str] = []
    top_level_keys: list[str] = []
    signal_keys: list[str] = []
    signal_shapes: list[str] = []
    signal_dtypes: list[str] = []

    try:
        with np.load(path, allow_pickle=False) as archive:
            top_level_keys = list(archive.files)
            for key in archive.files[:12]:
                signal_keys.append(key)
                signal_shapes.append(f"{key}:{format_shape(tuple(archive[key].shape))}")
                signal_dtypes.append(f"{key}:{archive[key].dtype}")
        readable = True
    except Exception as exc:  # pragma: no cover - runtime safety
        readable = False
        notes.append(f"Deep NPZ inspection failed: {exc}")

    return SampleInspection(
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="npz",
        readable=readable,
        top_level_keys=top_level_keys,
        signal_keys=signal_keys,
        signal_shapes=signal_shapes,
        signal_dtypes=signal_dtypes,
        notes=notes,
    )


def inspect_text_sample(path: Path) -> SampleInspection:
    readable, top_level, notes = inspect_text_metadata(path)
    return SampleInspection(
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format=detect_file_format(path),
        readable=readable,
        top_level_keys=top_level,
        signal_keys=[],
        signal_shapes=[],
        signal_dtypes=[],
        notes=notes,
    )


def inspect_sample(path: Path) -> SampleInspection:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return inspect_mat_sample(path)
    if suffix in {".h5", ".hdf5"}:
        return inspect_hdf5_sample(path)
    if suffix == ".npy":
        return inspect_npy_sample(path)
    if suffix == ".npz":
        return inspect_npz_sample(path)
    return inspect_text_sample(path)


def build_manifest_entries(files: list[tuple[str, Path, Path]]) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []

    for source_location, source_root, path in files:
        filename_info = parse_filename_metadata(path)
        bearing_code = infer_bearing_code_from_path(path, filename_info["bearing_code"])
        health_status_guess, damage_group_guess, label_source = infer_label(
            path,
            bearing_code,
        )
        readable, top_level_keys, notes = inspect_manifest_metadata(path)

        entries.append(
            ManifestEntry(
                source_location=source_location,
                source_root=source_root.relative_to(PROJECT_ROOT).as_posix() if source_root != PROJECT_ROOT else ".",
                relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                file_format=detect_file_format(path),
                readable=readable,
                condition_code=filename_info["condition_code"],
                speed_code=filename_info["speed_code"],
                torque_code=filename_info["torque_code"],
                radial_force_code=filename_info["radial_force_code"],
                bearing_code=bearing_code,
                repeat_index=filename_info["repeat_index"],
                health_status_guess=health_status_guess,
                damage_group_guess=damage_group_guess,
                label_source=label_source,
                top_level_keys=" | ".join(top_level_keys),
                notes=" | ".join(notes),
            )
        )

    return entries


def choose_sample_files(files: list[tuple[str, Path, Path]], sample_count: int) -> list[Path]:
    if not files:
        return []

    preferred = []
    for source_location, _, path in files:
        if source_location not in {"canonical_raw_root", "staging_extracted_root"}:
            continue
        if path.suffix.lower() not in SIGNAL_DATA_EXTENSIONS:
            continue
        preferred.append(path)

    selected: list[Path] = []
    selected_bearings: set[str] = set()

    priority_groups = ["healthy", "KA", "KI", "KB", "damaged", "other"]
    for group_name in priority_groups:
        if len(selected) >= sample_count:
            break
        for path in preferred:
            filename_info = parse_filename_metadata(path)
            bearing_code = infer_bearing_code_from_path(path, filename_info["bearing_code"])
            health_status_guess, damage_group_guess, _ = infer_label(path, bearing_code)
            inferred_group = damage_group_guess or health_status_guess or "other"
            if inferred_group != group_name:
                continue
            if bearing_code and bearing_code in selected_bearings:
                continue
            selected.append(path)
            if bearing_code:
                selected_bearings.add(bearing_code)
            break

    if len(selected) < sample_count:
        extras = [path for path in preferred if path not in selected]
        selected.extend(extras[: sample_count - len(selected)])
    return selected


def count_nonempty(values: list[str]) -> dict[str, int]:
    counter = Counter(value for value in values if value)
    return dict(sorted(counter.items()))


def determine_missing_requirements(entries: list[ManifestEntry]) -> list[str]:
    if not entries:
        return [
            "Place the official Paderborn raw measurement files under data/raw/paderborn/ (recursive subfolders are fine).",
            "Include the original measurement files rather than derived exports so preprocessing can mirror the CWRU pipeline.",
            "If available, place the accompanying measuring log, fact sheets, or README under data/raw/paderborn/ so bearing-code-to-damage mappings can be verified locally.",
        ]

    requirements: list[str] = []
    measurement_entries = [
        entry
        for entry in entries
        if entry.extension in SIGNAL_DATA_EXTENSIONS and entry.readable
    ]
    if not measurement_entries:
        requirements.append(
            "Add at least one readable measurement file under data/raw/paderborn/; current files are not in a supported or readable format."
        )

    if not any(entry.health_status_guess for entry in entries):
        requirements.append(
            "Add official label metadata or fact sheets so healthy vs damaged bearings can be verified instead of inferred from filenames."
        )

    if not any(entry.condition_code for entry in entries):
        requirements.append(
            "Add files or metadata that expose operating-condition codes so load-condition-aware splits can mirror the CWRU setup."
        )

    canonical_measurements = [
        entry
        for entry in entries
        if entry.source_location == "canonical_raw_root"
        and entry.extension in SIGNAL_DATA_EXTENSIONS
        and entry.readable
    ]
    if measurement_entries and not canonical_measurements:
        requirements.append(
            "Stage the extracted measurement tree under data/raw/paderborn/ or point the future preprocessing script at the current extracted source root."
        )

    return requirements


def is_ready_for_preprocessing(entries: list[ManifestEntry]) -> bool:
    return any(
        entry.extension in SIGNAL_DATA_EXTENSIONS and entry.readable
        for entry in entries
    )


def select_measurement_root(entries: list[ManifestEntry]) -> str:
    priorities = ("canonical_raw_root", "staging_extracted_root")
    for source_location in priorities:
        roots = [
            entry.source_root
            for entry in entries
            if entry.source_location == source_location
            and entry.extension in SIGNAL_DATA_EXTENSIONS
            and entry.readable
        ]
        if roots:
            return roots[0]
    return ""


def write_file_map_csv(entries: list[ManifestEntry], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def build_report(
    raw_root: Path,
    processed_root: Path,
    metadata_root: Path,
    entries: list[ManifestEntry],
    sample_inspections: list[SampleInspection],
    missing_requirements: list[str],
) -> str:
    extensions = count_nonempty([entry.extension for entry in entries])
    source_counts = count_nonempty([entry.source_location for entry in entries])
    health_counts = count_nonempty([entry.health_status_guess for entry in entries])
    damage_counts = count_nonempty([entry.damage_group_guess for entry in entries])
    condition_counts = count_nonempty([entry.condition_code for entry in entries])
    readable_count = sum(int(entry.readable) for entry in entries)
    ready = is_ready_for_preprocessing(entries)
    selected_measurement_root = select_measurement_root(entries)
    canonical_measurement_count = sum(
        1
        for entry in entries
        if entry.source_location == "canonical_raw_root"
        and entry.extension in SIGNAL_DATA_EXTENSIONS
        and entry.readable
    )

    lines = [
        "# Paderborn Preparation Report",
        "",
        "## Summary",
        "",
        f"- Raw root: `{raw_root.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Processed root: `{processed_root.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Metadata root: `{metadata_root.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Files discovered: {len(entries)}",
        f"- Readable files: {readable_count}",
        f"- Source locations: {json.dumps(source_counts, ensure_ascii=True)}",
        f"- Selected measurement root: `{selected_measurement_root or 'none detected'}`",
        f"- Canonical raw-root measurement files: {canonical_measurement_count}",
        f"- Ready for preprocessing: {'yes' if ready else 'no'}",
        "",
        "## Inventory",
        "",
        f"- Extensions: {json.dumps(extensions, ensure_ascii=True)}",
        f"- Healthy/damaged guesses: {json.dumps(health_counts, ensure_ascii=True)}",
        f"- Damage-group guesses: {json.dumps(damage_counts, ensure_ascii=True)}",
        f"- Operating-condition codes: {json.dumps(condition_counts, ensure_ascii=True)}",
        "",
    ]

    if entries:
        lines.extend(
            [
                "## Representative Files",
                "",
                "| File | Source | Format | Readable | Condition | Bearing | Health Guess | Top-Level Keys |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for entry in entries[: min(10, len(entries))]:
            top_level = entry.top_level_keys or "-"
            lines.append(
                f"| `{entry.relative_path}` | {entry.source_location} | {entry.file_format} | {entry.readable} | "
                f"{entry.condition_code or '-'} | {entry.bearing_code or '-'} | "
                f"{entry.health_status_guess or '-'} | {top_level} |"
            )
        lines.append("")

    if sample_inspections:
        lines.extend(["## Sample Inspection", ""])
        for sample in sample_inspections:
            lines.append(f"### `{sample.relative_path}`")
            lines.append("")
            lines.append(f"- File format: {sample.file_format}")
            lines.append(f"- Readable: {sample.readable}")
            lines.append(
                f"- Top-level keys: {', '.join(sample.top_level_keys) if sample.top_level_keys else 'none detected'}"
            )
            lines.append(
                f"- Signal keys/channels: {', '.join(sample.signal_keys) if sample.signal_keys else 'none detected'}"
            )
            lines.append(
                f"- Signal shapes: {', '.join(sample.signal_shapes) if sample.signal_shapes else 'none detected'}"
            )
            lines.append(
                f"- Signal dtypes: {', '.join(sample.signal_dtypes) if sample.signal_dtypes else 'none detected'}"
            )
            lines.append(
                f"- Notes: {'; '.join(sample.notes) if sample.notes else 'none'}"
            )
            lines.append("")

    lines.extend(["## Missing Requirements", ""])
    if missing_requirements:
        for item in missing_requirements:
            lines.append(f"- {item}")
    else:
        lines.append("- No blockers detected from the local file inventory.")
    lines.append("")

    lines.extend(
        [
            "## Readiness Note",
            "",
            (
                "- Paderborn has readable local measurement files in the canonical raw-data layout and is ready for a preprocessing script."
                if ready and canonical_measurement_count > 0
                else (
                    "- Paderborn has readable local measurement files and is ready for a preprocessing script, but the current extracted source should be staged under `data/raw/paderborn/` for a fully canonical CWRU-like layout."
                    if ready
                    else "- Paderborn is not ready for preprocessing yet because no readable measurement files were found locally."
                )
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    metadata_root = args.metadata_root.resolve()

    ensure_directories(raw_root, processed_root, metadata_root)

    files = discover_files(raw_root)
    entries = build_manifest_entries(files)
    sample_paths = choose_sample_files(files, args.sample_count)
    sample_inspections = [inspect_sample(path) for path in sample_paths]
    missing_requirements = determine_missing_requirements(entries)

    write_file_map_csv(entries, metadata_root / FILE_MAP_CSV_PATH.name)

    payload = {
        "dataset": "paderborn",
        "raw_root": raw_root.relative_to(PROJECT_ROOT).as_posix(),
        "processed_root": processed_root.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_root": metadata_root.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_present": bool(entries),
        "ready_for_preprocessing": is_ready_for_preprocessing(entries),
        "selected_measurement_root": select_measurement_root(entries),
        "discovered_file_count": len(entries),
        "source_location_counts": count_nonempty([entry.source_location for entry in entries]),
        "extension_counts": count_nonempty([entry.extension for entry in entries]),
        "health_status_guess_counts": count_nonempty([entry.health_status_guess for entry in entries]),
        "damage_group_guess_counts": count_nonempty([entry.damage_group_guess for entry in entries]),
        "condition_code_counts": count_nonempty([entry.condition_code for entry in entries]),
        "missing_requirements": missing_requirements,
        "files": [asdict(entry) for entry in entries],
        "sample_inspection": [asdict(sample) for sample in sample_inspections],
    }

    with (metadata_root / FILE_MAP_JSON_PATH.name).open("w", encoding="utf-8") as handle:
        json.dump(safe_json(payload), handle, indent=2)

    report = build_report(
        raw_root=raw_root,
        processed_root=processed_root,
        metadata_root=metadata_root,
        entries=entries,
        sample_inspections=sample_inspections,
        missing_requirements=missing_requirements,
    )
    (metadata_root / PREPARE_REPORT_PATH.name).write_text(report, encoding="utf-8")

    print(f"Discovered files: {len(entries)}")
    print(f"Ready for preprocessing: {is_ready_for_preprocessing(entries)}")
    print(f"Wrote {FILE_MAP_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Wrote {FILE_MAP_JSON_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Wrote {PREPARE_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")

    if missing_requirements:
        print("Missing requirements:")
        for item in missing_requirements:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
