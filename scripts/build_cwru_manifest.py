from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cwru"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "cwru"
FILE_MAP_PATH = METADATA_ROOT / "file_map.csv"
MANIFEST_PATH = METADATA_ROOT / "manifest.csv"

MANIFEST_COLUMNS = [
    "relpath",
    "filename",
    "class",
    "load_hp",
    "condition",
    "source_id",
    "subset_group",
    "exists",
]


def load_expected_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_manifest_rows(expected_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for record in expected_rows:
        relpath = Path("data") / "raw" / "cwru" / record["subset_group"] / record["filename"]
        exists = (PROJECT_ROOT / relpath).exists()
        row = {
            "relpath": relpath.as_posix(),
            "filename": record["filename"],
            "class": record["class"],
            "load_hp": record["load_hp"],
            "condition": record["condition"],
            "source_id": record["source_id"],
            "subset_group": record["subset_group"],
            "exists": str(exists).lower(),
        }
        rows.append(row)
        if not exists:
            missing.append(record["filename"])

    return rows, missing


def find_untracked_files(expected_rows: list[dict[str, str]]) -> list[Path]:
    expected_paths = {
        (Path("data") / "raw" / "cwru" / row["subset_group"] / row["filename"]).as_posix()
        for row in expected_rows
    }
    discovered = sorted(path.relative_to(PROJECT_ROOT) for path in RAW_ROOT.rglob("*.mat"))
    return [path for path in discovered if path.as_posix() not in expected_paths]


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not FILE_MAP_PATH.exists():
        print(f"file_map.csv not found: {FILE_MAP_PATH}")
        return 1

    expected_rows = load_expected_rows(FILE_MAP_PATH)
    rows, missing = build_manifest_rows(expected_rows)
    write_manifest(rows, MANIFEST_PATH)

    untracked = find_untracked_files(expected_rows)

    print(f"Wrote manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Expected files: {len(expected_rows)}")
    print(f"Missing expected files: {len(missing)}")
    if missing:
        for name in missing:
            print(f"  - {name}")

    print(f"Untracked discovered files: {len(untracked)}")
    if untracked:
        for path in untracked:
            print(f"  - {path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
