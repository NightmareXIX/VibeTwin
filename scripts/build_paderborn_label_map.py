from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
FILE_MAP_PATH = METADATA_ROOT / "file_map.csv"
OUTPUT_JSON_PATH = METADATA_ROOT / "bearing_label_map.json"
OUTPUT_MD_PATH = METADATA_ROOT / "bearing_label_map.md"

HEALTHY_PATTERN = re.compile(r"^K0\d+$", re.IGNORECASE)
DAMAGE_PATTERNS = {
    "KA": re.compile(r"^KA\d+$", re.IGNORECASE),
    "KB": re.compile(r"^KB\d+$", re.IGNORECASE),
    "KI": re.compile(r"^KI\d+$", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit per-bearing label map for the staged Paderborn dataset.",
    )
    parser.add_argument("--file-map", type=Path, default=FILE_MAP_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    return parser.parse_args()


def classify_bearing_code(bearing_code: str) -> tuple[str, str]:
    upper_code = bearing_code.upper()
    if HEALTHY_PATTERN.fullmatch(upper_code):
        return "healthy", ""

    for damage_group, pattern in DAMAGE_PATTERNS.items():
        if pattern.fullmatch(upper_code):
            return "damaged", damage_group

    return "", ""


def support_files_present(raw_root: Path, bearing_code: str) -> bool:
    bearing_dir = raw_root / bearing_code
    return (
        (bearing_dir / f"{bearing_code}.pdf").exists()
        and (bearing_dir / f"measuring_log_{bearing_code}.pdf").exists()
    )


def load_bearing_entries(file_map_path: Path, raw_root: Path) -> list[dict[str, Any]]:
    bearing_info: dict[str, dict[str, Any]] = {}

    with file_map_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["source_location"] != "canonical_raw_root":
                continue
            bearing_code = row["bearing_code"].strip().upper()
            if not bearing_code:
                continue

            entry = bearing_info.setdefault(
                bearing_code,
                {
                    "bearing_code": bearing_code,
                    "health_status": "",
                    "damage_group": "",
                    "label_source": "inferred_family_rule",
                    "support_files_present": support_files_present(raw_root, bearing_code),
                    "notes": "",
                    "measurement_file_count": 0,
                    "condition_codes": set(),
                },
            )

            if row["extension"].lower() == ".mat" and row["readable"].strip().lower() == "true":
                entry["measurement_file_count"] += 1
                if row["condition_code"]:
                    entry["condition_codes"].add(row["condition_code"])

    entries: list[dict[str, Any]] = []
    for bearing_code in sorted(bearing_info):
        entry = bearing_info[bearing_code]
        health_status, damage_group = classify_bearing_code(bearing_code)
        if not health_status:
            health_status = "unknown"
            damage_group = ""
            notes = "Bearing code did not match the current family-rule classifier; manual review is required."
        elif entry["support_files_present"]:
            notes = (
                "Support PDFs exist locally but were not parsed automatically in this pass; "
                "label remains inferred from the bearing-code family."
            )
        else:
            notes = (
                "No complete local support-file pair was found; label is inferred from the bearing-code family."
            )

        entries.append(
            {
                "bearing_code": bearing_code,
                "health_status": health_status,
                "damage_group": damage_group,
                "label_source": entry["label_source"],
                "support_files_present": bool(entry["support_files_present"]),
                "notes": notes,
                "measurement_file_count": int(entry["measurement_file_count"]),
                "condition_codes": sorted(entry["condition_codes"]),
            }
        )

    return entries


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_markdown(entries: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Paderborn Bearing Label Map",
        "",
        "## Summary",
        "",
        f"- Total bearings: `{summary['total_bearings']}`",
        f"- Verified from parsed PDFs: `{summary['verified_pdf_count']}`",
        f"- Inferred from family rule: `{summary['inferred_family_rule_count']}`",
        f"- Healthy bearings: `{summary['healthy_bearing_count']}`",
        f"- Damaged bearings: `{summary['damaged_bearing_count']}`",
        f"- Damage-group counts: `{json.dumps(summary['damage_group_counts'], ensure_ascii=True)}`",
        "",
        "## Bearing Table",
        "",
        "| bearing_code | health_status | damage_group | label_source | support_files_present | measurement_file_count | condition_codes | notes |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]

    for entry in entries:
        lines.append(
            "| "
            f"{entry['bearing_code']} | "
            f"{entry['health_status']} | "
            f"{entry['damage_group'] or '-'} | "
            f"{entry['label_source']} | "
            f"{str(entry['support_files_present']).lower()} | "
            f"{entry['measurement_file_count']} | "
            f"{', '.join(entry['condition_codes']) if entry['condition_codes'] else '-'} | "
            f"{entry['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Provenance Note",
            "",
            "- `verified_pdf` is reserved for a future pass that actually parses and uses the local support PDFs.",
            "- This version keeps every bearing explicit and ready for later manual verification without pretending that PDF verification already happened.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    file_map_path = args.file_map.resolve()
    raw_root = args.raw_root.resolve()
    metadata_root = args.metadata_root.resolve()
    metadata_root.mkdir(parents=True, exist_ok=True)

    if not file_map_path.exists():
        raise FileNotFoundError(f"file_map.csv was not found: {file_map_path.as_posix()}")
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root was not found: {raw_root.as_posix()}")

    entries = load_bearing_entries(file_map_path, raw_root)
    if not entries:
        raise RuntimeError("No Paderborn bearing entries were discovered from file_map.csv.")

    label_source_counts = Counter(entry["label_source"] for entry in entries)
    damage_group_counts = Counter(
        entry["damage_group"] for entry in entries if entry["damage_group"]
    )
    summary = {
        "total_bearings": len(entries),
        "verified_pdf_count": int(label_source_counts.get("verified_pdf", 0)),
        "inferred_family_rule_count": int(label_source_counts.get("inferred_family_rule", 0)),
        "healthy_bearing_count": int(sum(1 for entry in entries if entry["health_status"] == "healthy")),
        "damaged_bearing_count": int(sum(1 for entry in entries if entry["health_status"] == "damaged")),
        "damage_group_counts": dict(sorted(damage_group_counts.items())),
    }

    payload = {
        "dataset": "paderborn",
        "generated_from": {
            "file_map_path": file_map_path.as_posix(),
            "raw_root": raw_root.as_posix(),
        },
        "summary": summary,
        "bearings": entries,
    }
    write_json(metadata_root / OUTPUT_JSON_PATH.name, payload)
    markdown = build_markdown(entries, summary)
    (metadata_root / OUTPUT_MD_PATH.name).write_text(markdown, encoding="utf-8")

    print(f"Total bearings: {summary['total_bearings']}")
    print(f"Verified bearings: {summary['verified_pdf_count']}")
    print(f"Inferred bearings: {summary['inferred_family_rule_count']}")
    print(f"Wrote {(metadata_root / OUTPUT_JSON_PATH.name).as_posix()}")
    print(f"Wrote {(metadata_root / OUTPUT_MD_PATH.name).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
