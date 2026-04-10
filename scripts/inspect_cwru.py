from __future__ import annotations

import sys
from pathlib import Path

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - dependency guard
    loadmat = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cwru"


def find_signal_keys(keys: list[str]) -> list[str]:
    ordered: list[str] = []
    for suffix in ("_DE_time", "_FE_time", "_BA_time"):
        ordered.extend(key for key in keys if key.endswith(suffix))
    if ordered:
        return ordered
    return [key for key in keys if "time" in key.lower()]


def describe_mat_file(path: Path) -> bool:
    try:
        data = loadmat(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[FAIL] {path.relative_to(PROJECT_ROOT)}: {exc}")
        return False

    keys = sorted(key for key in data.keys() if not key.startswith("__"))
    signal_keys = find_signal_keys(keys)

    print(f"\n=== {path.relative_to(PROJECT_ROOT).as_posix()} ===")
    print(f"keys: {', '.join(keys) if keys else '(none)'}")
    print(f"likely signal keys: {', '.join(signal_keys) if signal_keys else '(none found)'}")

    for key in keys:
        value = data[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"  - {key}: shape={shape}, dtype={dtype}")

    return True


def main() -> int:
    if loadmat is None:
        print("scipy is not available; install scipy to inspect CWRU .mat files.")
        return 0

    if not RAW_ROOT.exists():
        print(f"Raw dataset folder not found: {RAW_ROOT}")
        return 0

    mat_files = sorted(RAW_ROOT.rglob("*.mat"))
    if not mat_files:
        print(f"No .mat files found under {RAW_ROOT}")
        return 0

    success_count = 0
    failure_count = 0
    for path in mat_files:
        if describe_mat_file(path):
            success_count += 1
        else:
            failure_count += 1

    print("\nSummary")
    print(f"  scanned: {len(mat_files)}")
    print(f"  loaded: {success_count}")
    print(f"  failed: {failure_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
