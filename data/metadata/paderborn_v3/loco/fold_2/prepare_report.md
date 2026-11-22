# Paderborn Preparation Report

## Summary

- Raw root: `data/raw/paderborn`
- Processed root: `data/processed/paderborn`
- Metadata root: `data/metadata/paderborn`
- Files discovered: 2657
- Readable files: 2561
- Source locations: {"canonical_raw_root": 2624, "project_root_archive": 32, "project_root_support": 1}
- Selected measurement root: `data/raw/paderborn`
- Canonical raw-root measurement files: 2560
- Ready for preprocessing: yes

## Inventory

- Extensions: {".mat": 2560, ".pdf": 64, ".rar": 32, ".txt": 1}
- Healthy/damaged guesses: {"damaged": 2158, "healthy": 498}
- Damage-group guesses: {"KA": 996, "KB": 249, "KI": 913}
- Operating-condition codes: {"N09_M07_F10": 640, "N15_M01_F10": 640, "N15_M07_F04": 640, "N15_M07_F10": 640}

## Representative Files

| File | Source | Format | Readable | Condition | Bearing | Health Guess | Top-Level Keys |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `data/raw/paderborn/K001/K001.pdf` | canonical_raw_root | pdf | False | - | K001 | healthy | - |
| `data/raw/paderborn/K001/measuring_log_K001.pdf` | canonical_raw_root | pdf | False | - | K001 | healthy | - |
| `data/raw/paderborn/K001/N09_M07_F10_K001_1.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_1:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_10.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_10:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_11.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_11:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_12.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_12:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_13.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_13:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_14.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_14:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_15.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_15:1x1:struct |
| `data/raw/paderborn/K001/N09_M07_F10_K001_16.mat` | canonical_raw_root | matlab | True | N09_M07_F10 | K001 | healthy | N09_M07_F10_K001_16:1x1:struct |

## Sample Inspection

### `data/raw/paderborn/K001/N09_M07_F10_K001_1.mat`

- File format: matlab
- Readable: True
- Top-level keys: N09_M07_F10_K001_1:1x1:struct
- Signal keys/channels: N09_M07_F10_K001_1.X[0].unnamed, N09_M07_F10_K001_1.X[1].unnamed, N09_M07_F10_K001_1.X[2].unnamed, N09_M07_F10_K001_1.Y[0].force, N09_M07_F10_K001_1.Y[1].phase_current_1, N09_M07_F10_K001_1.Y[2].phase_current_2, N09_M07_F10_K001_1.Y[3].speed, N09_M07_F10_K001_1.Y[4].temp_2_bearing_module, N09_M07_F10_K001_1.Y[5].torque, N09_M07_F10_K001_1.Y[6].vibration_1
- Signal shapes: N09_M07_F10_K001_1.X[0].unnamed:16008, N09_M07_F10_K001_1.X[1].unnamed:256823, N09_M07_F10_K001_1.X[2].unnamed:5, N09_M07_F10_K001_1.Y[0].force:16008, N09_M07_F10_K001_1.Y[1].phase_current_1:256823, N09_M07_F10_K001_1.Y[2].phase_current_2:256823, N09_M07_F10_K001_1.Y[3].speed:16008, N09_M07_F10_K001_1.Y[4].temp_2_bearing_module:5, N09_M07_F10_K001_1.Y[5].torque:16008, N09_M07_F10_K001_1.Y[6].vibration_1:256823
- Signal dtypes: N09_M07_F10_K001_1.X[0].unnamed:float64, N09_M07_F10_K001_1.X[1].unnamed:float64, N09_M07_F10_K001_1.X[2].unnamed:float64, N09_M07_F10_K001_1.Y[0].force:float64, N09_M07_F10_K001_1.Y[1].phase_current_1:float64, N09_M07_F10_K001_1.Y[2].phase_current_2:float64, N09_M07_F10_K001_1.Y[3].speed:float64, N09_M07_F10_K001_1.Y[4].temp_2_bearing_module:float64, N09_M07_F10_K001_1.Y[5].torque:float64, N09_M07_F10_K001_1.Y[6].vibration_1:float64
- Notes: none

### `data/raw/paderborn/KA01/N09_M07_F10_KA01_1.mat`

- File format: matlab
- Readable: True
- Top-level keys: N09_M07_F10_KA01_1:1x1:struct
- Signal keys/channels: N09_M07_F10_KA01_1.X[0].unnamed, N09_M07_F10_KA01_1.X[1].unnamed, N09_M07_F10_KA01_1.X[2].unnamed, N09_M07_F10_KA01_1.Y[0].force, N09_M07_F10_KA01_1.Y[1].phase_current_1, N09_M07_F10_KA01_1.Y[2].phase_current_2, N09_M07_F10_KA01_1.Y[3].speed, N09_M07_F10_KA01_1.Y[4].temp_2_bearing_module, N09_M07_F10_KA01_1.Y[5].torque, N09_M07_F10_KA01_1.Y[6].vibration_1
- Signal shapes: N09_M07_F10_KA01_1.X[0].unnamed:16001, N09_M07_F10_KA01_1.X[1].unnamed:256001, N09_M07_F10_KA01_1.X[2].unnamed:5, N09_M07_F10_KA01_1.Y[0].force:16001, N09_M07_F10_KA01_1.Y[1].phase_current_1:256001, N09_M07_F10_KA01_1.Y[2].phase_current_2:256001, N09_M07_F10_KA01_1.Y[3].speed:16001, N09_M07_F10_KA01_1.Y[4].temp_2_bearing_module:5, N09_M07_F10_KA01_1.Y[5].torque:16001, N09_M07_F10_KA01_1.Y[6].vibration_1:256001
- Signal dtypes: N09_M07_F10_KA01_1.X[0].unnamed:float64, N09_M07_F10_KA01_1.X[1].unnamed:float64, N09_M07_F10_KA01_1.X[2].unnamed:float64, N09_M07_F10_KA01_1.Y[0].force:float64, N09_M07_F10_KA01_1.Y[1].phase_current_1:float64, N09_M07_F10_KA01_1.Y[2].phase_current_2:float64, N09_M07_F10_KA01_1.Y[3].speed:float64, N09_M07_F10_KA01_1.Y[4].temp_2_bearing_module:float64, N09_M07_F10_KA01_1.Y[5].torque:float64, N09_M07_F10_KA01_1.Y[6].vibration_1:float64
- Notes: none

### `data/raw/paderborn/KI01/N09_M07_F10_KI01_1.mat`

- File format: matlab
- Readable: True
- Top-level keys: N09_M07_F10_KI01_1:1x1:struct
- Signal keys/channels: N09_M07_F10_KI01_1.X[0].unnamed, N09_M07_F10_KI01_1.X[1].unnamed, N09_M07_F10_KI01_1.X[2].unnamed, N09_M07_F10_KI01_1.Y[0].force, N09_M07_F10_KI01_1.Y[1].phase_current_1, N09_M07_F10_KI01_1.Y[2].phase_current_2, N09_M07_F10_KI01_1.Y[3].speed, N09_M07_F10_KI01_1.Y[4].temp_2_bearing_module, N09_M07_F10_KI01_1.Y[5].torque, N09_M07_F10_KI01_1.Y[6].vibration_1
- Signal shapes: N09_M07_F10_KI01_1.X[0].unnamed:16001, N09_M07_F10_KI01_1.X[1].unnamed:256001, N09_M07_F10_KI01_1.X[2].unnamed:5, N09_M07_F10_KI01_1.Y[0].force:16001, N09_M07_F10_KI01_1.Y[1].phase_current_1:256001, N09_M07_F10_KI01_1.Y[2].phase_current_2:256001, N09_M07_F10_KI01_1.Y[3].speed:16001, N09_M07_F10_KI01_1.Y[4].temp_2_bearing_module:5, N09_M07_F10_KI01_1.Y[5].torque:16001, N09_M07_F10_KI01_1.Y[6].vibration_1:256001
- Signal dtypes: N09_M07_F10_KI01_1.X[0].unnamed:float64, N09_M07_F10_KI01_1.X[1].unnamed:float64, N09_M07_F10_KI01_1.X[2].unnamed:float64, N09_M07_F10_KI01_1.Y[0].force:float64, N09_M07_F10_KI01_1.Y[1].phase_current_1:float64, N09_M07_F10_KI01_1.Y[2].phase_current_2:float64, N09_M07_F10_KI01_1.Y[3].speed:float64, N09_M07_F10_KI01_1.Y[4].temp_2_bearing_module:float64, N09_M07_F10_KI01_1.Y[5].torque:float64, N09_M07_F10_KI01_1.Y[6].vibration_1:float64
- Notes: none

## Missing Requirements

- No blockers detected from the local file inventory.

## Readiness Note

- Paderborn has readable local measurement files in the canonical raw-data layout and is ready for a preprocessing script.
