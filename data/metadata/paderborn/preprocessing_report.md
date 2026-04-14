# Paderborn Preprocessing Report

## Totals
| metric | value |
| --- | --- |
| healthy train windows | 82231 |
| healthy val windows | 16884 |
| healthy test windows | 16886 |
| fault test windows | 519082 |
| fault label count | 519082 |

## Signal Selection
| setting | value |
| --- | --- |
| selected signal channel | vibration_1 |
| files using fallback vibration selection | 0 |
| files skipped for missing signal | 1 |
| available channels example | force, phase_current_1, phase_current_2, speed, temp_2_bearing_module, torque, vibration_1 |

## Split and Normalization
| setting | value |
| --- | --- |
| window_size | 2048 |
| stride | 1024 |
| guard_gap | 2048 |
| split_ratios | {'train': 0.7, 'val': 0.15, 'test': 0.15} |
| dtype | float32 |

- Normalization method: zscore_global
- Healthy-train fit mean: -0.015629
- Healthy-train fit std: 0.294677
- Training samples used for normalization: 168409088

## Label Provenance
| item | value |
| --- | --- |
| measurement files with verified labels | 0 |
| measurement files with inferred labels | 2559 |
| bearings with support PDFs present | 32 |
| support PDF parsing status | local PDFs present but not parsed automatically |
| fault label map | KA=0, KB=1, KI=2 |

- Exact per-bearing damage verification from the local PDFs remains unresolved in this preprocessing pass because no PDF text extractor is available in the current environment.
- The train/val/test split uses bearing-code-family inference: `K0xx` as healthy, `KA/KB/KI` as damaged families.

## Counts by Bearing
| bearing_code | health_status | damage_group | measurement_files | train | val | test_healthy | test_fault | total_windows | label_verification | support_files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K001 | healthy | - | 80 | 13694 | 2815 | 2816 | 0 | 19325 | inferred_bearing_code_family | true |
| K002 | healthy | - | 80 | 13722 | 2811 | 2811 | 0 | 19344 | inferred_bearing_code_family | true |
| K003 | healthy | - | 80 | 13712 | 2815 | 2816 | 0 | 19343 | inferred_bearing_code_family | true |
| K004 | healthy | - | 80 | 13694 | 2809 | 2809 | 0 | 19312 | inferred_bearing_code_family | true |
| K005 | healthy | - | 80 | 13706 | 2816 | 2816 | 0 | 19338 | inferred_bearing_code_family | true |
| K006 | healthy | - | 80 | 13703 | 2818 | 2818 | 0 | 19339 | inferred_bearing_code_family | true |
| KA01 | damaged | KA | 80 | 0 | 0 | 0 | 19954 | 19954 | inferred_bearing_code_family | true |
| KA03 | damaged | KA | 80 | 0 | 0 | 0 | 20110 | 20110 | inferred_bearing_code_family | true |
| KA04 | damaged | KA | 80 | 0 | 0 | 0 | 19931 | 19931 | inferred_bearing_code_family | true |
| KA05 | damaged | KA | 80 | 0 | 0 | 0 | 19976 | 19976 | inferred_bearing_code_family | true |
| KA06 | damaged | KA | 80 | 0 | 0 | 0 | 19938 | 19938 | inferred_bearing_code_family | true |
| KA07 | damaged | KA | 80 | 0 | 0 | 0 | 20020 | 20020 | inferred_bearing_code_family | true |
| KA08 | damaged | KA | 79 | 0 | 0 | 0 | 19687 | 19687 | inferred_bearing_code_family | true |
| KA09 | damaged | KA | 80 | 0 | 0 | 0 | 19961 | 19961 | inferred_bearing_code_family | true |
| KA15 | damaged | KA | 80 | 0 | 0 | 0 | 19939 | 19939 | inferred_bearing_code_family | true |
| KA16 | damaged | KA | 80 | 0 | 0 | 0 | 19940 | 19940 | inferred_bearing_code_family | true |
| KA22 | damaged | KA | 80 | 0 | 0 | 0 | 19992 | 19992 | inferred_bearing_code_family | true |
| KA30 | damaged | KA | 80 | 0 | 0 | 0 | 19942 | 19942 | inferred_bearing_code_family | true |
| KB23 | damaged | KB | 80 | 0 | 0 | 0 | 19968 | 19968 | inferred_bearing_code_family | true |
| KB24 | damaged | KB | 80 | 0 | 0 | 0 | 19950 | 19950 | inferred_bearing_code_family | true |
| KB27 | damaged | KB | 80 | 0 | 0 | 0 | 20005 | 20005 | inferred_bearing_code_family | true |
| KI01 | damaged | KI | 80 | 0 | 0 | 0 | 19930 | 19930 | inferred_bearing_code_family | true |
| KI03 | damaged | KI | 80 | 0 | 0 | 0 | 19975 | 19975 | inferred_bearing_code_family | true |
| KI04 | damaged | KI | 80 | 0 | 0 | 0 | 19945 | 19945 | inferred_bearing_code_family | true |
| KI05 | damaged | KI | 80 | 0 | 0 | 0 | 19985 | 19985 | inferred_bearing_code_family | true |
| KI07 | damaged | KI | 80 | 0 | 0 | 0 | 19979 | 19979 | inferred_bearing_code_family | true |
| KI08 | damaged | KI | 80 | 0 | 0 | 0 | 19984 | 19984 | inferred_bearing_code_family | true |
| KI14 | damaged | KI | 80 | 0 | 0 | 0 | 19968 | 19968 | inferred_bearing_code_family | true |
| KI16 | damaged | KI | 80 | 0 | 0 | 0 | 20091 | 20091 | inferred_bearing_code_family | true |
| KI17 | damaged | KI | 80 | 0 | 0 | 0 | 20018 | 20018 | inferred_bearing_code_family | true |
| KI18 | damaged | KI | 80 | 0 | 0 | 0 | 19948 | 19948 | inferred_bearing_code_family | true |
| KI21 | damaged | KI | 80 | 0 | 0 | 0 | 19946 | 19946 | inferred_bearing_code_family | true |

## Counts by Operating Condition
| condition_code | train | val | test_healthy | test_fault | total_windows |
| --- | --- | --- | --- | --- | --- |
| N09_M07_F10 | 20538 | 4215 | 4215 | 129757 | 158725 |
| N15_M01_F10 | 20575 | 4222 | 4223 | 129572 | 158592 |
| N15_M07_F04 | 20561 | 4221 | 4222 | 129938 | 158942 |
| N15_M07_F10 | 20557 | 4226 | 4226 | 129815 | 158824 |

## Condition Inventory
- Measurement files per condition: {"N09_M07_F10": 640, "N15_M01_F10": 639, "N15_M07_F04": 640, "N15_M07_F10": 640}

## Skipped Files
- data/raw/paderborn/KA08/N15_M01_F10_KA08_2.mat: failed to load MAT file (Expecting matrix here)

## Validation
- Saved arrays verified with shapes {'train_healthy': (82231, 2048), 'val_healthy': (16884, 2048), 'test_healthy': (16886, 2048), 'test_fault': (519082, 2048), 'fault_labels': (519082,)}; fault label count matches fault window count.
