# Dataset Setup

## Official Dataset Names

- Case Western Reserve University Bearing Data Center dataset
- Paderborn University/KAt-DataCenter bearing dataset

Raw data is not redistributed in this repository. Users must obtain datasets from the official sources and follow the providers' terms.

## Expected Directory Layout

```text
data/raw/cwru/
data/raw/paderborn/
```

File maps and preprocessing metadata are provided under `data/metadata/`.

## CWRU Layout

Expected local file naming:

```text
normal_<load>.mat
ir007_<load>.mat
ball007_<load>.mat
or007_6_<load>.mat
```

Supported loads are `0`, `1`, `2`, and `3` hp. The selected CWRU signal is `*_DE_time`.

## Paderborn Layout

The selected Paderborn signal is `vibration_1`.

Healthy bearings:

```text
K001-K006
```

Damaged bearings:

```text
KA01, KA03, KA04, KA05, KA06, KA07, KA08, KA09, KA15, KA16, KA22, KA30
KB23, KB24, KB27
KI01, KI03, KI04, KI05, KI07, KI08, KI14, KI16, KI17, KI18, KI21
```

Operating condition codes:

```text
N09_M07_F10
N15_M01_F10
N15_M07_F04
N15_M07_F10
```

`data/metadata/paderborn/file_map.csv`, `bearing_label_map.json`, and related Markdown reports document the local file inventory and inferred labels.

The full Paderborn `window_manifest.csv` is regenerated locally because the metadata CSV is large. Run:

```powershell
python scripts/prepare_paderborn.py
python scripts/build_paderborn_label_map.py
python scripts/preprocess_paderborn.py
```

The release tracks a schema and sample instead:

```text
data/metadata/paderborn/window_manifest_schema.md
data/metadata/paderborn/window_manifest_sample.csv
```
