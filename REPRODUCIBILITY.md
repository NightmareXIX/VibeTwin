# Reproducibility

## Scope

This release is prepared for conference-paper review. It includes source code, preprocessing metadata, manifest-generation utilities, selected split/provenance metadata, evaluation scripts, reports, and compact metrics needed to inspect and rerun the VibeTwin experiments.

## Tracked And Omitted Files

Tracked materials include Python scripts, root documentation, `requirements.txt`, `environment.yml`, compact dataset metadata under `data/metadata/`, the CWRU `window_manifest.csv`, Markdown reports, selected JSON metrics, and compact paper-package provenance files.

Intentionally omitted materials include raw external datasets, processed NumPy arrays, trained checkpoints, local virtual environments, local wheels, generated score CSVs, ZIP packages, and bulky generated outputs.

## External Dataset Requirement

The CWRU and Paderborn datasets must be obtained separately from their official providers. Raw data should be placed under `data/raw/` using the layouts documented in [DATASET_SETUP.md](DATASET_SETUP.md). This repository does not redistribute those datasets.

## End-To-End Stages

1. Obtain the external datasets from the official providers.
2. Place files under `data/raw/cwru/` and `data/raw/paderborn/`.
3. Run dataset preparation and preprocessing to regenerate `data/processed/` arrays and metadata manifests.
4. Train and evaluate the baseline and ResDilatedAE models.
5. Regenerate metrics, tables, figures, deployment summaries, and the paper package.

The command sequence is summarized in [RUN_COMMANDS.md](RUN_COMMANDS.md).

## Storage Notes

Raw datasets are large and omitted from Git. Processed arrays are generated from raw data and omitted. Model checkpoints are generated from training runs and omitted. Bulky generated artifacts are also omitted unless they are compact provenance files explicitly allowed by `.gitignore`.

The CWRU `data/metadata/cwru/window_manifest.csv` is tracked because it is compact metadata. The Paderborn `data/metadata/paderborn/window_manifest.csv` is not tracked as a plain CSV because it is about 189 MB and can be regenerated locally with:

```powershell
python scripts/prepare_paderborn.py
python scripts/build_paderborn_label_map.py
python scripts/preprocess_paderborn.py
```

For review, the repository tracks `data/metadata/paderborn/window_manifest_schema.md` and `data/metadata/paderborn/window_manifest_sample.csv`.

## Known Limitations

- Raw dataset checksums are not yet provided.
- Paderborn KA/KB/KI labels are prefix-derived in code unless manually verified against official documentation.
- Trained checkpoints are not redistributed.
