# VibeTwin

VibeTwin is a research codebase for healthy-only bearing fault detection using compact and residual-dilated autoencoder models, with experiments on CWRU and Paderborn vibration datasets. The repository is organized to keep the GitHub version lightweight and reproducible: source code, manuscript assets, and small metadata files are tracked, while raw datasets, processed arrays, checkpoints, and generated experiment artifacts stay local.

## What is in this repository

- `scripts/`: preprocessing, training, evaluation, explanation, and paper-packaging utilities
- `data/metadata/cwru/`: CWRU metadata, inspection notes, and preprocessing audit files
- `data/metadata/paderborn/`: Paderborn metadata, label mappings, preparation notes, and preprocessing configuration
- `paper_latex/`: manuscript source, tables, figures, diagrams, and a compiled PDF snapshot
- `readme_versions.txt`: local writing/version note preserved from the research workflow

## What is intentionally not tracked

- `data/raw/`: original dataset files
- `data/processed/`: generated window arrays and labels
- `artifacts/`: checkpoints, metrics dumps, packaged exports, and generated figures
- local virtual environments, caches, and temporary folders

This follows normal GitHub practice for research repositories: publish the materials needed to understand and reproduce the pipeline, but do not upload large or license-sensitive datasets or machine-generated training outputs unless the project explicitly uses a release bucket, Git LFS, or an external data archive.

## Datasets

The experiments rely on external bearing datasets that should be obtained separately:

- CWRU
- Paderborn

After downloading the datasets, place them under `data/raw/` using the layout expected by the preparation scripts. The metadata files committed in this repository document the naming conventions, label mappings, and preprocessing assumptions used in the experiments.

## Reproducibility notes

1. Prepare the raw datasets.
2. Run the dataset preparation and preprocessing scripts in `scripts/`.
3. Train the baseline and generative models.
4. Run the evaluation scripts to regenerate metrics and figures.
5. Use `scripts/build_paper_package.py` if you want the full paper-support export locally.

The repository does not currently version generated checkpoints or large manifests. Those can be recreated from the tracked scripts and metadata.

## Python dependencies

Install dependencies from `requirements.txt`. PyTorch is required for the autoencoder and generative-model workflows.
