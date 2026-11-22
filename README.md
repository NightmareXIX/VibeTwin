# VibeTwin

VibeTwin is a healthy-reference reconstruction framework for bearing fault detection. It trains compact and residual-dilated autoencoder models on healthy vibration windows, then evaluates anomaly detection under CWRU load shift and on Paderborn, including the unseen-bearing and unseen-condition transfer protocols added for the camera-ready version (ICCIT 2026).

This repository does not redistribute raw CWRU/Paderborn data, processed arrays, trained checkpoints, or bulky generated outputs.

The repository contains source code, preprocessing metadata, manifest-generation utilities, selected split/provenance metadata, evaluation scripts, and compact result-provenance files.

## Repository Contents

- `scripts/`: dataset preparation, preprocessing, training, evaluation, explanation, and paper-package utilities.
- `data/metadata/`: compact dataset metadata, file maps, preprocessing configuration, label maps, and the tracked CWRU window manifest.
- `reports/`: small release/provenance notes after local path scrubbing.
- `artifacts/metrics/`: compact metrics and reports selected for provenance.
- `artifacts/paper_package_v1/`: compact final metrics, reports, and candidate tables selected for paper review.
- `artifacts/v3/analysis/`: compact summaries behind the camera-ready (ICCIT 2026) tables, the DC-offset check, and the CPU profile.

The manuscript itself (`paper/`: source, tables, figures, diagrams, compiled PDF) is a local,
gitignored artifact and is not part of this repository's tracked contents.

## Camera-Ready Version (ICCIT 2026)

The camera-ready revision of the paper "VibeTwin: A Calibrated Healthy-Reference Reconstruction
Framework for Bearing Fault Detection" reran every experiment. The main changes are:

- CWRU is rebuilt so that healthy and fault files are all sampled at 48 kHz, and each held-out load's
  whole healthy recording serves as healthy test data.
- Paderborn results use five seeds (42, 7, 123, 2024, 2026), 95% confidence intervals, and Welch
  tests with Holm correction.
- Two baselines are new: a classical RMS/kurtosis/crest-factor/envelope indicator and an RBF-kernel
  OC-SVM.
- Two Paderborn transfer protocols test unseen healthy bearings (3 folds) and unseen operating
  conditions (leave-one-condition-out, 4 folds).
- The ablation trains every topology with and without the frequency-aware loss.
- Persistence, condition-specific, and commissioning alarm rules, a check of the DC offset in the
  residual score, and a single-thread CPU latency profile are added.

The camera-ready runs write under `artifacts/v3/`; only the compact summaries in
`artifacts/v3/analysis/` are tracked. The commands are in the camera-ready section of
[RUN_COMMANDS.md](RUN_COMMANDS.md), and the table-to-file mapping is in
[RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md).

## Quick Start

Use Python 3.11 or 3.12 for the safest CPU reproduction path. The current local environment used during release preparation reports Python 3.13.7, but some scientific and PyTorch stacks can be more fragile on Python 3.13 unless locally tested.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For conda/mamba users, `environment.yml` provides a CPU-first environment.

## Dataset Setup

Raw datasets are not included. Obtain the Case Western Reserve University Bearing Data Center dataset and the Paderborn University/KAt-DataCenter bearing dataset from their official providers, then place local files under:

```text
data/raw/cwru/
data/raw/cwru48k/
data/raw/paderborn/
```

`scripts/download_datasets.sh` downloads all three from the official sources. See [DATASET_SETUP.md](DATASET_SETUP.md) for naming conventions, selected channels, Paderborn bearing lists, and metadata notes.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the release scope, tracked versus omitted files, storage expectations, known limitations, and end-to-end reproduction stages.

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for preprocessing, training, evaluation, deployment-metric, and paper-package commands.

See [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md) for the mapping between paper tables/figures and the scripts, metrics, reports, and generated artifacts that support them.

See [ARTIFACTS.md](ARTIFACTS.md) for the artifact retention policy and later Zenodo/OSF candidates.
