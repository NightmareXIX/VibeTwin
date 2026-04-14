# Data Layout

This repository tracks dataset metadata only.

- `data/raw/` is for original downloaded datasets and is ignored by Git.
- `data/processed/` is for generated arrays and labels and is ignored by Git.
- `data/metadata/` contains small, human-readable files that describe dataset organization, labels, preprocessing settings, and audits.

For the Paderborn dataset, the large `window_manifest.csv` file is intentionally excluded from version control because it is regenerated output and too large for a normal GitHub repository.
