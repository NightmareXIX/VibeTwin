# Artifacts

## Tracked Compact Artifacts

The release tracks compact provenance artifacts that are practical for code review:

- `artifacts/metrics/*.json`
- `artifacts/metrics/*.md`
- `artifacts/paper_package_v1/paper_package_manifest.csv`
- `artifacts/paper_package_v1/paper_package_overall_report.md`
- `artifacts/paper_package_v1/02_final_metrics/**/*.json`
- `artifacts/paper_package_v1/03_final_reports/**/*.md`
- `artifacts/paper_package_v1/05_candidate_tables/*`

## Omitted Bulky Artifacts

The release intentionally omits raw datasets, processed arrays, checkpoints, model binaries, local virtual environments, wheels, score CSV exports, ZIP packages, and bulky generated figures or run folders.

These files are omitted because they are generated, large, machine-specific, or governed by external dataset-provider terms. Raw CWRU and Paderborn data must be obtained from the official providers. Checkpoints are not redistributed because they are reproducible run outputs and can be large.

## Regeneration

Use [RUN_COMMANDS.md](RUN_COMMANDS.md) to regenerate omitted outputs. The main stages are dataset preparation, preprocessing, model training, evaluation, deployment benchmarking, and paper-package generation.

## Future Archive Candidates

Safe candidates for a future Zenodo/OSF archive include compact metrics JSON, Markdown reports, paper tables, final paper figures, CWRU metadata manifests, the Paderborn manifest schema/sample, and release documentation. Full raw datasets should remain outside the archive unless provider terms explicitly permit redistribution. Trained checkpoints and processed arrays can be archived later only if storage, licensing, and review requirements call for them.
