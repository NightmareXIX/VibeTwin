# CWRU Local Dataset Layout

This directory holds metadata and notes for the locally organized CWRU bearing dataset used by VibeTwin.

## Folder Overview

- `data/raw/cwru/normal/`: healthy bearing `.mat` files only
- `data/raw/cwru/ir_007/`: inner-race fault files with fault size `0.007`
- `data/raw/cwru/ball_007/`: ball fault files with fault size `0.007`
- `data/raw/cwru/or_007_6/`: outer-race fault files at the 6:00 position with fault size `0.007`
- `data/processed/cwru/train/`: placeholder for later healthy-only training windows
- `data/processed/cwru/val/`: placeholder for later validation windows
- `data/processed/cwru/test/`: placeholder for later mixed healthy/fault test windows
- `data/metadata/cwru/`: file maps, manifest, inspection notes, and dataset documentation

## Naming Convention

Files are renamed to a consistent, machine-friendly format:

- Healthy: `normal_<load>.mat`
- Inner race fault: `ir007_<load>.mat`
- Ball fault: `ball007_<load>.mat`
- Outer race fault at 6:00: `or007_6_<load>.mat`

The `<load>` suffix indicates the motor load condition in horsepower:

- `0` = 0 hp
- `1` = 1 hp
- `2` = 2 hp
- `3` = 3 hp

## Healthy vs Faulty Files

- Healthy files: `normal_0.mat`, `normal_1.mat`, `normal_2.mat`, `normal_3.mat`
- Faulty files:
  - `ir007_0.mat` to `ir007_3.mat`
  - `ball007_0.mat` to `ball007_3.mat`
  - `or007_6_0.mat` to `or007_6_3.mat`

## Metadata Files

- `file_map.csv`: canonical metadata table for expected files
- `file_map.json`: JSON version of the same mapping
- `manifest.csv`: generated file manifest with existence checks
- `mat_inspection_report.md`: representative `.mat` content inspection

## Project Usage Note

VibeTwin is currently set up for healthy-only training. The `processed/cwru/train`, `val`, and `test` folders are placeholders for later preprocessing steps such as fixed-length windowing, normalization, and optional filtering.
