# CWRU MAT Inspection Report

This report summarizes one representative file from each organized CWRU subset in `data/raw/cwru`.

## Scope

Inspected files:

- `data/raw/cwru/normal/normal_0.mat`
- `data/raw/cwru/ir_007/ir007_0.mat`
- `data/raw/cwru/ball_007/ball007_0.mat`
- `data/raw/cwru/or_007_6/or007_6_0.mat`

## Summary

- All inspected signal arrays are stored as `float64` column vectors.
- All inspected files include both drive-end (`*_DE_time`) and fan-end (`*_FE_time`) channels.
- The healthy sample does not expose a `*_BA_time` channel, while the inspected fault samples do include a `*_BA_time` channel.
- All inspected files include an RPM variable stored as a `uint16` scalar array.
- The drive-end time series is the most obvious primary vibration signal for later preprocessing and modeling.

## Full-scan Notes

The representative inspection above matches the expected CWRU layout, but the full recursive scan across all 16 files surfaced two healthy-file quirks that later preprocessing code should handle explicitly:

- `data/raw/cwru/normal/normal_1.mat` exposes `X098_DE_time` and `X098_FE_time` but no explicit RPM variable.
- `data/raw/cwru/normal/normal_2.mat` contains `X098_DE_time`, `X098_FE_time`, `X099_DE_time`, `X099_FE_time`, and an `ans` variable. In practice, this means loaders should prefer the variable name associated with the target source ID when multiple signal candidates are present.

## Per-file Details

### `normal_0.mat`

- Source ID: `97`
- Keys: `X097_DE_time`, `X097_FE_time`, `X097RPM`
- Signal arrays:
  - `X097_DE_time`: shape `(243938, 1)`, dtype `float64`
  - `X097_FE_time`: shape `(243938, 1)`, dtype `float64`
- Drive-end channel present: Yes
- Fan-end channel present: Yes
- Additional vibration channels: None detected in this file
- RPM / metadata variables:
  - `X097RPM`: shape `(1, 1)`, dtype `uint16`, value `1796`

### `ir007_0.mat`

- Source ID: `105`
- Keys: `X105_DE_time`, `X105_FE_time`, `X105_BA_time`, `X105RPM`
- Signal arrays:
  - `X105_DE_time`: shape `(121265, 1)`, dtype `float64`
  - `X105_FE_time`: shape `(121265, 1)`, dtype `float64`
  - `X105_BA_time`: shape `(121265, 1)`, dtype `float64`
- Drive-end channel present: Yes
- Fan-end channel present: Yes
- Additional vibration channels: `X105_BA_time`
- RPM / metadata variables:
  - `X105RPM`: shape `(1, 1)`, dtype `uint16`, value `1797`

### `ball007_0.mat`

- Source ID: `118`
- Keys: `X118_DE_time`, `X118_FE_time`, `X118_BA_time`, `X118RPM`
- Signal arrays:
  - `X118_DE_time`: shape `(122571, 1)`, dtype `float64`
  - `X118_FE_time`: shape `(122571, 1)`, dtype `float64`
  - `X118_BA_time`: shape `(122571, 1)`, dtype `float64`
- Drive-end channel present: Yes
- Fan-end channel present: Yes
- Additional vibration channels: `X118_BA_time`
- RPM / metadata variables:
  - `X118RPM`: shape `(1, 1)`, dtype `uint16`, value `1796`

### `or007_6_0.mat`

- Source ID: `130`
- Keys: `X130_DE_time`, `X130_FE_time`, `X130_BA_time`, `X130RPM`
- Signal arrays:
  - `X130_DE_time`: shape `(121991, 1)`, dtype `float64`
  - `X130_FE_time`: shape `(121991, 1)`, dtype `float64`
  - `X130_BA_time`: shape `(121991, 1)`, dtype `float64`
- Drive-end channel present: Yes
- Fan-end channel present: Yes
- Additional vibration channels: `X130_BA_time`
- RPM / metadata variables:
  - `X130RPM`: shape `(1, 1)`, dtype `uint16`, value `1796`

## Notes for Later Preprocessing

- For healthy-only training, `data/raw/cwru/normal/*.mat` is the clean starting point.
- For reusable preprocessing, later code should parameterize channel selection, window length, normalization, and optional filtering rather than baking those choices into the raw-data layout.
- Loader code should prefer drive-end channels first and should tolerate files with missing RPM metadata or extra array variables.
- The processed `train`, `val`, and `test` folders are placeholders only; no window generation has been performed yet.
