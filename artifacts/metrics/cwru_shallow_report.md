# Shallow Baseline Report

## Setup
- Processed root: `data/processed/cwru`
- Window size: `2048`
- Window stride: `1024`
- Processed normalization: `zscore_global` fit on `healthy_train_only`
- Feature count: `17`
- Features: `mean, std, rms, max, min, peak_to_peak, skewness, kurtosis, crest_factor, signal_energy, spectral_centroid, spectral_entropy, psd_band_00_ratio, psd_band_01_ratio, psd_band_02_ratio, psd_band_03_ratio, psd_band_04_ratio`

## PSD Band Features
- `psd_band_00_ratio` covers normalized frequency `0.0005` to `0.1001`
- `psd_band_01_ratio` covers normalized frequency `0.1006` to `0.2002`
- `psd_band_02_ratio` covers normalized frequency `0.2007` to `0.3003`
- `psd_band_03_ratio` covers normalized frequency `0.3008` to `0.4004`
- `psd_band_04_ratio` covers normalized frequency `0.4009` to `0.5000`

## Fault Window Counts
- `ball`: `469`
- `inner_race`: `472`
- `outer_race_6`: `471`

## OC-SVM
- Threshold rule: `mean_plus_3std`
- Threshold: `1.277889`
- Val score mean: `-1.303114`
- Val score std: `0.860334`
- AUROC: `1.000000`
- AUPRC: `1.000000`
- F1: `0.997880`
- Precision: `0.995769`
- Recall on fault windows: `1.000000`
- False alarm rate on healthy test windows: `0.024896`

## Isolation Forest
- Threshold rule: `mean_plus_3std`
- Threshold: `0.096668`
- Val score mean: `-0.051188`
- Val score std: `0.049285`
- AUROC: `1.000000`
- AUPRC: `1.000000`
- F1: `0.998939`
- Precision: `0.997880`
- Recall on fault windows: `1.000000`
- False alarm rate on healthy test windows: `0.012448`

## AE Comparison
| Model | AUROC | AUPRC | F1 | Precision | Recall Fault | False Alarm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AE | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| OC-SVM | 1.000000 | 1.000000 | 0.997880 | 0.995769 | 1.000000 | 0.024896 |
| Isolation Forest | 1.000000 | 1.000000 | 0.998939 | 0.997880 | 1.000000 | 0.012448 |

## Saved Artifacts
- OC-SVM metrics: `artifacts/metrics/cwru_ocsvm_metrics.json`
- Isolation Forest metrics: `artifacts/metrics/cwru_iforest_metrics.json`
- OC-SVM scores: `artifacts/metrics/cwru_ocsvm_scores.csv`
- Isolation Forest scores: `artifacts/metrics/cwru_iforest_scores.csv`
- Shared report: `artifacts/metrics/cwru_shallow_report.md`
- Shared plot: `artifacts/plots/cwru_shallow_score_hists.png`
