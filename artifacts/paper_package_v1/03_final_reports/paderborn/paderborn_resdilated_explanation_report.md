# Paderborn ResDilatedAE Explanation Report

## Setup
- Model: `ResDilatedAE`
- Seed used: `123`
- Checkpoint: `artifacts/generative_upgrades/resdilated_ae/seed_123/checkpoints/best.pt`
- Threshold rule: `percentile_99_5`
- Threshold value: `0.000492`
- Hardest operating condition under this setup: `N09_M07_F10`
- Explanations are inference-only and use the saved deterministic score outputs plus the saved `best.pt` checkpoint.

## Selected Cases
| Case | Split | Score | Prediction | Condition | Damage Group | Why Selected |
| --- | --- | ---: | --- | --- | --- | --- |
| tp_ka | test_fault | 0.009038 | abnormal | N15_M07_F04 | KA | Representative true-positive example from damage group KA. |
| tp_kb | test_fault | 0.026331 | abnormal | N15_M07_F04 | KB | Representative true-positive example from damage group KB. |
| tp_ki | test_fault | 0.006889 | abnormal | N15_M07_F04 | KI | Representative true-positive example from damage group KI. |
| hardest_condition_tp | test_fault | 0.012716 | abnormal | N09_M07_F10 | KA | True-positive example from hardest operating condition N09_M07_F10. |
| healthy_true_negative | test_healthy | 0.000088 | healthy | N09_M07_F10 | - | Representative healthy window with a comfortably sub-threshold score. |
| healthy_false_positive | test_healthy | 0.000492 | abnormal | N09_M07_F10 | - | Borderline healthy false-positive with the smallest score margin above threshold. |

## True Positive KA
![True Positive KA](figures/01_tp_ka.png)
- Score vs threshold: `0.009038` vs `0.000492`
- Prediction: `abnormal`
- Metadata: subset=`test_fault`, health_status=`damaged`, damage_group=`KA`, condition=`N15_M07_F04`, bearing=`KA22`, measurement=`N15_M07_F04_KA22_3`, window_index=`212874`, signal=`vibration_1`
- Source window: `data/raw/paderborn/KA22/N15_M07_F04_KA22_3.mat`
- Interpretation: Actual label: fault (KA); predicted label: abnormal. The reconstruction captures the coarse envelope but leaves a moderate oscillatory mismatch in the finer waveform detail. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the low-frequency band around 0.050, 0.090, 0.500 cycles/sample. That persistent mismatch against the healthy reconstruction is what likely pushed the anomaly score above threshold.

## True Positive KB
![True Positive KB](figures/02_tp_kb.png)
- Score vs threshold: `0.026331` vs `0.000492`
- Prediction: `abnormal`
- Metadata: subset=`test_fault`, health_status=`damaged`, damage_group=`KB`, condition=`N15_M07_F04`, bearing=`KB23`, measurement=`N15_M07_F04_KB23_10`, window_index=`249805`, signal=`vibration_1`
- Source window: `data/raw/paderborn/KB23/N15_M07_F04_KB23_10.mat`
- Interpretation: Actual label: fault (KB); predicted label: abnormal. The reconstruction captures the coarse envelope but leaves a moderate oscillatory mismatch in the finer waveform detail. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the mid-frequency band around 0.008, 0.051, 0.155 cycles/sample. That persistent mismatch against the healthy reconstruction is what likely pushed the anomaly score above threshold.

## True Positive KI
![True Positive KI](figures/03_tp_ki.png)
- Score vs threshold: `0.006889` vs `0.000492`
- Prediction: `abnormal`
- Metadata: subset=`test_fault`, health_status=`damaged`, damage_group=`KI`, condition=`N15_M07_F04`, bearing=`KI16`, measurement=`N15_M07_F04_KI16_7`, window_index=`453493`, signal=`vibration_1`
- Source window: `data/raw/paderborn/KI16/N15_M07_F04_KI16_7.mat`
- Interpretation: Actual label: fault (KI); predicted label: abnormal. The reconstruction closely follows the waveform and leaves only small residual ripples. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the low-frequency band around 0.154, 0.155, 0.156 cycles/sample. That persistent mismatch against the healthy reconstruction is what likely pushed the anomaly score above threshold.

## Hardest Condition TP
![Hardest Condition TP](figures/04_hardest_condition_tp.png)
- Score vs threshold: `0.012716` vs `0.000492`
- Prediction: `abnormal`
- Metadata: subset=`test_fault`, health_status=`damaged`, damage_group=`KA`, condition=`N09_M07_F10`, bearing=`KA15`, measurement=`N09_M07_F10_KA15_7`, window_index=`163926`, signal=`vibration_1`
- Source window: `data/raw/paderborn/KA15/N09_M07_F10_KA15_7.mat`
- Interpretation: Actual label: fault (KA); predicted label: abnormal. The reconstruction captures the coarse envelope but leaves a moderate oscillatory mismatch in the finer waveform detail. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the mid-frequency band around 0.002, 0.003, 0.500 cycles/sample. That persistent mismatch against the healthy reconstruction is what likely pushed the anomaly score above threshold.

## Healthy True Negative
![Healthy True Negative](figures/05_healthy_true_negative.png)
- Score vs threshold: `0.000088` vs `0.000492`
- Prediction: `healthy`
- Metadata: subset=`test_healthy`, health_status=`healthy`, damage_group=`-`, condition=`N09_M07_F10`, bearing=`K002`, measurement=`N09_M07_F10_K002_6`, window_index=`3394`, signal=`vibration_1`
- Source window: `data/raw/paderborn/K002/N09_M07_F10_K002_6.mat`
- Interpretation: Actual label: healthy; predicted label: healthy. The reconstruction closely follows the waveform and leaves only small residual ripples. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the high-frequency band around 0.001, 0.026, 0.500 cycles/sample. Because both the waveform and spectrum stay close to the learned healthy template, the score remains below the calibrated threshold.

## Healthy False Positive
![Healthy False Positive](figures/06_healthy_false_positive.png)
- Score vs threshold: `0.000492` vs `0.000492`
- Prediction: `abnormal`
- Metadata: subset=`test_healthy`, health_status=`healthy`, damage_group=`-`, condition=`N09_M07_F10`, bearing=`K002`, measurement=`N09_M07_F10_K002_7`, window_index=`3431`, signal=`vibration_1`
- Source window: `data/raw/paderborn/K002/N09_M07_F10_K002_7.mat`
- Interpretation: Actual label: healthy; predicted label: abnormal. The reconstruction closely follows the waveform and leaves only small residual ripples. FFT magnitudes for the original and reconstruction mostly overlap; the remaining residual spectrum is small and is concentrated most in the high-frequency band around 0.418, 0.420, 0.500 cycles/sample. Even though the window is labeled healthy, the remaining mismatch is large enough to push the anomaly score above threshold, so this behaves like a false alarm.

## Saved Artifacts
- Report: `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_report.md`
- Cases JSON: `artifacts/generative_upgrades/resdilated_ae/explanations/seed_123_percentile_99_5/resdilated_ae_explanation_cases.json`
