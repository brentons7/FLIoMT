"""
Physiological signal preprocessing pipelines.

Transforms raw sensor CSV files into standardized numpy arrays ready for
model training. Each pipeline produces .npy files in data/processed/.

Modules:
    ecg_pipeline — ECG: bandpass filter, resample, standardize
    ppg_pipeline — PPG: bandpass filter, AC/DC separation, standardize
    run_all      — Batch processor driven by data/manifests/sessions.csv
"""
