"""
ECG preprocessing pipeline.

Transforms raw AD8232/ADS1115 CSV files into standardized numpy arrays
suitable for model training.

Pipeline steps:
    1. Load CSV and parse ISO 8601 timestamps
    2. Compute realized sample rate from median inter-sample interval
    3. Resample to a regular grid at target_fs using scipy.signal.resample
    4. Apply Butterworth bandpass filter (default: 0.5–40 Hz, order 4)
    5. Fit StandardScaler on the scaler_fit_condition (default: resting)
       and apply to all conditions for this patient
    6. Save output as .npy array of shape [T, 1] to data/processed/{patient}/

Source reference: tslib/ecg_analyze.py (bandpass filter, realized fs)
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler


def load_ecg_csv(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a raw ECG CSV and return timestamps and voltage arrays.

    Args:
        file_path: Path to CSV with columns [timestamp, raw, voltage, patient, condition]

    Returns:
        timestamps: array of datetime64[ns] values
        voltage:    float32 array of voltage readings
    """
    df = pd.read_csv(file_path, header=0,
                     names=["timestamp", "raw", "voltage", "patient", "condition"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].values.astype("datetime64[ns]")
    voltage = df["voltage"].values.astype(np.float32)
    return timestamps, voltage


def compute_realized_fs(timestamps: np.ndarray) -> float:
    """
    Compute the realized sample rate from the median inter-sample interval.

    Args:
        timestamps: Array of datetime64[ns] timestamps

    Returns:
        Realized sample rate in Hz
    """
    dt_ns = np.diff(timestamps.astype(np.int64))
    dt_s = np.median(dt_ns) * 1e-9
    return 1.0 / dt_s


def resample_to_regular_grid(
    signal: np.ndarray,
    realized_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Resample a signal from its realized rate to a regular target rate.

    Uses scipy.signal.resample (FFT-based, appropriate for physiological signals).

    Args:
        signal:      1D array of signal values
        realized_fs: Realized sample rate in Hz
        target_fs:   Target sample rate in Hz (e.g., 100.0)

    Returns:
        Resampled signal, shape [N_resampled], dtype float32
    """
    n_samples = int(round(len(signal) * target_fs / realized_fs))
    return resample(signal, n_samples).astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Guardband clamps highcut to stay below Nyquist, matching tslib/ecg_analyze.py.

    Args:
        signal:  1D signal array
        fs:      Sample rate in Hz
        lowcut:  Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        order:   Filter order

    Returns:
        Filtered signal, same shape as input, dtype float32
    """
    nyq = fs / 2.0
    highcut = min(highcut, nyq - 1.0)
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def fit_scaler(signal: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on a 1D training signal.

    The scaler must be fit on the resting condition only, then applied
    to all other conditions for the same patient.

    Args:
        signal: 1D float array

    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(signal.reshape(-1, 1))
    return scaler


def run_ecg_pipeline(
    file_path: str | Path,
    output_dir: str | Path,
    target_fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    filter_order: int = 4,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Run the full ECG preprocessing pipeline on a single CSV file.

    Output file name is derived from the CSV name:
        brenton_ecg_resting_20260610_180351.csv
        → {output_dir}/ecg_resting.npy

    Args:
        file_path:     Path to raw ECG CSV
        output_dir:    Directory to write the output .npy file
        target_fs:     Target sample rate after resampling (Hz)
        bandpass_low:  Bandpass low cutoff (Hz)
        bandpass_high: Bandpass high cutoff (Hz)
        filter_order:  Butterworth filter order
        scaler:        Pre-fitted scaler to apply; if None, fits on this file

    Returns:
        processed: float32 array of shape [T, 1]
        scaler:    The scaler used (fitted on this file or passed in)
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps, voltage = load_ecg_csv(file_path)
    realized_fs = compute_realized_fs(timestamps)
    resampled = resample_to_regular_grid(voltage, realized_fs, target_fs)
    filtered = bandpass_filter(resampled, fs=target_fs,
                               lowcut=bandpass_low, highcut=bandpass_high,
                               order=filter_order)

    if scaler is None:
        scaler = fit_scaler(filtered)

    scaled = scaler.transform(filtered.reshape(-1, 1)).astype(np.float32)

    # Derive output name: "ecg_{condition}.npy" from file stem.
    # Stem format: {patient}_ecg_{condition...}_{YYYYMMDD}_{HHMMSS}
    # The date part is the first all-digit 8-char token; condition is
    # everything between "ecg" and that token.
    parts = file_path.stem.split("_")
    sensor_idx = parts.index("ecg")
    date_idx = next(i for i in range(sensor_idx + 1, len(parts)) if parts[i].isdigit() and len(parts[i]) == 8)
    condition = "_".join(parts[sensor_idx + 1 : date_idx])
    out_path = output_dir / f"ecg_{condition}.npy"
    np.save(out_path, scaled)
    print(f"Saved {out_path}  shape={scaled.shape}")

    return scaled, scaler


def process_patient_ecg(
    raw_dir: str | Path,
    patient: str,
    output_dir: str | Path,
    target_fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    filter_order: int = 4,
    scaler_fit_condition: str = "resting",
) -> None:
    """
    Preprocess all ECG conditions for a patient.

    Fits the scaler on `scaler_fit_condition` first, then applies it to
    all other conditions. Saves the scaler to {output_dir}/ecg_scaler.pkl.

    Args:
        raw_dir:              Directory containing raw ECG CSVs
        patient:              Patient ID (e.g., "brenton")
        output_dir:           Root processed dir; patient subdir is created
        target_fs:            Target sample rate (Hz)
        bandpass_low:         Bandpass low cutoff (Hz)
        bandpass_high:        Bandpass high cutoff (Hz)
        filter_order:         Butterworth filter order
        scaler_fit_condition: Condition to fit the scaler on (must be "resting")
    """
    raw_dir = Path(raw_dir)
    patient_out = Path(output_dir) / patient
    patient_out.mkdir(parents=True, exist_ok=True)

    csvs = sorted(raw_dir.glob(f"{patient}_ecg_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ECG CSVs found for patient {patient!r} in {raw_dir}")

    # Find resting file to fit scaler
    resting_files = [f for f in csvs if scaler_fit_condition in f.name]
    if not resting_files:
        raise FileNotFoundError(
            f"No {scaler_fit_condition!r} condition file found for patient {patient!r}"
        )

    # Fit scaler on resting condition
    _, scaler = run_ecg_pipeline(
        resting_files[0], patient_out,
        target_fs=target_fs,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        filter_order=filter_order,
        scaler=None,
    )

    # Save scaler
    scaler_path = patient_out / "ecg_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler → {scaler_path}")

    # Apply fitted scaler to all remaining conditions
    for csv_path in csvs:
        if scaler_fit_condition in csv_path.name:
            continue
        run_ecg_pipeline(
            csv_path, patient_out,
            target_fs=target_fs,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            filter_order=filter_order,
            scaler=scaler,
        )
