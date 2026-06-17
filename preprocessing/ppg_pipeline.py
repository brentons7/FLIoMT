"""
PPG preprocessing pipeline.

Transforms raw MAX30102 CSV files into standardized numpy arrays
suitable for model training.

Pipeline steps:
    1. Load CSV and parse ISO 8601 timestamps
    2. Compute realized sample rate from median inter-sample interval
    3. Resample red and IR channels to a regular grid at target_fs
    4. Apply Butterworth bandpass filter (default: 0.5–5 Hz, order 4)
    5. AC/DC separation: AC component via bandpass; DC via low-pass
    6. Fit StandardScaler on the scaler_fit_condition (default: resting)
       and apply to all conditions for this patient
    7. Save output as .npy array of shape [T, 2] (red_ac, ir_ac)
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler

from preprocessing.ecg_pipeline import compute_realized_fs, resample_to_regular_grid


def load_ppg_csv(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a raw PPG CSV and return timestamps, red, and IR arrays.

    Args:
        file_path: Path to CSV with columns [timestamp, red, ir, patient, condition]

    Returns:
        timestamps: datetime64[ns] array
        red:        float32 array of red LED intensity (660 nm)
        ir:         float32 array of IR LED intensity (880 nm)
    """
    df = pd.read_csv(file_path, header=0,
                     names=["timestamp", "red", "ir", "patient", "condition"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].values.astype("datetime64[ns]")
    red = df["red"].values.astype(np.float32)
    ir = df["ir"].values.astype(np.float32)
    return timestamps, red, ir


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 5.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter for PPG signals.

    Args:
        signal:  1D signal array
        fs:      Sample rate in Hz
        lowcut:  Low cutoff (default 0.5 Hz for PPG)
        highcut: High cutoff (default 5.0 Hz for PPG cardiac band)
        order:   Filter order

    Returns:
        Filtered signal, same shape as input, dtype float32
    """
    nyq = fs / 2.0
    highcut = min(highcut, nyq - 1.0)
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def _lowpass_filter(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(cutoff, nyq - 0.1)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


def separate_ac_dc(
    signal: np.ndarray,
    fs: float,
    dc_cutoff: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate AC (pulsatile) and DC (baseline) components of a PPG channel.

    AC component: the result of a bandpass filter (0.5–5 Hz)
    DC component: the result of a low-pass filter at dc_cutoff

    Args:
        signal:    1D PPG channel array (raw ADC counts)
        fs:        Sample rate in Hz
        dc_cutoff: Low-pass cutoff for DC component (Hz)

    Returns:
        ac: AC (pulsatile) component, float32
        dc: DC (baseline) component, float32
    """
    ac = bandpass_filter(signal, fs=fs)
    dc = _lowpass_filter(signal, fs=fs, cutoff=dc_cutoff)
    return ac, dc


def estimate_spo2(
    red_ac: np.ndarray,
    red_dc: np.ndarray,
    ir_ac: np.ndarray,
    ir_dc: np.ndarray,
) -> np.ndarray:
    """
    Estimate SpO2 from the ratio-of-ratios of red and IR AC/DC components.

    SpO2 ≈ 110 - 25 * R, where R = (AC_red/DC_red) / (AC_ir/DC_ir)

    Note: Empirical approximation only. Clinical use requires calibration
    against a reference pulse oximeter.

    Args:
        red_ac: AC component of red channel
        red_dc: DC component of red channel
        ir_ac:  AC component of IR channel
        ir_dc:  DC component of IR channel

    Returns:
        spo2: Estimated SpO2 percentage array, shape [T]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(
            (red_dc != 0) & (ir_dc != 0),
            (np.abs(red_ac) / np.abs(red_dc)) / (np.abs(ir_ac) / np.abs(ir_dc)),
            np.nan,
        )
    return (110.0 - 25.0 * r).astype(np.float32)


def run_ppg_pipeline(
    file_path: str | Path,
    output_dir: str | Path,
    target_fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 5.0,
    filter_order: int = 4,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Run the full PPG preprocessing pipeline on a single CSV file.

    Output file name is derived from the CSV name:
        brenton_ppg_resting_20260610_202410.csv
        → {output_dir}/ppg_resting.npy

    Args:
        file_path:     Path to raw PPG CSV
        output_dir:    Directory to write the output .npy file
        target_fs:     Target sample rate after resampling (Hz)
        bandpass_low:  Bandpass low cutoff (Hz)
        bandpass_high: Bandpass high cutoff (Hz)
        filter_order:  Butterworth filter order
        scaler:        Pre-fitted scaler; if None, fits on this file

    Returns:
        processed: float32 array of shape [T, 2] (columns: red_ac, ir_ac)
        scaler:    The scaler used
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps, red, ir = load_ppg_csv(file_path)
    realized_fs = compute_realized_fs(timestamps)

    red_r = resample_to_regular_grid(red, realized_fs, target_fs)
    ir_r = resample_to_regular_grid(ir, realized_fs, target_fs)

    red_ac, _ = separate_ac_dc(red_r, fs=target_fs)
    ir_ac, _ = separate_ac_dc(ir_r, fs=target_fs)

    # Stack into [T, 2]
    stacked = np.stack([red_ac, ir_ac], axis=1).astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(stacked)

    scaled = scaler.transform(stacked).astype(np.float32)

    parts = file_path.stem.split("_")
    sensor_idx = parts.index("ppg")
    date_idx = next(i for i in range(sensor_idx + 1, len(parts)) if parts[i].isdigit() and len(parts[i]) == 8)
    condition = "_".join(parts[sensor_idx + 1 : date_idx])
    out_path = output_dir / f"ppg_{condition}.npy"
    np.save(out_path, scaled)
    print(f"Saved {out_path}  shape={scaled.shape}")

    return scaled, scaler


def process_patient_ppg(
    raw_dir: str | Path,
    patient: str,
    output_dir: str | Path,
    target_fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 5.0,
    filter_order: int = 4,
    scaler_fit_condition: str = "resting",
) -> None:
    """
    Preprocess all PPG conditions for a patient.

    Fits the scaler on `scaler_fit_condition` first, then applies it to
    all other conditions. Saves the scaler to {output_dir}/ppg_scaler.pkl.
    """
    raw_dir = Path(raw_dir)
    patient_out = Path(output_dir) / patient
    patient_out.mkdir(parents=True, exist_ok=True)

    csvs = sorted(raw_dir.glob(f"{patient}_ppg_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No PPG CSVs found for patient {patient!r} in {raw_dir}")

    resting_files = [f for f in csvs if scaler_fit_condition in f.name]
    if not resting_files:
        raise FileNotFoundError(
            f"No {scaler_fit_condition!r} condition file found for patient {patient!r}"
        )

    _, scaler = run_ppg_pipeline(
        resting_files[0], patient_out,
        target_fs=target_fs,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        filter_order=filter_order,
        scaler=None,
    )

    scaler_path = patient_out / "ppg_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler → {scaler_path}")

    for csv_path in csvs:
        if scaler_fit_condition in csv_path.name:
            continue
        run_ppg_pipeline(
            csv_path, patient_out,
            target_fs=target_fs,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            filter_order=filter_order,
            scaler=scaler,
        )
