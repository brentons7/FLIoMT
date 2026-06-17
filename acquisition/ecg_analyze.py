import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None, help="CSV file to analyze.")
args = parser.parse_args()

if args.file:
    filepath = args.file
else:
    files = glob.glob("recordings/*.csv")
    if not files:
        print("No recordings found in ./recordings/")
        exit(1)
    filepath = max(files, key=os.path.getmtime)
    print(f"Using most recent file: {filepath}\n")

df = pd.read_csv(filepath)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

voltage = df["voltage"].values

# Calculate actual realized sample rate based on historical median time delta
dt = df["timestamp"].diff().dt.total_seconds().median()
fs = 1.0 / dt
print(f"Realized sample rate: {fs:.1f} Hz")
print(f"Total samples: {len(df)}")
print(f"Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds():.1f}s")

def bandpass(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    nyq = fs / 2
    # Guardband to ensure cuts never breach Nyquist limits
    highcut = min(highcut, nyq - 1)
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

filtered = bandpass(voltage, fs=fs)

print("\n--- Signal Statistics ---")
print(f"  Voltage Range: {np.min(voltage):.3f}V to {np.max(voltage):.3f}V")

# Robust thresholding: Median + 2.5 times the Standard Deviation of the AC signal
threshold = np.median(filtered) + 2.5 * np.std(filtered)
min_distance = int(fs * 0.4) # Minimum 400ms between heartbeats (~150 Max BPM)

peaks, _ = find_peaks(filtered, height=threshold, distance=min_distance)

print("\n--- Analysis Results ---")
if len(peaks) >= 2:
    rr_intervals = np.diff(peaks) / fs
    bpm_instantaneous = 60.0 / rr_intervals
    mean_bpm = np.mean(bpm_instantaneous)

    print(f"  R-peaks detected: {len(peaks)}")
    print(f"  Mean Heart Rate:  {mean_bpm:.1f} BPM")
    print(f"  Min Heart Rate:   {np.min(bpm_instantaneous):.1f} BPM")
    print(f"  Max Heart Rate:   {np.max(bpm_instantaneous):.1f} BPM")

    rr_ms = rr_intervals * 1000
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    print(f"\n--- HRV Metrics ---")
    print(f"  SDNN:   {sdnn:.1f} ms")
    print(f"  RMSSD:  {rmssd:.1f} ms")
else:
    print("  Error: Not enough R-peaks detected. Check pad contacts or physical grounding.")
print()
