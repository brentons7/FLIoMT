import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def analyze_spo2(file_path):
    # 1. Load data
    df = pd.read_csv(file_path)

    # Clean up column names in case of trailing spaces
    df.columns = [c.strip() for c in df.columns]

    # 2. Skip the first 100 samples (1 second) to clear the sensor startup ramp
    red_raw = df['red'].values[100:]
    ir_raw = df['ir'].values[100:]

    # 3. Calculate DC component (the steady-state baseline)
    red_dc = np.mean(red_raw)
    ir_dc = np.mean(ir_raw)

    # 4. Extract AC component (isolate the moving pulse waves)
    # We use a 100 Hz sampling rate based on your file intervals
    fs = 100.0
    # Filter between 0.5 Hz (30 BPM) and 4.0 Hz (240 BPM) to trap human pulse rates
    b, a = butter_bandpass(0.5, 4.0, fs, order=2)

    red_ac_wave = filtfilt(b, a, red_raw)
    ir_ac_wave = filtfilt(b, a, ir_raw)

    # Use root-mean-square (RMS) to calculate the stable average amplitude of the pulse waves
    red_ac = np.sqrt(np.mean(red_ac_wave**2))
    ir_ac = np.sqrt(np.mean(ir_ac_wave**2))

    # 5. Calculate the Ratio of Ratios (R)
    R = (red_ac / red_dc) / (ir_ac / ir_dc)

    # 6. Apply MAX30102 calibration formula
    spo2 = 110 - (25 * R)

    # Cap at 100% physically real limits
    if spo2 > 100.0:
        spo2 = 100.0

    return spo2, df['condition'].iloc[0]

# List of your files to loop through
files = [
    'recordings/brenton_ppg_resting_20260610_202410.csv',
    'recordings/brenton_ppg_light_activity_20260610_201109.csv',
    'recordings/brenton_ppg_post_exercise_20260610_203739.csv'
]

print("="*45)
print("       MAX30102 BLOOD OXYGEN ANALYSIS       ")
print("="*45)

for file in files:
    if os.path.exists(file):
        try:
            spo2_val, condition = analyze_spo2(file)
            print(f"Condition: {condition:<15} -> SpO2: {spo2_val:.2f}%")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    else:
        print(f"File missing: {file}")
print("="*45)
