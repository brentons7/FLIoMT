import time
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from collections import deque
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import board
import busio

WINDOW = 6         # Keep 6 seconds of data in memory for stable rolling BPM
FS = 100           # Match the 100 Hz recording frequency
BUFSIZE = WINDOW * FS

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
ads.gain = 1
ads.data_rate = 860
ecg = AnalogIn(ads, 0)

buf = deque(maxlen=BUFSIZE)

def bandpass(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

# Clear screen and print header
os.system('cls' if os.name == 'nt' else 'clear')
print("=" * 45)
print("        LIVE ECG REAL-TIME BPM MONITOR        ")
print("=" * 45)
print("Initializing buffer... (Takes ~6 seconds)")
print("-" * 45)

target_interval = 1.0 / FS
start = time.time()
next_sample_time = start
last_print_time = start

try:
    while True:
        current_time = time.time()

        # Precise time-slot pacing
        if current_time >= next_sample_time:
            v = ecg.voltage
            buf.append(v)

            # Only calculate and print updates once every 0.5 seconds to make it readable
            if current_time - last_print_time >= 0.5:
                elapsed = current_time - start
                bpm_str = "Calculating..."

                if len(buf) >= BUFSIZE:
                    signal = np.array(buf)
                    try:
                        filtered = bandpass(signal, fs=FS)

                        # Robust thresholding tracking local standard deviation
                        threshold = np.median(filtered) + 2.5 * np.std(filtered)
                        min_distance = int(FS * 0.4) # Max ~150 BPM limits

                        peaks, _ = find_peaks(filtered, height=threshold, distance=min_distance)

                        if len(peaks) >= 2:
                            rr = np.diff(peaks) / FS
                            bpm = 60.0 / np.mean(rr)
                            if 40 < bpm < 180:
                                bpm_str = f"{bpm:.1f} BPM"
                            else:
                                bpm_str = "Signal Noise"
                        else:
                            bpm_str = "No Peaks"
                    except Exception:
                        bpm_str = "Filtering..."

                # Dynamic terminal line update
                print(f"Time: {elapsed:>5.1f}s  |  Voltage: {v:>6.4f}V  |  Heart Rate: {bpm_str:<15}", end="\r", flush=True)
                last_print_time = current_time

            next_sample_time += target_interval
        else:
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\n\nMonitor stopped.")
