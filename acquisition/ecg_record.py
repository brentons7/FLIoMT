import time
import csv
import os
import argparse
from datetime import datetime
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=60) # Defaulting shorter for fast testing
parser.add_argument("--condition", type=str, default="unlabeled")
parser.add_argument("--patient", type=str, default="patient_01")
parser.add_argument("--fs", type=int, default=100, help="Target sample rate (Hz)")
args = parser.parse_args()

os.makedirs("recordings", exist_ok=True)
filename = f"recordings/{args.patient}_ecg_{args.condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
ads.gain = 1
ads.data_rate = 860 # High data rate on ADC to lower conversion latency
ecg = AnalogIn(ads, 0)

print(f"Recording ECG to {filename}")
print(f"Targeting {args.fs} Hz for {args.duration}s. Press Ctrl+C to stop.\n")
time.sleep(1)

f = open(filename, "w", newline="")
writer = csv.writer(f)
writer.writerow(["timestamp", "raw", "voltage", "patient", "condition"])

target_interval = 1.0 / args.fs
start = time.time()
next_sample_time = start
count = 0

try:
    while time.time() - start < args.duration:
        current_time = time.time()
        # Wait precisely until our next sample slot ticks
        if current_time >= next_sample_time:
            ts = datetime.now().isoformat()

            # CRITICAL: Pull voltage ONCE to prevent double I2C delays
            v = ecg.voltage
            raw_val = int(v * 32767 / 4.096) # Calculate matching raw bit estimation

            writer.writerow([ts, raw_val, v, args.patient, args.condition])
            count += 1

            if count % 100 == 0:
                print(f" {int(time.time() - start)}s, {count} samples collected")

            next_sample_time += target_interval
        else:
            # Short sleep to prevent burning 100% CPU while waiting micro-seconds
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    f.close()

duration = time.time() - start
rate = count / duration if duration > 0 else 0
print(f"\nDone. Saved {count} samples to {filename}")
print(f"Actual Average Sample Rate: {rate:.1f} Hz")
