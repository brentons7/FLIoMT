import time
import csv
import os
import argparse
from datetime import datetime
import max30102

# 1. Parse command-line arguments to mirror your ECG script
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=60)
parser.add_argument("--condition", type=str, default="unlabeled")
parser.add_argument("--patient", type=str, default="patient_01")
parser.add_argument("--fs", type=int, default=100, help="Target sample rate (Hz)")
args = parser.parse_args()

# 2. Setup the output directories and file name
os.makedirs("recordings", exist_ok=True)
filename = f"recordings/{args.patient}_ppg_{args.condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 3. Initialize your working MAX30102 driver
m = max30102.MAX30102()

print(f"Recording PPG to {filename}")
print(f"Targeting {args.fs} Hz for {args.duration}s. Press Ctrl+C to stop.\n")
time.sleep(1)

# 4. Open CSV and write headers matching your architecture
f = open(filename, "w", newline="")
writer = csv.writer(f)
writer.writerow(["timestamp", "red", "ir", "patient", "condition"])

# 5. Setup timing control loops
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

            # Pull the sequential sample from your verified driver
            red_val, ir_val = m.read_sequential()

            if red_val and ir_val:
                # Extract the first element from the returned data lists
                writer.writerow([ts, red_val[0], ir_val[0], args.patient, args.condition])
                count += 1

                # Console updates every 100 samples
                if count % 100 == 0:
                    print(f" {int(time.time() - start)}s, {count} samples collected")

            next_sample_time += target_interval
        else:
            # Prevent burning 100% CPU while waiting for the next slice
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    f.close()
    m.shutdown()

# 6. Print benchmarks on finish
duration = time.time() - start
rate = count / duration if duration > 0 else 0
print(f"\nDone. Saved {count} samples to {filename}")
print(f"Actual Average Sample Rate: {rate:.1f} Hz")
