import max30102
import time
import sys

m = max30102.MAX30102()

# Clear the screen and hide the blinking cursor for a cleaner look
print("\033[2J\033[H", end="")
print("====================================")
print("     MAX30102 REAL-TIME MONITOR     ")
print("====================================")
print(" Place finger on optical window.     ")
print(" Press Ctrl+C to exit.               ")
print("------------------------------------")
# Save the cursor position right here so we can loop back to it
print("\033[s", end="")

try:
    while True:
        red, ir = m.read_sequential()

        if red and ir:
            # Restore cursor to the saved position
            print("\033[u", end="")

            # Print the values on fixed lines, clearing to the end of each line (\033[K)
            print(f" RED Value : {red[0]:>7}\033[K")
            print(f" IR Value  : {ir[0]:>7}\033[K")
            print("------------------------------------")

            # Simple touch/presence detection visual indicator
            if red[0] > 50000:
                print(" STATUS    : [ FINGER DETECTED ]   \033[K")
            else:
                print(" STATUS    : [ NO FINGER ]         \033[K")

        # Keep the fast polling rate so the sensor doesn't lag
        time.sleep(0.01)

except KeyboardInterrupt:
    # Clean up the terminal appearance on exit
    print("\n\nStopped.")
    m.shutdown()
