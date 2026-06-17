#!/usr/bin/env bash
# Local end-to-end FL test — server + 3 clients all on localhost.
#
# Runs 3 FL rounds with 3 temporal partitions on the same machine.
# All clients use GPU if available (USE_GPU=true).
#
# Usage:
#   bash scripts/fl/test_local.sh
#
# Overrides:
#   ROUNDS=5 NUM_CLIENTS=2 USE_GPU=false bash scripts/fl/test_local.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

source venv/bin/activate

ROUNDS=${ROUNDS:-3}
NUM_CLIENTS=${NUM_CLIENTS:-3}
USE_GPU=${USE_GPU:-true}
PORT=${PORT:-8080}
CONFIG=${CONFIG:-configs/experiments/fl_ecg_3client.yaml}
LOG_DIR=${LOG_DIR:-/tmp/fliomt_local_test}

mkdir -p "$LOG_DIR"

cleanup() {
    echo ""
    echo "Stopping all FL processes..."
    kill "${SERVER_PID:-}" "${CLIENT_PIDS[@]:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ Server
echo "============================================================"
echo " FLIoMT Local End-to-End Test"
echo "  config:      $CONFIG"
echo "  rounds:      $ROUNDS"
echo "  num_clients: $NUM_CLIENTS"
echo "  use_gpu:     $USE_GPU"
echo "  port:        $PORT"
echo "  logs:        $LOG_DIR"
echo "============================================================"
echo ""
echo "[server] Starting..."

ROUNDS="$ROUNDS" MIN_CLIENTS="$NUM_CLIENTS" PORT="$PORT" \
    python fl/server.py \
        --config "$CONFIG" \
        --host   127.0.0.1 \
        --port   "$PORT" \
        --rounds "$ROUNDS" \
        --min_clients "$NUM_CLIENTS" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[server] PID=$SERVER_PID  log=$LOG_DIR/server.log"

# Give the server a moment to bind
sleep 2

# ------------------------------------------------------------------ Clients
CLIENT_PIDS=()
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    GPU_FLAG=""
    if [ "$USE_GPU" = "true" ]; then
        GPU_FLAG="--use_gpu"
    fi

    echo "[client $i] Starting (partition $i/$NUM_CLIENTS)..."
    python fl/run_client.py \
        --config         "$CONFIG" \
        --server_address "127.0.0.1:${PORT}" \
        --partition_id   "$i" \
        --num_partitions "$NUM_CLIENTS" \
        $GPU_FLAG \
    > "$LOG_DIR/client_${i}.log" 2>&1 &
    CLIENT_PIDS+=($!)
    echo "[client $i] PID=${CLIENT_PIDS[-1]}  log=$LOG_DIR/client_${i}.log"
done

echo ""
echo "All processes started. Tailing server log (Ctrl-C to stop)..."
echo "------------------------------------------------------------"

# Stream server log so progress is visible
tail -f "$LOG_DIR/server.log" &
TAIL_PID=$!

# Wait for server to finish (it exits after all rounds complete)
wait "$SERVER_PID" && SERVER_EXIT=0 || SERVER_EXIT=$?
kill "$TAIL_PID" 2>/dev/null || true

echo ""
echo "------------------------------------------------------------"
echo "[server] exited with code $SERVER_EXIT"

# Wait for clients
for i in "${!CLIENT_PIDS[@]}"; do
    wait "${CLIENT_PIDS[$i]}" && echo "[client $i] OK" || echo "[client $i] FAILED (exit $?)"
done

echo ""
echo "============================================================"
if [ "$SERVER_EXIT" -eq 0 ]; then
    echo " PASSED — $ROUNDS FL rounds completed on localhost"
else
    echo " FAILED — server exited with code $SERVER_EXIT"
    echo " Check logs in $LOG_DIR/"
fi
echo "============================================================"

# Print final round summary from server log
echo ""
echo "Server summary:"
grep -E "(val_loss|fit_loss|round|Round|WARNING|ERROR|FAILED)" "$LOG_DIR/server.log" | tail -20 || true
