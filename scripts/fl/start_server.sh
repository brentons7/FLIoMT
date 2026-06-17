#!/usr/bin/env bash
# Start the Flower FL aggregation server.
#
# The server has no local data. It only coordinates weight aggregation.
# It can run on any machine accessible to the clients on the network
# (Xavier, Raspberry Pi, workstation, etc.).
#
# Usage:
#   bash scripts/fl/start_server.sh
#
# Environment overrides:
#   HOST=0.0.0.0   Bind address (default: all interfaces)
#   PORT=8080
#   ROUNDS=20
#   MIN_CLIENTS=3
#   LOCAL_EPOCHS=2
#   LR=0.0001

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=${CONFIG:-configs/experiments/fl_ecg_3client.yaml}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
ROUNDS=${ROUNDS:-10}
MIN_CLIENTS=${MIN_CLIENTS:-3}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
LR=${LR:-0.0001}

# Detect the primary non-loopback IP so clients know what SERVER_IP to use.
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

echo "Starting FL server"
echo "  config:      $CONFIG"
echo "  host:        $HOST"
echo "  port:        $PORT"
echo "  rounds:      $ROUNDS"
echo "  min_clients: $MIN_CLIENTS"
echo ""
echo "  Clients should connect to SERVER_IP=${SERVER_IP:-<this-machine-ip>}"
echo ""

python fl/server.py \
    --config        "$CONFIG" \
    --host          "$HOST" \
    --port          "$PORT" \
    --rounds        "$ROUNDS" \
    --min_clients   "$MIN_CLIENTS" \
    --local_epochs  "$LOCAL_EPOCHS" \
    --learning_rate "$LR"
