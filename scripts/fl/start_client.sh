#!/usr/bin/env bash
# Generic FL client launcher — all device-specific parameters via environment.
#
# Usage:
#   SERVER_IP=192.168.1.10 PARTITION_ID=0 NUM_PARTITIONS=3 \
#     bash scripts/fl/start_client.sh
#
# Required:
#   SERVER_IP        IP address of the FL server
#   PARTITION_ID     This client's partition index (0-indexed)
#   NUM_PARTITIONS   Total number of clients
#
# Optional:
#   CONFIG           Path to experiment YAML (default: fl_ecg_3client.yaml)
#   PORT             Server port (default: 8080)
#   USE_GPU          "true" or "false" (default: false)
#   GPU_ID           GPU device index (default: 0)
#   BATCH_SIZE       Override batch size from config
#   NUM_WORKERS      DataLoader worker threads (default: 2)

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: SERVER_IP is required."
    echo "  Example: SERVER_IP=192.168.1.10 PARTITION_ID=0 bash scripts/fl/start_client.sh"
    exit 1
fi

CONFIG=${CONFIG:-configs/experiments/fl_ecg_2client.yaml}
PORT=${PORT:-8080}
PARTITION_ID=${PARTITION_ID:-0}
NUM_PARTITIONS=${NUM_PARTITIONS:-3}
USE_GPU=${USE_GPU:-false}

echo "Starting FL client"
echo "  config:         $CONFIG"
echo "  server:         ${SERVER_IP}:${PORT}"
echo "  partition:      ${PARTITION_ID}/${NUM_PARTITIONS}"
echo "  use_gpu:        $USE_GPU"
echo ""

python fl/run_client.py \
    --config          "$CONFIG" \
    --server_address  "${SERVER_IP}:${PORT}" \
    --partition_id    "$PARTITION_ID" \
    --num_partitions  "$NUM_PARTITIONS" \
    ${USE_GPU:+--use_gpu} \
    ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
    ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"}
