#!/usr/bin/env bash
# FL client preset for Jetson Nano — partition_id=1, CUDA GPU
#
# The Nano has a 128-core Maxwell GPU. USE_GPU=true enables CUDA inference.
# Batch size kept at 8 (Nano has 4 GB shared CPU/GPU RAM).
#
# Usage:
#   SERVER_IP=192.168.1.10 bash scripts/fl/start_client_nano.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: set SERVER_IP. Example:"
    echo "  SERVER_IP=192.168.1.10 bash scripts/fl/start_client_nano.sh"
    exit 1
fi

SERVER_IP="$SERVER_IP" \
PARTITION_ID=1 \
NUM_PARTITIONS=3 \
USE_GPU=true \
BATCH_SIZE=8 \
NUM_WORKERS=0 \
    bash scripts/fl/start_client.sh
