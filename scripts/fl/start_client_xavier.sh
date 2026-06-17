#!/usr/bin/env bash
# FL client preset for Jetson Xavier — partition_id=0, CUDA GPU
#
# Usage:
#   SERVER_IP=192.168.1.10 bash scripts/fl/start_client_xavier.sh
#
# Source: tslib/scripts/fl/start_client_xavier.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: set SERVER_IP. Example:"
    echo "  SERVER_IP=192.168.1.10 bash scripts/fl/start_client_xavier.sh"
    exit 1
fi

SERVER_IP="$SERVER_IP" \
PARTITION_ID=0 \
NUM_PARTITIONS=3 \
USE_GPU=true \
BATCH_SIZE=32 \
NUM_WORKERS=4 \
    bash scripts/fl/start_client.sh
