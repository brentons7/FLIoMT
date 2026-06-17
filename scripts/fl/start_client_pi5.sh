#!/usr/bin/env bash
# FL client preset for Raspberry Pi 5 — partition_id=2, CPU
#
# Usage:
#   SERVER_IP=192.168.1.10 bash scripts/fl/start_client_pi5.sh
#
# Source: tslib/scripts/fl/start_client_pi5.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: set SERVER_IP. Example:"
    echo "  SERVER_IP=192.168.1.10 bash scripts/fl/start_client_pi5.sh"
    exit 1
fi

SERVER_IP="$SERVER_IP" \
PARTITION_ID=2 \
NUM_PARTITIONS=3 \
USE_GPU=false \
BATCH_SIZE=16 \
NUM_WORKERS=2 \
    bash scripts/fl/start_client.sh
