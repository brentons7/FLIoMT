#!/usr/bin/env bash
# FL client preset for Jetson Orin Nano #2 — partition_id=0, CUDA GPU (Ampere SM87)
#
# JetPack 6, CUDA 12.6, Python 3.10. USE_GPU=true enables CUDA inference.
# Batch size 8 (Orin Nano has 8 GB shared LPDDR5 RAM).
#
# Usage:
#   SERVER_IP=192.168.1.10 bash scripts/fl/start_client_orin.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: set SERVER_IP. Example:"
    echo "  SERVER_IP=192.168.1.10 bash scripts/fl/start_client_orin.sh"
    exit 1
fi

SERVER_IP="$SERVER_IP" \
CONFIG=configs/experiments/fl_wesad_2client.yaml \
PARTITION_ID=0 \
NUM_PARTITIONS=1 \
PATIENT=wesad_S2 \
USE_GPU=true \
BATCH_SIZE=16 \
NUM_WORKERS=0 \
    bash scripts/fl/start_client.sh
