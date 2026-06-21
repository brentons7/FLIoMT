#!/usr/bin/env bash
# FL client — Raspberry Pi 5
# Edit variables below or pass as env overrides.
#
# Usage:
#   SERVER_IP=192.168.1.10 bash scripts/fl/start_client_pi5.sh
#   SERVER_IP=192.168.1.10 MODEL=TimesNet bash scripts/fl/start_client_pi5.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: set SERVER_IP. Example:"
    echo "  SERVER_IP=192.168.1.10 bash scripts/fl/start_client_pi5.sh"
    exit 1
fi

# ── Experiment (must match server) ────────────────────────────────────────────
MODEL=${MODEL:-PatchTST}

# ── This device ───────────────────────────────────────────────────────────────
PATIENT=${PATIENT:-wesad_S3}
PARTITION_ID=${PARTITION_ID:-0}
NUM_PARTITIONS=${NUM_PARTITIONS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
USE_GPU=${USE_GPU:-false}
NUM_WORKERS=${NUM_WORKERS:-0}

# ─────────────────────────────────────────────────────────────────────────────
SERVER_IP="$SERVER_IP" \
MODEL="$MODEL" \
PATIENT="$PATIENT" \
PARTITION_ID="$PARTITION_ID" \
NUM_PARTITIONS="$NUM_PARTITIONS" \
BATCH_SIZE="$BATCH_SIZE" \
USE_GPU="$USE_GPU" \
NUM_WORKERS="$NUM_WORKERS" \
    bash scripts/fl/start_client.sh
