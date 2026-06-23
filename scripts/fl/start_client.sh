#!/usr/bin/env bash
# Generic FL client launcher.
# Device-specific scripts (start_client_orin.sh, start_client_pi5.sh) set
# their defaults and call this script. You can also call this directly with
# env overrides for one-off experiments.
#
# Required:
#   SERVER_IP        IP address of the FL server
#
# All other variables have defaults — override any of them at runtime:
#   SERVER_IP=192.168.1.10 MODEL=TimesNet bash scripts/fl/start_client.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SERVER_IP:-}" ]; then
    echo "ERROR: SERVER_IP is required."
    echo "  Example: SERVER_IP=192.168.1.10 bash scripts/fl/start_client.sh"
    exit 1
fi

# ── Network ───────────────────────────────────────────────────────────────────
PORT=${PORT:-8080}

# ── Experiment (must match server) ───────────────────────────────────────────
MODEL=${MODEL:-PatchTST}

# ── This client ───────────────────────────────────────────────────────────────
PATIENT=${PATIENT:-mitbih_213}
PARTITION_ID=${PARTITION_ID:-0}
NUM_PARTITIONS=${NUM_PARTITIONS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
USE_GPU=${USE_GPU:-false}
NUM_WORKERS=${NUM_WORKERS:-0}
SENSOR=${SENSOR:-ecg}
SEQ_LEN=${SEQ_LEN:-128}

# ── Architecture overrides (leave unset to use model preset defaults) ─────────
# D_MODEL=128
# D_FF=256
# N_HEADS=8
# E_LAYERS=3
# DROPOUT=0.1
# PATCH_LEN=16   # PatchTST
# STRIDE=8       # PatchTST
# TOP_K=5        # TimesNet
# NUM_KERNELS=6  # TimesNet

# ─────────────────────────────────────────────────────────────────────────────
echo "Starting FL client"
echo "  model:       $MODEL"
echo "  server:      ${SERVER_IP}:${PORT}"
echo "  patient:     $PATIENT   sensor=$SENSOR   seq_len=$SEQ_LEN"
echo "  partition:   $((PARTITION_ID + 1))/$NUM_PARTITIONS   batch=$BATCH_SIZE   gpu=$USE_GPU"
echo ""

GPU_FLAG=""
[ "$USE_GPU" = "true" ] && GPU_FLAG="--use_gpu"

python fl/run_client.py \
    --server_address  "${SERVER_IP}:${PORT}" \
    --model           "$MODEL" \
    --patient         "$PATIENT" \
    --sensor          "$SENSOR" \
    --seq_len         "$SEQ_LEN" \
    --partition_id    "$PARTITION_ID" \
    --num_partitions  "$NUM_PARTITIONS" \
    --batch_size      "$BATCH_SIZE" \
    --num_workers     "$NUM_WORKERS" \
    $GPU_FLAG \
    ${D_MODEL:+--d_model      "$D_MODEL"} \
    ${D_FF:+--d_ff            "$D_FF"} \
    ${N_HEADS:+--n_heads      "$N_HEADS"} \
    ${E_LAYERS:+--e_layers    "$E_LAYERS"} \
    ${DROPOUT:+--dropout      "$DROPOUT"} \
    ${PATCH_LEN:+--patch_len  "$PATCH_LEN"} \
    ${STRIDE:+--stride        "$STRIDE"} \
    ${TOP_K:+--top_k          "$TOP_K"} \
    ${NUM_KERNELS:+--num_kernels "$NUM_KERNELS"}
