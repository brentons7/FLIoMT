#!/usr/bin/env bash
# PatchTST — tuned FL server for MIT-BIH ECG anomaly detection.
#
# Architecture: d_model=128, d_ff=256, n_heads=8, e_layers=4, patch_len=16, stride=8
# Training:     150 rounds | local_epochs=1 | cosine LR 1e-4 → 1e-5 | seq_len=128
#
# Run on Orin #1. Both clients must connect before round 1 begins.
#
# Orin #2  (partition 0):
#   SERVER_IP=<orin1_ip> MODEL=PatchTST BATCH_SIZE=32 bash scripts/fl/start_client_orin.sh
#
# Pi 5     (partition 1):
#   SERVER_IP=<orin1_ip> MODEL=PatchTST BATCH_SIZE=16 bash scripts/fl/start_client_pi5.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=PatchTST \
ROUNDS=150 \
LOCAL_EPOCHS=1 \
LR=0.0001 \
LR_MIN=0.00001 \
LR_SCHEDULE=cosine \
    bash scripts/fl/start_server.sh
