#!/usr/bin/env bash
# iTransformer — tuned FL server for MIT-BIH ECG anomaly detection.
#
# Architecture: d_model=128, d_ff=256, n_heads=8, e_layers=3
# Training:     200 rounds | local_epochs=2 | flat LR 1e-4 (no decay) | seq_len=128
#
# Note: iTransformer's inverted-attention treats channels as tokens. With 1 ECG
# channel the self-attention collapses to a 1×1 operation and does nothing.
# Flat LR prevents premature decay while the FFN layers are still learning;
# local_epochs=2 doubles gradient steps per round to speed convergence.
#
# Run on Orin #1. Both clients must connect before round 1 begins.
#
# Orin #2  (partition 0):
#   SERVER_IP=<orin1_ip> MODEL=iTransformer BATCH_SIZE=32 bash scripts/fl/start_client_orin.sh
#
# Pi 5     (partition 1):
#   SERVER_IP=<orin1_ip> MODEL=iTransformer BATCH_SIZE=16 bash scripts/fl/start_client_pi5.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=iTransformer \
ROUNDS=200 \
LOCAL_EPOCHS=2 \
LR=0.0001 \
LR_MIN=0.0001 \
LR_SCHEDULE=none \
    bash scripts/fl/start_server.sh
