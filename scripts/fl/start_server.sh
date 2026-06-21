#!/usr/bin/env bash
# FL aggregation server.
#
# Edit the variables below to configure an experiment, or pass any of them
# as environment overrides at runtime:
#
#   ROUNDS=200 bash scripts/fl/start_server.sh
#   MODEL=TimesNet LR=0.0005 bash scripts/fl/start_server.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Experiment ────────────────────────────────────────────────────────────────
MODEL=${MODEL:-PatchTST}
ROUNDS=${ROUNDS:-100}
MIN_CLIENTS=${MIN_CLIENTS:-2}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}

# ── Learning rate ─────────────────────────────────────────────────────────────
LR=${LR:-0.0001}
LR_MIN=${LR_MIN:-0.00001}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}     # cosine | none

# ── Network ───────────────────────────────────────────────────────────────────
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

# ─────────────────────────────────────────────────────────────────────────────
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

echo "Starting FL server"
echo "  model:       $MODEL"
echo "  rounds:      $ROUNDS   local_epochs=$LOCAL_EPOCHS   min_clients=$MIN_CLIENTS"
echo "  lr:          $LR → $LR_MIN ($LR_SCHEDULE)"
echo "  bind:        $HOST:$PORT"
echo ""
echo "  Clients should connect to SERVER_IP=${SERVER_IP:-<this-machine-ip>}"
echo ""

python fl/server.py \
    --model         "$MODEL" \
    --host          "$HOST" \
    --port          "$PORT" \
    --rounds        "$ROUNDS" \
    --min_clients   "$MIN_CLIENTS" \
    --local_epochs  "$LOCAL_EPOCHS" \
    --learning_rate "$LR" \
    --lr_min        "$LR_MIN" \
    --lr_schedule   "$LR_SCHEDULE"
