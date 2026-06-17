#!/usr/bin/env bash
# Batch preprocessing — runs ECG and PPG pipelines for all sessions in
# data/manifests/sessions.csv.
#
# Usage:
#   bash scripts/preprocess.sh
#   bash scripts/preprocess.sh --patient brenton --sensor ecg
#   bash scripts/preprocess.sh --overwrite

set -euo pipefail
cd "$(dirname "$0")/.."

echo "FLIoMT Preprocessing Pipeline"
echo "=============================="

python preprocessing/run_all.py "$@"
