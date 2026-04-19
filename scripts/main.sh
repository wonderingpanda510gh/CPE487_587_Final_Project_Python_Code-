#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message


KEYWORD="finalproject"

DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-results}"

mkdir -p "${OUTDIR}" # output directory
mkdir -p logs

LOG_FILE="logs/train.log"

echo "Running imagenet_impl.py" | tee -a "$LOG_FILE"
python -u scripts/imagenet_impl.py \
    2>&1 | tee -a "$LOG_FILE"
echo "Finished everything" | tee -a "$LOG_FILE"