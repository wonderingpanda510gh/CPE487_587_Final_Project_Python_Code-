#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message


KEYWORD="finalproject"

DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-results}"

mkdir -p "${OUTDIR}" # output directory
mkdir -p logs

LOG_FILE="logs/train.log"

echo "Running main.py" | tee -a "$LOG_FILE"
python -u scripts/main.py \
    2>&1 | tee -a "$LOG_FILE"
echo "Finished everything" | tee -a "$LOG_FILE"