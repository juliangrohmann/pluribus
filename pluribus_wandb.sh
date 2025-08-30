#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <preflop_bp> <sampled_bp> <run_name>"
  exit 1
fi

PREFLOP=$1
SAMPLED=$2
NAME=$3

source ../venv/bin/activate
python3 ../python/training/log_wandb.py -d metrics "$NAME"
./Pluribus server "$PREFLOP" "$SAMPLED"