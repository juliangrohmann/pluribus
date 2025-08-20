#!/usr/bin/env bash
set -euo pipefail

# Exit if not enough arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <num_clusters> <output_dir>"
  exit 1
fi

K=$1
OUTDIR=$2

mkdir -p "$OUTDIR"
source ../venv/bin/activate

for round in {1..3}; do
  ./Pluribus ochs-features --blueprint "$round" "$OUTDIR"
  python3 ../python/training/k_means.py "$round" --clusters "$K" --src "$OUTDIR" --out "$OUTDIR"
done
