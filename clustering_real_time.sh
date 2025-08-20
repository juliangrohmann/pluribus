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

./Pluribus ochs-features --real-time 3 "$OUTDIR"
python3 ../python/training/k_means.py 3 --clusters "$K" --flops --src "$OUTDIR" --out "$OUTDIR"

for start in {0..1600..100}; do
  end=$((start + 100))
  ./Pluribus emd-matrix "$start" "$end" "$OUTDIR"
  java -jar ../java/KMedoids/build/KMedoids.jar "$start" "$end" "$K" "$OUTDIR"
  rm -f "$OUTDIR"/emd_matrix*
done

./Pluribus emd-matrix 1700 1755 "$OUTDIR"
java -jar ../java/KMedoids/build/KMedoids.jar 1700 1755 "$K" "$OUTDIR"

./Pluribus build-rt-cluster-map "$K" "$OUTDIR"
