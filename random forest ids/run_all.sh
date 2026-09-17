#!/usr/bin/env bash
# End-to-end flow: CSV in, trained forest out.
#
#   ./run_all.sh path/to/DoS_dataset.csv [tag]
#
# Set TARGET to change the accuracy floor, MAX_NODES to cap comparators.
set -euo pipefail

CSV="${1:?usage: run_all.sh <dataset.csv> [tag]}"
TAG="${2:-run}"
OUT="results/$TAG"

mkdir -p "$OUT"

echo "== 1/4  features"
python3 src/prepare.py --csv "$CSV" \
    --out "$OUT/cache.npz" \
    --baseline-out "$OUT/baseline.json" \
    --raw-out "$OUT/raw_test.npz"

echo "== 2/4  sweep"
python3 src/sweep.py --cache "$OUT/cache.npz" --out "$OUT/sweep.csv" \
    --trees 1,3,5,7,9 --depths 2,3,4,5,6,8 \
    --sets timing,no_rate,timing_only,paper | tail -20

echo "== 3/4  select"
python3 src/select_model.py --cache "$OUT/cache.npz" --sweep "$OUT/sweep.csv" \
    --out "$OUT/model.json" --feature-set timing \
    --target "${TARGET:-0.995}" --policy robust --max-nodes "${MAX_NODES:-32}"

echo "== 4/4  the trees"
python3 src/show_trees.py --model "$OUT/model.json"
