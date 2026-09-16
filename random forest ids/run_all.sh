#!/usr/bin/env bash
# End-to-end flow: CSV in, verified Verilog out.
#
#   ./run_all.sh path/to/DoS_dataset.csv [tag]
#
# Drop the real HCRL DoS_dataset.csv in and nothing here changes.
set -euo pipefail

CSV="${1:?usage: run_all.sh <dataset.csv> [tag]}"
TAG="${2:-run}"
OUT="results/$TAG"
RTL="rtl/$TAG"

mkdir -p "$OUT" "$RTL"

echo "== 1/6  features"
python3 src/prepare.py --csv "$CSV" \
    --out "$OUT/cache.npz" \
    --baseline-out "$OUT/baseline.json" \
    --raw-out "$OUT/raw_test.npz"

echo "== 2/6  sweep"
python3 src/sweep.py --cache "$OUT/cache.npz" --out "$OUT/sweep.csv" \
    --trees 1,3,5,7,9 --depths 2,3,4,5,6 \
    --sets full,timing,timing_only,paper | tail -20

echo "== 3/6  select"
python3 src/select_model.py --cache "$OUT/cache.npz" --sweep "$OUT/sweep.csv" \
    --out "$OUT/model.json" --feature-set timing --target "${TARGET:-0.999}"

echo "== 4/6  verilog"
python3 src/export_rom.py --baseline "$OUT/baseline.json" --rtl-dir "$RTL"
python3 src/export_verilog.py --model "$OUT/model.json" --cache "$OUT/cache.npz" \
    --rtl-dir "$RTL" --raw-cache "$OUT/raw_test.npz" \
    --baseline "$OUT/baseline.json"
cp rtl/can_ids_features.v rtl/can_ids_top.v "$RTL/"

echo "== 5/6  simulate"
( cd "$RTL" \
  && iverilog -g2005 -o tb_forest.vvp tb_rf_forest.v rf_forest.v \
  && vvp tb_forest.vvp | tail -3 \
  && iverilog -g2005 -o tb_top.vvp tb_can_ids_top.v can_ids_top.v \
        can_ids_features.v rf_forest.v \
  && vvp tb_top.vvp | tail -3 )

echo "== 6/6  budget"
python3 src/hw_report.py --model "$OUT/model.json"
