#!/usr/bin/env bash
set -euo pipefail

# demo script for csv-cleaner
# runs a few endpoints and saves outputs you can copy into README

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
INPUT_CSV="${1:-data/raw/messy_IMDB_dataset.csv}"

OUT_DIR="${OUT_DIR:-reports/examples}"
mkdir -p "$OUT_DIR"

echo "Base URL: $BASE_URL"
echo "Input CSV: $INPUT_CSV"
echo "Output Dir: $OUT_DIR"
echo

echo "1) Health Check"
curl -sS "$BASE_URL/health" | python -m json.tool
echo

echo "2) Profile"
curl -sS -X POST "$BASE_URL/profile" \
  -F "file=@${INPUT_CSV}" \
  | tee "$OUT_DIR/profile_example.json" \
  | python -m json.tool
echo
echo "saved: $OUT_DIR/profile_example.json"
echo

echo "3) Plan (LLM)"
curl -sS -X POST "$BASE_URL/plan" \
  -F "file=@${INPUT_CSV}" \
  | tee "$OUT_DIR/plan_example.json" \
  | python -m json.tool
echo
echo "Saved: $OUT_DIR/plan_example.json"
echo

echo "4) Clean (LLM)"
curl -sS -X POST "$BASE_URL/clean/llm" \
  -F "file=@${INPUT_CSV}" \
  | tee "$OUT_DIR/report_example.json" \
  | python -m json.tool
echo
echo "Saved: $OUT_DIR/report_example.json"
echo

echo "Done."
echo "Next: Open $OUT_DIR/plan_example.json and $OUT_DIR/report_example.json for README snippets"
