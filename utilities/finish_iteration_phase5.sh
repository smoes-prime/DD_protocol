#!/usr/bin/env bash
# Finish phase 5 for one iteration (required before running iteration N+1 driver loop).
#
# Usage:
#   export DD_DATA_ROOT=/home/sebmo/HASTEN_proj/DD_data
#   export FILE_PATH=$DD_DATA_ROOT/projects
#   export PROTEIN=1B12
#   ITER=1 ./utilities/finish_iteration_phase5.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ITER="${ITER:-1}"
DD_DATA_ROOT="${DD_DATA_ROOT:-/home/sebmo/HASTEN_proj/DD_data}"
FILE_PATH="${FILE_PATH:-$DD_DATA_ROOT/projects}"
PROTEIN="${PROTEIN:-1B12}"
PROJECT_DIR="${PROJECT_DIR:-$FILE_PATH/$PROTEIN}"
N_CPU="${N_CPU:-$(nproc)}"
RECALL="${RECALL:-0.9}"
VAL_SIZE="${VAL_SIZE:-1000000}"

PRED_DIR="$PROJECT_DIR/iteration_${ITER}/morgan_1024_predictions"
ALL_MODELS="$PROJECT_DIR/iteration_${ITER}/all_models"

echo "=== Finish phase 5 for iteration $ITER ==="

if ! ls "$ALL_MODELS"/model_*.keras "$ALL_MODELS"/model_* 2>/dev/null | head -1 >/dev/null; then
  echo "ERROR: no trained models in $ALL_MODELS" >&2
  echo "Run all phase-4 jobs first:" >&2
  echo "  for f in $PROJECT_DIR/iteration_${ITER}/simple_job/simple_job_*.sh; do bash \"\$f\"; done" >&2
  exit 1
fi

python scripts_2/hyperparameter_result_evaluation.py \
  --n_iteration "$ITER" \
  --data_path "$PROJECT_DIR" \
  --morgan_directory "$DD_DATA_ROOT/library_morgan" \
  --number_mol "$VAL_SIZE" \
  --recall "$RECALL"

python scripts_2/simple_job_predictions_manual.py \
  --project_name "$PROTEIN" \
  --file_path "$FILE_PATH" \
  --n_iteration "$ITER" \
  --morgan_directory "$DD_DATA_ROOT/library_morgan"

JOBS_DIR="$PROJECT_DIR/iteration_${ITER}/simple_job_predictions"
shopt -s nullglob
jobs=("$JOBS_DIR"/simple_job_*.sh)
shopt -u nullglob
if [[ "${#jobs[@]}" -eq 0 ]]; then
  echo "ERROR: no prediction jobs in $JOBS_DIR" >&2
  exit 1
fi

mkdir -p "$PRED_DIR"
for job in "${jobs[@]}"; do
  echo "Running $job"
  bash "$job" | tee "${job%.sh}.log"
done

n_txt=$(find "$PRED_DIR" -maxdepth 1 -name '*.txt' | wc -l)
echo "Prediction shards in $PRED_DIR: $n_txt"
if [[ "$n_txt" -eq 0 ]]; then
  echo "ERROR: phase 5 finished but no *.txt files in $PRED_DIR" >&2
  exit 1
fi

echo "Iteration $ITER phase 5 complete."
