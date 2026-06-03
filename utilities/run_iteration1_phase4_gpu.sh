#!/usr/bin/env bash
# Iteration 1 phase 4: GPU-aware job generation + sequential training.
#
# Usage:
#   export DD_DATA_ROOT=/home/sebmo/HASTEN_proj/DD_data
#   export FILE_PATH=$DD_DATA_ROOT/projects
#   export PROTEIN=1B12
#   conda activate dd-env
#   ./utilities/run_iteration1_phase4_gpu.sh
#
set -euo pipefail

export DD_GPU_AUTO="${DD_GPU_AUTO:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DD_DATA_ROOT="${DD_DATA_ROOT:-/home/sebmo/HASTEN_proj/DD_data}"
FILE_PATH="${FILE_PATH:-$DD_DATA_ROOT/projects}"
PROTEIN="${PROTEIN:-1B12}"
PROJECT_DIR="${PROJECT_DIR:-$FILE_PATH/$PROTEIN}"

NUM_HYP="${NUM_HYP:-24}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-11}"
VAL_SIZE="${VAL_SIZE:-1000000}"
RECALL="${RECALL:-0.9}"
PERCENT_FIRST_MOLS="${PERCENT_FIRST_MOLS:-1}"
PERCENT_LAST_MOLS="${PERCENT_LAST_MOLS:-0.01}"

echo "=== Iteration 1 phase 4 (GPU-aware) ==="
echo "Project: $PROJECT_DIR"

python scripts_2/simple_job_models_manual.py \
  --iteration_no 1 \
  --morgan_directory "$DD_DATA_ROOT/library_morgan" \
  --file_path "$PROJECT_DIR" \
  --number_of_hyp "$NUM_HYP" \
  --total_iterations "$TOTAL_ITERATIONS" \
  --is_last False \
  --number_mol "$VAL_SIZE" \
  --percent_first_mols "$PERCENT_FIRST_MOLS" \
  --percent_last_mols "$PERCENT_LAST_MOLS" \
  --recall "$RECALL"

"$ROOT_DIR/utilities/run_gpu_jobs.sh" train 1 "$PROJECT_DIR"

echo "Phase 4 complete. Next: ITER=1 ./utilities/finish_iteration_phase5.sh"
