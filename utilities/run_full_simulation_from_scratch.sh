#!/usr/bin/env bash
# run_full_simulation_from_scratch.sh
# End-to-end driver for the Deep Docking simulation pipeline.
#
# Usage:
#   export DD_DATA_ROOT=/home/sebmo/HASTEN_proj/DD_data
#   export PROTEIN=1B12
#   export MASTER_SCORES_PATH=$DD_DATA_ROOT/master_scores/1B12_div_docked_DD_labels.csv
#   conda activate dd-env
#   ./utilities/run_full_simulation_from_scratch.sh

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Required env vars
DD_DATA_ROOT="${DD_DATA_ROOT:-/home/sebmo/HASTEN_proj/DD_data}"
FILE_PATH="${FILE_PATH:-$DD_DATA_ROOT/projects}"
PROTEIN="${PROTEIN:-1B12}"
PROJECT_DIR="${PROJECT_DIR:-$FILE_PATH/$PROTEIN}"
MASTER_SCORES_PATH="${MASTER_SCORES_PATH:-$DD_DATA_ROOT/master_scores/${PROTEIN}_div_docked_DD_labels.csv}"

# Pipeline config
NUM_HYP="${NUM_HYP:-24}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-11}"
TRAIN_SIZE="${TRAIN_SIZE:-3000000}"
VAL_SIZE="${VAL_SIZE:-1000000}"
RECALL="${RECALL:-0.9}"
PERCENT_FIRST_MOLS="${PERCENT_FIRST_MOLS:-1}"
PERCENT_LAST_MOLS="${PERCENT_LAST_MOLS:-0.01}"
N_CPU="${N_CPU:-$(nproc)}"
export DD_GPU_AUTO="${DD_GPU_AUTO:-1}"

echo "=== Deep Docking Full Simulation ==="
echo "Project: $PROJECT_DIR"
echo "CPUs: $N_CPU"
echo "Data Root: $DD_DATA_ROOT"
echo "Master Scores: $MASTER_SCORES_PATH"
echo "===================================="

# --- Check 0: Validate Prerequisites ---
if [[ ! -d "$DD_DATA_ROOT/library_smiles" ]] || [[ ! -d "$DD_DATA_ROOT/library_morgan" ]]; then
  echo "ERROR: library_smiles or library_morgan not found."
  echo "Please run prepare_dd_data.sh and morgan_fp.py first."
  exit 1
fi

if [[ ! -f "$MASTER_SCORES_PATH" ]]; then
  echo "ERROR: Master scores CSV not found at: $MASTER_SCORES_PATH"
  exit 1
fi

# --- Step 1: Iteration 1 Phase 1 (Sampling) ---
echo ">> [Step 1] Iteration 1 - Phase 1 (Sampling)"
python scripts_1/molecular_file_count_updated.py \
  -pt "$PROTEIN" -it 1 -cdd "$DD_DATA_ROOT/library_morgan" -t_pos "$N_CPU" -t_samp 5000000

python scripts_1/sampling.py \
  -pt "$PROTEIN" -fp "$FILE_PATH" -it 1 \
  -dd "$DD_DATA_ROOT/library_morgan" -t_pos "$N_CPU" \
  -tr_sz "$TRAIN_SIZE" -vl_sz "$VAL_SIZE"

python scripts_1/sanity_check.py -pt "$PROTEIN" -fp "$FILE_PATH" -it 1

python scripts_1/extracting_morgan.py \
  -pt "$PROTEIN" -fp "$FILE_PATH" -it 1 \
  -md "$DD_DATA_ROOT/library_morgan" -t_pos "$N_CPU"

python scripts_1/extracting_smiles.py \
  -pt "$PROTEIN" -fp "$FILE_PATH" -it 1 \
  -smd "$DD_DATA_ROOT/library_smiles" -t_pos "$N_CPU"

# --- Step 2: Iteration 1 Simulated Labels ---
echo ">> [Step 2] Iteration 1 - Simulated Labels"
python utilities/simulate_labels.py \
  --master_scores "$MASTER_SCORES_PATH" \
  --project_dir "$PROJECT_DIR" \
  --iteration 1

# --- Step 3: Iteration 1 Phase 4 (Training) ---
echo ">> [Step 3] Iteration 1 - Phase 4 (Training on GPU)"
./utilities/run_iteration1_phase4_gpu.sh

# --- Step 4: Iteration 1 Phase 5 (Predictions) ---
echo ">> [Step 4] Iteration 1 - Phase 5 (Predictions)"
ITER=1 ./utilities/finish_iteration_phase5.sh

# Verify predictions were generated
n_txt=$(find "$PROJECT_DIR/iteration_1/morgan_1024_predictions" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l || echo 0)
if [[ "$n_txt" -eq 0 ]]; then
  echo "ERROR: Iteration 1 predictions failed (no .txt files found)."
  exit 1
fi

# --- Step 5: Iterations 2-N (Simulation Driver) ---
echo ">> [Step 5] Iterations 2-${TOTAL_ITERATIONS} (Simulation Driver)"
START_ITER=2 END_ITER="$TOTAL_ITERATIONS" ./utilities/run_dd_simulation_pipeline.sh

# --- Step 6: Final Extraction ---
echo ">> [Step 6] Final Extraction"
python utilities/final_extraction.py \
  -smile_dir "$DD_DATA_ROOT/library_smiles" \
  -prediction_dir "$PROJECT_DIR/iteration_${TOTAL_ITERATIONS}/morgan_1024_predictions" \
  -processors "$N_CPU" \
  -mols_to_dock 100000

echo "=== Pipeline Completed Successfully! ==="