#!/usr/bin/env bash
# Deep Docking simulation-mode driver (iterations 2..N).
# - Uses precomputed scores via utilities/simulate_labels.py
# - Runs phase 1, phase 4, and phase 5 for each iteration
# - Stops on first error and writes per-step logs
#
# Usage:
#   chmod +x utilities/run_dd_simulation_pipeline.sh
#   export DD_DATA_ROOT=/home/sebmo/HASTEN_proj/DD_data
#   export FILE_PATH=$DD_DATA_ROOT/projects
#   export PROTEIN=1B12
#   conda activate dd-env
#   ./utilities/run_dd_simulation_pipeline.sh
#
# Optional overrides:
#   START_ITER=2 END_ITER=11 TRAIN_SIZE=3000000 VAL_SIZE=1000000 \
#   TOTAL_ITERATIONS=11 NUM_HYP=24 RECALL=0.9 \
#   MASTER_SCORES_PATH="$DD_DATA_ROOT/master_scores/1B12_div_docked_DD_labels.csv" \
#   ./utilities/run_dd_simulation_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Required env vars (with sensible defaults where possible)
DD_DATA_ROOT="${DD_DATA_ROOT:-/home/sebmo/HASTEN_proj/DD_data}"
FILE_PATH="${FILE_PATH:-$DD_DATA_ROOT/projects}"
PROTEIN="${PROTEIN:-1B12}"
PROJECT_DIR="${PROJECT_DIR:-$FILE_PATH/$PROTEIN}"
MASTER_SCORES_PATH="${MASTER_SCORES_PATH:-$DD_DATA_ROOT/master_scores/1B12_div_docked_DD_labels.csv}"

# Iteration and model config
START_ITER="${START_ITER:-2}"
END_ITER="${END_ITER:-11}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-11}"
NUM_HYP="${NUM_HYP:-24}"
RECALL="${RECALL:-0.9}"
PERCENT_FIRST_MOLS="${PERCENT_FIRST_MOLS:-1}"
PERCENT_LAST_MOLS="${PERCENT_LAST_MOLS:-0.01}"

# Sampling config
TRAIN_SIZE="${TRAIN_SIZE:-3000000}"
VAL_SIZE="${VAL_SIZE:-1000000}"      # validation and test each
TOTAL_SAMPLING="${TOTAL_SAMPLING:-$TRAIN_SIZE}"
N_CPU="${N_CPU:-$(nproc)}"

if [[ "$START_ITER" -lt 2 ]]; then
  echo "ERROR: START_ITER must be >= 2 (iteration 1 should already be completed)." >&2
  exit 1
fi
if [[ "$END_ITER" -lt "$START_ITER" ]]; then
  echo "ERROR: END_ITER must be >= START_ITER." >&2
  exit 1
fi
if [[ ! -f "$MASTER_SCORES_PATH" ]]; then
  echo "ERROR: master score file not found: $MASTER_SCORES_PATH" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_DIR/iteration_1" ]]; then
  echo "ERROR: missing $PROJECT_DIR/iteration_1 (run iteration 1 first)." >&2
  exit 1
fi

check_prev_predictions() {
  local prev="$1"
  local pred_dir="$PROJECT_DIR/iteration_${prev}/morgan_1024_predictions"
  if [[ ! -d "$pred_dir" ]]; then
    echo "ERROR: missing $pred_dir" >&2
    echo "Finish iteration $prev phase 5 first:" >&2
    echo "  ITER=$prev ./utilities/finish_iteration_phase5.sh" >&2
    exit 1
  fi
  local n_txt
  n_txt=$(find "$pred_dir" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
  if [[ "$n_txt" -eq 0 ]]; then
    echo "ERROR: $pred_dir has no *.txt prediction shards (found 0)." >&2
    echo "Finish iteration $prev phase 5 first:" >&2
    echo "  ITER=$prev ./utilities/finish_iteration_phase5.sh" >&2
    exit 1
  fi
  echo "Prereq OK: iteration $prev has $n_txt prediction shard(s)."
}

check_prev_predictions "$((START_ITER - 1))"

mkdir -p "$PROJECT_DIR/pipeline_logs"

run_step() {
  local iter="$1"
  local step="$2"
  shift 2
  local log="$PROJECT_DIR/pipeline_logs/iter_${iter}_${step}.log"
  echo "[$(date '+%F %T')] iter=$iter step=$step"
  echo "cmd: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
}

run_jobs_in_dir() {
  local iter="$1"
  local kind="$2"  # train or predict
  local jobs_dir="$3"

  shopt -s nullglob
  local jobs=("$jobs_dir"/simple_job_*.sh)
  shopt -u nullglob
  if [[ "${#jobs[@]}" -eq 0 ]]; then
    echo "ERROR: no jobs found in $jobs_dir" >&2
    exit 1
  fi

  local idx=0
  for job in "${jobs[@]}"; do
    idx=$((idx+1))
    run_step "$iter" "${kind}_job_${idx}" bash "$job"
  done
}

echo "=== DD simulation pipeline ==="
echo "Repo:             $ROOT_DIR"
echo "Project:          $PROJECT_DIR"
echo "Iterations:       $START_ITER .. $END_ITER"
echo "CPUs:             $N_CPU"
echo "Train/val size:   $TRAIN_SIZE / $VAL_SIZE"
echo "Master scores:    $MASTER_SCORES_PATH"
echo

for ((it=START_ITER; it<=END_ITER; it++)); do
  prev=$((it-1))
  is_last="False"
  if [[ "$it" -eq "$END_ITER" ]]; then
    is_last="True"
  fi

  echo "================ Iteration $it ================"
  check_prev_predictions "$prev"

  # Phase 1
  run_step "$it" "phase1_count" \
    python scripts_1/molecular_file_count_updated.py \
      -pt "$PROTEIN" -it "$it" \
      -cdd "$PROJECT_DIR/iteration_${prev}/morgan_1024_predictions" \
      -t_pos "$N_CPU" -t_samp "$TOTAL_SAMPLING"

  run_step "$it" "phase1_sampling" \
    python scripts_1/sampling.py \
      -pt "$PROTEIN" -fp "$FILE_PATH" -it "$it" \
      -dd "$PROJECT_DIR/iteration_${prev}/morgan_1024_predictions" \
      -t_pos "$N_CPU" -tr_sz "$TRAIN_SIZE" -vl_sz "$VAL_SIZE"

  run_step "$it" "phase1_sanity" \
    python scripts_1/sanity_check.py -pt "$PROTEIN" -fp "$FILE_PATH" -it "$it"

  run_step "$it" "phase1_extract_morgan" \
    python scripts_1/extracting_morgan.py \
      -pt "$PROTEIN" -fp "$FILE_PATH" -it "$it" \
      -md "$DD_DATA_ROOT/library_morgan" -t_pos "$N_CPU"

  run_step "$it" "phase1_extract_smiles" \
    python scripts_1/extracting_smiles.py \
      -pt "$PROTEIN" -fp "$FILE_PATH" -it "$it" \
      -smd "$DD_DATA_ROOT/library_smiles" -t_pos "$N_CPU"

  # Simulated docking labels
  run_step "$it" "simulate_labels" \
    python utilities/simulate_labels.py \
      --master_scores "$MASTER_SCORES_PATH" \
      --project_dir "$PROJECT_DIR" \
      --iteration "$it"

  # Phase 4: generate training jobs and run all
  run_step "$it" "phase4_make_jobs" \
    python scripts_2/simple_job_models_manual.py \
      --iteration_no "$it" \
      --morgan_directory "$DD_DATA_ROOT/library_morgan" \
      --file_path "$PROJECT_DIR" \
      --number_of_hyp "$NUM_HYP" \
      --total_iterations "$TOTAL_ITERATIONS" \
      --is_last "$is_last" \
      --number_mol "$VAL_SIZE" \
      --percent_first_mols "$PERCENT_FIRST_MOLS" \
      --percent_last_mols "$PERCENT_LAST_MOLS" \
      --recall "$RECALL"

  run_jobs_in_dir "$it" "train" "$PROJECT_DIR/iteration_${it}/simple_job"

  # Phase 5: pick best model and generate/run prediction jobs
  run_step "$it" "phase5_eval" \
    python scripts_2/hyperparameter_result_evaluation.py \
      --n_iteration "$it" \
      --data_path "$PROJECT_DIR" \
      --morgan_directory "$DD_DATA_ROOT/library_morgan" \
      --number_mol "$VAL_SIZE" \
      --recall "$RECALL"

  run_step "$it" "phase5_make_pred_jobs" \
    python scripts_2/simple_job_predictions_manual.py \
      --project_name "$PROTEIN" \
      --file_path "$FILE_PATH" \
      --n_iteration "$it" \
      --morgan_directory "$DD_DATA_ROOT/library_morgan"

  run_jobs_in_dir "$it" "predict" "$PROJECT_DIR/iteration_${it}/simple_job_predictions"

  echo "Iteration $it complete."
done

echo
echo "All iterations complete: $START_ITER..$END_ITER"
echo "Logs: $PROJECT_DIR/pipeline_logs"
