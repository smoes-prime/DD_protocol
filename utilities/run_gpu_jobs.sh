#!/usr/bin/env bash
# Run simple_job training or prediction scripts one at a time with GPU memory gating.
#
# Usage:
#   ./utilities/run_gpu_jobs.sh train 1 /path/to/projects/1B12
#   ./utilities/run_gpu_jobs.sh predict 1 /path/to/projects/1B12
#
set -euo pipefail

KIND="${1:?Usage: run_gpu_jobs.sh train|predict ITERATION PROJECT_DIR}"
ITER="${2:?missing iteration}"
PROJECT_DIR="${3:?missing project dir}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# shellcheck source=utilities/dd_gpu_profile.sh
source "$ROOT_DIR/utilities/dd_gpu_profile.sh"

if [[ "$KIND" == "train" ]]; then
  JOBS_DIR="$PROJECT_DIR/iteration_${ITER}/simple_job"
elif [[ "$KIND" == "predict" ]]; then
  JOBS_DIR="$PROJECT_DIR/iteration_${ITER}/simple_job_predictions"
else
  echo "ERROR: KIND must be train or predict" >&2
  exit 1
fi

if [[ ! -d "$JOBS_DIR" ]]; then
  echo "ERROR: jobs directory not found: $JOBS_DIR" >&2
  exit 1
fi

wait_for_gpu() {
  local min_free="${DD_MIN_FREE_MIB:-3500}"
  local max_wait="${DD_GPU_WAIT_MAX_SEC:-7200}"
  local waited=0
  while [[ "$waited" -lt "$max_wait" ]]; do
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      return 0
    fi
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$free_mib" && "$free_mib" -ge "$min_free" ]]; then
      # Optional: avoid starting if another python is using a lot of VRAM on GPU 0
      local used
      used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
      if [[ -z "$used" || "$used" -lt 500 ]]; then
        return 0
      fi
      if [[ "$free_mib" -ge "$((min_free + used))" ]]; then
        return 0
      fi
    fi
    echo "[gpu-wait] free=${free_mib:-?} MiB need>=${min_free} MiB (${waited}s)"
    sleep 15
    waited=$((waited + 15))
  done
  echo "ERROR: timed out waiting for GPU (>${max_wait}s)" >&2
  exit 1
}

shopt -s nullglob
jobs=("$JOBS_DIR"/simple_job_*.sh)
shopt -u nullglob
if [[ "${#jobs[@]}" -eq 0 ]]; then
  echo "ERROR: no simple_job_*.sh in $JOBS_DIR" >&2
  exit 1
fi

echo "=== run_gpu_jobs: $KIND iteration $ITER ==="
echo "Profile: ${DD_GPU_PROFILE:-unknown}  min_free_mib=${DD_MIN_FREE_MIB:-?}"
echo "Jobs: ${#jobs[@]} in $JOBS_DIR"
echo

for job in "${jobs[@]}"; do
  done_marker="${job}.done"
  if [[ -f "$done_marker" ]]; then
    echo "SKIP (done): $(basename "$job")"
    continue
  fi
  wait_for_gpu
  echo "RUN: $(basename "$job")"
  log="${job%.sh}.log"
  if bash "$job" 2>&1 | tee "$log"; then
    touch "$done_marker"
    echo "OK: $(basename "$job")"
  else
    echo "FAILED: $job (see $log)" >&2
    exit 1
  fi
done

echo "All $KIND jobs complete for iteration $ITER."
