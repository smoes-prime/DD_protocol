#!/usr/bin/env bash
# Source this file to set DD_* training grid variables from GPU VRAM.
#   source utilities/dd_gpu_profile.sh

_dd_profile_from_vram() {
  local total_mib="$1"
  if [[ "$total_mib" -le 10240 ]]; then
    DD_GPU_PROFILE="8gb"
    DD_BATCH_SIZES="${DD_BATCH_SIZES:-64,128}"
    DD_NUM_UNITS="${DD_NUM_UNITS:-100,256,512}"
    DD_OSS="${DD_OSS:-10}"
    DD_DROPOUT="${DD_DROPOUT:-0.2}"
    DD_MIN_FREE_MIB="${DD_MIN_FREE_MIB:-3500}"
    DD_MAX_PARALLEL="${DD_MAX_PARALLEL:-1}"
  elif [[ "$total_mib" -le 20480 ]]; then
    DD_GPU_PROFILE="16gb"
    DD_BATCH_SIZES="${DD_BATCH_SIZES:-128,256}"
    DD_NUM_UNITS="${DD_NUM_UNITS:-100,512,1500}"
    DD_OSS="${DD_OSS:-5,10}"
    DD_DROPOUT="${DD_DROPOUT:-0.2,0.5}"
    DD_MIN_FREE_MIB="${DD_MIN_FREE_MIB:-6000}"
    DD_MAX_PARALLEL="${DD_MAX_PARALLEL:-1}"
  else
    DD_GPU_PROFILE="24gb_plus"
    DD_BATCH_SIZES="${DD_BATCH_SIZES:-128,256}"
    DD_NUM_UNITS="${DD_NUM_UNITS:-100,1500,2000}"
    DD_OSS="${DD_OSS:-5,10,20}"
    DD_DROPOUT="${DD_DROPOUT:-0.2,0.5}"
    DD_MIN_FREE_MIB="${DD_MIN_FREE_MIB:-8000}"
    DD_MAX_PARALLEL="${DD_MAX_PARALLEL:-1}"
  fi
}

_dd_apply_cpu_safe() {
  DD_GPU_PROFILE="cpu_safe"
  DD_BATCH_SIZES="${DD_BATCH_SIZES:-64}"
  DD_NUM_UNITS="${DD_NUM_UNITS:-100,256}"
  DD_OSS="${DD_OSS:-5}"
  DD_DROPOUT="${DD_DROPOUT:-0.2}"
  DD_MIN_FREE_MIB="${DD_MIN_FREE_MIB:-0}"
  DD_MAX_PARALLEL="${DD_MAX_PARALLEL:-1}"
}

if [[ -n "${DD_GPU_PROFILE_FORCE:-}" ]]; then
  case "$DD_GPU_PROFILE_FORCE" in
    8gb|8GB) _dd_profile_from_vram 8192 ;;
    16gb|16GB) _dd_profile_from_vram 16384 ;;
    24gb|24GB|32gb) _dd_profile_from_vram 24576 ;;
    cpu) _dd_apply_cpu_safe ;;
    *) _dd_profile_from_vram 8192 ;;
  esac
elif command -v nvidia-smi >/dev/null 2>&1; then
  best_idx=0
  best_free=0
  best_total=0
  mapfile -t _gpu_lines < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  for line in "${_gpu_lines[@]}"; do
    line="${line// /}"
    IFS=',' read -r idx total free <<< "$line"
    [[ -z "${idx:-}" ]] && continue
    if [[ "$free" -gt "$best_free" ]]; then
      best_free=$free
      best_total=$total
      best_idx=$idx
    fi
  done
  if [[ "$best_total" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$best_idx}"
    _dd_profile_from_vram "$best_total"
    export DD_GPU_INDEX="$best_idx"
    export DD_GPU_TOTAL_MIB="$best_total"
    export DD_GPU_FREE_MIB="$best_free"
  else
    echo "dd_gpu_profile: no GPUs reported; CPU-safe defaults" >&2
    _dd_apply_cpu_safe
  fi
else
  echo "dd_gpu_profile: nvidia-smi not found; CPU-safe defaults" >&2
  _dd_apply_cpu_safe
fi

export DD_GPU_PROFILE DD_BATCH_SIZES DD_NUM_UNITS DD_OSS DD_DROPOUT DD_MIN_FREE_MIB DD_MAX_PARALLEL
