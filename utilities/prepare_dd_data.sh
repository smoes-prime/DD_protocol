#!/usr/bin/env bash
# Prepare HASTEN-style local data for Deep Docking (DD).
#
# Converts:
#   - Enamine_diversity21m.smi  (SMILES + ID, space- or tab-separated)
#   - 1B12_div_docked.csv         (score,ID comma-separated)
# into DD directory layout under DD_DATA_ROOT.
#
# Usage:
#   bash utilities/prepare_dd_data.sh
#   # or override defaults:
#   DATA_SRC=/home/sebmo/HASTEN_proj/data \
#   DD_DATA_ROOT=/home/sebmo/HASTEN_proj/DD_data \
#   PROJECT_NAME=1B12 \
#   bash utilities/prepare_dd_data.sh
#
set -euo pipefail

DATA_SRC="${DATA_SRC:-/home/sebmo/HASTEN_proj/data}"
DD_DATA_ROOT="${DD_DATA_ROOT:-/home/sebmo/HASTEN_proj/DD_data}"
SMILES_FILE="${SMILES_FILE:-Enamine_diversity21m.smi}"
SCORES_FILE="${SCORES_FILE:-1B12_div_docked.csv}"
PROJECT_NAME="${PROJECT_NAME:-1B12}"
LINES_PER_SHARD="${LINES_PER_SHARD:-1000000}"
SPLIT_LIBRARY="${SPLIT_LIBRARY:-1}"   # 1 = split 21M SMILES into shards; 0 = symlink only

LIB_SMILES="${DD_DATA_ROOT}/library_smiles"
LIB_MORGAN="${DD_DATA_ROOT}/library_morgan"
MASTER_SCORES="${DD_DATA_ROOT}/master_scores/${SCORES_FILE%.csv}_DD_labels.csv"
PROJECT_DIR="${DD_DATA_ROOT}/projects/${PROJECT_NAME}"

SRC_SMILES="${DATA_SRC}/${SMILES_FILE}"
SRC_SCORES="${DATA_SRC}/${SCORES_FILE}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "${SRC_SMILES}" ]] || die "Missing SMILES library: ${SRC_SMILES}"
[[ -f "${SRC_SCORES}" ]] || die "Missing scores file: ${SRC_SCORES}"

echo "=== Deep Docking data preparation ==="
echo "Source:      ${DATA_SRC}"
echo "DD data root: ${DD_DATA_ROOT}"
echo "Project:     ${PROJECT_NAME}"
echo ""

mkdir -p "${LIB_SMILES}" "${LIB_MORGAN}" "${DD_DATA_ROOT}/master_scores" "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# 1) Master score lookup (simulation / label generation)
#    DD format: header r_i_docking_score,ZINC_ID then score,id per line
# ---------------------------------------------------------------------------
echo "[1/4] Building master score lookup..."
if head -n1 "${SRC_SCORES}" | grep -qi 'r_i_docking_score'; then
  cp -f "${SRC_SCORES}" "${MASTER_SCORES}"
else
  { echo "r_i_docking_score,ZINC_ID"; cat "${SRC_SCORES}"; } > "${MASTER_SCORES}"
fi
SCORE_LINES=$(( $(wc -l < "${MASTER_SCORES}") - 1 ))
echo "      -> ${MASTER_SCORES} (${SCORE_LINES} scores)"

# ---------------------------------------------------------------------------
# 2) SMILES library shards (space-separated: SMILES ZINC_ID)
#    morgan_fp.py reads LIB_SMILES/*.txt via glob
# ---------------------------------------------------------------------------
echo "[2/4] Preparing SMILES library shards..."
CANONICAL_SMILES="${LIB_SMILES}/${SMILES_FILE%.smi}_canonical.smi"

# Normalize whitespace (space/tab) -> single space between SMILES and ID
awk 'NF>=2 {print $1, $NF}' "${SRC_SMILES}" > "${CANONICAL_SMILES}"
echo "      Canonical SMILES: ${CANONICAL_SMILES}"

if [[ "${SPLIT_LIBRARY}" == "1" ]]; then
  echo "      Splitting into shards of ${LINES_PER_SHARD} lines..."
  rm -f "${LIB_SMILES}"/smile_all_*.txt
  split -d -l "${LINES_PER_SHARD}" --additional-suffix=.txt \
    "${CANONICAL_SMILES}" "${LIB_SMILES}/smile_all_"
  N_SHARDS=$(ls -1 "${LIB_SMILES}"/smile_all_*.txt 2>/dev/null | wc -l)
  echo "      -> ${N_SHARDS} shards in ${LIB_SMILES}/"
else
  ln -sf "${CANONICAL_SMILES}" "${LIB_SMILES}/smile_all_00.txt"
  echo "      -> single shard symlink ${LIB_SMILES}/smile_all_00.txt"
fi

# ---------------------------------------------------------------------------
# 3) Project skeleton + logs.txt template for Slurm / manual runs
# ---------------------------------------------------------------------------
echo "[3/4] Creating project skeleton..."
mkdir -p "${PROJECT_DIR}/iteration_1"/{smile,morgan,docked,all_models,simple_job,simple_job_predictions}

LOGS="${PROJECT_DIR}/logs.txt"
if [[ ! -f "${LOGS}" ]]; then
  cat > "${LOGS}" <<EOF
${DD_DATA_ROOT}/projects
${PROJECT_NAME}
/path/to/your/receptor_grid.zip
${LIB_MORGAN}
${LIB_SMILES}
Glide
24
1000000

EOF
  echo "      Wrote ${LOGS} (edit grid path and docking program before Slurm runs)"
else
  echo "      Keeping existing ${LOGS}"
fi

# ---------------------------------------------------------------------------
# 4) Helper scripts + README snippet in DD_data
# ---------------------------------------------------------------------------
echo "[4/4] Installing helper references..."
cat > "${DD_DATA_ROOT}/README_DD_data.txt" <<EOF
Deep Docking data layout
========================

library_smiles/     SMILES shards for morgan_fp.py (space-separated: SMILES ID)
library_morgan/     Morgan fingerprints (create with morgan_fp.py, then add_morgan_headers.sh)
master_scores/      Pre-docked scores for simulation mode (lookup by ZINC_ID)
projects/${PROJECT_NAME}/   DD project (iteration_1, logs.txt, ...)

Next steps (from your DD_protocol clone):

  conda activate dd-env
  cd /path/to/DD_protocol

  # A) Morgan fingerprints (CPU-heavy; 21M compounds takes a long time)
  python utilities/morgan_fp.py \\
    --smile_folder_path ${LIB_SMILES} \\
    --folder_name ${LIB_MORGAN} \\
    --tot_process \$(nproc)

  bash utilities/add_morgan_headers.sh ${LIB_MORGAN}

  # B) Iteration 1 phase 1 (example)
  python scripts_1/molecular_file_count_updated.py \\
    -pt ${PROJECT_NAME} -it 1 -cdd ${LIB_MORGAN} -t_pos \$(nproc) -t_samp 3000000
  ...

  # C) After phase 1 sampling, simulated docking labels:
  python utilities/simulate_labels.py \\
    --master_scores ${MASTER_SCORES} \\
    --project_dir ${PROJECT_DIR} \\
    --iteration 1

See utilities/RUN_UBUNTU_CUDA.md for full command sequence.
EOF

echo ""
echo "=== Done ==="
echo "Library SMILES:  ${LIB_SMILES}"
echo "Morgan (empty):  ${LIB_MORGAN}  (run morgan_fp.py next)"
echo "Master scores:   ${MASTER_SCORES}"
echo "Project:         ${PROJECT_DIR}"
echo ""
echo "Edit ${LOGS} and set your Glide/FRED grid path before automated phases."
