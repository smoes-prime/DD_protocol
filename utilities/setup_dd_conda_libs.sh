#!/usr/bin/env bash
# Fix scipy/sklearn CXXABI_1.3.15 ImportError on Ubuntu (use conda libstdc++, not system /lib).
#
# Run once:
#   conda activate dd-env
#   bash utilities/setup_dd_conda_libs.sh
#
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate dd-env first (conda activate dd-env)" >&2
  exit 1
fi

conda install -y -c conda-forge libstdcxx-ng libgcc-ng scipy scikit-learn

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/dd_env_libs.sh" <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -c "import scipy, sklearn; print('scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'OK')"
echo "Done. Re-activate dd-env in new shells: conda deactivate && conda activate dd-env"
