# Deep Docking on Ubuntu (CUDA 12/13 driver) with pre-docked scores

This guide matches data under `/home/sebmo/HASTEN_proj/data` and simulation mode
(lookup scores instead of Glide/FRED) for project `1B12`.

## 0. Paths (adjust if needed)

```bash
export HASTEN_ROOT=/home/sebmo/HASTEN_proj
export DATA_SRC=$HASTEN_ROOT/data
export DD_DATA_ROOT=$HASTEN_ROOT/DD_data
export DD_REPO=$HASTEN_ROOT/DD_protocol   # git clone of your fork here
export PROJECT_NAME=1B12
```

## 1. Clone DD and create conda env

CUDA **13.0 driver** on the host is fine; do **not** use the old `environment.yml`
(cuda10 + tensorflow-gpu 1.15). Use the modern env:

```bash
cd $HASTEN_ROOT
git clone <your-dd-fork-url> DD_protocol
cd DD_protocol

conda env create -f utilities/environment_modern.yml
conda activate dd-env

python -c "import tensorflow as tf; print('TF', tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -c "from rdkit import Chem; print('RDKit OK')"
```

If GPU is empty, training still runs on CPU (slow). For driver/CUDA issues see:
https://www.tensorflow.org/install/pip#linux

## 2. Prepare data directories

```bash
cd $DD_REPO
chmod +x utilities/prepare_dd_data.sh utilities/add_morgan_headers.sh

bash utilities/prepare_dd_data.sh
```

This creates:

| Path | Purpose |
|------|---------|
| `$DD_DATA_ROOT/library_smiles/` | Sharded `SMILES ZINC_ID` `.txt` files |
| `$DD_DATA_ROOT/library_morgan/` | Morgan fingerprints (you generate) |
| `$DD_DATA_ROOT/master_scores/..._DD_labels.csv` | All pre-docked scores for simulation |
| `$DD_DATA_ROOT/projects/1B12/` | DD project + `logs.txt` template |

Edit `$DD_DATA_ROOT/projects/1B12/logs.txt` — set line 3 to your receptor grid zip.

## 3. Morgan fingerprints (long step for 21M compounds)

```bash
conda activate dd-env
cd $DD_REPO

python utilities/morgan_fp.py \
  --smile_folder_path $DD_DATA_ROOT/library_smiles \
  --folder_name $DD_DATA_ROOT/library_morgan \
  --tot_process $(nproc)

bash utilities/add_morgan_headers.sh $DD_DATA_ROOT/library_morgan
```

Expect days/weeks for ~21M molecules depending on CPU cores. You can test on one shard first:

```bash
mkdir -p /tmp/dd_test_smiles /tmp/dd_test_morgan
head -n 100000 $DD_DATA_ROOT/library_smiles/smile_all_00.txt > /tmp/dd_test_smiles/chunk.txt
python utilities/morgan_fp.py --smile_folder_path /tmp/dd_test_smiles --folder_name /tmp/dd_test_morgan --tot_process 8
```

## 4. Iteration 1 — phase 1 (sampling)

Example: 3M train + 1M valid + 1M test on first iteration (`t_samp = 5M`):

```bash
FILE_PATH=$DD_DATA_ROOT/projects
PROTEIN=$PROJECT_NAME
N_CPU=$(nproc)

python scripts_1/molecular_file_count_updated.py \
  -pt $PROTEIN -it 1 -cdd $DD_DATA_ROOT/library_morgan -t_pos $N_CPU -t_samp 5000000

python scripts_1/sampling.py \
  -pt $PROTEIN -fp $FILE_PATH -it 1 \
  -dd $DD_DATA_ROOT/library_morgan -t_pos $N_CPU \
  -tr_sz 3000000 -vl_sz 1000000

python scripts_1/sanity_check.py -pt $PROTEIN -fp $FILE_PATH -it 1

python scripts_1/extracting_morgan.py \
  -pt $PROTEIN -fp $FILE_PATH -it 1 \
  -md $DD_DATA_ROOT/library_morgan -t_pos $N_CPU

python scripts_1/extracting_smiles.py \
  -pt $PROTEIN -fp $FILE_PATH -it 1 \
  -smd $DD_DATA_ROOT/library_smiles -t_pos $N_CPU
```

## 5. Simulated docking (skip phases 2–3)

```bash
python utilities/simulate_labels.py \
  --master_scores $DD_DATA_ROOT/master_scores/1B12_div_docked_DD_labels.csv \
  --project_dir $DD_DATA_ROOT/projects/$PROJECT_NAME \
  --iteration 1
```

If exit code 2: some sampled IDs are missing from your docked CSV — expand
`1B12_div_docked.csv` or reduce sample sizes.

**Do not run** `extract_labels.py` unless you have real SDFs in `iteration_1/docked/`.

## 6. Phase 4 — train models (GPU-aware, one job at a time)

**Do not** run `for f in simple_job_*.sh; do bash "$f"; done` in parallel — it exhausts VRAM on 8GB cards.

```bash
export DD_GPU_AUTO=1
bash utilities/setup_dd_conda_libs.sh   # once, if CXXABI ImportError appears

# Iteration 1 (auto-detects GPU, uses bs 64/128 and units 100–512 on 8GB)
./utilities/run_iteration1_phase4_gpu.sh
```

Manual overrides:

| Variable | Example | Effect |
|----------|---------|--------|
| `DD_GPU_PROFILE_FORCE` | `8gb` | Force profile without nvidia-smi |
| `DD_BATCH_SIZES` | `64,128` | Batch sizes in generated jobs |
| `DD_NUM_UNITS` | `100,256,512` | Hidden layer sizes |
| `DD_MIN_FREE_MIB` | `3500` | Wait until GPU has this much free memory |
| `DD_CAP_JOBS` | `12` | Cap number of training jobs |

Or regenerate + run for any iteration:

```bash
DD_GPU_AUTO=1 python scripts_2/simple_job_models_manual.py ... # same args as before
./utilities/run_gpu_jobs.sh train 1 $FILE_PATH/$PROTEIN
```

Completed jobs get a `simple_job_N.sh.done` marker; re-running skips them.

For later iterations: repeat phase 1 with `-cdd` pointing to previous
`iteration_N/morgan_1024_predictions`, then `simulate_labels.py --iteration N`.

## 7. Phase 5 — screen library (GPU)

```bash
ITER=1 ./utilities/finish_iteration_phase5.sh
```

This runs hyperparameter evaluation, creates prediction jobs, and runs them
sequentially via `run_gpu_jobs.sh predict`.

Or manually:

```bash
python scripts_2/hyperparameter_result_evaluation.py \
  --n_iteration 1 \
  --data_path $FILE_PATH/$PROTEIN \
  --morgan_directory $DD_DATA_ROOT/library_morgan \
  --number_mol 1000000 \
  --recall 0.9

python scripts_2/simple_job_predictions_manual.py \
  --project_name $PROTEIN \
  --file_path $FILE_PATH \
  --n_iteration 1 \
  --morgan_directory $DD_DATA_ROOT/library_morgan

./utilities/run_gpu_jobs.sh predict 1 $FILE_PATH/$PROTEIN
```

## 8. Iterations 2–11 (simulation driver)

```bash
./utilities/run_dd_simulation_pipeline.sh
```

## Important notes

1. **Scored subset vs 21M library**: `1B12_div_docked.csv` must contain every
   compound ID that phase 1 might sample, or simulation will fail for missing IDs.
2. **Simulation is not built into DD** — `simulate_labels.py` replaces phases 2–3 only.
3. **CUDA 13 host driver**: use `environment_modern.yml` (TensorFlow 2.15+ with bundled CUDA).
4. **Project path**: `simple_job_models_manual.py` expects `--file_path` to include
   the project folder name (e.g. `.../projects/1B12`), not the parent `projects/` alone.
