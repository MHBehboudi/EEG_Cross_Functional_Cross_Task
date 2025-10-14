#!/bin/bash -l
#SBATCH --job-name=eeg2025_mini
#SBATCH --partition=a30-2.12gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/mxb190076/EEG_Cross_Functional_Cross_Task
#SBATCH --output=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.out
#SBATCH --error=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.err

set -euo pipefail

# --- Env ---
module purge
# If your cluster requires a Python module, uncomment and set the correct version:
# module load python/3.12

# Activate the repo-local venv created at /work/mxb190076/EEG_Cross_Functional_Cross_Task/.venv
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Make sure Python can find the package under src/
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

# Threading sanity
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# --- Run ---
mkdir -p data output

# Mini run to validate end-to-end; scale up later as needed
python cli.py \
  --mini --epochs 3 --batch_size 128 --num_workers 4 \
  --data_dir ./data --out_dir ./output --save_zip \
  --cluster_mode ssl_pca --n_clusters 20 --pcs_per_cluster 3 \
  --ssl_epochs 5 --ssl_steps 80 --ssl_batch 16 --ssl_crop 150
