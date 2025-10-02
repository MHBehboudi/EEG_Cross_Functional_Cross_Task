#!/bin/bash -l
#SBATCH --job-name=eeg2025_mini
#SBATCH --partition=a30-2.12gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/mxb190076/EEG_Cross_Functional_Cross_Task
#SBATCH --output=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.out
#SBATCH --error=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.err

set -euo pipefail

# --- Env ---
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/mxb190076/eeg2025

# Threading sanity (don’t oversubscribe CPUs)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# --- Run ---
mkdir -p data output

# Mini run (fast, startkit-like). For full training, remove --mini and bump --epochs.
python cli.py \
  --mini --epochs 1 --batch_size 128 --num_workers 4 \
  --data_dir ./data --out_dir ./output --save_zip \
  --use_clusters --n_clusters 50 --projector_init kmeans \
  --amp auto --tf32 --compile
