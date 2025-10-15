#!/bin/bash -l
#SBATCH --job-name=eeg2025_demega
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

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/mxb190076/eeg2025_venv  # use your existing env name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

mkdir -p data output

python cli.py \
  --mini --epochs 30 --batch_size 128 --num_workers 4 \
  --data_dir ./data --out_dir ./output --save_zip \
  --arch demega --use_projector 1 \
  --n_clusters 20 --pcs_per_cluster 3 \
  --ssl_epochs 10 --ssl_steps 150 --ssl_batch 16 --ssl_crop 150
