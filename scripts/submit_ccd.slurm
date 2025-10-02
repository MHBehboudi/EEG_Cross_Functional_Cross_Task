#!/bin/bash -l
#SBATCH --job-name=eeg2025_ccd
#SBATCH --partition=a30-2.12gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mxb190076/EEG_Cross_Functional_Cross_Task
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

# activate env in /work
conda activate /work/mxb190076/eeg2025_env

# sanity pins for CPU threads on Codabench-like CPUs (harmless here)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p data output
srun python scripts/train_ccd.py --mini --epochs 100 --batch_size 128 --num_workers 4
