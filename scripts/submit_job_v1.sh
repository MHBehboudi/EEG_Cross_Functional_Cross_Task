#!/bin/bash -l
#SBATCH --job-name=eeg2025_train_ch1_mini
#SBATCH --partition=a30-2.12gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/mxb190076/work/EEG2025
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/mxb190076/Libribrain_Test/libribrain_env
PYTHON=/work/mxb190076/Libribrain_Test/libribrain_env/bin/python

mkdir -p data
srun $PYTHON train_and_submit_v1.py --mini --epochs 100 --batch_size 128 --num_workers 4
