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

set -Eeuo pipefail

# ---- EDIT THESE IF NEEDED ----
# Your repo on the HPC:
REPO_DIR="${REPO_DIR:-$HOME/EEG2025/EEG_Cross_Functional_Cross_Task}"

# Your Python env on the HPC (the one you used before):
ENV_PATH="${ENV_PATH:/work/mxb190076/Libribrain_Test/libribrain_env}"
PYTHON="${PYTHON:/work/mxb190076/Libribrain_Test/libribrain_env}"

# Default training args (can be overridden via sbatch --export)
ARGS="${ARGS:---mini --epochs 100 --batch-size 128 --num-workers ${SLURM_CPUS_PER_TASK:-4}}"
# --------------------------------

# Activate environment if present
if [ -d "$ENV_PATH" ] && [ -x "$ENV_PATH/bin/activate" ]; then
  source "$ENV_PATH/bin/activate"
fi

# Hygiene
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Paths & logs
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/output"
cd "$REPO_DIR"

echo "[INFO] Using python: $PYTHON"
"$PYTHON" --version

echo "[INFO] Launching training:"
echo "       $PYTHON scripts/train_ccd.py $ARGS"
"$PYTHON" scripts/train_ccd.py $ARGS

echo "[INFO] Done. Check output/ for weights and ZIP."
