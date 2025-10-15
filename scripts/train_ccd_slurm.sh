#!/bin/bash
#SBATCH --job-name=eegcfct-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --partition=gpu             # <-- keep your original partition if different
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

# --- If you load modules on your cluster, keep the same lines you used before ---
# module purge
# module load cuda/12.2  # (Only if you previously loaded CUDA; otherwise leave it out)

# Go to the directory where you ran `sbatch`
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate your existing virtualenv (unchanged)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Make sure Python can import the src/ package (safe even if PYTHONPATH is empty)
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# (Optional) keep threads low for deterministic behavior
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Where outputs go
OUT_DIR="$PWD/output"
DATA_DIR="$PWD/data"
mkdir -p "$OUT_DIR" "$DATA_DIR"

# ---- Tunables (edit if you want) ----
ARCH="demega"          # demega | eegnex | transformer
USE_PROJECTOR=1        # 1 = use SSL+cluster+PCA projector, 0 = no projector
EPOCHS=30
BATCH_SIZE=128
WORKERS=4
SEED=2025

# SSL/Projector
SSL_EPOCHS=10
SSL_STEPS=150
SSL_BATCH=16
SSL_CROP=150
N_CLUSTERS=20
PCS_PER_CLUSTER=3

# Use --mini for quick runs; remove it for full data
MINI_FLAG="--mini"

echo "==== RUN CONFIG ===="
echo "ARCH=${ARCH}  USE_PROJECTOR=${USE_PROJECTOR}"
echo "EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  WORKERS=${WORKERS}  SEED=${SEED}"
echo "SSL: epochs=${SSL_EPOCHS} steps=${SSL_STEPS} batch=${SSL_BATCH} crop=${SSL_CROP}"
echo "Projector: K=${N_CLUSTERS}  PCs/cluster=${PCS_PER_CLUSTER}"
echo "Mini=$([ -n "$MINI_FLAG" ] && echo 1 || echo 0)  PWD=$PWD"
echo "===================="

# Kick off training and build the Codabench zip in ./output
python cli.py \
  ${MINI_FLAG} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${WORKERS} \
  --seed ${SEED} \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --save_zip \
  --n_clusters ${N_CLUSTERS} \
  --pcs_per_cluster ${PCS_PER_CLUSTER} \
  --ssl_epochs ${SSL_EPOCHS} \
  --ssl_steps ${SSL_STEPS} \
  --ssl_batch ${SSL_BATCH} \
  --ssl_crop ${SSL_CROP} \
  --arch ${ARCH} \
  --use_projector ${USE_PROJECTOR}
