#!/bin/bash -l
#SBATCH --job-name=eeg2025_train
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

# ---------- Config (override at submit time) ----------
: "${ARCH:=transformer}"     # transformer | gnn | eegnex
: "${USE_PROJECTOR:=1}"      # 1 for transformer/eegnex, 0 for gnn
: "${EPOCHS:=30}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=4}"
: "${SEED:=2025}"
: "${SSL_EPOCHS:=10}"
: "${SSL_STEPS:=150}"
: "${SSL_BATCH:=16}"
: "${SSL_CROP:=150}"
: "${N_CLUSTERS:=20}"
: "${PCS_PER_CLUSTER:=3}"
: "${MINI:=1}"               # 1 = mini mode, 0 = full

# ---------- Environment ----------
module purge || true
# Prefer project venv if it exists; otherwise fall back to your conda env
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  module load miniconda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /work/mxb190076/eeg2025
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# Make your package importable
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Where Codabench will look if you test locally with their runner
export EEG2025_RES_DIR="$PWD/output"

mkdir -p data output

# ---------- Build CLI flags ----------
MINI_FLAG=
if [[ "$MINI" == "1" ]]; then
  MINI_FLAG="--mini"
fi

# If user picked GNN, ignore projector (it’s electrode-graph by default)
if [[ "$ARCH" == "gnn" ]]; then
  USE_PROJECTOR=0
fi

echo "==== RUN CONFIG ===="
echo "ARCH=$ARCH  USE_PROJECTOR=$USE_PROJECTOR"
echo "EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  WORKERS=$NUM_WORKERS  SEED=$SEED"
echo "SSL: epochs=$SSL_EPOCHS steps=$SSL_STEPS batch=$SSL_BATCH crop=$SSL_CROP"
echo "Projector: K=$N_CLUSTERS  PCs/cluster=$PCS_PER_CLUSTER"
echo "Mini=$MINI  PWD=$PWD"
echo "===================="

# ---------- Train + build ZIP ----------
srun python cli.py \
  $MINI_FLAG \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --data_dir ./data \
  --out_dir ./output \
  --save_zip \
  --arch "$ARCH" \
  --use_projector "$USE_PROJECTOR" \
  --n_clusters "$N_CLUSTERS" \
  --pcs_per_cluster "$PCS_PER_CLUSTER" \
  --ssl_epochs "$SSL_EPOCHS" \
  --ssl_steps "$SSL_STEPS" \
  --ssl_batch "$SSL_BATCH" \
  --ssl_crop "$SSL_CROP"

echo "Artifacts in ./output:"
ls -lh output || true
