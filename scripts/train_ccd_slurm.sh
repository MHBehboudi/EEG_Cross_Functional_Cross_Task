#!/bin/bash -l
#SBATCH --job-name=eeg2025_deme
#SBATCH --partition=a30-2.12gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/mxb190076/EEG_Cross_Functional_Cross_Task
#SBATCH --output=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.out
#SBATCH --error=/work/mxb190076/EEG_Cross_Functional_Cross_Task/slurm-%j.err

set -euo pipefail

# --- Env selection (no surprises) ---
# Prefer the repo's local venv if present; otherwise try your fixed conda env.
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  module purge || true
  module load miniconda || true
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "/work/mxb190076/eeg2025"; then
      conda activate /work/mxb190076/eeg2025
    else
      echo "[WARN] Conda env /work/mxb190076/eeg2025 not found; continuing without conda."
    fi
  fi
fi

# Threads / logging hygiene
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

mkdir -p data output

# Quick diag (helps debug CUDA/torch mismatches fast)
echo "=== RUNTIME DIAG ==="
which python || true
python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (torch):", getattr(torch.version, "cuda", None))
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY
echo "===================="

# --- Run (DeMEGA backbone + SSL projector), unchanged knobs ---
python cli.py \
  --mini \
  --epochs 30 --batch_size 128 --num_workers 4 \
  --data_dir ./data --out_dir ./output --save_zip \
  --arch demega --use_projector 1 \
  --d_model 128 --tf_depth 2 --tf_heads 4 --tf_mlp 2.0 --dropout 0.10 --token_k 9 \
  --n_clusters 20 --pcs_per_cluster 3 \
  --ssl_epochs 10 --ssl_steps 150 --ssl_batch 16 --ssl_crop 150
