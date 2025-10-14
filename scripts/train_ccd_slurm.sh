#!/bin/bash -l
#SBATCH --job-name=eeg2025_train_ccd
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
# If your cluster needs a Python module, uncomment and set it:
# module load python/3.12

# Activate the repo-local venv
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Expose src/ to Python
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p data output

# --- Train a quick mini run to produce weights + projection ---
python one_click_train_and_package.py \
  --mini --epochs 3 --batch_size 128 --num_workers 4 \
  --n_clusters 20 --pcs_per_cluster 3 \
  --ssl_epochs 5 --ssl_steps 80 --ssl_batch 16 --ssl_crop 150

# --- Build a flat ZIP from CURRENT repo + generated artifacts ---
rm -f output/submission-to-upload.zip
tmpdir="$(mktemp -d -p "$SLURM_SUBMIT_DIR" tmp_submit.XXXXXX)"
cp "$SLURM_SUBMIT_DIR/submission.py" "$tmpdir/"

# include weights if present
[ -f output/weights_challenge_1.pt ] && cp output/weights_challenge_1.pt "$tmpdir/"
[ -f output/weights_challenge_2.pt ] && cp output/weights_challenge_2.pt "$tmpdir/"

# include the projection (needed to match channel dim at inference)
[ -f output/projection.npy ] && cp output/projection.npy "$tmpdir/"

# guard: ensure submission.py does NOT reference /app/output
if grep -q "/app/output" "$tmpdir/submission.py"; then
  echo "[ERROR] submission.py still references /app/output. Use /app/input/res." >&2
  exit 3
fi

# create flat zip
( cd "$tmpdir" && zip -r "$SLURM_SUBMIT_DIR/output/submission-to-upload.zip" . )

# verify contents
echo "[INFO] ZIP contents:"
unzip -l "$SLURM_SUBMIT_DIR/output/submission-to-upload.zip"

echo "[DONE] Upload: $SLURM_SUBMIT_DIR/output/submission-to-upload.zip"
