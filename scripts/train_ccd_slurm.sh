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
# module load python/3.12     # uncomment if your cluster requires
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Make sure Python finds src/
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p data output

# --- Train a quick mini run to produce weights ---
python cli.py \
  --mini --epochs 3 --batch_size 128 --num_workers 4 \
  --data_dir ./data --out_dir ./output --save_zip \
  --cluster_mode ssl_pca --n_clusters 20 --pcs_per_cluster 3 \
  --ssl_epochs 5 --ssl_steps 80 --ssl_batch 16 --ssl_crop 150

# --- Build Codabench ZIP from CURRENT repo files (flat layout) ---
rm -f output/submission-to-upload.zip
tmpdir="$(mktemp -d -p "$SLURM_SUBMIT_DIR" tmp_submit.XXXXXX)"
cp "$SLURM_SUBMIT_DIR/submission.py" "$tmpdir/"

# copy weights if they exist
[ -f output/weights_challenge_1.pt ] && cp output/weights_challenge_1.pt "$tmpdir/"
[ -f output/weights_challenge_2.pt ] && cp output/weights_challenge_2.pt "$tmpdir/"

# guard: ensure submission.py does NOT reference /app/output
if grep -q "/app/output" "$tmpdir/submission.py"; then
  echo "[ERROR] submission.py still references /app/output/* . Fix it to use /app/input/res/ and re-run." >&2
  exit 3
fi

# create flat zip
( cd "$tmpdir" && zip -r "$SLURM_SUBMIT_DIR/output/submission-to-upload.zip" . )

# verify contents
echo "[INFO] ZIP contents:"
unzip -l "$SLURM_SUBMIT_DIR/output/submission-to-upload.zip"

echo "[INFO] Checking embedded submission.py for /app/output (should be none):"
if unzip -p "$SLURM_SUBMIT_DIR/output/submission-to-upload.zip" submission.py | grep -n "/app/output" ; then
  echo "[ERROR] Bad path found inside the ZIP. Aborting." >&2
  exit 4
else
  echo "[OK] submission.py inside ZIP correctly uses /app/input/res"
fi

echo "[DONE] Upload this file to Codabench:"
echo "       $SLURM_SUBMIT_DIR/output/submission-to-upload.zip"
