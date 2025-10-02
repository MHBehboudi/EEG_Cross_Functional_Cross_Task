# EEG_Cross_Functional_Cross_Task

Reproducible training and Codabench submission for EEG 2025.

## Layout
- \src/eegcfct\: library code (models, data, train utils)
- \scripts/\: entrypoints & HPC job scripts
- \submission.py\: Codabench entry (kept at repo root)
- \configs/\: experiment configs (YAML)
- \output/\: artifacts (ignored)

## Quickstart (local)
\\\ash
conda env create -f environment.yml     # or: pip install -r requirements.txt
python scripts/train_ccd.py --mini
\\\

## Submit
- Zip: \submission.py\, \weights_challenge_1.pt\, \weights_challenge_2.pt\ (flat, no folders)
