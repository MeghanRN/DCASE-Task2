# DCASE-Task2

# DCASE 2025 Task 2 – BEATs + K-NN Baseline

Minimal, first-shot system that beats the official auto-encoder baseline
using a pre-trained BEATs transformer as feature extractor and a
cosine-distance K-nearest-neighbour detector.

## Quick start

```bash
# create env and install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# build reference embeddings
bash scripts/01_build_indices.sh                       # loops all machines

# run inference
bash scripts/02_infer_all.sh                           # writes work/scores/*

# package submission
bash scripts/03_make_submission.sh beats_knn_run1      # → submission/run1.zip
