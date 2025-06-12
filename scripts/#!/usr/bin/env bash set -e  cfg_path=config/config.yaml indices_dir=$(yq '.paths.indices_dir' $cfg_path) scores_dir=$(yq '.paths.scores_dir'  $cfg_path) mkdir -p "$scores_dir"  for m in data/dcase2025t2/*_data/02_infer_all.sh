#!/usr/bin/env bash
set -e

cfg_path=config/config.yaml
indices_dir=$(yq '.paths.indices_dir' $cfg_path)
scores_dir=$(yq '.paths.scores_dir'  $cfg_path)
mkdir -p "$scores_dir"

for m in data/dcase2025t2/*_data/raw/*; do
  [[ -d "${m}/test" ]] || continue
  name=$(basename "$m")
  echo "=== infer $name ==="
  python -m src.beats_knn_asd infer \
         --wav-dir   "${m}/test" \
         --index-file "${indices_dir}/${name}.npz" \
         --csv       "${scores_dir}/${name}_scores.csv"
done