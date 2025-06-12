#!/usr/bin/env bash
set -e

cfg_path=config/config.yaml
indices_dir=$(yq '.paths.indices_dir' $cfg_path)

mkdir -p "$indices_dir"

for m in data/dcase2025t2/*_data/raw/*; do
  [[ -d "${m}/train" ]] || continue
  name=$(basename "$m")
  echo "=== index $name ==="
  python -m src.beats_knn_asd build-index \
         --wav-dir "${m}/train" \
         --index-file "${indices_dir}/${name}.npz"
done