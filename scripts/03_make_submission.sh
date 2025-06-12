
#!/usr/bin/env bash
set -e
run=$1
python -m src.make_submission "$run" --scores-glob "work/scores/*_scores.csv"