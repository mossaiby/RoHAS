#!/usr/bin/env bash
# Scale study, step 1b (docs/roadmap.md): corpus v2 with 600 biographies per person, then the
# 5e17 and 5e16 points again.
#   nohup bash experiments/track_c/run_scale_v2f.sh > logs/scale_v2f.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
export TRACK_C_DATA=track_c_v2f HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p logs
[ -f data/track_c_v2f/meta.json ] || $PY experiments/track_c/data.py --grow-facts track_c_v2 --bios 600
$PY experiments/track_c/run_scale.py --point 17f 16f
echo "scale v2f: all done"
