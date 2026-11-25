#!/usr/bin/env bash
# Scale study, first step (docs/roadmap.md, "Scale study"): build corpus v2, check the pipeline
# on a tiny point, then run the 5e16 and 5e17 points. Logs in logs/.
#   nohup bash experiments/track_c/run_scale_v2.sh > logs/scale_v2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
export TRACK_C_DATA=track_c_v2 HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p logs
[ -f data/track_c_v2/meta.json ] || $PY experiments/track_c/data.py --corpus v2
[ -f data/track_c_v2/ctxv_code_train.bin ] || $PY experiments/track_c/data_context.py --varied --tokens 42e6
[ -f data/track_c_v2/mixed_marked_val.bin ] || $PY experiments/track_c/data_mixed_ood.py
$PY experiments/track_c/run_scale.py --point smoke
$PY experiments/track_c/run_scale.py --point 16 17
echo "scale v2: all done"
