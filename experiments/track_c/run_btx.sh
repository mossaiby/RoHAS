#!/usr/bin/env bash
# Scale study: the Branch-Train-MiX baseline (docs/roadmap.md, pre-registered 2026-11-25).
#   setsid nohup bash experiments/track_c/run_btx.sh > logs/btx.log 2>&1 &
# For each point: merge the four branched experts, fine-tune at the gate's training FLOPs (btx)
# and at ten times them (btx10), then evaluate.py and diag_mixed.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TRACK_C_DATA=track_c_v2f HF_HUB_OFFLINE=1
PY=.venv/bin/python
CK=checkpoints/track_c
mkdir -p logs

wait_gpu() {   # the GPU is shared: start only with $1 MiB free
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 120; done
}
retry() {      # $1 MiB free, then the command; on failure (an out-of-memory error on the shared GPU) try again
  local need=$1 n=0; shift
  until wait_gpu $need && "$@"; do
    n=$((n + 1)); [ $n -ge 40 ] && { echo "btx: giving up after $n failures: $*"; exit 1; }
    echo "btx: failed ($n), retrying in 120 s: $*"; sleep 120
  done
}

for pt in 16f 17f; do
  need=2500; [ $pt = 17f ] && need=4500   # MiB free before starting (micro-batch 4); retry covers the rest
  p=sc$pt
  gate_flops=$($PY -c "import json; print(json.load(open('results/track_c/train_${p}_gate_conf.json'))['data']['flops'])")
  [ -f $CK/${p}_btx_init.pt ] || $PY experiments/track_c/btx.py --point $pt
  for v in btx btx10; do
    [ $v = btx ] && f=$gate_flops || f=$($PY -c "print(10 * $gate_flops)")
    if [ ! -f $CK/${p}_$v.pt ]; then
      retry $need $PY experiments/track_c/train.py --name ${p}_$v --init $CK/${p}_btx_init.pt \
        --mix mixed=4 lang=1 facts=1 math=1 code=1 --flops "$f" --lr 1e-3 --batch-tokens 8192 --micro 4
    fi
  done
  retry $need $PY experiments/track_c/evaluate.py --tag ${p}_btx --systems model:${p}_btx model:${p}_btx10
  for dset in mixed_val mixed_long_val mixed_inline_val mixed_marked_val; do
    tag=${p}_btx; [ $dset = mixed_val ] || tag=${dset:6:-4}_${p}_btx
    retry $need $PY experiments/track_c/diag_mixed.py --tag $tag --data $dset --models ${p}_btx ${p}_btx10 --gates ${p}_gate_conf
  done
done
echo "btx: all done"
