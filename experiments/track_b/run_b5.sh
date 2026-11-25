#!/bin/sh
# Track B, step 5 (pre-registered in docs/roadmap.md): GSM8K with 1.5B experts and the circuit.
# Cache the three experts, train the gate, fit the circuit head, loss check, then 8-shot GSM8K.
# Single-expert systems run first (about 5 GB of GPU memory); the gate systems load all three
# experts (about 14 GB) and wait until 16 GB are free, so they never crowd out another job.
set -e
cd ~/atom-space
P=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q DL-DONE logs/b5_download.log; do sleep 60; done
for e in base15 coder15 math; do $P experiments/track_b/cache.py --expert $e --step 5 --bs 4; done
$P experiments/track_b/gate.py --name b5_gate_s0 --experts base15 coder15 math --step 5
$P experiments/track_b/gsm8k.py --only head loss
for s in base15 base15_circuit base15_calc coder15 math; do
  $P experiments/track_b/gsm8k.py --only gen --systems $s
done
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for s in gate gate_circuit gate_calc; do
  while [ "$(free_mib)" -lt 16000 ]; do sleep 300; done
  $P experiments/track_b/gsm8k.py --only gen --systems $s
done
echo "b5: all done"
