"""Track C: robustness of the circuit result (seeds, held-out items, independent experts).

    python experiments/track_c/run_robustness.py [--only heldout gates eval heldout_new]

The final gate design (two-level atom gate, circuit confidence in nats, vetoes) was chosen
after looking at the arithmetic results of eval seed 0. This script measures it on items it
was never tuned on and adds variance estimates:

  heldout      arithmetic and word problems with --eval-seed 1000, 100 items per cell, for
               the existing systems (tag main_heldout)
  gates        gate seeds 1 and 2 of the seeded confidence-only control and of the final
               atom gate; the confidence-only control and the atom gate on the independent
               experts (seed 0)
  eval         full evaluation of those gates on the development items (tag main_circuit_seeds)
  heldout_new  the held-out items for those gates (tag main_heldout_new)

Gate seeds vary the gate's initialisation and its training batches; the experts are fixed
(retraining them per seed is out of budget), so the spread is gate variance only.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
HELD = ["--eval-seed", "1000", "--per-cell", "100", "--skip", "loss", "facts", "routing"]


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def todo(tag, systems):
    """Systems not yet in eval_{tag}.json (a run interrupted part-way resumes where it stopped)."""
    path = os.path.join(ROOT, "results", "track_c", f"eval_{tag}.json")
    done = json.load(open(path))["data"]["systems"] if os.path.exists(path) else {}
    return [s for s in systems if s not in done]


def evaluate(tag, systems, extra=()):
    rest = todo(tag, systems)
    if rest:
        run([os.path.join(HERE, "evaluate.py"), "--tag", tag, *extra, "--systems", *rest])


def gate(name, experts, extra, seed):
    if not os.path.exists(os.path.join(CKPT, f"{name}.pt")):
        run([os.path.join(HERE, "gate.py"), "--name", name, "--mode", "prob", "--conf", *extra,
             "--tokens", "6e6", "--seed", str(seed), "--experts", *experts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["heldout", "gates", "eval", "heldout_new"]
    seeded = [f"main_seeded_{d}" for d in ["lang", "facts", "math", "code"]]
    indep = [f"main_{d}" for d in ["lang", "facts", "math", "code"]]
    atom = ["--atom", "--veto"]
    new = {}                                            # gate name -> (experts, extra, seed)
    for s in (1, 2):
        new[f"main_seeded_gate_conf_s{s}"] = (seeded, [], s)
        new[f"main_seeded_gate_atom_s{s}"] = (seeded + ["circuit"], atom, s)
    new["main_gate_conf"] = (indep, [], 0)
    new["main_gate_atom"] = (indep + ["circuit"], atom, 0)
    new_systems = [f"gate:{g}:prob" for g in new] + ["gate:main_gate_atom:prob_noveto"]

    if "heldout" in steps:
        evaluate("main_heldout", [
             "model:main_mono_s", "model:main_mono_m", "model:main_seeded_math", "model:main_math",
             "gate:main_seeded_gate_prob:prob", "gate:main_gate_prob:prob",
             "gate:main_seeded_gate_conf:prob", "gate:main_seeded_gate_split:prob",
             "gate:main_seeded_gate_atom:prob", "gate:main_seeded_gate_atom:prob_noveto",
             "gate:main_seeded_gate_atom:hard"], HELD)
    if "gates" in steps:
        for g, (ex, extra, s) in new.items():
            gate(g, ex, extra, s)
    if "eval" in steps:
        evaluate("main_circuit_seeds", new_systems)
    if "heldout_new" in steps:
        evaluate("main_heldout_new", new_systems, HELD)


if __name__ == "__main__":
    main()
