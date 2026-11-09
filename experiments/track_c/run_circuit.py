"""Track C + Track A: the compiled arithmetic circuits as a fifth expert under the learned gate.

    python experiments/track_c/run_circuit.py --prefix main [--variant seeded]

Gates (prob mixing, trained on the same mixed documents and token budget as the main gates;
no routing labels):
  {ep}_gate_conf   the four trained experts, with expert-confidence features (control)
  {ep}_gate_circ   the four trained experts plus the compiled circuits, same features, one
                   joint head (ablation: its trust in the circuit fades after about 6 answer
                   digits, the longest answers in the gate's training data)
  {ep}_gate_split  as gate_circ, but the circuit's weight comes from a separate head on the
                   confidence features only (gate.py --split; ablation: the head still sees
                   the LM experts' confidence, which rises, wrongly, on long numbers)
  {ep}_gate_atom   the final system: two-level gate, the circuit weighted by its own
                   confidence only, plus the circuit's vetoes (gate.py --atom --veto);
                   evaluated with and without the vetoes; linear atom head with its own
                   learning rate, reading the circuit's confidence in nats (tag {p}_circuit_atom)
  {ep}_gate_atom_lin  ablation: linear head on confidence divided by log V; weight 0.40 where
                   active (tag {p}_circuit_atom_lin)
  {ep}_gate_atom_mlp  ablation: the same with a small MLP atom head at the gate's learning
                   rate; the circuit's weight stays at 0.39 where it is active (recorded in
                   tag {p}_circuit under this name; trained before --atom-head existed)
Evaluation tag {p}_circuit: losses, fact recall, arithmetic up to 32 digits, word problems,
routing. Arithmetic is also re-scored for the main systems (tag {p}) with the corrected
prompt (see evaluate.py), so every system is measured the same way.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="main")
    ap.add_argument("--variant", default="seeded", choices=["independent", "seeded"])
    ap.add_argument("--only", nargs="*", help="gates eval atom_eval rescore")
    a = ap.parse_args()
    p = a.prefix
    ep = p if a.variant == "independent" else f"{p}_seeded"
    steps = a.only or ["gates", "eval", "atom_eval", "rescore"]
    experts = [f"{ep}_{d}" for d in ["lang", "facts", "math", "code"]]
    gates = {"conf": (experts, []), "circ": (experts + ["circuit"], []), "split": (experts + ["circuit"], ["--split"]),
             "atom": (experts + ["circuit"], ["--atom", "--veto"])}
    if "gates" in steps:
        for g, (ex, extra) in gates.items():
            if not os.path.exists(os.path.join(CKPT, f"{ep}_gate_{g}.pt")):
                run([os.path.join(HERE, "gate.py"), "--name", f"{ep}_gate_{g}", "--mode", "prob", "--conf",
                     *extra, "--tokens", "6e6", "--experts", *ex])
    if "eval" in steps:
        run([os.path.join(HERE, "evaluate.py"), "--tag", f"{p}_circuit", "--systems",
             f"gate:{ep}_gate_conf:prob", f"gate:{ep}_gate_circ:prob", f"gate:{ep}_gate_circ:hard",
             f"gate:{ep}_gate_split:prob", f"gate:{ep}_gate_split:hard",
             f"gate:{ep}_gate_atom_mlp:prob", f"gate:{ep}_gate_atom_mlp:prob_noveto", f"gate:{ep}_gate_atom_mlp:hard"])
    if "atom_eval" in steps:
        run([os.path.join(HERE, "evaluate.py"), "--tag", f"{p}_circuit_atom", "--systems",
             f"gate:{ep}_gate_atom:prob", f"gate:{ep}_gate_atom:prob_noveto", f"gate:{ep}_gate_atom:hard"])
    if "rescore" in steps:
        systems = [f"model:{p}_mono_s", f"model:{p}_mono_m", f"model:{ep}_math", f"gate:{ep}_gate_prob:prob"]
        run([os.path.join(HERE, "evaluate.py"), "--tag", p, "--skip", "loss", "facts", "--systems", *systems])


if __name__ == "__main__":
    main()
