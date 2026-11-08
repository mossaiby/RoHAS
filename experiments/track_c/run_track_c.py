"""Track C end to end at one scale: experts, compute-matched monolithic baselines, gates, eval.

    python experiments/track_c/run_track_c.py --scale pilot     # T = 10M tokens per expert
    python experiments/track_c/run_track_c.py --scale main      # T = 100M tokens per expert

Compute matching. Each expert (size s) trains on T tokens of its own data. The gate is
trained on G tokens of mixed documents, which costs G forward passes of every expert plus
the gate itself. Both monolithic baselines get the same total training FLOPs:
  mono_s  one size-s model, the same architecture as an expert, about 4T tokens
  mono_m  one size-m model (about 4x the parameters), about T tokens
and they see the same data: the four sources in equal shares plus the G mixed tokens.
Variants of the experts (the baselines and their FLOPs are the same for both)
  independent  each expert trains from scratch on its own data only
  seeded       as in Branch-Train-Merge (Li et al., 2022): one size-s seed model trains on the
               four sources for a fraction f of the expert budget (4fT tokens), then four
               copies branch and each trains (1-f)T further tokens on its own data
Steps whose checkpoint exists are skipped, so the script can be re-run after a stop.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c import model as M  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
SCALES = {"smoke": 0.3e6, "pilot": 10e6, "main": 100e6}
GATE_TOKENS = {"smoke": 0.1e6, "pilot": 6e6, "main": 6e6}


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="pilot", choices=list(SCALES))
    ap.add_argument("--only", nargs="*", help="subset of steps: experts mono gates eval")
    ap.add_argument("--variant", default="independent", choices=["independent", "seeded"])
    ap.add_argument("--seed-frac", type=float, default=0.5)
    ap.add_argument("--prefix", help="run name prefix (default: the scale)")
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    T = SCALES[a.scale]
    p = a.prefix or a.scale
    steps = a.only or ["experts", "mono", "gates", "eval"]
    vocab = json.load(open(os.path.join(ROOT, "data", "track_c", "meta.json")))["vocab"]
    fs = M.make("s", vocab).flops_per_token()
    fm = M.make("m", vocab).flops_per_token()
    gate_params = 2 * 4 * 384 * 2 + (2 * 4 * 384) * 256 + 256 + 256 * 4 + 4   # LayerNorm + MLP
    experts_flops = 4 * fs * T
    gate_flops = GATE_TOKENS[a.scale] * (4 * fs / 3 + 6 * gate_params)
    budget = experts_flops + gate_flops
    G = GATE_TOKENS[a.scale]

    def mix(f):
        """Equal shares of the four sources plus exactly G mixed tokens for a model with f FLOPs/token."""
        share = G / (budget / f)
        return ["lang=1", "facts=1", "math=1", "code=1", f"mixed={4 * share / (1 - share):.5f}"]
    print(f"scale {p}: T={T / 1e6:.0f}M, experts {experts_flops:.3e} FLOPs + gate {gate_flops:.3e} "
          f"= {budget:.3e}; mono_s {budget / fs / 1e6:.0f}M tokens, mono_m {budget / fm / 1e6:.0f}M tokens",
          flush=True)

    def have(n):
        return os.path.exists(os.path.join(CKPT, f"{n}.pt"))

    hp = ["--batch-tokens", str(a.batch_tokens)] + (["--lr", str(a.lr)] if a.lr else [])
    ep = p if a.variant == "independent" else f"{p}_seeded"
    if "experts" in steps:
        if a.variant == "independent":
            for d in ["lang", "facts", "math", "code"]:
                if not have(f"{p}_{d}"):
                    run([os.path.join(HERE, "train.py"), "--name", f"{p}_{d}", "--size", "s",
                         "--mix", f"{d}=1", "--tokens", str(T), *hp])
        else:
            f = a.seed_frac
            if not have(f"{ep}_seed"):
                run([os.path.join(HERE, "train.py"), "--name", f"{ep}_seed", "--size", "s",
                     "--mix", "lang=1", "facts=1", "math=1", "code=1", "--tokens", str(4 * f * T), *hp])
            for d in ["lang", "facts", "math", "code"]:
                if not have(f"{ep}_{d}"):
                    run([os.path.join(HERE, "train.py"), "--name", f"{ep}_{d}", "--init",
                         os.path.join(CKPT, f"{ep}_seed.pt"), "--mix", f"{d}=1", "--tokens", str((1 - f) * T),
                         *hp, *([] if a.lr else ["--lr", "1e-3"])])
    if "mono" in steps:
        if not have(f"{p}_mono_s"):
            run([os.path.join(HERE, "train.py"), "--name", f"{p}_mono_s", "--size", "s",
                 "--mix", *mix(fs), "--flops", str(budget), *hp])
        if not have(f"{p}_mono_m"):
            run([os.path.join(HERE, "train.py"), "--name", f"{p}_mono_m", "--size", "m", "--micro", "8",
                 "--mix", *mix(fm), "--flops", str(budget), "--batch-tokens", str(a.batch_tokens)])
    experts = [f"{ep}_{d}" for d in ["lang", "facts", "math", "code"]]
    if "gates" in steps:
        for mode in ["prob", "logit"]:
            if not have(f"{ep}_gate_{mode}"):
                run([os.path.join(HERE, "gate.py"), "--name", f"{ep}_gate_{mode}", "--mode", mode,
                     "--tokens", str(G), "--experts", *experts])
    if "eval" in steps:
        systems = [f"model:{e}" for e in experts] + \
                  [f"gate:{ep}_gate_prob:prob", f"gate:{ep}_gate_prob:hard", f"gate:{ep}_gate_logit:logit"]
        if a.variant == "independent":
            systems = [f"model:{p}_mono_s", f"model:{p}_mono_m"] + systems
        else:
            systems = [f"model:{ep}_seed"] + systems
        run([os.path.join(HERE, "evaluate.py"), "--tag", p, "--systems", *systems])


if __name__ == "__main__":
    main()
