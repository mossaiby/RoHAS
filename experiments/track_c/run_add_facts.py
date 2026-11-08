"""Track C, modularity: the cost of adding 500 new people after training.

    python experiments/track_c/run_add_facts.py --prefix pilot8k [--variant seeded]

Updates (same new-fact tokens for every system: as many passes over facts_new_train as
the original facts expert made over facts_train)
  expert   fine-tune only the facts expert; the gate and the other experts are unchanged
  mono_s   fine-tune the size-s monolithic model
  mono_m   fine-tune the size-m monolithic model
each in two variants: new facts only, and new facts plus an equal share of old facts (replay).
A third variant, replay_all, is for the monolithic models only: half the update is new facts
and the other half replays all four original sources in equal shares, at the same token
count as replay. It is the fair replay baseline for a model that holds every data type.
The evaluation reports recall of new and old facts and the loss on every data type, so
forgetting outside facts is visible; train_*.json gives the FLOPs of each update.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

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
    ap.add_argument("--prefix", default="pilot8k", help="run prefix used by run_track_c.py")
    ap.add_argument("--variant", default="independent", choices=["independent", "seeded"])
    ap.add_argument("--only", nargs="*", help="train eval")
    ap.add_argument("--variants", nargs="*", help="subset of: new replay replay_all (default: all)")
    a = ap.parse_args()
    p = a.prefix
    ep = p if a.variant == "independent" else f"{p}_seeded"
    tag = f"{ep}_add_facts"
    steps = a.only or ["train", "eval"]
    meta = json.load(open(os.path.join(ROOT, "data", "track_c", "meta.json")))
    ck = torch.load(os.path.join(CKPT, f"{ep}_facts.pt"), map_location="cpu", weights_only=False)
    passes = ck["tokens"] / meta["tokens"]["facts_train"]
    new_tokens = passes * meta["tokens"]["facts_new_train"]
    print(f"facts expert made {passes:.1f} passes; update = {new_tokens / 1e6:.2f}M new-fact tokens", flush=True)

    variants = {"new": (["facts_new=1"], 1.0), "replay": (["facts_new=1", "facts=1"], 2.0),
                "replay_all": (["facts_new=2", "facts=0.5", "lang=0.5", "math=0.5", "code=0.5"], 2.0)}
    bases = {"expert": (f"{ep}_facts", "s"), "mono_s": (f"{p}_mono_s", "s"), "mono_m": (f"{p}_mono_m", "m")}
    if a.variants:
        variants = {v: variants[v] for v in a.variants}
    if "train" in steps:
        for v, (mix, scale_tokens) in variants.items():
            for b, (init, size) in bases.items():
                if v == "replay_all" and b == "expert":
                    continue
                name = f"{ep if b == 'expert' else p}_add_{v}_{b}"
                if os.path.exists(os.path.join(CKPT, f"{name}.pt")):
                    continue
                run([os.path.join(HERE, "train.py"), "--name", name, "--size", size, "--init",
                     os.path.join(CKPT, f"{init}.pt"), "--mix", *mix, "--tokens", str(new_tokens * scale_tokens),
                     "--micro", "8" if size == "m" else "16", "--batch-tokens", "8192", "--evals", "4"])
            # the gated system after the update: same gate weights, facts expert replaced
            for mode in (() if v == "replay_all" else ("prob", "logit")):
                g = torch.load(os.path.join(CKPT, f"{ep}_gate_{mode}.pt"), map_location="cpu", weights_only=False)
                g["experts"] = [f"{ep}_add_{v}_expert" if e == f"{ep}_facts" else e for e in g["experts"]]
                torch.save(g, os.path.join(CKPT, f"{ep}_gate_{mode}_add_{v}.pt"))
    if "eval" in steps:
        systems = [] if a.variants else [f"gate:{ep}_gate_prob:prob", f"model:{p}_mono_s", f"model:{p}_mono_m"]
        for v in variants:
            systems += ([] if v == "replay_all" else [f"gate:{ep}_gate_prob_add_{v}:prob"]) + \
                       [f"model:{p}_add_{v}_mono_s", f"model:{p}_add_{v}_mono_m"]
        run([os.path.join(HERE, "evaluate.py"), "--tag", tag, "--skip", "math", "--systems", *systems])


if __name__ == "__main__":
    main()
