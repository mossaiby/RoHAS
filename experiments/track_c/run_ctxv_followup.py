"""Track C: gate seeds and the add-facts test for the varied-exposure experts (main_ctxv20).

    python experiments/track_c/run_ctxv_followup.py [--only gates add eval heldout diag]

  gates    gate seeds 1 and 2 of main_ctxv20_gate_conf and main_ctxv20_gate_atom (experts fixed)
  add      add 500 people by updating only the facts expert, with the seeded recipe of
           run_add_facts.py unchanged: "new" = facts_new only, 12.5M tokens; "replay" =
           facts_new and facts in equal shares, 25M tokens. The gates keep their weights
           and get the updated facts expert. The seeded conf gate gets the seeded updated
           experts (main_seeded_add_<v>_expert) the same way, for a like-for-like comparison.
           The update has no context exposure: it tests whether the mixed-text gain survives
           an ordinary facts update. "replay_ctx" keeps the exposure: the same 25M tokens and
           the same new-fact share (1/2), with 0.3 old facts and 0.2 ctxvu_facts_train
           (data_context.py --update; own people old and new, loss on facts tokens only).
  eval     tag main_ctxv20_seeds (gate seeds, development items) and main_ctxv20_add_facts
           (updated systems; arithmetic skipped, as in run_add_facts.py)
  heldout  held-out arithmetic and word problems for the three atom-gate seeds (tag
           main_ctxv20_heldout; eval seed 1000, 100 items per cell)
  diag     mixed-text loss on mixed_val and the data_mixed_ood.py formats for the new conf
           gates and the updated systems (tags <format>_main_ctxv20_followup)
"""
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c.data_mixed_ood import EVAL  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
P = "main_ctxv20"
HELD = ["--eval-seed", "1000", "--per-cell", "100", "--skip", "loss", "facts", "routing"]
UPDATES = {"new": (["facts_new=1"], 12509184, []), "replay": (["facts_new=1", "facts=1"], 25026560, []),
           "replay_ctx": (["facts_new=0.5", "facts=0.3", "ctxvu_facts=0.2"], 25026560, ["--own", "facts"])}


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def have(n):
    return os.path.exists(os.path.join(CKPT, f"{n}.pt"))


def evaluate(tag, systems, extra=()):
    path = os.path.join(ROOT, "results", "track_c", f"eval_{tag}.json")
    done = json.load(open(path))["data"]["systems"] if os.path.exists(path) else {}
    rest = [s for s in systems if s not in done]
    if rest:
        run([os.path.join(HERE, "evaluate.py"), "--tag", tag, *extra, "--systems", *rest])


def swap(gate, old, new, out):
    g = torch.load(os.path.join(CKPT, f"{gate}.pt"), map_location="cpu", weights_only=False)
    assert old in g["experts"], (gate, g["experts"])
    g["experts"] = [new if e == old else e for e in g["experts"]]
    torch.save(g, os.path.join(CKPT, f"{out}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["gates", "add", "eval", "heldout", "diag"]
    experts = [f"{P}_{t}" for t in ["lang", "facts", "math", "code"]]
    seeds = [f"{P}_gate_{g}_s{s}" for s in (1, 2) for g in ("conf", "atom")]
    if "gates" in steps:
        for s in (1, 2):
            for g, extra, ex in [("conf", [], experts), ("atom", ["--atom", "--veto"], experts + ["circuit"])]:
                if not have(f"{P}_gate_{g}_s{s}"):
                    run([os.path.join(HERE, "gate.py"), "--name", f"{P}_gate_{g}_s{s}", "--mode", "prob", "--conf",
                         *extra, "--tokens", "6e6", "--seed", str(s), "--experts", *ex])
    if "add" in steps:
        for v, (mix, n, extra) in UPDATES.items():
            name = f"{P}_add_{v}_expert"
            if not have(name):
                run([os.path.join(HERE, "train.py"), "--name", name, "--size", "s", "--init",
                     os.path.join(CKPT, f"{P}_facts.pt"), "--mix", *mix, "--tokens", str(n),
                     "--micro", "16", "--batch-tokens", "8192", "--evals", "4", *extra])
            for g in ("conf", "atom"):
                swap(f"{P}_gate_{g}", f"{P}_facts", name, f"{P}_gate_{g}_add_{v}")
            if v != "replay_ctx":
                swap("main_seeded_gate_conf", "main_seeded_facts", f"main_seeded_add_{v}_expert",
                     f"main_seeded_gate_conf_add_{v}")
    added = [f"gate:{P}_gate_conf_add_{v}:prob" for v in UPDATES] + [f"gate:{P}_gate_atom_add_replay:prob"] + \
            [f"gate:main_seeded_gate_conf_add_{v}:prob" for v in ("new", "replay")] + \
            [f"gate:{P}_gate_atom_add_replay_ctx:prob"]
    if "eval" in steps:
        evaluate(f"{P}_seeds", [f"gate:{g}:prob" for g in seeds])
        evaluate(f"{P}_add_facts", added, ["--skip", "math"])
    if "heldout" in steps:
        evaluate(f"{P}_heldout", [f"gate:{P}_gate_atom:prob", f"gate:{P}_gate_atom_s1:prob",
                                  f"gate:{P}_gate_atom_s2:prob", f"gate:{P}_gate_conf:prob"], HELD)
    if "diag" in steps:
        gates = [f"{P}_gate_conf_s1", f"{P}_gate_conf_s2", f"{P}_gate_conf_add_replay", "main_seeded_gate_conf_add_replay",
                 f"{P}_gate_conf_add_replay_ctx"]
        for d in ["mixed_val"] + [f"mixed_{f}_val" for f in EVAL]:
            tag = f"{d[6:-4] if d != 'mixed_val' else 'mixed'}_{P}_followup"
            run([os.path.join(HERE, "diag_mixed.py"), "--tag", tag, "--data", d, "--gates", *gates])


if __name__ == "__main__":
    main()
