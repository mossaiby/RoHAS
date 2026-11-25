"""Track C scale study: the paper's from-scratch recipe at several compute budgets, corpus v2.

    TRACK_C_DATA=track_c_v2 python experiments/track_c/run_scale.py --point 16 17 [--only ...]

Each point runs, in this order (steps whose checkpoint exists are skipped):
  seed     one expert-size seed model on 4fT tokens of the four types (f = 1/2)
  experts  four branches from the seed, (1-f)T tokens each, a fraction --frac from
           ctxv_<type>_train with the loss on their own type only (run_context.py's recipe)
  mono     mono_s (expert size) and mono_m (4x the parameters) at the systems' total FLOPs,
           the four types in equal shares plus exactly G tokens of mixed_train
  gates    the confidence gate (prob) and the two-level atom gate with the circuit and vetoes,
           on G tokens of mixed_train
  eval     evaluate.py (tag sc<point>)
  diag     diag_mixed.py on mixed_val and the held-out formats (tags sc<point>, <format>_sc<point>)
The recipe (8k-token batches, lr 1e-3 for expert size and 6e-4 for the larger model) is that of
the 5e16 runs in the paper; see docs/roadmap.md, "Scale study".
Requires: data.py --corpus v2, data_context.py --varied --tokens 42e6, data_mixed_ood.py, all
with TRACK_C_DATA=track_c_v2.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c import model as M  # noqa: E402
from experiments.track_c.data import OUT as DATA  # noqa: E402
from experiments.track_c.data_mixed_ood import EVAL  # noqa: E402
from experiments.track_c.gate import Gate  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
TYPES = ["lang", "facts", "math", "code"]
POINTS = {
    "smoke": dict(s="xs", m="s", T=0.5e6, G=0.02e6),       # pipeline check only, not reported
    "16": dict(s="s", m="m", T=100e6, G=6e6),
    "17": dict(s="s17", m="m17", T=350e6, G=60e6),
}
POINTS["16f"], POINTS["17f"] = POINTS["16"], POINTS["17"]   # on track_c_v2f (600 biographies per person)
BATCH = "8192"
LR_S, LR_M = "1e-3", "6e-4"


def run(args):
    print(">>", " ".join(os.path.basename(a) if os.path.isabs(a) else a for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def have(n):
    return os.path.exists(os.path.join(CKPT, f"{n}.pt"))


def point(pt, steps, frac, f=0.5):
    c = POINTS[pt]
    p = f"sc{pt}"
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    assert meta.get("corpus") == "v2", "run with TRACK_C_DATA=track_c_v2"
    assert pt.endswith("f") == ("facts_bios" in meta), "points 16f and 17f use track_c_v2f, the others track_c_v2"
    V = meta["vocab"]
    fs, fm = M.make(c["s"], V).flops_per_token(), M.make(c["m"], V).flops_per_token()
    d = M.SIZES[c["s"]]["d"]
    gp = sum(q.numel() for q in Gate(4, d, n_hidden=4, conf=True).parameters())
    T, G = c["T"], c["G"]
    budget = 4 * fs * T + G * (4 * fs / 3 + 6 * gp)
    share = {"s": G / (budget / fs), "m": G / (budget / fm)}
    assert max(share.values()) < 0.5, f"G is too large a share of the monolithic runs: {share}"
    print(f"point {p}: {c['s']} / {c['m']}, T={T / 1e6:.0f}M, G={G / 1e6:.0f}M, budget {budget:.3e} FLOPs; "
          f"mono_s {budget / fs / 1e6:.0f}M tokens, mono_m {budget / fm / 1e6:.0f}M tokens", flush=True)

    def mix(k):
        return ["lang=1", "facts=1", "math=1", "code=1", f"mixed={4 * share[k] / (1 - share[k]):.5f}"]

    common = ["--batch-tokens", BATCH, "--micro", "16", "--compile"]
    experts = [f"{p}_{t}" for t in TYPES]
    if "seed" in steps and not have(f"{p}_seed"):
        run([os.path.join(HERE, "train.py"), "--name", f"{p}_seed", "--size", c["s"], "--mix", *mix("s")[:4],
             "--tokens", str(4 * f * T), "--lr", LR_S, *common])
    if "experts" in steps:
        for t in TYPES:
            if not have(f"{p}_{t}"):
                run([os.path.join(HERE, "train.py"), "--name", f"{p}_{t}", "--init",
                     os.path.join(CKPT, f"{p}_seed.pt"), "--mix", f"{t}={1 - frac}", f"ctxv_{t}={frac}",
                     "--own", t, "--tokens", str((1 - f) * T), "--lr", LR_S, *common])
    if "mono" in steps:
        if not have(f"{p}_mono_s"):
            run([os.path.join(HERE, "train.py"), "--name", f"{p}_mono_s", "--size", c["s"], "--mix", *mix("s"),
                 "--flops", str(budget), "--lr", LR_S, *common])
        if not have(f"{p}_mono_m"):
            run([os.path.join(HERE, "train.py"), "--name", f"{p}_mono_m", "--size", c["m"], "--mix", *mix("m"),
                 "--flops", str(budget), "--lr", LR_M, *common])
    if "gates" in steps:
        for g, extra, ex in [("conf", [], experts), ("atom", ["--atom", "--veto"], experts + ["circuit"])]:
            if not have(f"{p}_gate_{g}"):
                run([os.path.join(HERE, "gate.py"), "--name", f"{p}_gate_{g}", "--mode", "prob", "--conf", *extra,
                     "--tokens", str(G), "--experts", *ex])
    if "eval" in steps:
        run([os.path.join(HERE, "evaluate.py"), "--tag", p, "--systems", f"model:{p}_mono_s", f"model:{p}_mono_m",
             f"model:{p}_seed", *[f"model:{e}" for e in experts], f"gate:{p}_gate_conf:prob",
             f"gate:{p}_gate_atom:prob"])
    if "diag" in steps:
        for dset in ["mixed_val"] + [f"mixed_{fm_}_val" for fm_ in EVAL]:
            tag = p if dset == "mixed_val" else f"{dset[6:-4]}_{p}"
            run([os.path.join(HERE, "diag_mixed.py"), "--tag", tag, "--data", dset,
                 "--models", f"{p}_mono_s", f"{p}_mono_m", "--gates", f"{p}_gate_conf"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs="+", default=["16", "17"], choices=list(POINTS))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--frac", type=float, default=0.2)
    a = ap.parse_args()
    steps = a.only or ["seed", "experts", "mono", "gates", "eval", "diag"]
    for pt in a.point:
        point(pt, steps, a.frac)


if __name__ == "__main__":
    main()
