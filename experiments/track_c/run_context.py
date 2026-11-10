"""Track C: context exposure for the seeded experts (the mixed-text gap).

    python experiments/track_c/run_context.py --frac 0.2 [--varied] [--only experts gates eval diag]

Same recipe and FLOPs as the seeded experts (branch from main_seeded_seed, 50M tokens each,
8k-token batches, lr 1e-3), but a fraction --frac of each branch's tokens comes from
ctx_<type>_train (data_context.py): mixed documents with the loss on the expert's own type
only. Then the confidence-only gate and the final atom gate (with the circuit), the
standard evaluation (tag main_ctx<pct>), and the mixed-text decomposition (diag_mixed.py) on
mixed_val and on the data_mixed_ood.py formats (tags <format>_main_ctx<pct>).
--varied uses the ctxv_<type> files (formats varied; prefix main_ctxv<pct>).
Requires: python experiments/track_c/data_context.py [--varied]; data_mixed_ood.py
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c.data_mixed_ood import EVAL  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
TYPES = ["lang", "facts", "math", "code"]


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def have(n):
    return os.path.exists(os.path.join(CKPT, f"{n}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--varied", action="store_true", help="ctxv_<type> data (data_context.py --varied)")
    a = ap.parse_args()
    steps = a.only or ["experts", "gates", "eval", "diag"]
    src = "ctxv" if a.varied else "ctx"
    p = f"main_{src}{round(100 * a.frac)}"
    experts = [f"{p}_{t}" for t in TYPES]
    if "experts" in steps:
        for t in TYPES:
            if not have(f"{p}_{t}"):
                run([os.path.join(HERE, "train.py"), "--name", f"{p}_{t}", "--init",
                     os.path.join(CKPT, "main_seeded_seed.pt"), "--mix", f"{t}={1 - a.frac}", f"{src}_{t}={a.frac}",
                     "--own", t, "--tokens", "50e6", "--batch-tokens", "8192", "--lr", "1e-3"])
    if "gates" in steps:
        for g, extra, ex in [("conf", [], experts), ("atom", ["--atom", "--veto"], experts + ["circuit"])]:
            if not have(f"{p}_gate_{g}"):
                run([os.path.join(HERE, "gate.py"), "--name", f"{p}_gate_{g}", "--mode", "prob", "--conf", *extra,
                     "--tokens", "6e6", "--experts", *ex])
    if "eval" in steps:
        run([os.path.join(HERE, "evaluate.py"), "--tag", p, "--systems", *[f"model:{e}" for e in experts],
             f"gate:{p}_gate_conf:prob", f"gate:{p}_gate_atom:prob"])
    if "diag" in steps:
        for d in ["mixed_val"] + [f"mixed_{f}_val" for f in EVAL]:
            tag = p if d == "mixed_val" else f"{d[6:-4]}_{p}"
            run([os.path.join(HERE, "diag_mixed.py"), "--tag", tag, "--data", d, "--gates", f"{p}_gate_conf"])


if __name__ == "__main__":
    main()
