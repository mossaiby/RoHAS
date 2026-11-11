"""Track C: a second seed of the monolithic baselines (mono_s, mono_m).

    python experiments/track_c/run_baseline_seed.py [--seed 1] [--only train eval diag]

Same settings as main_mono_s and main_mono_m (train_main_mono_*.json: mixture, token budget,
8k-token batches, default learning rates), with another seed (initialisation and data order).
  train  main_mono_s_r<seed>, main_mono_m_r<seed>
  eval   losses and fact recall (tag main_r<seed>); held-out arithmetic and word problems for
         mono_s (tag main_r<seed>_heldout, eval seed 1000, 100 items per cell)
  diag   mixed-text loss on mixed_val and the three held-out formats (tags <fmt>_main_r<seed>)
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c.data_mixed_ood import EVAL  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["train", "eval", "diag"]
    names = {}
    for size in ("s", "m"):
        src = json.load(open(os.path.join(ROOT, "results", "track_c", f"train_main_mono_{size}.json")))["data"]
        name = f"main_mono_{size}_r{a.seed}"
        names[size] = name
        if "train" in steps and not os.path.exists(os.path.join(CKPT, f"{name}.pt")):
            run([os.path.join(HERE, "train.py"), "--name", name, "--size", size,
                 "--mix", *[f"{k}={v}" for k, v in src["mix"].items()], "--tokens", str(src["tokens"]),
                 "--batch-tokens", str(src["batch_tokens"]), "--lr", str(src["lr"]), "--seed", str(a.seed),
                 *(["--micro", "8"] if size == "m" else [])])
    tag = f"main_r{a.seed}"
    if "eval" in steps:
        run([os.path.join(HERE, "evaluate.py"), "--tag", tag, "--skip", "math", "routing",
             "--systems", *[f"model:{n}" for n in names.values()]])
        run([os.path.join(HERE, "evaluate.py"), "--tag", f"{tag}_heldout", "--eval-seed", "1000", "--per-cell", "100",
             "--skip", "loss", "facts", "routing", "--systems", f"model:{names['s']}"])
    if "diag" in steps:
        for d in ["mixed_val"] + [f"mixed_{f}_val" for f in EVAL]:
            t = tag if d == "mixed_val" else f"{d[6:-4]}_{tag}"
            run([os.path.join(HERE, "diag_mixed.py"), "--tag", t, "--data", d, "--models", *names.values()])


if __name__ == "__main__":
    main()
