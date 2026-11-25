"""Scale study: how the gate weighs its experts on plain biographies (facts_val) and on the
biography segments of mixed text (mixed_val).

    TRACK_C_DATA=track_c_v2f python experiments/track_c/diag_facts_routing.py --tags sc16f sc17f

Added after the step 1b results (docs/roadmap.md): at sc17f the atom gate's facts loss is twice
mono_s's although the facts expert alone is better than mono_s. For each gate (gate_conf,
gate_atom) and data set, on the windows of evaluate.val_losses (40 batches of 8 x 512, rng 4321)
or, for mixed_val, the same draw over the biography-labelled tokens: the loss of the gate and of
each LM expert alone, the gate's mean weight on each expert, the share of tokens where the facts
expert has the largest weight, and the facts expert's mean weight by position in the window
(the gate sees a 32-token mean, so the first positions have little context).
Writes results/track_c/diag_facts_routing.json.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c import evaluate as E  # noqa: E402
from experiments.track_c.gate import DATA, combine, expert_outputs  # noqa: E402
from experiments.track_c.train import batch, tokens  # noqa: E402

BUCKETS = [(0, 16), (16, 64), (64, 512)]


@torch.no_grad()
def one(sys_, data, n_batches=40, bs=8, ctx=512):
    arr = tokens(data)
    lab = np.memmap(os.path.join(DATA, f"{data}_labels.bin"), dtype=np.uint8, mode="r") if data == "mixed_val" else None
    rng = np.random.default_rng(4321)
    names = sys_.expert_names
    fi = names.index(next(n for n in names if n.endswith("_facts")))
    lm_idx = [k for k, n in enumerate(names) if n != "circuit"]
    acc = {"n": 0, "gate": 0.0, "lam": np.zeros(len(names)), "top_facts": 0,
           "experts": np.zeros(len(lm_idx)), "pos": np.zeros((len(BUCKETS), 2))}
    for _ in range(n_batches):
        if lab is None:
            xy = batch(arr, bs, ctx, rng).cuda()
            m = torch.ones(bs, ctx, dtype=torch.bool)
        else:
            ix = rng.integers(0, len(arr) - ctx - 1, bs)
            xy = torch.from_numpy(np.stack([arr[i:i + ctx + 1] for i in ix]).astype(np.int64)).cuda()
            m = torch.from_numpy(np.stack([lab[i + 1:i + ctx + 1] for i in ix]) == 1)   # biography tokens
        x, y = xy[:, :-1], xy[:, 1:]
        logps, hs = expert_outputs(sys_.experts, x)
        ll = sys_.gate(hs, logps)
        lp = combine(logps, ll, sys_.mode)
        m = m.cuda()
        tgt = y.unsqueeze(-1)
        acc["gate"] += float(-lp.gather(-1, tgt)[..., 0][m].sum())
        for j, k in enumerate(lm_idx):
            acc["experts"][j] += float(-logps[:, :, k].gather(-1, tgt)[..., 0][m].sum())
        lam = ll.exp()
        acc["lam"] += lam[m].sum(0).cpu().numpy()
        acc["top_facts"] += int((lam.argmax(-1) == fi)[m].sum())
        for b, (lo, hi) in enumerate(BUCKETS):
            mb = m.clone()
            mb[:, :lo] = False
            mb[:, hi:] = False
            acc["pos"][b] += [float(lam[..., fi][mb].sum()), int(mb.sum())]
        acc["n"] += int(m.sum())
    n = acc["n"]
    return {"tokens": n, "loss_gate": acc["gate"] / n,
            "loss_expert": {names[k]: acc["experts"][j] / n for j, k in enumerate(lm_idx)},
            "weight": {nm: float(v / n) for nm, v in zip(names, acc["lam"])},
            "top_facts_share": acc["top_facts"] / n,
            "facts_weight_by_position": {f"{lo}-{hi}": float(s / max(c, 1)) for (lo, hi), (s, c) in
                                         zip(BUCKETS, acc["pos"])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["sc16f", "sc17f"])
    a = ap.parse_args()
    out = {}
    for tag in a.tags:
        for g in ("gate_conf", "gate_atom"):
            s = E.make_system(f"gate:{tag}_{g}:prob")
            for data in ("facts_val", "mixed_val"):
                r = one(s, data)
                out[f"{tag}_{g}/{data}"] = r
                print(f"{tag}_{g} {data}: gate {r['loss_gate']:.3f} experts "
                      + " ".join(f"{k.split('_')[-1]} {v:.3f}" for k, v in r["loss_expert"].items())
                      + " | weight " + " ".join(f"{k.split('_')[-1]} {v:.2f}" for k, v in r["weight"].items())
                      + f" | facts on top {r['top_facts_share']:.2f} | by position "
                      + " ".join(f"{v:.2f}" for v in r["facts_weight_by_position"].values()), flush=True)
            del s
            E.Gated._cache.clear()
            torch.cuda.empty_cache()
    save("track_c", "diag_facts_routing", out)


if __name__ == "__main__":
    main()
