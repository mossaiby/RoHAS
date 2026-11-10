"""Track C: where does the mixed-text loss come from?

    python experiments/track_c/diag_mixed.py --tag seeded --models main_mono_s \
        --gates main_seeded_gate_conf

Per-token loss on mixed_val (or --data, e.g. mixed_long_val), using its segment labels, split by segment type and by
position in the segment (the first 16 tokens vs the rest), for:
  each --models system and each --gates gate (prob mixing), and, for every gate, its LM
  experts alone, the label oracle (each token scored by the expert of its segment's type)
  and the per-token oracle (the best expert for each token, chosen with hindsight).
A per-token oracle above mono_s means routing alone cannot close the gap.
Writes results/track_c/diag_mixed_<tag>.json.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c import evaluate as E  # noqa: E402
from experiments.track_c.gate import DATA, expert_outputs  # noqa: E402
from experiments.track_c.train import tokens  # noqa: E402

TYPES = ["lang", "facts", "math", "code"]
EARLY = 16


def windows(data="mixed_val", n=320, ctx=512, seed=4321):
    arr = tokens(data)
    lab = np.fromfile(os.path.join(DATA, f"{data}_labels.bin"), dtype=np.uint8)
    starts = np.random.default_rng(seed).integers(0, len(arr) - ctx - 1, size=n)
    X = torch.from_numpy(np.stack([arr[s:s + ctx + 1].astype(np.int64) for s in starts]))
    L = np.stack([lab[s + 1:s + ctx + 1] for s in starts])          # segment type of each target
    pos = np.zeros_like(L, dtype=np.int64)                          # position in its segment run
    for t in range(1, L.shape[1]):
        pos[:, t] = np.where(L[:, t] == L[:, t - 1], pos[:, t - 1] + 1, 0)
    return X, L, pos


@torch.no_grad()
def nll(fn, X, bs=8):
    out = []
    for i in range(0, len(X), bs):
        xy = X[i:i + bs].cuda()
        out.append(fn(xy[:, :-1], xy[:, 1:]).float().cpu())
    return torch.cat(out).numpy()


def summary(v, L, pos):
    r = {"all": float(v.mean())}
    for k, t in enumerate(TYPES):
        m = L == k
        r[t] = float(v[m].mean())
        r[f"{t}_early"] = float(v[m & (pos < EARLY)].mean())
        r[f"{t}_late"] = float(v[m & (pos >= EARLY)].mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--gates", nargs="*", default=[])
    ap.add_argument("--data", default="mixed_val", help="mixed_val, or a data_mixed_ood.py set")
    a = ap.parse_args()
    t0 = time.time()
    X, L, pos = windows(a.data)
    res = {}
    for m in a.models:
        s = E.make_system(f"model:{m}")
        res[m] = summary(nll(lambda x, y: -s.logprobs(x)[0].gather(-1, y[..., None])[..., 0], X), L, pos)
        del s
    for g in a.gates:
        s = E.make_system(f"gate:{g}:prob")
        res[f"{g}:prob"] = summary(nll(lambda x, y: -s.logprobs(x)[0].gather(-1, y[..., None])[..., 0], X), L, pos)
        lms = [m for m in s.experts if isinstance(m, torch.nn.Module)][:4]

        def per_expert(x, y):
            lp, _ = expert_outputs(lms, x)
            return -lp.gather(-1, y[..., None, None].expand(*y.shape, lp.shape[2], 1))[..., 0]
        per = nll(per_expert, X)                                     # (N, T, 4)
        for k, t in enumerate(TYPES):
            res[f"{g}/expert_{t}"] = summary(per[..., k], L, pos)
        res[f"{g}/oracle_label"] = summary(np.take_along_axis(per, L[..., None].astype(np.int64), -1)[..., 0],
                                           L, pos)
        res[f"{g}/oracle_token"] = summary(per.min(-1), L, pos)
        del s
        torch.cuda.empty_cache()
    share = {t: float((L == k).mean()) for k, t in enumerate(TYPES)}
    print(f"{'system':42s} {'all':>6s} " + " ".join(f"{t:>6s}" for t in TYPES) + "  | early | late")
    for n, r in res.items():
        print(f"{n[:42]:42s} {r['all']:6.3f} " + " ".join(f"{r[t]:6.3f}" for t in TYPES) + "  | " +
              " ".join(f"{r[t + '_early']:5.2f}" for t in TYPES) + " | " +
              " ".join(f"{r[t + '_late']:5.2f}" for t in TYPES))
    save("track_c", f"diag_mixed_{a.tag}", {"systems": res, "token_share": share,
                                            "early_share": float((pos < EARLY).mean()), "early_tokens": EARLY,
                                            "data": a.data},
         wall_time=time.time() - t0)


if __name__ == "__main__":
    main()
