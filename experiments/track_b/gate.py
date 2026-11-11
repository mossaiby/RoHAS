"""Track B: the Track C confidence gate over cached pretrained experts; training and evaluation.

    python experiments/track_b/gate.py --name b_gate --experts base coder [--seed 0]

Training: mixed_train windows (cache.py), probability mixing, no routing labels; 3 epochs,
batches of 8 windows, AdamW lr 1e-3 (cosine), weight decay 0.01. The gate is Track C's
(gate.Gate with conf=True): every expert's hidden state and its 32-token causal mean, after
a LayerNorm, plus each expert's entropy and largest log-probability divided by log V.

Evaluation, on every validation file: mean next-token loss of each expert alone, the gate,
the per-token oracle (the best expert for each token in hindsight) and, where tokens are
labelled, the label oracle (prose / markdown -> the first expert, code -> the second), each
also split by label; the gate's mean weight on each expert by label. Control (added after
the pre-registered criteria were evaluated): a constant mixture, the fixed expert weights
that minimise the loss on mixed_train (grid of 101 values per weight, two experts), so the
gain of the gate over plain ensembling can be read off.
Writes checkpoints/track_b/<name>.pt and results/track_b/<name>.json.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_b.cache import CACHE, CTX, FILES  # noqa: E402
from experiments.track_b.data import OUT  # noqa: E402
from experiments.track_c.gate import Gate  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "track_b")
LOGV = math.log(151936)


def load(expert, f):
    return {k: np.load(os.path.join(CACHE, expert, f"{f}.{k}.npy"), mmap_mode="r") for k in ("h", "tlp", "ent", "mx")}


def labels(f, n_win):
    p = os.path.join(OUT, f"{f}_labels.bin")
    if not os.path.exists(p):
        return None
    lab = np.fromfile(p, dtype=np.uint8)
    return np.stack([lab[i * CTX + 1:(i + 1) * CTX + 1] for i in range(n_win)])   # label of each target


def batch(caches, idx):
    """Gate inputs and target log-probs (B, T, E) for windows idx."""
    idx = np.sort(idx)
    hs = [torch.from_numpy(np.asarray(c["h"][idx])).cuda().float() for c in caches]
    cf = torch.stack([torch.from_numpy(np.asarray(c["ent"][idx])) for c in caches] +
                     [torch.from_numpy(np.asarray(c["mx"][idx])) for c in caches], dim=-1).cuda() / LOGV
    tlp = torch.stack([torch.from_numpy(np.asarray(c["tlp"][idx])) for c in caches], dim=-1).cuda()
    return hs, cf, tlp


def gate_logw(gate, hs, cf):
    """log lambda (B, T, E): Gate.forward in conf mode, with precomputed confidence features."""
    h = torch.cat(hs, dim=-1)
    c = torch.cumsum(h, dim=1)
    w = gate.window
    prev = torch.cat([torch.zeros_like(c[:, :w]), c[:, :-w]], dim=1)
    n = torch.clamp(torch.arange(1, h.shape[1] + 1, device=h.device), max=w).view(1, -1, 1)
    z = gate.net(torch.cat([gate.ln(torch.cat([h, (c - prev) / n], dim=-1)), cf], dim=-1))
    return torch.log_softmax(z, dim=-1)


def train(a, experts, d):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    caches = [load(e, "mixed_train") for e in experts]
    W = caches[0]["tlp"].shape[0]
    gate = Gate(len(experts), d, window=32, conf=True).cuda()
    opt = torch.optim.AdamW(gate.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (W // a.bs)
    t0, run, curve = time.time(), 0.0, []
    for step in range(steps):
        if step % (W // a.bs) == 0:
            order = rng.permutation(W)
        for g in opt.param_groups:
            g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        j = step % (W // a.bs)
        hs, cf, tlp = batch(caches, order[j * a.bs:(j + 1) * a.bs])
        loss = -torch.logsumexp(gate_logw(gate, hs, cf) + tlp, dim=-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % max(1, steps // 20) == 0:
            curve.append({"step": step + 1, "loss": run})
            print(f"  step {step + 1}/{steps} loss {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return gate, {"steps": steps, "tokens": steps * a.bs * CTX, "seconds": round(time.time() - t0), "curve": curve,
                  "gate_params": sum(p.numel() for p in gate.parameters())}


def constant_weights(experts):
    t = torch.stack([torch.from_numpy(np.array(load(e, "mixed_train")["tlp"])) for e in experts], -1)
    assert len(experts) == 2
    grid = torch.linspace(0, 1, 101)
    loss = [float(-torch.logsumexp(torch.log(torch.stack([1 - w, w]).clamp_min(1e-12)) + t, -1).mean()) for w in grid]
    w = float(grid[int(np.argmin(loss))])
    return torch.log(torch.tensor([1 - w, w]).clamp_min(1e-12))


@torch.no_grad()
def evaluate(gate, experts):
    out = {}
    const = constant_weights(experts)
    for f in FILES[1:]:
        caches = [load(e, f) for e in experts]
        W = caches[0]["tlp"].shape[0]
        lab = labels(f, W)
        tl, lw = [], []
        for i in range(0, W, 16):
            hs, cf, tlp = batch(caches, np.arange(i, min(W, i + 16)))
            tl.append(tlp.cpu())
            lw.append(gate_logw(gate, hs, cf).cpu())
        tlp, logw = torch.cat(tl), torch.cat(lw)                    # (W, T, E)
        sysl = {e: -tlp[..., k] for k, e in enumerate(experts)}
        sysl["gate"] = -torch.logsumexp(logw + tlp, dim=-1)
        sysl["constant"] = -torch.logsumexp(const + tlp, dim=-1)
        sysl["oracle_token"] = -tlp.amax(-1)
        if lab is not None:
            L = torch.from_numpy(lab.astype(np.int64))
            sysl["oracle_label"] = -tlp.gather(-1, L.unsqueeze(-1)).squeeze(-1)
        r = {"tokens": int(W * CTX), "loss": {k: float(v.mean()) for k, v in sysl.items()}}
        lam = logw.exp()
        if lab is not None:
            for t, nm in enumerate(["prose", "code"]):
                m = L == t
                r[f"loss_{nm}"] = {k: float(v[m].mean()) for k, v in sysl.items()}
                r[f"weight_{nm}"] = {e: float(lam[..., k][m].mean()) for k, e in enumerate(experts)}
                r[f"share_{nm}"] = float(m.float().mean())
        else:
            r["weight"] = {e: float(lam[..., k].mean()) for k, e in enumerate(experts)}
        out[f] = r
        print(f, {k: round(v, 4) for k, v in r["loss"].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--experts", nargs="+", default=["base", "coder"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    d = load(a.experts[0], "mixed_train")["h"].shape[-1]
    path = os.path.join(CKPT, f"{a.name}.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        gate = Gate(len(a.experts), d, window=32, conf=True).cuda()
        gate.load_state_dict(ck["gate"])
        info = ck["info"]
    else:
        gate, info = train(a, a.experts, d)
        torch.save({"gate": gate.state_dict(), "experts": a.experts, "d": d, "info": info}, path)
    gate.eval()
    res = evaluate(gate, a.experts)
    save("track_b", a.name, {"experts": a.experts, "train": info, "eval": res,
                             "constant_weights": constant_weights(a.experts).exp().tolist()}, seed=a.seed)


if __name__ == "__main__":
    main()
