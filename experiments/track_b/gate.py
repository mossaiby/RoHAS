"""Track B: the Track C confidence gate over cached pretrained experts; training and evaluation.

    python experiments/track_b/gate.py --name b_gate_s0 --experts base coder [--seed 0]
    python experiments/track_b/gate.py --name b3_gate_s0 --experts base coder math --step 2

Training: the step's training file (mixed_train, or m3_train for step 2) as cached windows
(cache.py), probability mixing, no routing labels; 3 epochs, batches of 8 windows, AdamW lr
1e-3 (cosine), weight decay 0.01. The gate is Track C's confidence gate (gate.Gate with
conf=True), written here for experts of different widths: every expert's hidden state and
its 32-token causal mean, after one LayerNorm, plus each expert's entropy and largest
log-probability divided by log V, into a 256-unit MLP. With two experts of width 896 its
parameters are those of gate.Gate, so step-1 checkpoints load unchanged.

Evaluation, on every validation file: mean next-token loss of each expert alone, the gate,
the per-token oracle (the best expert for each token in hindsight) and, where tokens are
labelled, the label oracle (label k -> expert k: prose -> base, code -> coder, math ->
math), each also split by label; the gate's mean weight on each expert by label. Control
(added in step 1 after its pre-registered criteria were evaluated; pre-registered for step
2): the constant mixture, the fixed expert weights that minimise the loss on the training
file (two experts: 101-point grid; three: simplex grid with step 0.02).
Writes checkpoints/track_b/<name>.pt and results/track_b/<name>.json.
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_b.cache import CACHE, CTX, FILES, FILES2, FILES3, FILES4  # noqa: E402
from experiments.track_b.data import OUT, TYPES  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "track_b")
LOGV = math.log(151936)


class GateB(nn.Module):
    def __init__(self, dims, window=32, hidden=256):
        super().__init__()
        self.window = window
        n_h = 2 * sum(dims)
        self.ln = nn.LayerNorm(n_h)
        self.net = nn.Sequential(nn.Linear(n_h + 2 * len(dims), hidden), nn.GELU(), nn.Linear(hidden, len(dims)))


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
    """log lambda (B, T, E): Track C's Gate.forward in conf mode, with precomputed confidence features."""
    h = torch.cat(hs, dim=-1)
    c = torch.cumsum(h, dim=1)
    w = gate.window
    prev = torch.cat([torch.zeros_like(c[:, :w]), c[:, :-w]], dim=1)
    n = torch.clamp(torch.arange(1, h.shape[1] + 1, device=h.device), max=w).view(1, -1, 1)
    z = gate.net(torch.cat([gate.ln(torch.cat([h, (c - prev) / n], dim=-1)), cf], dim=-1))
    return torch.log_softmax(z, dim=-1)


def train(a, experts, dims, train_file):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    caches = [load(e, train_file) for e in experts]
    W = caches[0]["tlp"].shape[0]
    gate = GateB(dims).cuda()
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
                  "gate_params": sum(p.numel() for p in gate.parameters()), "train_file": train_file}


def constant_weights(experts, train_file):
    t = torch.stack([torch.from_numpy(np.array(load(e, train_file)["tlp"])) for e in experts], -1).cuda()
    if len(experts) == 2:
        grid = [torch.tensor([1 - w, w]) for w in torch.linspace(0, 1, 101)]
    else:
        k = 50                                          # simplex grid, step 0.02
        grid = [torch.tensor(c, dtype=torch.float32) / k for c in itertools.product(range(k + 1), repeat=len(experts))
                if sum(c) == k]
    best, wbest = float("inf"), None
    for w in grid:
        loss = float(-torch.logsumexp(torch.log(w.clamp_min(1e-12)).cuda() + t, -1).mean())
        if loss < best:
            best, wbest = loss, w
    return torch.log(wbest.clamp_min(1e-12))


@torch.no_grad()
def evaluate(gate, experts, files, const):
    out = {}
    for f in files:
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
            sysl["oracle_label"] = -tlp.gather(-1, L.clamp(max=len(experts) - 1).unsqueeze(-1)).squeeze(-1)
        r = {"tokens": int(W * CTX), "loss": {k: float(v.mean()) for k, v in sysl.items()}}
        lam = logw.exp()
        if lab is not None:
            for t, nm in enumerate(TYPES):
                m = L == t
                if not m.any():
                    continue
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
    ap.add_argument("--step", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--frozen", help="evaluate this trained gate (checkpoint name) over --experts, without training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    files = {1: FILES, 2: FILES2, 3: FILES3, 4: FILES4}[a.step]
    os.makedirs(CKPT, exist_ok=True)
    dims = [load(e, files[1])["h"].shape[-1] for e in a.experts]
    path = os.path.join(CKPT, f"{a.frozen or a.name}.pt")
    if a.frozen:
        ck = torch.load(path, map_location="cuda", weights_only=False)
        assert (ck.get("dims") or dims) == dims, "a replacement expert must have the width of the one it replaces"
        gate = GateB(dims).cuda()
        gate.load_state_dict(ck["gate"])
        info = {**ck["info"], "frozen_from": a.frozen, "frozen_experts": ck["experts"]}
    elif os.path.exists(path):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        gate = GateB(dims).cuda()
        gate.load_state_dict(ck["gate"])
        info = ck["info"]
    else:
        gate, info = train(a, a.experts, dims, files[0])
        torch.save({"gate": gate.state_dict(), "experts": a.experts, "dims": dims, "info": info}, path)
    gate.eval()
    const = constant_weights(a.experts, files[0])
    res = evaluate(gate, a.experts, files[1:], const)
    save("track_b", a.name, {"experts": a.experts, "step": a.step, "train": info, "eval": res,
                             "constant_weights": const.exp().tolist()}, seed=a.seed)


if __name__ == "__main__":
    main()
