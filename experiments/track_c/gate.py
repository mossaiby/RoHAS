"""Train the token-level gate over frozen experts on mixed documents (no routing labels).

    python experiments/track_c/gate.py --experts lang facts math code --mode prob --name gate_prob

The gate sees, at each position, every expert's final hidden state and its causal mean
over the last --window positions, and outputs weights lambda over the experts.
  prob   log p = logsumexp_e(log lambda_e + log p_e)            (probability mixture)
  logit  log p = log_softmax(sum_e lambda_e log p_e)            (weighted product of experts,
                                                                 the Level-B blend)
  hard   evaluation only: the expert with the largest lambda (from a prob-trained gate)
Writes checkpoints/track_c/<name>.pt and results/track_c/train_<name>.json.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c import model as M  # noqa: E402
from experiments.track_c.train import CKPT, tokens, batch  # noqa: E402


class Gate(nn.Module):
    def __init__(self, n_experts, d, window=32, hidden=256):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(nn.LayerNorm(2 * n_experts * d), nn.Linear(2 * n_experts * d, hidden),
                                 nn.GELU(), nn.Linear(hidden, n_experts))

    def forward(self, hs):
        h = torch.cat(hs, dim=-1).float()                       # (B, T, E*d)
        c = torch.cumsum(h, dim=1)
        w = self.window
        prev = torch.cat([torch.zeros_like(c[:, :w]), c[:, :-w]], dim=1) if c.shape[1] > w else torch.zeros_like(c)
        n = torch.clamp(torch.arange(1, h.shape[1] + 1, device=h.device), max=w).view(1, -1, 1)
        mean = (c - prev) / n
        return torch.log_softmax(self.net(torch.cat([h, mean], dim=-1)), dim=-1)   # log lambda


def load_experts(names, device="cuda"):
    ex = []
    for n in names:
        m, _ = M.load(os.path.join(CKPT, f"{n}.pt"), device)
        for p in m.parameters():
            p.requires_grad_(False)
        ex.append(m)
    return ex


@torch.no_grad()
def expert_outputs(experts, x):
    logps, hs = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for m in experts:
            lg, h = m(x, hidden=True)
            logps.append(torch.log_softmax(lg.float(), dim=-1))
            hs.append(h)
    return torch.stack(logps, dim=2), hs                    # (B, T, E, V), list of (B, T, d)


def combine(logps, log_lam, mode):
    """Next-token log-probabilities (B, T, V) from expert log-probs and gate log-weights."""
    if mode == "prob":
        return torch.logsumexp(logps + log_lam.unsqueeze(-1), dim=2)
    if mode == "logit":
        return torch.log_softmax((logps * log_lam.exp().unsqueeze(-1)).sum(2), dim=-1)
    if mode == "hard":
        idx = log_lam.argmax(-1, keepdim=True).unsqueeze(-1).expand(*logps.shape[:2], 1, logps.shape[-1])
        return logps.gather(2, idx).squeeze(2)
    raise ValueError(mode)


def load_gate(name, device="cuda"):
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), map_location=device, weights_only=False)
    g = Gate(len(ck["experts"]), ck["d"], ck["window"]).to(device)
    g.load_state_dict(ck["gate"])
    g.eval()
    return g, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="+", default=["lang", "facts", "math", "code"])
    ap.add_argument("--mode", default="prob", choices=["prob", "logit"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=float, default=6e6, help="gate training tokens (2 epochs of mixed_train)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    experts = load_experts(args.experts)
    d = experts[0].c.d
    gate = Gate(len(experts), d, args.window).cuda()
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=0.01)
    arr = tokens("mixed_train")
    steps = int(args.tokens // (args.micro * args.ctx))
    t0 = time.time()
    curve = []
    run = 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        xy = batch(arr, args.micro, args.ctx, rng).cuda()
        logps, hs = expert_outputs(experts, xy[:, :-1])
        lp = combine(logps, gate(hs), args.mode)
        loss = F.nll_loss(lp.reshape(-1, lp.shape[-1]), xy[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % max(1, steps // 20) == 0:
            curve.append({"step": step + 1, "loss": run})
            print(f"  step {step + 1}/{steps} loss {run:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # FLOPs: forward of every expert (2N per token) + gate forward and backward
    fwd = sum(m.flops_per_token() / 3 for m in experts)
    gp = sum(p.numel() for p in gate.parameters())
    trained = steps * args.micro * args.ctx
    torch.save({"gate": gate.state_dict(), "experts": args.experts, "d": d, "window": args.window,
                "mode": args.mode}, os.path.join(CKPT, f"{args.name}.pt"))
    save("track_c", f"train_{args.name}",
         {"name": args.name, "experts": args.experts, "mode": args.mode, "gate_params": gp,
          "tokens": trained, "flops": (fwd + 6 * gp) * trained, "curve": curve},
         seed=args.seed, wall_time=time.time() - t0)


if __name__ == "__main__":
    main()
