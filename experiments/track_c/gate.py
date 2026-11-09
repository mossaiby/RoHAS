"""Train the token-level gate over frozen experts on mixed documents (no routing labels).

    python experiments/track_c/gate.py --experts lang facts math code --mode prob --name gate_prob

The gate sees, at each position, every expert's final hidden state and its causal mean
over the last --window positions, and outputs weights lambda over the experts.
  prob   log p = logsumexp_e(log lambda_e + log p_e)            (probability mixture)
  logit  log p = log_softmax(sum_e lambda_e log p_e)            (weighted product of experts,
                                                                 the Level-B blend)
  hard   evaluation only: the expert with the largest lambda (from a prob-trained gate)
--conf adds two confidence features per expert (entropy and largest log-probability of its
next-token distribution, both divided by log V). The expert name "circuit" is the compiled
arithmetic expert (circuit_expert.py); it has no hidden state, so the gate sees it only
through these features.
--split gives the experts without hidden states (the circuit; they must come last) their
own small head on the confidence features alone, so their weight cannot depend on the
LM hidden states (which, for example, encode how long the current number is).
--atom makes the gate two-level: each hidden-state-free expert (atom) gets its weight from
its OWN confidence features only, against a fixed "rest" logit; the LM experts share the
rest, split by the hidden-state gate. An atom's weight then depends on nothing but what
the atom itself reports. The atom head is linear (--atom-head linear, the default) with its
own learning rate (--atom-lr) and no weight decay: with the small MLP at the gate's learning
rate, the 97% of positions where the circuit abstains held its weight at 0.39 where it is
active, although the data optimum is close to 1 (the MLP run is kept as an ablation). The
atom head reads the atom's confidence in nats, not divided by log V: divided, the active and
abstaining states differ by about 1 per feature, the head's weights stay small, and one bias
has to trade the active weight against the abstaining one (both linear-head runs settled at
0.40; kept as the ablation gate_atom_lin). --veto applies the atoms' vetoes (circuit_expert.py) after mixing:
vetoed tokens get probability ~0 and the rest is renormalised.
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
from experiments.track_c.train import CKPT, DATA, tokens, batch  # noqa: E402


def confidence(logps):
    """(B, T, 2E) entropy and max log-prob of each expert's distribution, divided by log V."""
    logv = math.log(logps.shape[-1])
    ent = torch.stack([-(lp.exp() * lp).sum(-1) for lp in logps.unbind(2)], dim=-1)
    return torch.cat([ent / logv, logps.amax(-1) / logv], dim=-1)


class Gate(nn.Module):
    def __init__(self, n_experts, d, window=32, hidden=256, n_hidden=None, conf=False, split=False, atom=False,
                 atom_head="mlp", atom_nats=False):
        super().__init__()
        self.window, self.conf, self.split, self.atom_mode = window, conf, split, atom
        self.atom_nats = atom_nats
        n_hidden = n_experts if n_hidden is None else n_hidden
        self.n_atoms = n_experts - n_hidden
        n_h = 2 * n_hidden * d
        if atom:
            assert conf and self.n_atoms > 0
            self.ln = nn.LayerNorm(n_h)
            self.net = nn.Sequential(nn.Linear(n_h + 2 * n_experts, hidden), nn.GELU(), nn.Linear(hidden, n_hidden))
            self.atom = (nn.Linear(2 * self.n_atoms, self.n_atoms) if atom_head == "linear" else
                         nn.Sequential(nn.Linear(2 * self.n_atoms, 32), nn.GELU(), nn.Linear(32, self.n_atoms)))
        elif split:
            assert conf and n_hidden < n_experts
            self.ln = nn.LayerNorm(n_h)
            self.net = nn.Sequential(nn.Linear(n_h + 2 * n_experts, hidden), nn.GELU(), nn.Linear(hidden, n_hidden))
            self.atom = nn.Sequential(nn.Linear(2 * n_experts, 32), nn.GELU(), nn.Linear(32, n_experts - n_hidden))
        elif conf:   # LayerNorm on the hidden-state part only; confidence features enter as they are
            self.ln = nn.LayerNorm(n_h)
            self.net = nn.Sequential(nn.Linear(n_h + 2 * n_experts, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
        else:
            self.net = nn.Sequential(nn.LayerNorm(n_h), nn.Linear(n_h, hidden), nn.GELU(), nn.Linear(hidden, n_experts))

    def forward(self, hs, logps=None, last=False):
        """log lambda (B, T, E); with last=True only for the final position (logps then has T=1)."""
        h = torch.cat(hs, dim=-1).float()                       # (B, T, E*d)
        c = torch.cumsum(h, dim=1)
        w = self.window
        prev = torch.cat([torch.zeros_like(c[:, :w]), c[:, :-w]], dim=1) if c.shape[1] > w else torch.zeros_like(c)
        n = torch.clamp(torch.arange(1, h.shape[1] + 1, device=h.device), max=w).view(1, -1, 1)
        mean = (c - prev) / n
        if last:
            h, mean = h[:, -1:], mean[:, -1:]
        if not self.conf:
            return torch.log_softmax(self.net(torch.cat([h, mean], dim=-1)), dim=-1)
        cf = confidence(logps)
        z = self.net(torch.cat([self.ln(torch.cat([h, mean], dim=-1)), cf], dim=-1))
        if self.atom_mode:
            E, A = cf.shape[-1] // 2, self.n_atoms
            own = torch.cat([cf[..., E - A:E], cf[..., 2 * E - A:]], dim=-1)    # atoms' entropy, max log-prob
            if self.atom_nats:
                own = own * math.log(logps.shape[-1])
            a = self.atom(own)
            top = torch.log_softmax(torch.cat([a, torch.zeros_like(a[..., :1])], dim=-1), dim=-1)
            return torch.cat([top[..., -1:] + torch.log_softmax(z, dim=-1), top[..., :-1]], dim=-1)
        if self.split:
            z = torch.cat([z, self.atom(cf)], dim=-1)
        return torch.log_softmax(z, dim=-1)


def load_experts(names, device="cuda"):
    ex = []
    for n in names:
        if n == "circuit":
            from tokenizers import Tokenizer
            from experiments.track_c.circuit_expert import CircuitExpert
            ex.append(CircuitExpert(Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))))
            continue
        m, _ = M.load(os.path.join(CKPT, f"{n}.pt"), device)
        for p in m.parameters():
            p.requires_grad_(False)
        ex.append(m)
    return ex


@torch.no_grad()
def expert_outputs(experts, x, last=False):
    """Expert log-probs (B, T, E, V), or (B, 1, E, V) with last=True, and LM hidden states (B, T, d)."""
    logps, hs = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for m in experts:
            if not isinstance(m, nn.Module):                # compiled circuit: log-probs only
                logps.append(m.logprobs(x, last=last))
                continue
            lg, h = m(x, hidden=True)
            logps.append(torch.log_softmax((lg[:, -1:] if last else lg).float(), dim=-1))
            hs.append(h)
    return torch.stack(logps, dim=2), hs                    # (B, T, E, V), list of (B, T, d) per LM expert


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


def apply_vetoes(lp, experts):
    """Zero out (log-prob -30) the tokens vetoed by atom experts in their last call; renormalise."""
    for m in experts:
        mask = getattr(m, "last_veto", None)
        if mask is not None and not isinstance(m, nn.Module):
            lp = lp.masked_fill(mask, -30.0)
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
    return lp


def load_gate(name, device="cuda"):
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), map_location=device, weights_only=False)
    g = Gate(len(ck["experts"]), ck["d"], ck["window"], n_hidden=ck.get("n_hidden"),
             conf=ck.get("conf", False), split=ck.get("split", False), atom=ck.get("atom", False),
             atom_head=ck.get("atom_head", "mlp"), atom_nats=ck.get("atom_nats", False)).to(device)
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
    ap.add_argument("--conf", action="store_true", help="add expert confidence features")
    ap.add_argument("--split", action="store_true", help="gate hidden-state-free experts from confidence only")
    ap.add_argument("--atom", action="store_true", help="two-level gate: atoms weighted by their own confidence")
    ap.add_argument("--atom-head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--atom-units", default="nats", choices=["nats", "logv"], help="scale of the atom head's input")
    ap.add_argument("--atom-lr", type=float, default=3e-2, help="peak learning rate of the atom head")
    ap.add_argument("--veto", action="store_true", help="apply the atoms' vetoes after mixing")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    experts = load_experts(args.experts)
    lms = [m for m in experts if isinstance(m, nn.Module)]
    d = lms[0].c.d
    if len(lms) < len(experts) and not args.conf:
        raise SystemExit("an expert without hidden states needs --conf")
    if args.split and any(isinstance(m, nn.Module) for m in experts[len(lms):]):
        raise SystemExit("--split needs the experts without hidden states last")
    gate = Gate(len(experts), d, args.window, n_hidden=len(lms), conf=args.conf, split=args.split,
                atom=args.atom, atom_head=args.atom_head,
                atom_nats=args.atom_units == "nats").cuda()
    head = list(gate.atom.parameters()) if args.atom else []
    rest = [q for q in gate.parameters() if all(q is not h for h in head)]
    groups = [{"params": rest, "lr": args.lr, "peak": args.lr, "weight_decay": 0.01}]
    if head:
        groups.append({"params": head, "lr": args.atom_lr, "peak": args.atom_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    arr = tokens("mixed_train")
    steps = int(args.tokens // (args.micro * args.ctx))
    t0 = time.time()
    curve = []
    run = 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["peak"] * 0.5 * (1 + math.cos(math.pi * step / steps))
        xy = batch(arr, args.micro, args.ctx, rng).cuda()
        logps, hs = expert_outputs(experts, xy[:, :-1])
        lp = combine(logps, gate(hs, logps), args.mode)
        if args.veto:
            lp = apply_vetoes(lp, experts)
        loss = F.nll_loss(lp.reshape(-1, lp.shape[-1]), xy[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % max(1, steps // 20) == 0:
            curve.append({"step": step + 1, "loss": run})
            print(f"  step {step + 1}/{steps} loss {run:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # FLOPs: forward of every LM expert (2N per token) + gate forward and backward; the
    # circuit runs once per equation on the CPU and is not counted here (see circuit_calls)
    fwd = sum(m.flops_per_token() / 3 for m in lms)
    gp = sum(p.numel() for p in gate.parameters())
    trained = steps * args.micro * args.ctx
    torch.save({"gate": gate.state_dict(), "experts": args.experts, "d": d, "window": args.window,
                "mode": args.mode, "n_hidden": len(lms), "conf": args.conf, "split": args.split,
                "atom": args.atom, "veto": args.veto,
                "atom_head": args.atom_head if args.atom else None,
                "atom_nats": args.atom and args.atom_units == "nats"}, os.path.join(CKPT, f"{args.name}.pt"))
    calls = sum(getattr(m, "calls", 0) for m in experts if not isinstance(m, nn.Module))
    save("track_c", f"train_{args.name}",
         {"name": args.name, "experts": args.experts, "mode": args.mode, "conf": args.conf, "split": args.split,
          "atom": args.atom, "veto": args.veto,
          "atom_head": args.atom_head if args.atom else None, "atom_lr": args.atom_lr if args.atom else None,
          "atom_units": args.atom_units if args.atom else None,
          "circuit_calls": calls, "gate_params": gp,
          "tokens": trained, "flops": (fwd + 6 * gp) * trained, "curve": curve},
         seed=args.seed, wall_time=time.time() - t0)


if __name__ == "__main__":
    main()
