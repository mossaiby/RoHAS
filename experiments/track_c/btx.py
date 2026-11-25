"""Branch-Train-MiX baseline (Sukhbaatar et al., 2024) for the scale study.

    TRACK_C_DATA=track_c_v2f python experiments/track_c/btx.py --point 16f   # merge only
    then train.py --init checkpoints/track_c/sc16f_btx_init.pt ...           # see run_btx.sh

The four branched experts' feed-forward blocks become, layer by layer, the experts of a top-2
mixture-of-experts layer with a new linear router; all other weights are the element-wise mean
of the experts'. The model has GPT's interface (forward(idx, hidden), n_params,
flops_per_token), so train.py, evaluate.py and diag_mixed.py use it through model.load/save.
See docs/roadmap.md, "Scale study: a Branch-Train-MiX baseline".
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.track_c import model as M  # noqa: E402
from experiments.track_c.train import CKPT  # noqa: E402

TYPES = ["lang", "facts", "math", "code"]
AUX_COEF = 0.01


class MoEBlock(nn.Module):
    def __init__(self, c, n_experts, k):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d, bias=False), nn.LayerNorm(c.d, bias=False)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.router = nn.Linear(c.d, n_experts, bias=False)
        self.fc = nn.Parameter(torch.empty(n_experts, 4 * c.d, c.d))     # each expert's Block.fc.weight
        self.out = nn.Parameter(torch.empty(n_experts, c.d, 4 * c.d))    # each expert's Block.out.weight
        self.h, self.k, self.E = c.heads, k, n_experts

    def forward(self, x):
        B, T, D = x.shape
        q, kk, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, kk, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, kk, v))
        y = F.scaled_dot_product_attention(q, kk, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        u = self.ln2(x)
        logits = self.router(u).float()                                   # (B, T, E)
        top, idx = logits.topk(self.k, dim=-1)
        w = torch.zeros_like(logits).scatter(-1, idx, torch.softmax(top, dim=-1))   # (B, T, E), k non-zero
        # Switch-style balance: fraction of routing slots per expert times its mean router probability
        frac = torch.zeros_like(logits).scatter(-1, idx, 1.0).mean((0, 1)) / self.k
        self.aux = self.E * (frac * torch.softmax(logits, dim=-1).mean((0, 1))).sum()
        # every expert on every token, weighted; the unchosen ones have weight 0 (a dense
        # implementation of the sparse layer: same outputs, FLOPs counted for k experts only)
        hid = F.gelu(torch.einsum("btd,efd->btef", u, self.fc))
        o = torch.einsum("btef,edf->bted", hid, self.out)
        return x + (o * w.to(o.dtype).unsqueeze(-1)).sum(2)


class BTX(nn.Module):
    kind = "btx"

    def __init__(self, c, n_experts=4, k=2):
        super().__init__()
        self.c, self.n_experts, self.k = c, n_experts, k
        self.wte = nn.Embedding(c.vocab, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(MoEBlock(c, n_experts, k) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.d, bias=False)

    @property
    def save_extra(self):
        return {"kind": self.kind, "n_experts": self.n_experts, "k": self.k}

    def forward(self, idx, hidden=False):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x)
        logits = h @ self.wte.weight.T
        return (logits, h) if hidden else logits

    def aux_loss(self):
        """Mean load-balancing loss over the layers of the last forward pass, times its coefficient."""
        return AUX_COEF * sum(b.aux for b in self.blocks) / len(self.blocks)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_active(self):
        """Parameters used per token: all but the (E - k) unchosen feed-forward blocks per layer."""
        per_ffn = 8 * self.c.d ** 2
        return self.n_params() - self.c.layers * (self.n_experts - self.k) * per_ffn

    def flops_per_token(self):
        """As GPT.flops_per_token, with the active parameters: 6 x (active excluding positions) + attention."""
        c = self.c
        return 6 * (self.n_active() - self.wpe.weight.numel()) + 12 * c.layers * c.d * c.ctx


def merge(experts, k=2):
    """A BTX model from GPT experts with the same configuration (branched from one seed)."""
    c = experts[0].c
    assert all(e.c == c for e in experts), "experts differ in configuration"
    m = BTX(c, len(experts), k)
    sds = [e.state_dict() for e in experts]
    new = m.state_dict()
    for name in new:
        parts = name.split(".")
        if parts[0] == "blocks" and parts[2] in ("fc", "out"):
            new[name] = torch.stack([sd[f"blocks.{parts[1]}.{parts[2]}.weight"] for sd in sds])
        elif parts[0] == "blocks" and parts[2] == "router":
            new[name] = torch.randn_like(new[name]) * 0.02
        else:
            new[name] = torch.stack([sd[name] for sd in sds]).float().mean(0).to(new[name].dtype)
    m.load_state_dict(new)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, help="scale point, e.g. 16f (experts sc16f_<type>)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    experts = [M.load(os.path.join(CKPT, f"sc{a.point}_{t}.pt"), a.device)[0] for t in TYPES]
    m = merge(experts, a.k)
    out = os.path.join(CKPT, f"sc{a.point}_btx_init.pt")
    M.save(out, m, experts=[f"sc{a.point}_{t}" for t in TYPES])
    print(f"{out}: {m.n_params() / 1e6:.1f}M parameters, {m.n_active() / 1e6:.1f}M active, "
          f"{m.flops_per_token() / 1e6:.0f} MFLOP/token (expert {experts[0].flops_per_token() / 1e6:.0f})")


if __name__ == "__main__":
    main()
