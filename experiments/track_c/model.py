"""A small GPT (pre-LayerNorm, GELU MLP, learned positions, tied embeddings, no biases)."""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 8192
    ctx: int = 512
    d: int = 384
    layers: int = 8
    heads: int = 6
    dropout: float = 0.0


SIZES = {
    "s": dict(d=384, layers=8, heads=6),     # about 17M parameters: one expert
    "m": dict(d=768, layers=9, heads=12),    # about 4x: the larger monolithic baseline
    "xs": dict(d=128, layers=2, heads=2),    # smoke tests
}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d, bias=False), nn.LayerNorm(c.d, bias=False)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.fc = nn.Linear(c.d, 4 * c.d, bias=False)
        self.out = nn.Linear(4 * c.d, c.d, bias=False)
        self.h = c.heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.d, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, hidden=False):
        """Logits (B, T, V); with hidden=True also the final normalised state (B, T, d)."""
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x)
        logits = h @ self.wte.weight.T
        return (logits, h) if hidden else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self):
        """Training FLOPs per token: 6 x (weights excluding position embeddings) plus attention."""
        c = self.c
        n = self.n_params() - self.wpe.weight.numel()
        return 6 * n + 12 * c.layers * c.d * c.ctx


def make(size, vocab, ctx=512):
    return GPT(Config(vocab=vocab, ctx=ctx, **SIZES[size]))


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(Config(**ck["config"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def save(path, model, **extra):
    torch.save({"config": asdict(model.c), "model": model.state_dict(), **extra}, path)
