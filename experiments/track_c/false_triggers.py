"""Track C: false triggers of the circuit expert on held-out text, and text with wrong arithmetic.

    python experiments/track_c/false_triggers.py [--only scan stress]

The circuit expert (circuit_expert.py) makes two kinds of claim about the next token:
  prediction  after "<non-digit> a op b =" (or "=-"), the next answer digit (weight ~0.97 in the gate)
  veto        after a digit of b, the wrong one of "=" / "=-" for the result's sign; after a
              complete answer, every digit. A vetoed token gets log-prob -30 after mixing.
A claim is false when the text's actual next token differs (prediction) or is vetoed (veto).

  scan     every held-out set (lang, facts, math, code val and the four mixed formats) in
           non-overlapping 512-token windows: counts of true and false predictions and vetoes.
           On the windows that contain a false claim (and up to 200 windows with true ones), the
           per-token loss of the atom gate (with and without vetoes) and of the conf gate of the
           same experts (no circuit); the extra loss at false claims is reported per set and per
           million tokens of that set.
  stress   prose that states arithmetic (200 items per kind): "correct", "wrong" (the stated
           result is off by a small amount), "sign" (a - b with a < b written without the minus,
           e.g. 12-20=8). Summed loss of the tokens from "=" to the end of the stated result,
           for the atom gate (vetoes on and off), the conf gate and mono_s.
Writes results/track_c/false_triggers.json.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_c import evaluate as E, synth  # noqa: E402
from experiments.track_c.circuit_expert import CircuitExpert, DIGITS, EQ, EQ_NEG  # noqa: E402
from experiments.track_c.data import OUT as DATA, EOT  # noqa: E402
from experiments.track_c.train import tokens  # noqa: E402

SETS = ["lang_val", "facts_val", "math_val", "code_val", "mixed_val", "mixed_long_val", "mixed_inline_val",
        "mixed_marked_val"]
P = "main_ctxv20"
SYSTEMS = {"atom": f"gate:{P}_gate_atom:prob", "atom_noveto": f"gate:{P}_gate_atom:prob_noveto",
           "conf": f"gate:{P}_gate_conf:prob"}
CTX = 512


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))


def classify(circ, x, y):
    """Per position of one window: 0 none, 1 true prediction, 2 false prediction, 3 true veto
    (a veto that the text respects), 4 false veto. A position with both a prediction and a veto
    is classed by the prediction."""
    tgt, vt = circ.analyse(x.tolist())
    c = np.zeros(len(x), np.int8)
    digit = np.isin(y, list(circ.digit))
    viol = ((vt == DIGITS) & digit) | ((vt == EQ) & (y == circ.eq)) | ((vt == EQ_NEG) & (y == circ.eq_neg))
    c[(vt != 0) & ~viol] = 3
    c[viol] = 4
    c[(tgt >= 0) & (tgt == y)] = 1
    c[(tgt >= 0) & (tgt != y)] = 2
    return c


@torch.no_grad()
def token_losses(sys_, X, bs=16):
    out = []
    for i in range(0, len(X), bs):
        xy = torch.from_numpy(np.stack(X[i:i + bs]).astype(np.int64)).cuda()
        lp, _ = sys_.logprobs(xy[:, :-1])
        out.append(-lp.gather(-1, xy[:, 1:].unsqueeze(-1)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, CTX))


def scan(tok, systems):
    circ = CircuitExpert(tok, backend="python")
    names = ["none", "true_pred", "false_pred", "true_veto", "false_veto"]
    res, keep = {}, {}
    rng = np.random.default_rng(0)
    for s in SETS:
        arr = np.asarray(tokens(s))
        n_win = (len(arr) - 1) // CTX
        counts = np.zeros(5, np.int64)
        false_w, true_w = [], []
        for w in range(n_win):
            win = arr[w * CTX:(w + 1) * CTX + 1].astype(np.int64)
            c = classify(circ, win[:-1], win[1:])
            counts += np.bincount(c, minlength=5)
            if (c == 2).any() or (c == 4).any():
                false_w.append((win, c))
            elif (c == 1).any() or (c == 3).any():
                true_w.append((win, c))
        if len(true_w) > 200:
            true_w = [true_w[i] for i in rng.choice(len(true_w), 200, replace=False)]
        res[s] = {"tokens": int(n_win * CTX), "windows": n_win, "counts": dict(zip(names, map(int, counts))),
                  "windows_with_false": len(false_w)}
        keep[s] = false_w + true_w
        print(s, json.dumps(res[s]), flush=True)
    for key, spec in systems.items():
        sys_ = E.make_system(spec)
        for s in SETS:
            if not keep[s]:
                continue
            L = token_losses(sys_, [w for w, _ in keep[s]])
            C = np.stack([c for _, c in keep[s]])
            for k, nm in enumerate(names[1:], start=1):
                m = C == k
                res[s].setdefault("loss_sum", {}).setdefault(key, {})[nm] = float(L[m].sum())
                res[s].setdefault("n_scored", {})[nm] = int(m.sum())
        del sys_
        torch.cuda.empty_cache()
    for s in SETS:
        ls = res[s].get("loss_sum")
        if not ls:
            continue
        extra = {}
        for v in ("atom", "atom_noveto"):
            for nm in ("false_pred", "false_veto", "true_pred"):
                extra[f"{v}/{nm}"] = ls[v][nm] - ls["conf"][nm]
        res[s]["extra_nats_vs_conf"] = extra
        # false claims are all scored (true ones are sampled), so this is the set's whole cost
        false_cost = extra["atom/false_pred"] + extra["atom/false_veto"]
        res[s]["false_cost_nats_per_token"] = false_cost / res[s]["tokens"]
    return res


def stress_items(n=200, seed=3000):
    rng = random.Random(seed)
    items = []
    for kind in ("correct", "wrong", "sign"):
        for _ in range(n):
            k = rng.choice(synth.KIDS)
            if kind == "sign":
                a = rng.randint(10, 899)
                b = rng.randint(a + 1, 999)
                q, shown = f"{a}-{b}", str(b - a)
            else:
                op = rng.choice("+-*")
                a, b = rng.randint(10, 999), rng.randint(10, 999)
                if op == "-" and a < b:
                    a, b = b, a
                if op == "*":
                    b = rng.randint(2, 99)
                right = a + b if op == "+" else a - b if op == "-" else a * b
                shown = right if kind == "correct" else right + rng.choice([-1, 1]) * rng.choice([1, 2, 10, 100])
                q, shown = f"{a}{op}{b}", str(shown)
            pre = f"{k} wrote {q}"
            items.append({"kind": kind, "prefix": pre, "text": f"{pre}={shown} on the board.", "shown": shown})
    return items


def stress(tok, systems):
    eot = tok.token_to_id(EOT)
    items = stress_items()
    enc = [[eot] + tok.encode(it["text"]).ids for it in items]
    # scored tokens: from "=" (or "=-") to the end of the stated result
    spans = []
    for it, e in zip(items, enc):
        p = len(tok.encode(it["prefix"]).ids) + 1          # index of "=" in e (after <|eot|>)
        spans.append((p, p + len(tok.encode("=" + it["shown"]).ids)))
    res = {}
    for key, spec in list(systems.items()) + [("mono_s", "model:main_mono_s")]:
        sys_ = E.make_system(spec)
        per = []
        for i in range(0, len(enc), 64):
            sub = enc[i:i + 64]
            L = max(map(len, sub))
            x = torch.tensor([e + [eot] * (L - len(e)) for e in sub]).cuda()
            lp, _ = sys_.logprobs(x[:, :-1])
            nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1).float().cpu().numpy()
            for r, (a, b) in enumerate(spans[i:i + 64]):
                per.append(float(nll[r, a - 1:b - 1].sum()))      # target t is predicted at t-1
        res[key] = {kind: float(np.mean([v for v, it in zip(per, items) if it["kind"] == kind]))
                    for kind in ("correct", "wrong", "sign")}
        print(key, json.dumps(res[key]), flush=True)
        del sys_
        torch.cuda.empty_cache()
    return {"nats_per_item": res, "n_per_kind": 200, "example": {k: next(i["text"] for i in items if i["kind"] == k)
                                                                  for k in ("correct", "wrong", "sign")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["scan", "stress"]
    tok = tokenizer()
    path = os.path.join(ROOT, "results", "track_c", "false_triggers.json")
    out = json.load(open(path))["data"] if os.path.exists(path) else {}
    if "scan" in steps:
        out["scan"] = scan(tok, SYSTEMS)
        save("track_c", "false_triggers", out)
    if "stress" in steps:
        out["stress"] = stress(tok, SYSTEMS)
        save("track_c", "false_triggers", out)


if __name__ == "__main__":
    main()
