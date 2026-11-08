"""Evaluate track C systems: single models and gated sets of experts.

    python experiments/track_c/evaluate.py --tag pilot \
        --systems model:mono_s model:mono_m model:lang model:math gate:gate_prob:prob \
                  gate:gate_prob:hard gate:gate_logit:logit

Metrics (same items for every system)
  loss      mean next-token loss (nats) on lang / facts / math / code / mixed validation
  facts     recall: the correct value must outrank every other value of that attribute
            (summed log-probability of " value"); prompts from held-out and training templates
  arith     exact match of greedy a op b = ..., by operator and digit count (training: +,- up
            to 5 digits, * up to 3)
  word      exact match of the final answer in held-out word problems
  routing   (gated systems) mean gate weight per true segment type on mixed validation text
Writes results/track_c/eval_<tag>.json.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c import model as M, synth  # noqa: E402
from experiments.track_c.data import OUT as DATA, TYPES  # noqa: E402
from experiments.track_c.gate import load_gate, load_experts, expert_outputs, combine  # noqa: E402
from experiments.track_c.train import CKPT, tokens, batch  # noqa: E402


class Single:
    def __init__(self, name):
        self.name = f"model:{name}"
        self.m, _ = M.load(os.path.join(CKPT, f"{name}.pt"))

    @torch.no_grad()
    def logprobs(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.log_softmax(self.m(x).float(), dim=-1), None


class Gated:
    _cache = {}

    def __init__(self, gate_name, mode):
        self.name = f"gate:{gate_name}:{mode}"
        self.gate, ck = load_gate(gate_name)
        key = tuple(ck["experts"])
        if key not in Gated._cache:
            Gated._cache[key] = load_experts(ck["experts"])
        self.experts, self.mode, self.expert_names = Gated._cache[key], mode, ck["experts"]

    @torch.no_grad()
    def logprobs(self, x):
        logps, hs = expert_outputs(self.experts, x)
        ll = self.gate(hs)
        return combine(logps, ll, self.mode), ll


def make_system(spec):
    kind, *rest = spec.split(":")
    return Single(rest[0]) if kind == "model" else Gated(rest[0], rest[1])


# ---- metrics -----------------------------------------------------------------------
@torch.no_grad()
def val_losses(sys_, n_batches=40, bs=8, ctx=512):
    out = {}
    for s in TYPES + ["mixed"]:
        arr = tokens(f"{s}_val")
        rng = np.random.default_rng(4321)
        tot = 0.0
        for _ in range(n_batches):
            xy = batch(arr, bs, ctx, rng).cuda()
            lp, _ = sys_.logprobs(xy[:, :-1])
            tot += F.nll_loss(lp.reshape(-1, lp.shape[-1]), xy[:, 1:].reshape(-1)).item()
        out[s] = tot / n_batches
    return out


@torch.no_grad()
def routing(sys_, n_batches=40, bs=8, ctx=512):
    arr = tokens("mixed_val")
    lab = np.memmap(os.path.join(DATA, "mixed_val_labels.bin"), dtype=np.uint8, mode="r")
    rng = np.random.default_rng(99)
    sums = np.zeros((len(TYPES), len(sys_.expert_names)))
    hits = np.zeros(len(TYPES))
    cnt = np.zeros(len(TYPES))
    for _ in range(n_batches):
        ix = rng.integers(0, len(arr) - ctx - 1, bs)
        x = torch.from_numpy(np.stack([arr[i:i + ctx] for i in ix]).astype(np.int64)).cuda()
        y = np.stack([lab[i + 1:i + ctx + 1] for i in ix])           # type of the predicted token
        _, ll = sys_.logprobs(x)
        lam = ll.exp().cpu().numpy()
        for t in range(len(TYPES)):
            mask = y == t
            if mask.any():
                sums[t] += lam[mask].sum(0)
                cnt[t] += mask.sum()
                if len(sys_.expert_names) == len(TYPES):      # experts are ordered as TYPES
                    hits[t] += (lam[mask].argmax(-1) == t).sum()
    return {"mean_lambda": {TYPES[t]: dict(zip(sys_.expert_names, (sums[t] / max(1, cnt[t])).round(4).tolist()))
                            for t in range(len(TYPES))},
            "argmax_accuracy": {TYPES[t]: float(hits[t] / max(1, cnt[t])) for t in range(len(TYPES))}}


@torch.no_grad()
def score_continuations(sys_, tok, pairs, bs=None):
    """Summed log p(cont | <eot> prefix) for each (prefix, cont) string pair."""
    bs = bs or (64 if isinstance(sys_, Gated) else 256)
    eot = tok.token_to_id("<|eot|>")
    cache = {}

    def enc_text(t):
        if t not in cache:
            cache[t] = tok.encode(t).ids
        return cache[t]
    enc = [([eot] + enc_text(p), enc_text(" " + c)) for p, c in pairs]
    out = np.zeros(len(enc))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(enc[i][0]) + len(enc[i][1]) for i in idx)
        x = torch.full((len(idx), L), eot, dtype=torch.long)
        mask = torch.zeros((len(idx), L - 1))
        for r, i in enumerate(idx):
            pi, ci = enc[i]
            x[r, :len(pi) + len(ci)] = torch.tensor(pi + ci)
            mask[r, len(pi) - 1:len(pi) + len(ci) - 1] = 1       # positions predicting cont tokens
        x, mask = x.cuda(), mask.cuda()
        lp, _ = sys_.logprobs(x[:, :-1])
        tgt = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        out[idx] = (tgt * mask).sum(1).cpu().numpy()
    return out


def fact_recall(sys_, tok, persons, heldout, n_people=300, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(persons, min(n_people, len(persons)))
    prompts = []
    for p in sample:
        fp = synth.fact_prompts(p, heldout=heldout)
        if not heldout:  # one training template per attribute
            by = {}
            for a, pr, v in fp:
                by.setdefault(a, []).append((a, pr, v))
            fp = [rng.choice(v) for v in by.values()]
        prompts += fp
    pairs, spans = [], []
    for a, pr, v in prompts:
        cands = sorted(set(synth.ATTRS[a]))
        spans.append((len(pairs), cands.index(v), len(cands)))
        pairs += [(pr, c) for c in cands]
    sc = score_continuations(sys_, tok, pairs)
    by_attr = {}
    for (a, _, _), (s0, ci, n) in zip(prompts, spans):
        by_attr.setdefault(a, []).append(int(np.argmax(sc[s0:s0 + n]) == ci))
    acc = {a: float(np.mean(v)) for a, v in by_attr.items()}
    acc["all"] = float(np.mean([x for v in by_attr.values() for x in v]))
    return acc


@torch.no_grad()
def greedy(sys_, tok, prompts, max_new, stop="\n"):
    eot = tok.token_to_id("<|eot|>")
    enc = [[eot] + tok.encode(p).ids for p in prompts]
    outs = [None] * len(prompts)
    groups = {}
    for i, e in enumerate(enc):
        groups.setdefault(len(e), []).append(i)
    for L, idx in groups.items():
        for s in range(0, len(idx), 128):
            sub = idx[s:s + 128]
            x = torch.tensor([enc[i] for i in sub]).cuda()
            for _ in range(max_new):
                lp, _ = sys_.logprobs(x[:, -512:])
                x = torch.cat([x, lp[:, -1].argmax(-1, keepdim=True)], dim=1)
            for r, i in enumerate(sub):
                ids = x[r, L:].tolist()
                if eot in ids:
                    ids = ids[:ids.index(eot)]
                outs[i] = tok.decode(ids).split(stop)[0]
    return outs


def arithmetic(sys_, tok, per_cell=50, seed=0):
    rng = random.Random(seed)
    cells = [("+", d) for d in range(1, 9)] + [("-", d) for d in range(1, 9)] + [("*", d) for d in range(1, 5)]
    items = []
    for op, d in cells:
        for _ in range(per_cell):
            q, a = synth.arith_line(rng, op, d, d)
            items.append((op, d, q + "=", a))
    outs = greedy(sys_, tok, [q for _, _, q, _ in items], max_new=18)
    res = {}
    for (op, d, _, a), o in zip(items, outs):
        res.setdefault(op, {}).setdefault(str(d), []).append(int(o.strip() == a))
    return {op: {d: float(np.mean(v)) for d, v in ds.items()} for op, ds in res.items()}


def word_problems(sys_, tok, n=200, seed=1):
    rng = random.Random(10_000 + seed)
    items = [synth.word_problem(rng) for _ in range(n)]
    outs = greedy(sys_, tok, [q for q, _ in items], max_new=40)
    ok = []
    for (q, a), o in zip(items, outs):
        want = re.search(r"The answer is (-?\d+)", a).group(1)
        got = re.search(r"The answer is (-?\d+)", o)
        ok.append(int(bool(got) and got.group(1) == want))
    return float(np.mean(ok))


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--people", default="old", choices=["old", "new"])
    args = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    persons = json.load(open(os.path.join(DATA, "people.json")))
    path = os.path.join(os.path.dirname(DATA), "..", "results", "track_c", f"eval_{args.tag}.json")
    results = {}
    if os.path.exists(path):
        results = json.load(open(path))["data"]["systems"]
    t0 = time.time()
    for spec in args.systems:
        s = make_system(spec)
        r = results.get(s.name, {})
        t = time.time()
        if "loss" not in args.skip:
            r["loss"] = val_losses(s)
        if "facts" not in args.skip:
            r["facts_heldout"] = fact_recall(s, tok, persons["old"], heldout=True)
            r["facts_train_templates"] = fact_recall(s, tok, persons["old"], heldout=False)
            if persons.get("new"):
                r["facts_new_heldout"] = fact_recall(s, tok, persons["new"], heldout=True)
                r["facts_new_train_templates"] = fact_recall(s, tok, persons["new"], heldout=False)
        if "math" not in args.skip:
            r["arith"] = arithmetic(s, tok)
            r["word"] = word_problems(s, tok)
        if isinstance(s, Gated):
            r["routing"] = routing(s)
        results[s.name] = r
        print(s.name, f"({time.time() - t:.0f}s)", json.dumps(r), flush=True)
        del s
        torch.cuda.empty_cache()
    save("track_c", f"eval_{args.tag}", {"systems": results}, wall_time=time.time() - t0)


if __name__ == "__main__":
    main()
