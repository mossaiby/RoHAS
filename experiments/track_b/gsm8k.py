"""Track B, step 5: GSM8K accuracy with 1.5B experts under the gate, with the compiled circuit.

    python experiments/track_b/gsm8k.py [--only head loss gen] [--systems ...] [--limit N] [--bs 16]

Experts: Qwen2.5-1.5B (base15), Qwen2.5-Coder-1.5B (coder15), Qwen2.5-Math-1.5B (math), frozen,
bf16; their outputs are cached by cache.py --step 5 and the gate b5_gate_s0 is trained by
gate.py --step 5 (m3c_train: GSM8K test even items without annotations). Pre-registered in
docs/roadmap.md (Track B, step 5).

  head  the circuit's two-level head (3 parameters), fitted as in step 3 (circuit_gate.py) on
        a4_train with the Track C trigger, against b5_gate_s0, 10,000 full-batch Adam steps
  loss  next-token loss of each expert, the gate and the gate with the circuit on prose_val,
        code_val, notebooks_val, a4_blank_val, a4_inline_val, mathc_val (Track C trigger)
  gen   8-shot GSM8K, greedy, KV-cached, on the 659 odd test items. The circuit is GsmCircuit:
        active only on `<<a op b=` (a, b non-negative decimals, op + - * /). Systems: base15,
        coder15, math, gate, gate_circuit, base15_circuit, gate_calc, base15_calc (the
        calculator of the original GSM8K paper: an expression inside an open "<<" is evaluated
        exactly when "=" is written, and its value and ">>" are inserted).
Writes checkpoints/track_b/b5_circuit_head.pt, results/track_b/gsm8k.json and the decoded
outputs results/track_b/gsm8k_outputs.json.
"""
import argparse
import ast
import glob
import json
import os
import random
import re
import sys
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from circuits import arith  # noqa: E402
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_b.cache import CACHE, CTX, EXPERTS  # noqa: E402
from experiments.track_b.circuit_gate import LOGV, V, circuit, head_logw  # noqa: E402
from experiments.track_b.data import TYPES, tokenizer  # noqa: E402
from experiments.track_b.gate import CKPT, GateB, batch, gate_logw, labels, load  # noqa: E402
from experiments.track_c.circuit_expert import DIGITS, EQ, EQ_NEG, CircuitExpert  # noqa: E402
from experiments.track_c.data import HUB  # noqa: E402

LM = ["base15", "coder15", "math"]
GATE = "b5_gate_s0"
HEAD = os.path.join(CKPT, "b5_circuit_head.pt")
SYSTEMS = ["base15", "coder15", "math", "gate", "gate_circuit", "base15_circuit", "gate_calc", "base15_calc"]
SHOTS, MAX_NEW = 8, 384


# ---- the circuit on GSM8K annotations ---------------------------------------------------
class GsmCircuit(CircuitExpert):
    """Track C's circuit expert, active only on `<<a op b=` with a, b non-negative decimals and op
    one of + - * /, "<<" immediately before a and "=" (or "=-") immediately after b. The answer is
    circuits.arith.evaluate (division to 6 places), trailing fractional zeros and "." removed."""

    def __init__(self, tok):
        super().__init__(tok.backend_tokenizer, backend="circuit")
        self.V = V
        self.lp_on, self.lp_off = circuit().lp_on, circuit().lp_off
        vocab = tok.get_vocab()
        self.ops = {tok.convert_tokens_to_ids(o): o for o in "+-*/"}
        self.dot = tok.convert_tokens_to_ids(".")
        self.open = {i for t, i in vocab.items() if t.endswith("<<")}
        assert None not in self.ops and self.dot is not None and self.open
        self.sym_id = {**self.digit_id, ".": self.dot}

    def answer(self, a, op, b):
        key = (a, op, b)
        if key not in self.cache:
            self.calls += 1
            try:
                r = arith.evaluate(a, op, b)
            except ZeroDivisionError:
                r = None
            if r is not None and "." in r:
                r = r.rstrip("0").rstrip(".")
                r = "0" if r in ("", "-", "-0") else r
            self.cache[key] = r
        return self.cache[key]

    def _number(self, ids, i):
        """End of the decimal starting at ids[i] (digits, then optionally "." and digits), or i."""
        D, n, j = self.digit, len(ids), i
        while j < n and ids[j] in D:
            j += 1
        if j > i and j + 1 < n and ids[j] == self.dot and ids[j + 1] in D:
            j += 1
            while j < n and ids[j] in D:
                j += 1
        return j

    def _text(self, ids):
        return "".join(self.digit.get(t, ".") for t in ids)

    def analyse(self, ids):
        n = len(ids)
        out = np.full(n, -1, dtype=np.int64)
        veto = np.zeros(n, dtype=np.int8)
        D = self.digit
        for i in range(1, n):
            if ids[i - 1] not in self.open or ids[i] not in D:
                continue
            j = self._number(ids, i)
            if j >= n or ids[j] not in self.ops:
                continue
            a, op = self._text(ids[i:j]), self.ops[ids[j]]
            k, dot = j + 1, False
            while k < n:
                if ids[k] in D:                       # b so far ends here: the sign rules out "=" or "=-"
                    neg = op == "-" and (self.answer(a, op, self._text(ids[j + 1:k + 1])) or "").startswith("-")
                    veto[k] = EQ if neg else EQ_NEG
                elif ids[k] == self.dot and not dot and k > j + 1 and k + 1 < n and ids[k + 1] in D:
                    dot = True
                else:
                    break
                k += 1
            if k == j + 1 or k >= n or ids[k] not in (self.eq, self.eq_neg):
                continue
            ans = self.answer(a, op, self._text(ids[j + 1:k]))
            if ans is None or (ids[k] == self.eq_neg) != ans.startswith("-"):
                continue
            p = k
            for c in ans.lstrip("-"):
                out[p] = self.sym_id[c]
                if p + 1 >= n or ids[p + 1] != self.sym_id[c]:
                    break
                p += 1
            else:
                veto[p] = DIGITS                       # complete: no digit and no "." follows
        return out, veto

    def last(self, rows, device):
        """Log-probs (B, V) and veto mask (B, V) at the last position of each token list."""
        lp = torch.full((len(rows), self.V), -LOGV, device=device)
        mask = torch.zeros((len(rows), self.V), dtype=torch.bool, device=device)
        dig = torch.tensor(sorted(self.digit) + [self.dot], device=device)
        for r, ids in enumerate(rows):
            tgt, vt = self.analyse(ids[-200:])        # an annotation is far shorter than 200 tokens
            t, v = int(tgt[-1]), int(vt[-1])
            if t >= 0:
                lp[r] = self.lp_off
                lp[r, t] = self.lp_on
            if v == DIGITS:
                mask[r, dig] = True
            elif v == EQ:
                mask[r, self.eq] = True
            elif v == EQ_NEG:
                mask[r, self.eq_neg] = True
        return lp, mask


# ---- the calculator tool ----------------------------------------------------------------
def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        x, y = _ev(node.left), _ev(node.right)
        return x + y if isinstance(node.op, ast.Add) else x - y if isinstance(node.op, ast.Sub) else \
            x * y if isinstance(node.op, ast.Mult) else x / y
    raise ValueError("unsupported")


def calc_value(expr):
    expr = expr.replace(",", "").strip()
    if not expr or not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        return None
    try:
        return _ev(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def fmt(v):
    if v.denominator == 1:
        return str(v.numerator)
    q = round(abs(v) * 10 ** 6)
    s = f"{q // 10 ** 6}.{q % 10 ** 6:06d}".rstrip("0").rstrip(".")
    return ("-" if v < 0 and s.strip("0.") else "") + s


# ---- GSM8K ------------------------------------------------------------------------------
def gsm8k_raw(split):
    import pyarrow.parquet as pq
    f = glob.glob(os.path.join(HUB, "datasets--openai--gsm8k", "snapshots", "*", "main", f"{split}-*.parquet"))[0]
    return pq.read_table(f).to_pylist()


def prompt_prefix():
    train = gsm8k_raw("train")
    shots = [train[i] for i in random.Random(0).sample(range(len(train)), SHOTS)]
    return "".join(f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}\n\n" for r in shots)


def number(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def score(out, gold):
    g = number(gold)
    m = re.search(r"####\s*(-?[\d,.$]+)", out)
    strict = int(m is not None and number(m.group(1)) == g)
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", out)
    flexible = int(bool(nums) and number(nums[-1]) == g)
    return strict, flexible


COVERED = re.compile(r"\d+(?:\.\d+)?[+\-*/]\d+(?:\.\d+)?")


def annotations(out):
    """Complete annotations <<expr=value>>: (covered by the circuit, value correct)."""
    res = []
    for m in re.finditer(r"<<([^<>=]*)=([^<>]*)>>", out):
        expr, val = m.group(1), m.group(2).strip()
        v, r = calc_value(expr), number(val)
        places = len(val.split(".")[1]) if "." in val else 0
        ok = v is not None and r is not None and abs(v - r) < Fraction(1, 10 ** places)
        res.append((bool(COVERED.fullmatch(expr.strip())), bool(ok)))
    return res


# ---- live generation, KV-cached ---------------------------------------------------------
class Live:
    def __init__(self, gate, wb, tok):
        self.loaded = {}                               # loaded when a system first needs them
        self.gate, self.wb, self.tok = gate, wb.cuda(), tok
        self.circ = GsmCircuit(tok)

    def model(self, e):
        if e not in self.loaded:
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(EXPERTS[e], dtype=torch.bfloat16).cuda().eval()
            assert m.get_output_embeddings().weight.shape[0] == V, e
            self.loaded[e] = m
        return self.loaded[e]

    @staticmethod
    def need(system):
        base = system.split("_")[0]
        return LM if base == "gate" else [base]

    @torch.no_grad()
    def prefill(self, names, rows):
        B, L = len(rows), max(map(len, rows))
        pad = self.tok.eos_token_id
        ids = torch.full((B, L), pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for r, x in enumerate(rows):
            ids[r, L - len(x):] = torch.tensor(x)
            mask[r, L - len(x):] = 1
        ids, mask = ids.cuda(), mask.cuda()
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        st = {}
        for e in names:
            m = self.model(e)
            o = m.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
            h = o.last_hidden_state
            st[e] = {"past": o.past_key_values, "h": h[:, -self.gate.window:].float(),
                     "lp": torch.log_softmax(m.get_output_embeddings()(h[:, -1]).float(), -1)}
        return st, mask, pos[:, -1]

    @torch.no_grad()
    def advance(self, names, st, nxt, mask, last):
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        last = last + 1
        for e in names:
            m = self.model(e)
            o = m.model(input_ids=nxt.unsqueeze(1), attention_mask=mask, position_ids=last.unsqueeze(1),
                        past_key_values=st[e]["past"], use_cache=True)
            h = o.last_hidden_state
            st[e] = {"past": o.past_key_values, "h": torch.cat([st[e]["h"][:, 1:], h.float()], 1),
                     "lp": torch.log_softmax(m.get_output_embeddings()(h[:, -1]).float(), -1)}
        return mask, last

    @torch.no_grad()
    def logprobs(self, system, st, rows):
        base = system.split("_")[0]
        if base == "gate":
            lp = torch.stack([st[e]["lp"] for e in LM], 1)                      # (B, E, V)
            hl = torch.cat([st[e]["h"][:, -1] for e in LM], -1)
            mean = torch.cat([st[e]["h"].mean(1) for e in LM], -1)
            cf = torch.cat([-(lp.exp() * lp).sum(-1), lp.amax(-1)], -1) / LOGV  # entropies, then maxima
            z = self.gate.net(torch.cat([self.gate.ln(torch.cat([hl, mean], -1)), cf], -1))
            lm = torch.logsumexp(torch.log_softmax(z, -1).unsqueeze(-1) + lp, 1)
        else:
            lm = st[base]["lp"]
        if not system.endswith("_circuit"):
            return lm
        lc, veto = self.circ.last(rows, lm.device)
        feat = torch.stack([-(lc.exp() * lc).sum(-1), lc.amax(-1)], -1)
        a = feat @ self.wb[:2] + self.wb[2]
        mix = torch.logaddexp(F.logsigmoid(a).unsqueeze(-1) + lc, F.logsigmoid(-a).unsqueeze(-1) + lm)
        mix = mix.masked_fill(veto, -30.0)
        return mix - torch.logsumexp(mix, -1, keepdim=True)


def stopped(text):
    return "\n\n" in text or "Question:" in text


def decode_batch(live, system, prompts, max_new):
    tok, eos = live.tok, live.tok.eos_token_id
    names = live.need(system)
    rows = [[eos] + tok(p, add_special_tokens=False).input_ids for p in prompts]
    gen = [[] for _ in prompts]
    queue = [[] for _ in prompts]
    done = [False] * len(prompts)
    calls = 0
    st, mask, last = live.prefill(names, rows)
    for _ in range(max_new):
        nxt = live.logprobs(system, st, rows).argmax(-1).tolist()
        for r in range(len(rows)):
            if done[r]:
                nxt[r] = eos
                continue
            if queue[r]:
                nxt[r] = queue[r].pop(0)
            rows[r].append(nxt[r])
            gen[r].append(nxt[r])
            text = tok.decode(gen[r])
            if nxt[r] == eos or stopped(text):
                done[r] = True
            elif system.endswith("_calc") and not queue[r] and text.endswith("="):
                o, c = text.rfind("<<"), text.rfind(">>")
                if o > c:
                    v = calc_value(text[o + 2:-1])
                    if v is not None:
                        queue[r] = tok(fmt(v) + ">>", add_special_tokens=False).input_ids
                        calls += 1
        if all(done):
            break
        mask, last = live.advance(names, st, torch.tensor(nxt, device="cuda"), mask, last)
    outs = []
    for g in gen:
        t = tok.decode(g)
        outs.append(re.split(r"\n\n|Question:", t)[0].strip())
    return outs, calls


def gen_eval(gate, wb, systems, limit, bs):
    tok = tokenizer()
    live = Live(gate, wb, tok)
    items = gsm8k_raw("test")[1::2]                       # odd items: never seen by the gate
    if limit:
        items = items[:limit]
    prefix = prompt_prefix()
    prompts = [prefix + f"Question: {r['question'].strip()}\nAnswer:" for r in items]
    gold = [r["answer"].split("####")[-1].strip() for r in items]
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    opath = os.path.join(ROOT, "results", "track_b", "gsm8k_outputs.json")
    outputs = json.load(open(opath)) if os.path.exists(opath) else {}
    res = {}
    for system in systems:
        t0 = time.time()
        outs, calls = [None] * len(prompts), 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            o, c = decode_batch(live, system, [prompts[i] for i in idx], MAX_NEW)
            calls += c
            for i, x in zip(idx, o):
                outs[i] = x
            print(f"  {system}: {min(s + bs, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
        sc = [score(o, g) for o, g in zip(outs, gold)]
        ann = [annotations(o) for o in outs]
        flat = [x for a in ann for x in a]
        cov = [ok for c, ok in flat if c]
        unc = [ok for c, ok in flat if not c]
        res[system] = {"strict": float(np.mean([s for s, _ in sc])), "flexible": float(np.mean([f for _, f in sc])),
                       "correct": [s for s, _ in sc],
                       "annotations": {"covered": len(cov), "covered_correct": int(sum(cov)),
                                       "uncovered": len(unc), "uncovered_correct": int(sum(unc))},
                       "calculator_calls": calls, "items": len(items), "seconds": round(time.time() - t0)}
        outputs[system] = outs
        print(system, json.dumps({k: v for k, v in res[system].items() if k != "correct"}), flush=True)
        if not limit:
            json.dump(outputs, open(opath, "w"), indent=0)
    return res


def paired(a, b, n=10000, seed=0):
    """Mean of b - a and its 95% bootstrap interval over items."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "b_only": int(((d > 0)).sum()), "a_only": int((d < 0).sum())}


# ---- head and loss (as step 3, over the 1.5B gate) --------------------------------------
def load_gate():
    ck = torch.load(os.path.join(CKPT, f"{GATE}.pt"), map_location="cuda", weights_only=False)
    assert ck["experts"] == LM, ck["experts"]
    g = GateB(ck["dims"]).cuda()
    g.load_state_dict(ck["gate"])
    return g.eval()


@torch.no_grad()
def per_token(f, gate, circ):
    caches = [load(e, f) for e in LM]
    X = np.load(os.path.join(CACHE, f"{f}.x.npy"))
    W = X.shape[0]
    tl, lg = [], []
    for i in range(0, W, 16):
        hs, cf, t = batch(caches, np.arange(i, min(W, i + 16)))
        tl.append(t.cpu())
        lg.append(torch.logsumexp(gate_logw(gate, hs, cf) + t, -1).cpu())
    tlp, lp_gate = torch.cat(tl), torch.cat(lg)
    act = np.zeros((W, CTX), bool)
    lp_c = np.full((W, CTX), -LOGV, np.float32)
    for w in range(W):
        tgt, _ = circ.analyse(X[w, :-1].tolist())
        on = tgt >= 0
        act[w] = on
        lp_c[w, on] = np.where(tgt[on] == X[w, 1:][on], circ.lp_on, circ.lp_off)
    return tlp, lp_gate, torch.from_numpy(act), torch.from_numpy(lp_c), labels(f, W)


def train_head(gate, circ):
    _, lp_gate, act, lp_c, _ = per_token("a4_train", gate, circ)
    lp_gate, act, lp_c = lp_gate.cuda().flatten(), act.cuda().flatten(), lp_c.cuda().flatten()
    wb = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([wb], lr=3e-2)
    for _ in range(10000):
        lc, lr = head_logw(wb, act)
        loss = -torch.logaddexp(lc + lp_c, lr + lp_gate).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    lc, _ = head_logw(wb.detach(), act)
    info = {"w": wb.detach().cpu().tolist(), "train_loss": float(loss), "active_frac": float(act.float().mean()),
            "lambda_active": float(lc[act].exp().mean()), "lambda_inactive": float(lc[~act].exp().mean())}
    print("head", json.dumps(info), flush=True)
    torch.save({"wb": wb.detach().cpu(), "info": info}, HEAD)
    return wb.detach().cpu(), info


def loss_eval(gate, circ, wb):
    out = {}
    for f in ["prose_val", "code_val", "notebooks_val", "a4_blank_val", "a4_inline_val", "mathc_val"]:
        tlp, lp_gate, act, lp_c, lab = per_token(f, gate, circ)
        lc, lr = head_logw(wb, act)
        lp_gc = torch.logaddexp(lc + lp_c, lr + lp_gate)
        sysl = {e: -tlp[..., k] for k, e in enumerate(LM)}
        sysl.update({"gate": -lp_gate, "gate_circuit": -lp_gc})
        r = {"tokens": int(tlp.shape[0] * CTX), "active_tokens": int(act.sum()),
             "loss": {k: float(v.mean()) for k, v in sysl.items()},
             "lambda_active": float(lc[act].exp().mean()) if act.any() else None,
             "lambda_inactive": float(lc[~act].exp().mean())}
        if act.any():
            r["loss_active"] = {k: float(v[act].mean()) for k, v in sysl.items()}
        if lab is not None:
            L = torch.from_numpy(lab.astype(np.int64))
            for t, nm in enumerate(TYPES):
                m = L == t
                if m.any():
                    r[f"loss_{nm}"] = {k: float(v[m].mean()) for k, v in sysl.items()}
        out[f] = r
        print(f, {k: round(v, 4) for k, v in r["loss"].items()}, "active", r["active_tokens"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--systems", nargs="*", default=SYSTEMS)
    ap.add_argument("--limit", type=int, help="first N odd items only (smoke test; not written to gsm8k.json)")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    steps = a.only or ["head", "loss", "gen"]
    gate, circ = load_gate(), circuit()
    path = os.path.join(ROOT, "results", "track_b", "gsm8k.json")
    out = json.load(open(path))["data"] if os.path.exists(path) else {}
    if "head" in steps or not os.path.exists(HEAD):
        wb, out["head"] = train_head(gate, circ)
        save("track_b", "gsm8k", out)
    else:
        wb = torch.load(HEAD, weights_only=False)["wb"]
    if "loss" in steps:
        out["loss"] = loss_eval(gate, circ, wb)
        save("track_b", "gsm8k", out)
    if "gen" in steps:
        res = gen_eval(gate, wb, a.systems, a.limit, a.bs)
        if a.limit:
            print("smoke test: not saved", flush=True)
            return
        out.setdefault("gen", {}).update(res)
        g = out["gen"]
        out["compare"] = {f"{x}->{y}": paired(g[x]["correct"], g[y]["correct"])
                          for x, y in [("gate", "gate_circuit"), ("base15", "base15_circuit"), ("gate", "gate_calc"),
                                       ("gate_circuit", "gate_calc"), ("base15", "gate")] if x in g and y in g}
        save("track_b", "gsm8k", out)


if __name__ == "__main__":
    main()
