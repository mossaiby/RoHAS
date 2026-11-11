"""Track B, step 3: the compiled arithmetic circuit under the frozen pretrained gate.

    python experiments/track_b/circuit_gate.py [--only head loss gen]

Experts: Qwen2.5-0.5B and Qwen2.5-Coder-0.5B (frozen) and the compiled circuit
(track_c/circuit_expert.py, unchanged except for the vocabulary size, 151,936 rows). The LM
gate is the step-1 gate b_gate_s0, frozen. The circuit enters through the two-level gate of
Track C: its weight lambda_c = sigmoid(w . c_circ + b) depends only on its own confidence
features c_circ (entropy and largest log-probability, in nats); the LM experts share 1 - lambda_c.
Only (w, b) are trained.

  head  fit (w, b) on a4_train (cached LM outputs; the circuit is computed from the tokens):
        full-batch Adam, lr 3e-2, 10,000 steps, minimising the mixture's next-token loss (500
        steps, the first setting, had not converged: weight 0.72 where the circuit is active,
        loss still falling; at 10,000 steps the loss changes by less than 1e-5)
  loss  next-token loss on prose_val, code_val, notebooks_val, a4_blank_val, a4_inline_val for
        base, coder, the gate, and the gate with the circuit (without vetoes: for text the
        circuit gets right they can only lower the loss), by segment type and on positions
        where the circuit is active; the circuit's mean weight where active / inactive
  gen   live greedy decoding, vetoes on: held-out equations "a op b" (the first generated
        token restricted to "=" or "=-", as in Track C; eval seed 1000), + and - at 1, 4, 8,
        16, 32 digits and * at 1, 2, 4, 8, 40 per cell; and the same inside other text (a prose
        or code segment, then "\\n\\n" or " ", then the equation; +, -, * at 4 and 16 digits, 20
        per cell). Exact = the output starts with "=" + answer and no digit follows.
Writes checkpoints/track_b/b_circuit_head.pt and results/track_b/circuit_gate.json.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_b.cache import CACHE, CTX, EXPERTS  # noqa: E402
from experiments.track_b.data import OUT, TYPES, prose, segment, tokenizer  # noqa: E402
from experiments.track_b.gate import CKPT, GateB, batch, gate_logw, labels, load  # noqa: E402
from experiments.track_c import synth  # noqa: E402
from experiments.track_c.circuit_expert import CircuitExpert, EPS  # noqa: E402
from experiments.track_c.data import code_files  # noqa: E402
from experiments.track_c.data_mixed_ood import FORMATS  # noqa: E402

V = 151936
LOGV = math.log(V)
LM = ["base", "coder"]
GATE = "b_gate_s0"
H_ON = -(1 - EPS) * math.log(1 - EPS) - EPS * math.log(EPS / (V - 1))
FEAT_ON, FEAT_OFF = (H_ON, math.log(1 - EPS)), (LOGV, -LOGV)


def circuit():
    c = CircuitExpert(tokenizer().backend_tokenizer, backend="circuit")
    c.V = V
    c.lp_on, c.lp_off = math.log(1 - EPS), math.log(EPS / (V - 1))
    return c


def load_gate():
    ck = torch.load(os.path.join(CKPT, f"{GATE}.pt"), map_location="cuda", weights_only=False)
    g = GateB(ck.get("dims") or [ck["d"]] * len(ck["experts"])).cuda()     # step-1 checkpoints store d
    g.load_state_dict(ck["gate"])
    return g.eval()


@torch.no_grad()
def per_token(f, gate, circ):
    """For every target of file f: log p of base, coder, the LM gate, the circuit, and whether the
    circuit is active there."""
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


def head_logw(wb, act):
    on = torch.tensor(FEAT_ON, device=act.device)
    off = torch.tensor(FEAT_OFF, device=act.device)
    f = torch.where(act.unsqueeze(-1), on, off).to(wb.device)
    a = f @ wb[:2] + wb[2]
    return torch.nn.functional.logsigmoid(a), torch.nn.functional.logsigmoid(-a)


def train_head(gate, circ):
    _, lp_gate, act, lp_c, _ = per_token("a4_train", gate, circ)
    lp_gate, act, lp_c = lp_gate.cuda().flatten(), act.cuda().flatten(), lp_c.cuda().flatten()
    wb = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([wb], lr=3e-2)
    for step in range(10000):
        lc, lr = head_logw(wb, act)
        loss = -torch.logaddexp(lc + lp_c, lr + lp_gate).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    lc, _ = head_logw(wb.detach(), act)
    info = {"w": wb.detach().cpu().tolist(), "train_loss": float(loss), "active_frac": float(act.float().mean()),
            "lambda_active": float(lc[act].exp().mean()), "lambda_inactive": float(lc[~act].exp().mean())}
    print("head", json.dumps(info), flush=True)
    torch.save({"wb": wb.detach().cpu(), "info": info}, os.path.join(CKPT, "b_circuit_head.pt"))
    return wb.detach().cpu(), info


def loss_eval(gate, circ, wb):
    out = {}
    for f in ["prose_val", "code_val", "notebooks_val", "a4_blank_val", "a4_inline_val"]:
        tlp, lp_gate, act, lp_c, lab = per_token(f, gate, circ)
        lc, lr = head_logw(wb, act)
        lp_gc = torch.logaddexp(lc + lp_c, lr + lp_gate)
        sysl = {"base": -tlp[..., 0], "coder": -tlp[..., 1], "gate": -lp_gate, "gate_circuit": -lp_gc}
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
        print(f, {k: round(v, 4) for k, v in r["loss"].items()}, "active", r["active_tokens"],
              r.get("loss_active") and {k: round(v, 3) for k, v in r["loss_active"].items()}, flush=True)
    return out


# ---- live generation ---------------------------------------------------------------------
class Live:
    def __init__(self, gate, circ, wb):
        from transformers import AutoModelForCausalLM
        self.models = [AutoModelForCausalLM.from_pretrained(EXPERTS[e], dtype=torch.bfloat16).cuda().eval()
                       for e in LM]
        self.gate, self.circ, self.wb = gate, circ, wb.cuda()

    @torch.no_grad()
    def next_logprobs(self, x, system):
        """(B, V) next-token log-probs at the last position of x (B, T)."""
        need = [0] if system == "base" else [1] if system == "coder" else [0, 1]
        hs, lps = {}, {}
        for k in need:
            m = self.models[k]
            h = m.model(input_ids=x).last_hidden_state
            hs[k] = h.float()
            lps[k] = torch.log_softmax(m.get_output_embeddings()(h[:, -1]).float(), -1)
        if system in ("base", "coder"):
            return lps[need[0]]
        h = torch.cat([hs[0], hs[1]], -1)
        w = min(self.gate.window, h.shape[1])
        mean = h[:, -w:].mean(1)
        lp = torch.stack([lps[0], lps[1]], 1)                                   # (B, 2, V)
        cf = torch.cat([-(lp.exp() * lp).sum(-1), lp.amax(-1)], -1) / LOGV     # entropies, then maxima
        z = self.gate.net(torch.cat([self.gate.ln(torch.cat([h[:, -1], mean], -1)), cf], -1))
        lm = torch.logsumexp(torch.log_softmax(z, -1).unsqueeze(-1) + lp, 1)
        if system == "gate":
            return lm
        lc = self.circ.logprobs(x, last=True)[:, -1].float()
        veto = self.circ.last_veto[:, -1]
        feat = torch.stack([-(lc.exp() * lc).sum(-1), lc.amax(-1)], -1)
        a = feat @ self.wb[:2] + self.wb[2]
        mix = torch.logaddexp(torch.nn.functional.logsigmoid(a).unsqueeze(-1) + lc,
                              torch.nn.functional.logsigmoid(-a).unsqueeze(-1) + lm)
        mix = mix.masked_fill(veto, -30.0)
        return mix - torch.logsumexp(mix, -1, keepdim=True)


def greedy(live, tok, prompts, system, first, max_new):
    eos = tok.eos_token_id
    newline = torch.tensor([i for t, i in tok.get_vocab().items() if "Ċ" in t] + [eos]).cuda()
    enc = [[eos] + tok(p, add_special_tokens=False).input_ids for p in prompts]
    outs = [None] * len(prompts)
    groups = {}
    for i, e in enumerate(enc):
        groups.setdefault(len(e), []).append(i)
    for L, idx in groups.items():
        for s in range(0, len(idx), 32):
            sub = idx[s:s + 32]
            x = torch.tensor([enc[i] for i in sub]).cuda()
            done = torch.zeros(len(sub), dtype=torch.bool, device="cuda")
            for step in range(max_new):
                lp = live.next_logprobs(x, system)
                if step == 0:
                    allowed = torch.tensor(first, device="cuda")
                    lp = torch.full_like(lp, -float("inf")).index_copy(1, allowed, lp[:, allowed])
                x = torch.cat([x, lp.argmax(-1, keepdim=True)], 1)
                done |= torch.isin(x[:, -1], newline)
                if done.all():
                    break
            for r, i in enumerate(sub):
                outs[i] = tok.decode(x[r, L:].tolist()).split("\n")[0]
    return outs


def exact(o, a):
    o, want = o.strip(), "=" + a
    return int(o.startswith(want) and not o[len(want):len(want) + 1].isdigit())


def gen_eval(gate, circ, wb):
    tok = tokenizer()
    live = Live(gate, circ, wb)
    first = [tok.convert_tokens_to_ids("="), tok.convert_tokens_to_ids("=-")]
    rng = random.Random(1000)
    cells = [(op, d) for op in "+-" for d in (1, 4, 8, 16, 32)] + [("*", d) for d in (1, 2, 4, 8)]
    eq = [(op, d) + synth.arith_line(rng, op, d, d) for op, d in cells for _ in range(40)]
    code_val = [t for split, t in code_files() if split == "val"]
    pools = {"prose": prose(range(700, 720)), "code": code_val}
    rng = random.Random(2000)
    ctx = []
    for kind in ("prose", "code"):
        for join, jn in (("\n\n", "blank"), (" ", "inline")):
            for op in "+-*":
                for d in (4, 16):
                    for _ in range(20):
                        seg = ""
                        while not seg:
                            seg = segment(kind, rng, pools, FORMATS["blank"])
                        seg = seg[-600:].rstrip()
                        if seg and seg[-1].isdigit():
                            seg += "."
                        q, a = synth.arith_line(rng, op, d, d)
                        ctx.append((kind, jn, op, d, seg + join + q, a))
    res = {}
    for system in ("base", "coder", "gate", "gate_circuit"):
        t0 = time.time()
        o1 = greedy(live, tok, [q for *_, q, _ in eq], system, first, 2 * 32 + 2)
        cell = {}
        for (op, d, q, a), o in zip(eq, o1):
            cell.setdefault(f"{op}/{d}", []).append(exact(o, a))
        o2 = greedy(live, tok, [c[4] for c in ctx], system, first, 2 * 16 + 4)
        cc = {}
        for c, o in zip(ctx, o2):
            cc.setdefault(f"{c[0]}/{c[1]}/{c[2]}/{c[3]}", []).append(exact(o, c[5]))
        res[system] = {"equations": {k: float(np.mean(v)) for k, v in cell.items()},
                       "in_text": {k: float(np.mean(v)) for k, v in cc.items()},
                       "in_text_all": float(np.mean([x for v in cc.values() for x in v])),
                       "seconds": round(time.time() - t0)}
        print(system, json.dumps({"eq": res[system]["equations"], "in_text_all": res[system]["in_text_all"]}),
              flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["head", "loss", "gen"]
    gate, circ = load_gate(), circuit()
    path = os.path.join(ROOT, "results", "track_b", "circuit_gate.json")
    out = json.load(open(path))["data"] if os.path.exists(path) else {}
    hp = os.path.join(CKPT, "b_circuit_head.pt")
    if "head" in steps or not os.path.exists(hp):
        wb, out["head"] = train_head(gate, circ)
        save("track_b", "circuit_gate", out)
    else:
        wb = torch.load(hp, weights_only=False)["wb"]
    if "loss" in steps:
        out["loss"] = loss_eval(gate, circ, wb)
        save("track_b", "circuit_gate", out)
    if "gen" in steps:
        out["gen"] = gen_eval(gate, circ, wb)
        save("track_b", "circuit_gate", out)


if __name__ == "__main__":
    main()
