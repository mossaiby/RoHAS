"""The compiled arithmetic circuits as a track C expert over the shared BPE vocabulary.

At every position the expert either predicts the next token of an arithmetic answer or
abstains with the null atom (the uniform distribution). It is active when the context
ends with

    <non-digit> a op b =  [answer prefix]        op in + - *, a and b digit strings
    <non-digit> a op b =- [answer prefix]        (the BPE merges "=-"; negative results)

where the answer prefix is a proper prefix of the circuit's own answer. The answer
itself is computed by the hand-weighted transformers of Track A (circuits.arith), once
per equation and cached. The glue here only reads token ids (digits are single tokens
in this tokenizer) to find the equation; it never inspects digit values, the same rule
as circuits/arith.py. The expert also emits vetoes (the anti-atom): after each digit of b it rules out
whichever of "=" and "=-" has the wrong sign for the result (the sign comes from the
subtraction circuit), and after a complete answer it rules out every digit. Which
non-digit follows, and whether b is complete, are left to the other experts.

    backend="circuit"  the compiled circuits (default)
    backend="python"   Python integers, a tool with identical outputs (sanity check)
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from circuits import arith  # noqa: E402

EPS = 1e-4          # probability mass the active expert leaves for all other tokens
NONE, DIGITS, EQ, EQ_NEG = 0, 1, 2, 3   # veto codes


class CircuitExpert:
    def __init__(self, tok, backend="circuit"):
        self.V = tok.get_vocab_size()
        self.digit = {tok.token_to_id(str(d)): str(d) for d in range(10)}
        self.digit_id = {v: k for k, v in self.digit.items()}
        self.ops = {tok.token_to_id(o): o for o in "+-*"}
        self.eq, self.eq_neg = tok.token_to_id("="), tok.token_to_id("=-")
        assert None not in self.digit and None not in self.ops and None not in (self.eq, self.eq_neg)
        # tokens whose text ends in "-": a preceding minus would make a negative
        self.minus_end = {i for t, i in tok.get_vocab().items() if t.endswith("-")}
        self.backend = backend
        self.cache = {}
        self.calls = 0
        self.lp_on, self.lp_off = math.log(1 - EPS), math.log(EPS / (self.V - 1))

    def answer(self, a, op, b):
        key = (a, op, b)
        if key not in self.cache:
            self.calls += 1
            if self.backend == "circuit":
                self.cache[key] = arith.evaluate(a, op, b)
            else:
                x, y = int(a), int(b)
                self.cache[key] = str(x + y if op == "+" else x - y if op == "-" else x * y)
        return self.cache[key]

    def targets(self, ids):
        """Next-token id predicted at each position of one sequence, or -1 (abstain)."""
        return self.analyse(ids)[0]

    def analyse(self, ids):
        """(targets, vetoes) for one sequence. targets[p] is the predicted next-token id or -1;
        vetoes[p] is NONE, DIGITS (the answer is complete: the next token is not a digit),
        EQ or EQ_NEG (after a digit of b: the result's sign rules out "=" or "=-")."""
        out = np.full(len(ids), -1, dtype=np.int64)
        veto = np.zeros(len(ids), dtype=np.int8)
        D, ops = self.digit, self.ops
        i, n = 0, len(ids)
        while i < n:
            # find a maximal digit run a, then op, digit run b, then "=" or "=-"
            # a must be preceded by a known non-digit: at position 0 it may be cut off by the window
            if ids[i] not in D or i == 0 or ids[i - 1] in D or ids[i - 1] in self.minus_end:
                i += 1
                continue
            j = i
            while j < n and ids[j] in D:
                j += 1
            if j >= n or ids[j] not in ops:
                i = j
                continue
            a, op = "".join(D[t] for t in ids[i:j]), ops[ids[j]]
            k = j + 1
            while k < n and ids[k] in D:
                # b so far is ids[j+1:k+1]; if it ends here, the sign picks "=" or "=-"
                if op == "-":
                    neg = self.answer(a, op, "".join(D[t] for t in ids[j + 1:k + 1])).startswith("-")
                else:
                    neg = False                       # + and * of non-negative numbers
                veto[k] = EQ if neg else EQ_NEG
                k += 1
            if k == j + 1 or k >= n or ids[k] not in (self.eq, self.eq_neg):
                i = max(k, j + 1)
                continue
            b = "".join(D[t] for t in ids[j + 1:k])
            ans = self.answer(a, op, b)
            if (ids[k] == self.eq_neg) != ans.startswith("-"):
                i = k + 1
                continue
            digits = ans.lstrip("-")
            p = k                                      # position whose next token is digits[0]
            for c in digits:
                out[p] = self.digit_id[c]
                if p + 1 >= n or ids[p + 1] != self.digit_id[c]:
                    break                              # sequence ends, or it wrote another digit
                p += 1
            else:
                veto[p] = DIGITS                       # p is the last answer digit
            i = k + 1
        return out, veto

    @torch.no_grad()
    def logprobs(self, x, last=False):
        """(B, T, V) next-token log-probabilities for token ids x (B, T); (B, 1, V) if last.
        The veto mask (B, T, V) of the same call is kept in self.last_veto."""
        res = [self.analyse(r.tolist()) for r in x.cpu().numpy()]
        tgt = torch.from_numpy(np.stack([r[0] for r in res])).to(x.device)
        vt = torch.from_numpy(np.stack([r[1] for r in res])).to(x.device)
        if last:
            tgt, vt = tgt[:, -1:], vt[:, -1:]
        lp = torch.full((*tgt.shape, self.V), -math.log(self.V), device=x.device)
        on = tgt >= 0
        if on.any():
            rows = torch.full((int(on.sum()), self.V), self.lp_off, device=x.device)
            rows.scatter_(1, tgt[on].unsqueeze(1), self.lp_on)
            lp[on] = rows
        mask = torch.zeros((*tgt.shape, self.V), dtype=torch.bool, device=x.device)
        dig = torch.tensor(sorted(self.digit), device=x.device)
        m = vt == DIGITS
        if m.any():
            mask[m] = mask[m].index_fill(1, dig, True)
        mask[..., self.eq] |= vt == EQ
        mask[..., self.eq_neg] |= vt == EQ_NEG
        self.last_veto = mask
        return lp
