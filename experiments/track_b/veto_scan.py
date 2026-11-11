"""Track B: the circuit expert's claims on the validation text, checked against the text (CPU only).

    python experiments/track_b/veto_scan.py

The loss evaluation of circuit_gate.py is without vetoes. This scan counts, on the cached
512-token windows of every validation set, the circuit's predictions and vetoes and how many
the text contradicts (a predicted digit that is not the next token; a vetoed token that is the
next token), as false_triggers.py does for Track C. Writes results/track_b/veto_scan.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_b.cache import CACHE, CTX  # noqa: E402
from experiments.track_b.data import tokenizer  # noqa: E402
from experiments.track_c.circuit_expert import CircuitExpert, DIGITS, EQ, EQ_NEG  # noqa: E402

PLAIN = ["prose_val", "code_val", "notebooks_val"]          # no arithmetic segments
MIXED = ["a4_blank_val", "a4_inline_val"]
NAMES = ["none", "true_pred", "false_pred", "true_veto", "false_veto"]


def main():
    tok = tokenizer()
    circ = CircuitExpert(tok.backend_tokenizer, backend="python")
    out = {}
    for f in PLAIN + MIXED:
        X = np.load(os.path.join(CACHE, f"{f}.x.npy"))
        counts, examples = np.zeros(5, np.int64), []
        for w in range(X.shape[0]):
            x, y = X[w, :-1], X[w, 1:]
            tgt, vt = circ.analyse(x.tolist())
            viol = ((vt == DIGITS) & np.isin(y, list(circ.digit))) | ((vt == EQ) & (y == circ.eq)) | \
                   ((vt == EQ_NEG) & (y == circ.eq_neg))
            c = np.zeros(len(x), np.int8)
            c[(vt != 0) & ~viol] = 3
            c[viol] = 4
            c[(tgt >= 0) & (tgt == y)] = 1
            c[(tgt >= 0) & (tgt != y)] = 2
            counts += np.bincount(c, minlength=5)
            for p in np.nonzero((c == 2) | (c == 4))[0]:
                examples.append(tok.decode(X[w, max(0, p - 12):p + 4].tolist()))
        out[f] = {"tokens": int(X.shape[0] * CTX), "counts": dict(zip(NAMES, map(int, counts))),
                  "false_examples": examples[:10]}
        print(f, out[f], flush=True)
    tot = lambda fs, k: sum(out[f]["counts"][k] for f in fs)  # noqa: E731
    macros = {"VetoScanTokens": f"{sum(out[f]['tokens'] for f in PLAIN) / 1e6:.1f}M",
              "VetoScanRespected": f"{tot(PLAIN, 'true_veto'):,}".replace(",", "{,}"),
              "VetoScanViolated": str(tot(PLAIN, "false_veto") + tot(PLAIN, "false_pred")),
              "VetoMixedViolated": str(tot(MIXED, "false_veto")),
              "VetoMixedFalsePred": str(tot(MIXED, "false_pred"))}
    save("track_b", "veto_scan", out, macros=macros)


if __name__ == "__main__":
    main()
