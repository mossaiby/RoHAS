"""Track A: size of each compiled circuit, and scratchpad length / time against operand size.

    python experiments/track_a/run_cost.py

Timing is single-process wall clock of the NumPy reference implementation; run it on an
otherwise idle machine.
"""
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save, timer  # noqa: E402
from circuits import addsub, mul, div, parser  # noqa: E402


def size(c):
    arrays = [c.E, c.W_U]
    for kind, b in c.blocks:
        if kind == "attn":
            for h in b:
                arrays += [h.Q, h.K, h.V, h.O]
        else:
            arrays += [b.Win, b.bias, b.Wout]
    heads = sum(len(b) for k, b in c.blocks if k == "attn")
    units = sum(b.Win.shape[1] for k, b in c.blocks if k == "mlp")
    return {"params": int(sum(a.size for a in arrays)), "nonzero": int(sum(np.count_nonzero(a) for a in arrays)),
            "layers": len(c.blocks), "heads": heads, "mlp_units": units, "d_model": c.L.D, "vocab": c.L.V}


def main(seed=0):
    rng = random.Random(seed)
    circs = {"add": addsub.build("add"), "sub": addsub.build("sub"), "mul": mul.build(),
             "div": div.build(), "parser": parser.build()}
    table = [dict(name=n, **size(c)) for n, c in circs.items()]
    for row in table:
        print(row)

    def digits(n):
        return str(rng.randint(10 ** (n - 1), 10 ** n - 1))

    scaling = []
    with timer() as total:
        for op, sizes in (("add", (4, 16, 64, 256)), ("sub", (4, 16, 64, 256)),
                          ("mul", (2, 4, 8, 16, 32)), ("div", (4, 8, 16, 32))):
            for n in sizes:
                reps = 5 if n <= 16 else 2
                toks, secs = [], []
                for _ in range(reps):
                    a = digits(n)
                    b = digits(n if op != "div" else max(1, n // 2))
                    if op in ("add", "sub"):
                        pr = addsub.prompt(a, b, op)
                    elif op == "mul":
                        pr = mul.prompt(a, b)
                    else:
                        pr = div.prompt(a, b)
                    t = time.time()
                    g = circs[op].run(pr)
                    secs.append(time.time() - t)
                    toks.append(len(pr) + len(g) + 1)
                scaling.append({"op": op, "digits": n, "tokens": float(np.mean(toks)),
                                "seconds": float(np.mean(secs))})
                print(scaling[-1])
        for d in (1, 2, 3, 4, 5):
            def tree(k):
                return str(rng.randint(1, 50)) if k == 0 else "(" + tree(k - 1) + rng.choice("+-*") + tree(k - 1) + ")"
            e = tree(d)
            t = time.time()
            _, g, calls = parser.evaluate(e, trace=True)
            scaling.append({"op": "parser", "depth": d, "operators": 2 ** d - 1, "calls": len(calls),
                            "tokens": len(g) + len(e) + 2, "seconds": time.time() - t})
            print(scaling[-1])

    by = {r["name"]: r for r in table}
    macros = {f"{n}Params": f"{by[n]['params']:,}" for n in by}
    macros.update({f"{n}Nonzero": f"{by[n]['nonzero']:,}" for n in by})
    save("track_a", "cost", {"circuits": table, "scaling": scaling}, macros=macros, seed=seed,
         wall_time=total["elapsed"])


if __name__ == "__main__":
    main()
