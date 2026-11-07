"""Track A: correctness of the compiled circuits and the parser atom.

    python experiments/track_a/run_correctness.py                 # all quick suites
    python experiments/track_a/run_correctness.py --suite exhaustive3   # ~4.5 h on 15 workers

Suites (each adds/replaces its rows in results/track_a/correctness.json):
  exhaustive2   every pair of integers 0..99 (div: divisor 1..99), circuit level
  exhaustive3   every pair of integers 0..999 (div: divisor 1..999), circuit level
  long          random integers with many 9s, up to 200 digits (add/sub), 32 (mul/div)
  decimal       random signed decimals through circuits/arith.py vs Decimal
  parser        random expressions and balanced trees vs the reference evaluator
"""
import argparse
import os
import random
import sys
from decimal import Decimal, ROUND_DOWN, getcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save, timer, pool_map, ROOT  # noqa: E402

getcontext().prec = 1000
_C = {}


def _circ(name):
    if name not in _C:
        from circuits import addsub, mul, div
        _C[name] = {"add": lambda: addsub.build("add"), "sub": lambda: addsub.build("sub"),
                    "mul": mul.build, "div": div.build}[name]()
    return _C[name]


def run_int(job):
    """One integer case at circuit level. job = (op, a, b) as strings; returns 1 if wrong."""
    from circuits import addsub, mul, div
    op, a, b = job
    if op in ("add", "sub"):
        w = max(len(a), len(b))
        g = _circ(op).run(addsub.prompt(a.zfill(w), b.zfill(w), op))
        body = "".join(g[1:])
        if op == "add":
            got = int(("1" if g[0] == "1" else "") + body)
            want = int(a) + int(b)
        else:
            got = -int(body) if g[0] == "NEG" else int(body)
            want = int(a) - int(b)
    elif op == "mul":
        got = int(mul.answer(_circ("mul").run(mul.prompt(a, b))))
        want = int(a) * int(b)
    else:
        got = int(div.answer(_circ("div").run(div.prompt(a, b))))
        want = int(a) // int(b)
    return int(got != want)


def run_decimal(job):
    from circuits import arith
    op, a, b, places = job
    da, db = Decimal(a), Decimal(b)
    if op == "/":
        want = (da / db).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)
    else:
        want = {"+": da + db, "-": da - db, "*": da * db}[op]
    return int(Decimal(arith.evaluate(a, op, b, places=places)) != want)


def run_expr(expr):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_parser import reference
    from circuits import parser
    try:
        want = reference(expr)
    except ZeroDivisionError:
        try:
            parser.evaluate(expr)
            return 1
        except ZeroDivisionError:
            return 0
    return int(Decimal(parser.evaluate(expr)) != want)


def suite_exhaustive(k):
    hi = 10 ** k
    rows = []
    for op in ("add", "sub", "mul", "div"):
        jobs = [(op, str(a), str(b)) for a in range(hi) for b in range(1 if op == "div" else 0, hi)]
        with timer() as t:
            errs = sum(pool_map(run_int, jobs, chunksize=512))
        rows.append({"circuit": op, "suite": f"exhaustive, operands < 10^{k}", "cases": len(jobs),
                     "errors": errs, "seconds": round(t["elapsed"], 1)})
        print(rows[-1])
    return rows


def suite_long(rng):
    def num(n):
        s = "".join(rng.choice("9999000123456789") for _ in range(n)).lstrip("0")
        return s or "0"
    rows = []
    for op, lengths, per in (("add", (10, 50, 100, 200), 50), ("sub", (10, 50, 100, 200), 50),
                             ("mul", (8, 16, 32), 20), ("div", (8, 16, 32), 20)):
        jobs = []
        for n in lengths:
            for _ in range(per):
                a = num(n)
                b = num(rng.randint(1, n))
                if op == "div" and int(b) == 0:
                    b = "7"
                jobs.append((op, a, b))
        with timer() as t:
            errs = sum(pool_map(run_int, jobs, chunksize=2))
        rows.append({"circuit": op, "suite": f"random, up to {max(lengths)} digits", "cases": len(jobs),
                     "errors": errs, "seconds": round(t["elapsed"], 1)})
        print(rows[-1])
    return rows


def suite_decimal(rng):
    def num():
        i = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 8)))
        f = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 6)))
        s = i + ("." + f if f else "")
        return ("-" + s) if rng.random() < 0.4 else s
    rows = []
    for op in ("+", "-", "*", "/"):
        jobs = []
        while len(jobs) < 1000:
            a, b = num(), num()
            if op == "/" and Decimal(b) == 0:
                continue
            jobs.append((op, a, b, rng.randint(0, 8)))
        with timer() as t:
            errs = sum(pool_map(run_decimal, jobs, chunksize=16))
        rows.append({"circuit": {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op],
                     "suite": "signed decimals (arith)", "cases": len(jobs), "errors": errs,
                     "seconds": round(t["elapsed"], 1)})
        print(rows[-1])
    return rows


def suite_parser(rng):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_parser import random_expr

    def tree(d):
        if d == 0:
            return str(rng.randint(1, 50))
        return "(" + tree(d - 1) + rng.choice("+-*/") + tree(d - 1) + ")"
    rows = []
    exprs = [random_expr(rng) for _ in range(1000)]
    with timer() as t:
        errs = sum(pool_map(run_expr, exprs, chunksize=8))
    rows.append({"circuit": "parser", "suite": "random expressions", "cases": len(exprs),
                 "errors": errs, "seconds": round(t["elapsed"], 1)})
    print(rows[-1])
    trees = [tree(d) for d in (1, 2, 3, 4) for _ in range(10)]
    with timer() as t:
        errs = sum(pool_map(run_expr, trees, chunksize=1))
    rows.append({"circuit": "parser", "suite": "balanced trees, depth 1-4", "cases": len(trees),
                 "errors": errs, "seconds": round(t["elapsed"], 1)})
    print(rows[-1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="*", default=["exhaustive2", "long", "decimal", "parser"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    path = os.path.join(ROOT, "results", "track_a", "correctness.json")
    suites = []
    if os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as f:
            suites = json.load(f)["data"]["suites"]

    with timer() as t:
        for s in args.suite:
            new = {"exhaustive2": lambda: suite_exhaustive(2), "exhaustive3": lambda: suite_exhaustive(3),
                   "long": lambda: suite_long(rng), "decimal": lambda: suite_decimal(rng),
                   "parser": lambda: suite_parser(rng)}[s]()
            names = {(r["circuit"], r["suite"]) for r in new}
            suites = [r for r in suites if (r["circuit"], r["suite"]) not in names] + new

    total = sum(r["cases"] for r in suites)
    errors = sum(r["errors"] for r in suites)
    save("track_a", "correctness", {"suites": suites},
         macros={"CorrectnessCases": f"{total:,}", "CorrectnessErrors": str(errors)},
         seed=args.seed, wall_time=t["elapsed"])


if __name__ == "__main__":
    main()
