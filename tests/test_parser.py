"""Checks the expression-parser atom against a reference evaluator.

The reference is a recursive-descent evaluator over Decimal with the same semantics:
usual precedence, left associativity, unary minus, and '/' truncated toward zero to
`places` fractional digits at every division.

Run:  python -m pytest tests -q      or      python tests/test_parser.py
"""
import os
import random
import sys
from decimal import Decimal, ROUND_DOWN, getcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuits import parser  # noqa: E402

getcontext().prec = 400
PLACES = 6


def reference(expr, places=PLACES):
    toks = [c for c in expr if not c.isspace()]
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def number():
        s = ""
        while peek() is not None and (peek().isdigit() or peek() == "."):
            s += take()
        return Decimal(s)

    def primary():
        if peek() == "-":
            take()
            return -primary()
        if peek() == "(":
            take()
            v = expr_()
            assert take() == ")"
            return v
        return number()

    def term():
        v = primary()
        while peek() in ("*", "/"):
            op, r = take(), primary()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise ZeroDivisionError
                v = (v / r).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)
        return v

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            op, r = take(), term()
            v = v + r if op == "+" else v - r
        return v

    v = expr_()
    assert i == len(toks)
    return v


def random_expr(rng, depth=0):
    if depth > 2 or rng.random() < 0.35:
        n = str(rng.randint(0, 99))
        if rng.random() < 0.3:
            n += "." + str(rng.randint(0, 99))
        e = n
    elif rng.random() < 0.3:
        e = "(" + random_expr(rng, depth + 1) + ")"
    else:
        e = random_expr(rng, depth + 1) + rng.choice("+-*/") + random_expr(rng, depth + 1)
    if rng.random() < 0.15 and (e[0] == "(" or e.replace(".", "").isdigit()):
        e = "-" + e                       # unary minus on a number or a group
    return e


def check(expr):
    try:
        want = reference(expr)
    except ZeroDivisionError:
        try:
            parser.evaluate(expr, places=PLACES)
        except ZeroDivisionError:
            return
        raise AssertionError(f"{expr}: expected ZeroDivisionError")
    got = parser.evaluate(expr, places=PLACES)
    assert Decimal(got) == want, (expr, got, want)


def test_fixed_cases():
    for e in ["5", "(5)", "((2))", "2+3", "1+2*3", "(1+2)*3", "8/4*2", "1-2+3", "10-4-3",
              "-(2+3)", "-(2-5)", "2--3", "2*-3", "-3*-3", "-(1.5+2)*3/(4-6)",
              "((1+2)*(3+4))-((5-6)*(7+8))", "1/3*3", "0.1+0.2", "100/7", "-(-(1))"[1:]]:
        check(e)


def test_random_expressions():
    rng = random.Random(7)
    for _ in range(150):
        check(random_expr(rng))


if __name__ == "__main__":
    test_fixed_cases()
    print("test_fixed_cases: ok")
    test_random_expressions()
    print("test_random_expressions: ok")
