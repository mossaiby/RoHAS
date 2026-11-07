"""Checks the compiled circuits against Python's Decimal.

Run:  python -m pytest tests -q      or      python tests/test_circuits.py
"""
import os
import random
import sys
from decimal import Decimal, ROUND_DOWN, getcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuits import arith, mul, div  # noqa: E402

getcontext().prec = 400
rng = random.Random(1234)


def _digits(n, heavy=True):
    pool = "9999000123456789" if heavy else "0123456789"
    return "".join(rng.choice(pool) for _ in range(n))


def _num(max_int=6, max_frac=6, signed=True):
    i = _digits(rng.randint(1, max_int))
    f = _digits(rng.randint(0, max_frac))
    s = i + ("." + f if f else "")
    return ("-" + s) if signed and rng.random() < 0.4 else s


def _same(got, expected):
    return Decimal(got) == expected


def test_add_sub_signed():
    for _ in range(300):
        a, b = _num(), _num()
        assert _same(arith.add(a, b), Decimal(a) + Decimal(b)), (a, b)
        assert _same(arith.sub(a, b), Decimal(a) - Decimal(b)), (a, b)


def test_add_long_carry():
    a, b = "0." + "9" * 120, "0." + "0" * 119 + "1"
    assert arith.add(a, b) == "1." + "0" * 120


def test_mul_signed_decimals():
    for _ in range(200):
        a, b = _num(5, 4), _num(5, 4)
        assert _same(arith.mul(a, b), Decimal(a) * Decimal(b)), (a, b)


def test_mul_integers_long():
    c = mul.build()
    for n, m in ((1, 1), (30, 1), (1, 30), (25, 20)):
        a, b = "9" * n, "9" * m
        assert int(mul.answer(c.run(mul.prompt(a, b)))) == int(a) * int(b)


def test_div_signed_decimals():
    for _ in range(150):
        a, b = _num(6, 3), _num(4, 3)
        if Decimal(b) == 0:
            continue
        places = rng.randint(0, 8)
        q = (Decimal(a) / Decimal(b)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)
        assert _same(arith.div(a, b, places), q), (a, b, places)


def test_div_integers():
    c = div.build()
    for _ in range(100):
        a = str(rng.randint(0, 10 ** rng.randint(1, 12)))
        b = str(rng.randint(1, 10 ** rng.randint(1, 6)))
        assert int(div.answer(c.run(div.prompt(a, b)))) == int(a) // int(b), (a, b)


def test_compare():
    for _ in range(100):
        a, b = _num(4, 3), _num(4, 3)
        if rng.random() < 0.2:
            b = a
        for op, fn in ((">", lambda x, y: x > y), ("<", lambda x, y: x < y), ("==", lambda x, y: x == y),
                       (">=", lambda x, y: x >= y), ("<=", lambda x, y: x <= y)):
            assert arith.compare(a, op, b) == fn(Decimal(a), Decimal(b)), (a, op, b)


def test_div_by_zero():
    try:
        arith.div("1", "0.000")
    except ZeroDivisionError:
        return
    raise AssertionError("expected ZeroDivisionError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
