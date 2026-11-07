"""
Signed-decimal arithmetic on top of the compiled circuits.

All arithmetic (carries, borrows, digit products, quotient digits, sign of a
difference) is done by the hand-weighted transformers. This module only does fixed
string formatting around them, none of which inspects digit values:

* split sign / integer part / fraction part, and zero-pad two operands to one shape
  (add, sub);
* move the decimal point: the integer circuits see digit strings with the point
  removed, and the point is re-inserted at a position computed from the operand
  string lengths (mul: |frac a| + |frac b| places; div: zeros are appended so that the
  integer quotient has `places` fractional digits);
* combine signs: for + and - the sign of the result comes from the subtraction
  circuit; for * and / it is the XOR of the operand signs;
* strip leading zeros.

Division truncates toward zero to `places` fractional digits.
"""
from . import addsub, mul as _mul, div as _div

_cache = {}


def _circuit(name):
    if name not in _cache:
        _cache[name] = {"add": lambda: addsub.build("add"), "sub": lambda: addsub.build("sub"),
                        "mul": _mul.build, "div": _div.build}[name]()
    return _cache[name]


def _parse(x):
    s = str(x).strip()
    neg = s.startswith("-")
    s = s.lstrip("+-")
    ip, _, fp = s.partition(".")
    ip = ip or "0"
    if not (ip.isdigit() and (fp == "" or fp.isdigit())):
        raise ValueError(f"not a decimal number: {x!r}")
    return neg, ip, fp


def _fmt(neg, ip, fp):
    ip = ip.lstrip("0") or "0"
    s = ip + ("." + fp if fp else "")
    if neg and s.strip("0.") != "":
        s = "-" + s
    return s


def pad(a, b):
    """Zero-pad two unsigned decimals to the same shape (string preprocessing)."""
    (ai, _, af), (bi, _, bf) = a.partition("."), b.partition(".")
    wi, wf = max(len(ai), len(bi)), max(len(af), len(bf))
    f = lambda i, fr: i.zfill(wi) + ("." + fr.ljust(wf, "0") if wf else "")
    return f(ai, af), f(bi, bf)


def _unsigned(ip, fp):
    return ip + ("." + fp if fp else "")


def add(a, b):
    na, ia, fa = _parse(a)
    nb, ib, fb = _parse(b)
    A, B = _unsigned(ia, fa), _unsigned(ib, fb)
    if na == nb:
        pa, pb = pad(A, B)
        g = _circuit("add").run(addsub.prompt(pa, pb, "add"))
        body = ("1" if g[0] == "1" else "") + "".join(g[1:])
        neg = na
    else:
        pos, negv = (A, B) if not na else (B, A)
        pa, pb = pad(pos, negv)
        g = _circuit("sub").run(addsub.prompt(pa, pb, "sub"))
        body = "".join(g[1:])
        neg = g[0] == "NEG"
    ip, _, fp = body.partition(".")
    return _fmt(neg, ip, fp)


def sub(a, b):
    nb, ib, fb = _parse(b)
    return add(a, ("" if nb else "-") + _unsigned(ib, fb))


def mul(a, b):
    na, ia, fa = _parse(a)
    nb, ib, fb = _parse(b)
    A = (ia + fa).lstrip("0") or "0"
    B = (ib + fb).lstrip("0") or "0"
    digits = _mul.answer(_circuit("mul").run(_mul.prompt(A, B)))
    scale = len(fa) + len(fb)
    if scale == 0:
        return _fmt(na != nb, digits, "")
    digits = digits.zfill(scale + 1)
    return _fmt(na != nb, digits[:-scale], digits[-scale:])


def div(a, b, places=6):
    na, ia, fa = _parse(a)
    nb, ib, fb = _parse(b)
    N = ((ia + fa) + "0" * (len(fb) + places)).lstrip("0") or "0"
    D = ((ib + fb) + "0" * len(fa)).lstrip("0")
    if not D:
        raise ZeroDivisionError("division by zero")
    q = _div.answer(_circuit("div").run(_div.prompt(N, D)))
    if places == 0:
        return _fmt(na != nb, q, "")
    q = q.zfill(places + 1)
    return _fmt(na != nb, q[:-places], q[-places:])


def cmp(a, b):
    """-1, 0, or 1, from the subtraction circuit's sign and digits."""
    d = sub(a, b)
    if d.startswith("-"):
        return -1
    return 0 if d.strip("0.") == "" else 1


def compare(a, op, b):
    c = cmp(a, b)
    return {">": c > 0, "<": c < 0, ">=": c >= 0, "<=": c <= 0, "==": c == 0, "!=": c != 0}[op]


def evaluate(a, op, b, places=6):
    """Result string of `a op b`; comparisons return 'True' / 'False'."""
    if op == "+":
        return add(a, b)
    if op == "-":
        return sub(a, b)
    if op in ("*", "x", "×"):
        return mul(a, b)
    if op in ("/", "÷"):
        return div(a, b, places)
    if op in (">", "<", ">=", "<=", "==", "!="):
        return str(compare(a, op, b))
    raise ValueError(f"unsupported operator {op!r}")
