"""
Decimal addition of ANY length, by a transformer whose weights are all written by hand.

    "^1.234 + 2.345 ="   ->   " 3.579"         ('^' = start token, '$' = end token)

No weight depends on the number of digits. Positions are encoded as the pair (p, p^2),
so attention can do an *exact* "give me the token at position t" lookup for any t.

Requirement: both operands have the same shape (same digits before and after the '.').
Use pad() to zero-pad -- that is string preprocessing, it is not part of the model.

How the carry works (the interesting part)
------------------------------------------
Output is generated most-significant digit first, so the carry into a digit depends on
everything below it. The carry into digit i is decided by the FIRST later digit pair
whose sum is not 9:  sum >= 10 -> carry,  sum <= 8 -> no carry  (9s just pass it along).
Finding "the first non-9 after i" with attention:
  1. every digit pair computes its sum s and a flag g = (s != 9)
  2. u[j]    = position of the nearest g=1 pair at or before j        (causal argmax)
  3. prev[j] = u[j-1]  -> each g=1 pair now knows the g=1 pair before it (a linked list)
  4. to find the first g=1 pair after i: look up u[i], then attend to the g=1 pair
     whose prev equals u[i].  Its flag (s >= 10) is the carry.
"""
import numpy as np

VOCAB = list("0123456789.+= ^$")
tok = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)

LAM = KAP = 200.0   # attention sharpness: right key beats wrong keys by e^-100 or more
BIG = 1e12          # "never pick this key"
M = 100.0           # gate size for ReLU units

# ---- residual stream layout -------------------------------------------------
FEATURES = ["VAL", "ISDIG", "ISDOT", "ISPLUS", "ISEQ", "ISSPACE", "ISBOS", "ONE",
            "POS", "POS2",                      # position p and p^2
            "PLUSPOS", "EQPOS",                 # where the '+' and '=' are
            "HOLD",                             # 1 if this token is a digit of the 2nd operand
            "AL",                               # the 1st operand's digit aligned with this one
            "G", "T",                           # g = (sum != 9), t = (sum >= 10)
            "U", "U2",                          # position (and its square) of nearest g=1 at/before me
            "PREV", "PREV2",                    # same, but for the position just before me
            "SQ", "UQ", "UQ2", "QDOT", "QDIG", "QSPACE",   # what I read at the digit I'm about to write
            "CAR"]                              # carry into that digit
R = {n: i for i, n in enumerate(FEATURES)}
for c in VOCAB:
    R[("out", c)] = len(FEATURES) + tok[c]
D = len(FEATURES) + V

def vec(d):
    v = np.zeros(D)
    for k, c in d.items():
        v[R[k]] += c
    return v

# ---- embedding: token table + a (p, p^2) positional signal ------------------
E = np.zeros((V, D))
for c, i in tok.items():
    E[i, R["ONE"]] = 1
    if c.isdigit():
        E[i, R["VAL"]] = int(c); E[i, R["ISDIG"]] = 1
E[tok["."], R["ISDOT"]] = 1
E[tok["+"], R["ISPLUS"]] = 1
E[tok["="], R["ISEQ"]] = 1
E[tok[" "], R["ISSPACE"]] = 1
E[tok["^"], R["ISBOS"]] = 1

def embed(ids):
    x = E[ids].copy()
    p = np.arange(len(ids), dtype=float)
    x[:, R["POS"]], x[:, R["POS2"]] = p, p * p
    return x

# ---- attention --------------------------------------------------------------
class Attn:
    def __init__(self):
        self.h = []
    def head(self, dk, dv):
        h = dict(Q=np.zeros((D, dk)), K=np.zeros((D, dk)), V=np.zeros((D, dv)), O=np.zeros((dv, D)))
        self.h.append(h)
        return h
    def __call__(self, x):
        n = len(x)
        mask = np.triu(np.full((n, n), -np.inf), 1)          # causal
        out = np.zeros_like(x)
        for h in self.h:
            s = (x @ h["Q"]) @ (x @ h["K"]).T + mask
            a = np.exp(s - s.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            out += (a @ (x @ h["V"])) @ h["O"]
        return out

def lookup(attn, tgt, reads, writes):
    """Head that fetches the token at position tgt (a linear function of my residual).
    score = -LAM (pos_k - tgt)^2, written as a bilinear form in (p^2, p)."""
    h = attn.head(2, len(reads))
    h["K"][R["POS2"], 0] = 1
    h["K"][R["POS"], 1] = 1
    h["Q"][R["ONE"], 0] = -LAM
    for k, c in tgt.items():
        h["Q"][R[k], 1] += 2 * LAM * c
    for j, (rd, wr) in enumerate(zip(reads, writes)):
        for k, c in rd.items():
            h["V"][R[k], j] += c
        h["O"][j, R[wr]] = 1

def sentinel_argmax(attn, key_terms, bos_score, value_reads, value_writes):
    """Head that attends to the key with the largest score(key); BOS is a fallback whose
    value is 0, so 'nothing qualifies' reads back as 0."""
    h = attn.head(1, len(value_reads))
    h["Q"][R["ONE"], 0] = 1
    for k, c in key_terms.items():
        h["K"][R[k], 0] += c
    h["K"][R["ISBOS"], 0] += bos_score
    for j, (rd, wr) in enumerate(zip(value_reads, value_writes)):
        for k, c in rd.items():
            h["V"][R[k], j] += c
        h["O"][j, R[wr]] = 1

class MLP:
    def __init__(self):
        self.W, self.b, self.O = [], [], []
    def unit(self, inputs, bias, outputs):
        self.W.append(vec(inputs)); self.b.append(bias); self.O.append(vec(outputs))
    def done(self):
        self.Win, self.bias, self.Wout = np.array(self.W).T, np.array(self.b), np.array(self.O)
        return self
    def __call__(self, x):
        return np.maximum(x @ self.Win + self.bias, 0) @ self.Wout

# ---- block 1: find the '+' and '=' ------------------------------------------
a1 = Attn()
sentinel_argmax(a1, {"ISPLUS": KAP}, KAP / 2, [{"POS": 1}], ["PLUSPOS"])   # 0 if no '+' yet
sentinel_argmax(a1, {"ISEQ": KAP},   KAP / 2, [{"POS": 1}], ["EQPOS"])
m1 = MLP()      # HOLD = is a digit, after the '+', before the '='   (= digit of 2nd operand)
B = 1e9
for shift, sign in ((0, +1), (-1, -1)):
    m1.unit({"PLUSPOS": 1, "ISDIG": B, "EQPOS": -B}, shift - B, {"HOLD": sign})
m1.done()

# ---- block 2: fetch the 1st operand's aligned digit, then g and t -----------
a2 = Attn()
lookup(a2, {"POS": 1, "PLUSPOS": -1, "ONE": -1}, [{"VAL": 1}], ["AL"])    # position p - plus - 1
m2 = MLP()      # s = VAL + AL.   G = HOLD and s != 9.   T = HOLD and s >= 10.
for bias, out in ((9, {"G": +1}), (8, {"G": -1})):                          # [s <= 8]
    m2.unit({"VAL": -1, "AL": -1, "HOLD": M}, bias - M, out)
for bias, s in ((-9, +1), (-10, -1)):                                       # [s >= 10]
    m2.unit({"VAL": 1, "AL": 1, "HOLD": M}, bias - M, {"G": s, "T": s})
m2.done()

# ---- block 3: u[j] = position of nearest g=1 at or before j (0 if none) -----
a3 = Attn()
sentinel_argmax(a3, {"POS": KAP, "G": BIG}, BIG / 2,
                [{"POS": 1}, {"POS2": 1}], ["U", "U2"])

# ---- block 4: prev[j] = u[j-1] ----------------------------------------------
a4 = Attn()
lookup(a4, {"POS": 1, "ONE": -1}, [{"U": 1}, {"U2": 1}], ["PREV", "PREV2"])

# ---- block 5: at an output step, read the digit pair I'm about to write -----
# Current token at position p writes output slot m = p - eq. That slot lines up with the
# 2nd operand's token at position q = (2p - eq + 1)/2.  (m = 0 is the carry-out slot.)
a5 = Attn()
lookup(a5, {"POS": 1, "EQPOS": -0.5, "ONE": 0.5},
       [{"VAL": 1, "AL": 1}, {"U": 1}, {"U2": 1}, {"ISDOT": 1}, {"ISDIG": 1}, {"ISSPACE": 1}],
       ["SQ", "UQ", "UQ2", "QDOT", "QDIG", "QSPACE"])

# ---- block 6: the carry = (s >= 10) of the g=1 pair whose prev == u[q] -------
a6 = Attn()
h = a6.head(4, 1)
h["K"][R["PREV2"], 0] = 1;   h["Q"][R["ONE"], 0] = -LAM
h["K"][R["PREV"], 1] = 1;    h["Q"][R["UQ"], 1] = 2 * LAM          # -LAM (prev - uq)^2
h["K"][R["ONE"], 2] = 1; h["K"][R["G"], 2] = -1; h["K"][R["ISBOS"], 2] = -1
h["Q"][R["ONE"], 2] = -BIG                                         # skip g=0 pairs
h["K"][R["ISBOS"], 3] = 1                                          # BOS: fallback, carry 0
h["Q"][R["UQ2"], 3] = LAM;   h["Q"][R["ONE"], 3] = -LAM / 2
h["V"][R["T"], 0] = 1;       h["O"][0, R["CAR"]] = 1

# ---- final MLP: digit = (s + carry) mod 10, or ' '/'1' for the carry-out slot
m6 = MLP()
for v in range(20):                         # triangle (3 ReLUs) = 1 exactly when s+carry == v
    for shift, coef in ((1, 1), (0, -2), (-1, 1)):
        m6.unit({"SQ": 1, "CAR": 1, "QDIG": M}, shift - v - M, {("out", str(v % 10)): 10 * coef})
for v, c in ((0, " "), (1, "1")):
    for shift, coef in ((1, 1), (0, -2), (-1, 1)):
        m6.unit({"CAR": 1, "ISEQ": M}, shift - v - M, {("out", c): 10 * coef})
m6.done()

# '.' and '$' are copied straight from what was read at q
W_U = np.zeros((D, V))
for c in VOCAB:
    W_U[R[("out", c)], tok[c]] = 1
W_U[R["QDOT"], tok["."]] = 10
W_U[R["QSPACE"], tok["$"]] = 10
W_U[R["ISEQ"], tok["$"]] = -10

LAYERS = [(a1, m1), (a2, m2), (a3, None), (a4, None), (a5, None), (a6, m6)]

def forward(ids):
    x = embed(ids)
    for attn, mlp in LAYERS:
        x = x + attn(x)
        if mlp is not None:
            x = x + mlp(x)
    return x @ W_U

def generate(a, b):
    ids = [tok[c] for c in f"^{a} + {b} ="]
    start, out = len(ids), ""
    while len(ids) < 2 * start:                       # safety cap for malformed input
        nxt = VOCAB[int(forward(ids)[-1].argmax())]
        if nxt == "$":
            break
        ids.append(tok[nxt]); out += nxt
    return out

def pad(a, b):
    """NOT part of the model: zero-pad two decimals to the same shape."""
    (ai, _, af), (bi, _, bf) = a.partition("."), b.partition(".")
    wi, wf = max(len(ai), len(bi)), max(len(af), len(bf))
    f = lambda i, fr: i.zfill(wi) + ("." + fr.ljust(wf, "0") if wf else "")
    return f(ai, af), f(bi, bf)

def add(a, b):
    return generate(*pad(a, b))

# ---- tests ------------------------------------------------------------------
def expected(a, b):
    a, b = pad(a, b)
    ai, _, af = a.partition(".")
    s = str(int(a.replace(".", "")) + int(b.replace(".", ""))).zfill(len(ai + af) + 1)
    head, tail = s[1:1 + len(ai)], s[1 + len(ai):]
    return (" " if s[0] == "0" else "1") + head + ("." + tail if af else "")

def run_tests():
    import random, time
    for a, b in [("1.234", "2.345"), ("0.999", "0.001"), ("9.999", "9.999"),
                 ("12.5", "0.0034"), ("99", "1")]:
        print(f"{a} + {b} = {add(a, b).strip()}")

    random.seed(1)
    digit = lambda: random.choice("9999000123456789")      # lots of 9s: stress the carries
    def num(i, f):
        return "".join(digit() for _ in range(i)) + ("." + "".join(digit() for _ in range(f)) if f else "")
    bad = 0
    N = 1500
    for _ in range(N):
        a, b = num(random.randint(1, 6), random.randint(0, 8)), num(random.randint(1, 6), random.randint(0, 8))
        bad += add(a, b) != expected(a, b)
    print(f"\n{N} random tests (mixed shapes, 9-heavy): {bad} wrong")

    for n in (50, 200):
        a, b = "0." + "9" * n, "0." + "0" * (n - 1) + "1"
        t = time.time()
        ok = add(a, b) == expected(a, b)
        print(f"0.{'9'*3}...9 ({n} nines) + 0.{'0'*3}...1: {'ok' if ok else 'WRONG'}  [{time.time()-t:.1f}s]")
    a = "".join(random.choice("0123456789") for _ in range(150)) + "." + "".join(random.choice("0123456789") for _ in range(150))
    b = "".join(random.choice("0123456789") for _ in range(150)) + "." + "".join(random.choice("0123456789") for _ in range(150))
    print("150.150-digit random:", "ok" if add(a, b) == expected(a, b) else "WRONG")


# ---- REPL -------------------------------------------------------------------
import re
_NUM = r"(\d+\.?\d*|\.\d+)"
_EXPR = re.compile(rf"^\s*{_NUM}\s*\+\s*{_NUM}\s*=?\s*$")

def repl():
    try:
        import readline  # noqa: F401  (arrow-key history, if available)
    except ImportError:
        pass
    raw = False
    print("Hand-weighted decimal adder. Type e.g.  1.234 + 2.345   (any number of digits)")
    print("Commands: :raw toggle exact model output, :test run self-tests, :quit")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", "quit", "exit"):
            break
        if line == ":raw":
            raw = not raw
            print(f"raw output {'on' if raw else 'off'}")
            continue
        if line == ":test":
            run_tests()
            continue
        m = _EXPR.match(line)
        if not m:
            print("  expected: <number> + <number>   e.g. 12.5 + 0.0034")
            continue
        a, b = m.groups()
        pa, pb = pad(a, b)
        out = generate(pa, pb)
        if raw:
            print(f"  model input : ^{pa} + {pb} =")
            print(f"  model output: {out!r}")
        else:
            print(f"  {out.strip()}")

if __name__ == "__main__":
    import sys
    run_tests() if "--test" in sys.argv else repl()