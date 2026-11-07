"""
Addition and subtraction of decimals of any length, most significant digit first,
in one forward pass per output digit (no scratchpad).

    "^1.234 + 2.345 ="   ->   " 3.579"      (first slot: carry-out, ' ' or '1')
    "^1.234 - 2.345 ="   ->   "NEG1.111"    (first slot: sign, ' ' or 'NEG')

Both operands must have the same shape (same digits before and after '.'); zero
padding is string preprocessing done by the caller (see arith.pad).

The carry construction is the one from misc/adder.py: the carry into digit i is the
flag t of the FIRST later digit pair whose sum is not 9 (9s propagate). Each pair
computes g = (sum != 9); u[j] = nearest g=1 pair at or before j; prev[j] = u[j-1]
links the g=1 pairs into a list; the first g=1 pair after i is the one whose prev
equals u[i]. If there is none, the carry is the initial carry, read from BOS.

Subtraction is ten's complement through the same machinery: a - b = a + (9..9 - b) + 1,
i.e. the second operand's digits are complemented and BOS supplies an initial carry
of 1. A second chain computes b + (9..9 - a) + 1 = b - a; both chains share g
(a_i + 9 - b_i != 9 iff a_i != b_i), so only the carry flag t differs. The carry-out
of the first chain is the sign: 1 means a >= b. Later digits use the chain the sign
selects; they read the sign back from the first emitted token.
"""
from .core import (Layout, Circuit, MLP, lookup, only, last, lin, gate, out,
                   KAP, LAM, BIG)

FEATURES = ["VAL", "ISDIG", "ISDOT", "ISOP", "ISEQ", "ISSPACE", "ISNEG",
            "OPPOS", "EQPOS", "HOLD", "AL", "G", "T1", "T2", "U", "U2", "PREV", "PREV2",
            "SQ1", "SQ2", "UQ", "UQ2", "QDOT", "QDIG", "QSPACE", "NEGF", "CAR1", "CAR2"]


def build(mode):
    assert mode in ("add", "sub")
    op = "+" if mode == "add" else "-"
    vocab = list("0123456789") + [".", op, "=", " ", "^", "$"] + (["NEG"] if mode == "sub" else [])
    L = Layout(vocab, FEATURES)
    embed = {str(d): {"VAL": d, "ISDIG": 1} for d in range(10)}
    embed.update({".": {"ISDOT": 1}, op: {"ISOP": 1}, "=": {"ISEQ": 1}, " ": {"ISSPACE": 1}})
    if mode == "sub":
        embed["NEG"] = {"ISNEG": 1}
        embed["^"] = {"T1": 1, "T2": 1}          # initial carry of the complement trick

    # block 1: where the operator and '=' are
    a1 = [only(L, "ISOP", "OPPOS"), only(L, "ISEQ", "EQPOS")]
    # HOLD = 1 on a digit after the operator and before '=' (a digit of the 2nd operand)
    m1 = MLP(L)
    B = 1e9
    for shift, sign in ((0, +1), (-1, -1)):
        m1.unit({"OPPOS": 1, "ISDIG": B, "EQPOS": -B}, shift - B, {"HOLD": sign})

    # block 2: fetch the 1st operand's aligned digit (position p - op - 1), then g and t
    a2 = [lookup(L, {"POS": 1, "OPPOS": -1, "ONE": -1}, [({"VAL": 1}, "AL")])]
    m2 = MLP(L)
    if mode == "add":
        s = {"VAL": 1, "AL": 1}
        m2.ge(lin((-1, s)), -8, {"G": 1}, g={"HOLD": 1})                 # s <= 8
        m2.ge(s, 10, {"G": 1, "T1": 1}, g={"HOLD": 1})                   # s >= 10
    else:
        d = {"AL": 1, "VAL": -1}                                          # a_i - b_i
        m2.ge(d, 1, {"G": 1, "T1": 1}, g={"HOLD": 1})                    # a_i > b_i
        m2.ge(lin((-1, d)), 1, {"G": 1, "T2": 1}, g={"HOLD": 1})         # b_i > a_i

    # block 3: u[j] = position of nearest g=1 at or before j (0 = BOS if none)
    a3 = [last(L, {"G": 1}, [({"POS": 1}, "U"), ({"POS2": 1}, "U2")])]
    # block 4: prev[j] = u[j-1]
    a4 = [lookup(L, {"POS": 1, "ONE": -1}, [({"U": 1}, "PREV"), ({"U2": 1}, "PREV2")])]

    # block 5: the token at p writes output slot p - eq, aligned with the 2nd operand
    # token at q = (2p - eq + 1) / 2. Read that pair's sums and its u.
    if mode == "add":
        sq = [({"VAL": 1, "AL": 1}, "SQ1")]
    else:
        sq = [({"AL": 1, "VAL": -1, "ONE": 9}, "SQ1"), ({"VAL": 1, "AL": -1, "ONE": 9}, "SQ2")]
    a5 = [lookup(L, {"POS": 1, "EQPOS": -0.5, "ONE": 0.5},
                 sq + [({"U": 1}, "UQ"), ({"U2": 1}, "UQ2"), ({"ISDOT": 1}, "QDOT"),
                       ({"ISDIG": 1}, "QDIG"), ({"ISSPACE": 1}, "QSPACE")])]
    if mode == "sub":    # sign = was the first emitted token 'NEG'?
        a5.append(lookup(L, {"EQPOS": 1, "ONE": 1}, [({"ISNEG": 1}, "NEGF")]))

    # block 6: carry = t of the g=1 pair whose prev == u[q]; BOS is the fallback
    from .core import Head
    h = Head(L, 4, 2 if mode == "sub" else 1)
    R = L.R
    h.K[R["PREV2"], 0] = 1;  h.Q[R["ONE"], 0] = -LAM
    h.K[R["PREV"], 1] = 1;   h.Q[R["UQ"], 1] = 2 * LAM                # -LAM (prev - uq)^2
    h.K[R["ONE"], 2] = 1; h.K[R["G"], 2] = -1; h.K[R["ISBOS"], 2] = -1
    h.Q[R["ONE"], 2] = -BIG                                           # skip g=0 pairs
    h.K[R["ISBOS"], 3] = 1
    h.Q[R["UQ2"], 3] = LAM;  h.Q[R["ONE"], 3] = -LAM / 2              # BOS: just below a match
    h.V[R["T1"], 0] = 1;     h.O[0, R["CAR1"]] = 1
    if mode == "sub":
        h.V[R["T2"], 1] = 1; h.O[1, R["CAR2"]] = 1
    a6 = [h]

    # final MLP: digit = (s + carry) mod 10; slot 0 is carry-out / sign
    m6 = MLP(L)
    if mode == "add":
        for v in range(20):
            m6.eq({"SQ1": 1, "CAR1": 1}, v, out(str(v % 10)), g={"QDIG": 1})
        m6.eq({"CAR1": 1}, 0, out(" "), g={"ISEQ": 1})
        m6.eq({"CAR1": 1}, 1, out("1"), g={"ISEQ": 1})
    else:
        for v in range(20):
            m6.eq({"SQ1": 1, "CAR1": 1}, v, out(str(v % 10)), g=gate("QDIG", neg=("NEGF",)))
            m6.eq({"SQ2": 1, "CAR2": 1}, v, out(str(v % 10)), g=gate("QDIG", "NEGF"))
        m6.eq({"CAR1": 1}, 1, out(" "), g={"ISEQ": 1})
        m6.eq({"CAR1": 1}, 0, out("NEG"), g={"ISEQ": 1})

    # '.' is copied from q; '$' when q reaches the space after the 2nd operand
    extra = [("QDOT", ".", 10.0), ("QSPACE", "$", 10.0), ("ISEQ", "$", -10.0)]
    blocks = [("attn", a1), ("mlp", m1), ("attn", a2), ("mlp", m2), ("attn", a3),
              ("attn", a4), ("attn", a5), ("attn", a6), ("mlp", m6)]
    return Circuit(L, embed, blocks, extra, name=mode)


def prompt(a, b, mode):
    op = "+" if mode == "add" else "-"
    return ["^"] + list(a) + [" ", op, " "] + list(b) + [" ", "="]
