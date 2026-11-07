"""
Integer long division of any length with a scratchpad, by hand-written weights.

    "^A/B="  ->  scratchpad rows ... "#"  quotient digits (n+1, MSD first)  "$"

Restoring division: for each digit of A (most significant first) bring the digit
down (X = 10 R + a), then subtract B repeatedly until the result goes negative;
the quotient digit is the number of successful subtractions.

Every row has the same layout and length Lr = 2n + m + 4 (n = |A|, m = |B|, W = m+1):

    header (1)       'D' (bring-down row) or 'S1'..'S10' (c-th subtraction of this step)
    Areg   (n)       digits of A not yet consumed, left-aligned, '_' fill
    Qreg   (n+1)     quotient digits found so far, right-aligned, '0' fill
    num    (W)       the working number, least significant first, one token per digit:
                     n0..n9 (no borrow) or n10..n19 (borrow out); the borrow out of
                     the last column is the sign of a subtraction row
    '|' (or '#' after the final bring-down row)

Which row comes next depends only on the previous row:
    previous D            -> S1
    previous S_c, >= 0    -> S_{c+1}
    previous S_c, < 0     -> D, which appends q = c - 1 to Qreg and brings down the
                             next digit into the last non-negative row, i.e. the row
                             before the previous one (always 2 rows back).
The bring-down row that finds Areg empty is the last; its Qreg is the quotient.

Small state (row type, subtraction count, borrow) lives in token identities; the
unbounded state (registers, numbers) is copied row to row at linear offsets, so no
weight depends on n or m.
"""
from .core import Layout, Circuit, MLP, lookup, only, last, lin, gate, out

NUMS = [f"n{v}" for v in range(20)]
HDRS = ["D"] + [f"S{c}" for c in range(1, 11)]
FEATURES = ["VAL", "ISDIG", "ISSLASH", "ISEQ", "ISSEP", "ISHASH", "HD", "HC", "ND", "NB",
            "SLPOS", "EQPOS", "SEPPOS", "HPOS",
            "isH", "geA", "geQ", "geN", "geT", "ALAST", "QLAST", "N0", "ROW1", "OUT", "TEND",
            "CURD", "PHD", "PHC", "PNEG", "RA_VAL", "RA_DIG", "PS_VAL", "PS_DIG", "PS_ND",
            "PN_VAL", "PN_DIG", "PA0_VAL", "PA0_DIG", "PP_ND", "BK", "A0_VAL", "OQ_VAL", "FINAL"]


def build():
    vocab = list("0123456789") + ["/", "=", "^", "|", "#", "_", "$"] + HDRS + NUMS
    L = Layout(vocab, FEATURES)
    embed = {str(d): {"VAL": d, "ISDIG": 1} for d in range(10)}
    embed.update({"/": {"ISSLASH": 1}, "=": {"ISEQ": 1}, "|": {"ISSEP": 1}, "#": {"ISHASH": 1}})
    embed["D"] = {"HD": 1}
    for c in range(1, 11):
        embed[f"S{c}"] = {"HC": c}
    for v, t in enumerate(NUMS):
        embed[t] = {"ND": v % 10, "NB": v // 10}

    # Shapes:  n = SL - 1,  Lr = SL + EQ + 1,  slot o = POS - SEPPOS.
    # Slots: o=0 header; [1, n+1) Areg; [n+1, 2n+2) Qreg; [2n+2, Lr-1) num; Lr-1 terminator.
    Lr = {"SLPOS": 1, "EQPOS": 1, "ONE": 1}
    o = {"POS": 1, "SEPPOS": -1}

    a1 = [only(L, "ISSLASH", "SLPOS"), only(L, "ISEQ", "EQPOS"),
          last(L, {"ISEQ": 1, "ISSEP": 1}, [({"POS": 1}, "SEPPOS")]),
          only(L, "ISHASH", "HPOS")]

    m1 = MLP(L)
    m1.eq(o, 0, {"isH": 1})
    m1.ge(o, 1, {"geA": 1})
    m1.ge(lin(o, {"SLPOS": -1}), 0, {"geQ": 1})                       # o >= n + 1
    m1.ge(lin(o, {"SLPOS": -2}), 0, {"geN": 1})                       # o >= 2n + 2
    m1.ge(lin(o, {"SLPOS": -1, "EQPOS": -1}), 0, {"geT": 1})          # o == Lr - 1
    m1.eq(lin(o, {"SLPOS": -1, "ONE": 1}), 0, {"ALAST": 1})           # o == n
    m1.eq(lin(o, {"SLPOS": -2, "ONE": 1}), 0, {"QLAST": 1})           # o == 2n + 1
    m1.eq(lin(o, {"SLPOS": -2}), 0, {"N0": 1})                        # o == 2n + 2
    m1.ge({"SEPPOS": 1, "EQPOS": -1}, 1, {"ROW1": 1})
    m1.ge({"HPOS": 1}, 1, {"OUT": 1})
    m1.ge({"POS": 1, "HPOS": -1, "SLPOS": -1}, 0, {"TEND": 1})        # n + 1 digits emitted

    nxt = {"POS": 1, "ONE": 1}
    a2 = [
        lookup(L, {"SEPPOS": 1, "ONE": 1}, [({"HD": 1}, "CURD")]),                  # this row's header
        lookup(L, lin({"SEPPOS": 1, "ONE": 1}, (-1, Lr)),                           # previous header
               [({"HD": 1}, "PHD"), ({"HC": 1}, "PHC")]),
        lookup(L, {"POS": 1, "ONE": -1}, [({"NB": 1}, "PNEG")]),                    # prev row's sign
        lookup(L, lin(o, {"ONE": 1}), [({"VAL": 1}, "RA_VAL"), ({"ISDIG": 1}, "RA_DIG")]),
        lookup(L, lin(nxt, (-1, Lr)),                                               # prev row, same slot
               [({"VAL": 1}, "PS_VAL"), ({"ISDIG": 1}, "PS_DIG"), ({"ND": 1}, "PS_ND")]),
        lookup(L, lin(nxt, {"ONE": 1}, (-1, Lr)),                                   # prev row, next slot
               [({"VAL": 1}, "PN_VAL"), ({"ISDIG": 1}, "PN_DIG")]),
        lookup(L, lin({"SEPPOS": 1, "ONE": 2}, (-1, Lr)),                           # prev row Areg[0]
               [({"VAL": 1}, "PA0_VAL"), ({"ISDIG": 1}, "PA0_DIG")]),
        lookup(L, lin({"POS": 1}, (-2, Lr)), [({"ND": 1}, "PP_ND")]),               # 2 rows back, k-1
        lookup(L, {"EQPOS": 1, "SLPOS": 2, "ONE": -1, "POS": -1, "SEPPOS": 1},      # B[k] (0 past B)
               [({"VAL": 1}, "BK")]),
        lookup(L, {"ONE": 1}, [({"VAL": 1}, "A0_VAL")]),                            # A[0]
        lookup(L, {"POS": 1, "EQPOS": -1}, [({"VAL": 1}, "OQ_VAL")]),               # final Qreg[t]
    ]

    m2 = MLP(L)      # the final bring-down row: a D row (not row 0) whose consumed digit is '_'
    m2.unit({"CURD": 1, "ROW1": 1, "PA0_DIG": -1}, -1, {"FINAL": 1})

    isA = {"geA": 1, "geQ": -1}
    isQ = {"geQ": 1, "geN": -1}
    isN = {"geN": 1, "geT": -1}
    isT = {"geT": 1}
    m3 = MLP(L)
    # header
    m3.when(gate("isH", neg=("ROW1", "OUT")), out("D"))
    m3.when(gate("isH", "ROW1", "PNEG", neg=("PHD", "OUT")), out("D"))
    m3.when(gate("isH", "ROW1", "PHD", neg=("OUT",)), out("S1"))
    for c in range(1, 10):
        m3.eq({"PHC": 1}, c, out(f"S{c + 1}"), g=gate("isH", "ROW1", neg=("PHD", "PNEG", "OUT")))
    # Areg: row 0 = A without its first digit; D rows shift left; S rows copy
    for d in range(10):
        m3.eq({"RA_VAL": 1}, d, out(str(d)), g=gate(isA, "RA_DIG", neg=("ROW1", "OUT")))
        m3.eq({"PN_VAL": 1}, d, out(str(d)), g=gate(isA, "ROW1", "CURD", "PN_DIG", neg=("ALAST", "OUT")))
        m3.eq({"PS_VAL": 1}, d, out(str(d)), g=gate(isA, "ROW1", "PS_DIG", neg=("CURD", "OUT")))
    m3.when(gate(isA, neg=("ROW1", "RA_DIG", "OUT")), out("_"))
    m3.when(gate(isA, "ROW1", "CURD", neg=("ALAST", "PN_DIG", "OUT")), out("_"))
    m3.when(gate(isA, "ROW1", "CURD", "ALAST", neg=("OUT",)), out("_"))
    m3.when(gate(isA, "ROW1", neg=("CURD", "PS_DIG", "OUT")), out("_"))
    # Qreg: row 0 zeros; D rows shift left and append q = c - 1; S rows copy
    m3.when(gate(isQ, neg=("ROW1", "OUT")), out("0"))
    for d in range(10):
        m3.eq({"PN_VAL": 1}, d, out(str(d)), g=gate(isQ, "ROW1", "CURD", neg=("QLAST", "OUT")))
        m3.eq({"PHC": 1}, d + 1, out(str(d)), g=gate(isQ, "ROW1", "CURD", "QLAST", neg=("OUT",)))
        m3.eq({"PS_VAL": 1}, d, out(str(d)), g=gate(isQ, "ROW1", neg=("CURD", "OUT")))
    # num, bring-down rows: column 0 = the consumed digit, column k = X[k-1] from 2 rows back
    for d in range(10):
        m3.eq({"A0_VAL": 1}, d, out(NUMS[d]), g=gate(isN, "N0", neg=("ROW1", "OUT")))
        m3.eq({"PA0_VAL": 1}, d, out(NUMS[d]), g=gate(isN, "ROW1", "CURD", "N0", "PA0_DIG", neg=("OUT",)))
        m3.eq({"PP_ND": 1}, d, out(NUMS[d]), g=gate(isN, "ROW1", "CURD", neg=("N0", "OUT")))
    m3.when(gate(isN, neg=("ROW1", "N0", "OUT")), out(NUMS[0]))
    m3.when(gate(isN, "ROW1", "CURD", "N0", neg=("PA0_DIG", "OUT")), out(NUMS[0]))
    # num, subtraction rows: s = X[k] - B[k] - borrow_in, borrow_in = this token's NB
    for s in range(-10, 10):
        tok = NUMS[s % 10 + (10 if s < 0 else 0)]
        m3.eq({"PS_ND": 1, "BK": -1, "NB": -1}, s, out(tok), g=gate(isN, "ROW1", neg=("CURD", "OUT")))
    # terminator
    m3.when(gate(isT, neg=("FINAL", "OUT")), out("|"))
    m3.when(gate(isT, "FINAL", neg=("OUT",)), out("#"))
    # output phase: the final row's Qreg, then '$'
    for d in range(10):
        m3.eq({"OQ_VAL": 1}, d, out(str(d)), g=gate("OUT", neg=("TEND",)))
    m3.when(gate("OUT", "TEND"), out("$"))

    blocks = [("attn", a1), ("mlp", m1), ("attn", a2), ("mlp", m2), ("mlp", m3)]
    return Circuit(L, embed, blocks, name="div")


def prompt(a, b):
    return ["^"] + list(a) + ["/"] + list(b) + ["="]


def answer(tokens):
    """Quotient digits from a generated token list (everything after '#')."""
    return "".join(tokens[tokens.index("#") + 1:])
