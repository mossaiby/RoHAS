"""
Integer multiplication of any length with a scratchpad, by hand-written weights.

    "^A*B="  ->  scratchpad rows ... "#"  product digits (MSD first)  "$"

A has n digits, B has m digits, W = n + m columns. One row per digit b_r of B
(least significant first). Row r holds, in order:

    Bcopy  (m tokens)  B shifted right by r with '_' fill, so its last char is b_r
    Ashift (W tokens)  digits of A * 10^r, least significant first
    cells  (W tokens)  S_r = S_{r-1} + A * b_r * 10^r, least significant first
    '|' (or '#' after the last row)

A cell is one token c00..c99 whose value is the column total
S_{r-1}[k] + Ashift_r[k] * b_r + carry_in <= 9 + 81 + 9; its embedding exposes the
digit (total mod 10) and the carry (total div 10), so the carry chain needs no
extra tokens.

Every row has the same length Lr = m + 2W + 1, so every value a token needs sits at
a position that is a LINEAR function of the current position and of the positions of
'*', '=', and the last row separator:
    previous row, same column:   p + 1 - Lr
    this row's Ashift digit:     p + 1 - W
    this row's b digit:          sep + m
That is what lets fixed weights handle any n and m. After the last row, the product
is the last row's cells read back in reverse.
"""
from .core import Layout, Circuit, MLP, lookup, only, last, lin, gate, out

CELLS = [f"c{v:02d}" for v in range(100)]
FEATURES = ["VAL", "ISDIG", "ISSTAR", "ISEQ", "ISSEP", "ISHASH", "CD", "CC",
            "STARPOS", "EQPOS", "SEPPOS", "HPOS",
            "geA", "geC", "geT", "ROW1", "OUT", "TEND", "A0",
            "PV_VAL", "PV_DIG", "RB_VAL", "RB_DIG", "RA_VAL", "RA_DIG",
            "PC_CD", "ASH", "BR", "BM2DIG", "OUT_CD", "PROD"] + [f"IB{v}" for v in range(1, 10)]


def build():
    vocab = list("0123456789") + ["*", "=", "^", "|", "#", "_", "$"] + CELLS
    L = Layout(vocab, FEATURES)
    embed = {str(d): {"VAL": d, "ISDIG": 1} for d in range(10)}
    embed.update({"*": {"ISSTAR": 1}, "=": {"ISEQ": 1}, "|": {"ISSEP": 1}, "#": {"ISHASH": 1}})
    for v, c in enumerate(CELLS):
        embed[c] = {"CD": v % 10, "CC": v // 10}

    # Shapes as linear expressions:  n = STAR - 1,  m = EQ - STAR - 1,  W = EQ - 2,
    # row length Lr = m + 2W + 1 = 3 EQ - STAR - 4,  slot o = POS - SEPPOS.
    m_ = {"EQPOS": 1, "STARPOS": -1, "ONE": -1}
    W_ = {"EQPOS": 1, "ONE": -2}
    Lr = {"EQPOS": 3, "STARPOS": -1, "ONE": -4}
    o = {"POS": 1, "SEPPOS": -1}

    a1 = [only(L, "ISSTAR", "STARPOS"), only(L, "ISEQ", "EQPOS"),
          last(L, {"ISEQ": 1, "ISSEP": 1}, [({"POS": 1}, "SEPPOS")]),
          only(L, "ISHASH", "HPOS")]

    m1 = MLP(L)      # which slot of the row the NEXT token is in
    m1.ge(lin(o, (-1, m_)), 0, {"geA": 1})                     # o >= m        (Ashift or later)
    m1.ge(lin(o, (-1, m_), (-1, W_)), 0, {"geC": 1})           # o >= m + W    (cells or later)
    m1.ge(lin(o, (-1, m_), (-2, W_)), 0, {"geT": 1})           # o == m + 2W   (terminator)
    m1.eq(lin(o, (-1, m_)), 0, {"A0": 1})                      # first Ashift slot
    m1.ge({"SEPPOS": 1, "EQPOS": -1}, 1, {"ROW1": 1})          # row index >= 1
    m1.ge({"HPOS": 1}, 1, {"OUT": 1})                          # past '#': output phase
    m1.ge(lin({"POS": 1, "HPOS": -1}, (-1, W_)), 0, {"TEND": 1})   # all W digits emitted

    nxt = {"POS": 1, "ONE": 1}                                 # position being written
    a2 = [
        lookup(L, lin({"POS": 1}, (-1, Lr)), [({"VAL": 1}, "PV_VAL"), ({"ISDIG": 1}, "PV_DIG")]),
        lookup(L, lin({"STARPOS": 1, "ONE": 1}, o), [({"VAL": 1}, "RB_VAL"), ({"ISDIG": 1}, "RB_DIG")]),
        lookup(L, {"SEPPOS": 1, "EQPOS": 1, "POS": -1, "ONE": -2},
               [({"VAL": 1}, "RA_VAL"), ({"ISDIG": 1}, "RA_DIG")]),
        lookup(L, lin(nxt, (-1, Lr)), [({"CD": 1}, "PC_CD")]),
        lookup(L, lin(nxt, (-1, W_)), [({"VAL": 1}, "ASH")]),
        lookup(L, lin({"SEPPOS": 1}, m_), [({"VAL": 1}, "BR")]),
        lookup(L, lin({"SEPPOS": 1, "ONE": -1}, m_), [({"ISDIG": 1}, "BM2DIG")]),
        lookup(L, {"HPOS": 2, "POS": -1, "ONE": -1}, [({"CD": 1}, "OUT_CD")]),
    ]

    m2 = MLP(L)      # one-hot of the b digit
    for v in range(1, 10):
        m2.eq({"BR": 1}, v, {f"IB{v}": 1})
    m3 = MLP(L)      # PROD = Ashift digit * b digit (a gated sum over the nine b values)
    for v in range(1, 10):
        m3.unit({"ASH": v, f"IB{v}": 1e6}, -1e6, {"PROD": 1})

    isB = {"ONE": 1, "geA": -1}
    isA = {"geA": 1, "geC": -1}
    isC = {"geC": 1, "geT": -1}
    isT = {"geT": 1}
    m4 = MLP(L)
    # Bcopy: row 0 copies B; later rows copy the previous row shifted right by one.
    for d in range(10):
        m4.eq({"RB_VAL": 1}, d, out(str(d)), g=gate(isB, "RB_DIG", neg=("ROW1", "OUT")))
        m4.eq({"PV_VAL": 1}, d, out(str(d)), g=gate(isB, "ROW1", "PV_DIG", neg=("OUT",)))
    m4.when(gate(isB, neg=("ROW1", "RB_DIG", "OUT")), out("_"))
    m4.when(gate(isB, "ROW1", neg=("PV_DIG", "OUT")), out("_"))
    # Ashift: row 0 copies A (LSD first, 0 beyond its length); later rows shift by one.
    for d in range(10):
        m4.eq({"RA_VAL": 1}, d, out(str(d)), g=gate(isA, "RA_DIG", neg=("ROW1", "OUT")))
        m4.eq({"PV_VAL": 1}, d, out(str(d)), g=gate(isA, "ROW1", "PV_DIG", neg=("A0", "OUT")))
    m4.when(gate(isA, neg=("ROW1", "RA_DIG", "OUT")), out("0"))
    m4.when(gate(isA, "ROW1", "A0", neg=("OUT",)), out("0"))
    m4.when(gate(isA, "ROW1", neg=("A0", "PV_DIG", "OUT")), out("0"))
    # cells: total = previous row's digit + a*b + incoming carry (carry = this token's CC)
    for v in range(100):
        m4.eq({"PROD": 1, "CC": 1}, v, out(CELLS[v]), g=gate(isC, neg=("ROW1", "OUT")))
        m4.eq({"PC_CD": 1, "PROD": 1, "CC": 1}, v, out(CELLS[v]), g=gate(isC, "ROW1", neg=("OUT",)))
    # terminator: '#' after the row whose Bcopy[m-2] is already '_' (the last b digit)
    m4.when(gate(isT, "BM2DIG", neg=("OUT",)), out("|"))
    m4.when(gate(isT, neg=("BM2DIG", "OUT")), out("#"))
    # output phase: the last row's cells, most significant first, then '$'
    for d in range(10):
        m4.eq({"OUT_CD": 1}, d, out(str(d)), g=gate("OUT", neg=("TEND",)))
    m4.when(gate("OUT", "TEND"), out("$"))

    blocks = [("attn", a1), ("mlp", m1), ("attn", a2), ("mlp", m2), ("mlp", m3), ("mlp", m4)]
    return Circuit(L, embed, blocks, name="mul")


def prompt(a, b):
    return ["^"] + list(a) + ["*"] + list(b) + ["="]


def answer(tokens):
    """Product digits from a generated token list (everything after '#')."""
    return "".join(tokens[tokens.index("#") + 1:])
