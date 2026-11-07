"""
An expression-evaluating atom: a hand-weighted transformer that parses
( ) + - * / and unary minus, and calls the arithmetic circuits for each operation.

    "^-(1.5+2)*3/(4-6)="  ->  rows of rewrites, calls  "[a op b?"  ...  "#"  result  "$"

The transformer never computes a number itself. It rewrites the expression one step
per row and, for every binary operation, emits a call block "[a op b?". The
orchestrator (evaluate() below, the same latching the gate does in the engine) hands
"a op b" to the matching arithmetic atom and appends its answer and "]". The parser
then copies that answer into the next row. Routing is the orchestrator's only job;
all arithmetic happens in the compiled circuits of circuits/arith.py.

Rows (row 0 is the input, with unary '-' rewritten as the sign token '~'):

    rule BIN  innermost group, first '*' or '/' in it, else first '+' or '-':
              "... a op b ..."  ->  call "[a op b?" R "]"  then  "... R ..."
    rule PAR  a group holding one number: "(n)" -> "n";  "~(n)" -> "~n";  "~(~n)" -> "n"
    rule DONE the row is one number: "#" then that number, then "$"

How the redex is found without counting brackets (counting would need weights that
depend on the maximum length):
  * The first ')' of a row closes an innermost group. Its partner is the nearest
    opener before it, where openers are '(' and the row-start markers. A row without
    ')' is its own group, closed by the row terminator.
  * Each '*' or '/' records whether it is the first one since its nearest opener
    (the previous one is older than that opener). Inside an innermost group exactly
    one operator carries the flag, so "the latest flagged operator before the closer"
    is the group's first '*' or '/' if it lies after the opener. The same applies to '+' and '-'.
  * Operands: the left one starts after the nearest non-number token before the
    operator. The right one ends at the first non-number token whose own "nearest
    earlier non-number" is the operator (the linked-list trick of the adder).
All of this is computed at the row terminator '|'. The writer of the next row reads
it from there, so every copy reads a position that is a linear function of marker
positions.
"""
from .core import Layout, Circuit, MLP, lookup, only, last, lin, gate, out, LAM, BIG, Head, EMIT

DIGITS = list("0123456789")
COPY = DIGITS + [".", "~", "+", "-", "*", "/", "(", ")"]          # tokens a row can hold
FLAGS = {".": "ISDOT", "~": "ISNEG", "+": "ISPLUS", "-": "ISMINUS", "*": "ISSTAR", "/": "ISSLASH",
         "(": "ISLP", ")": "ISRP", "=": "ISEQ", "|": "ISSEP", "[": "ISCB", "?": "ISQ", "]": "ISCE",
         "#": "ISHASH"}
VOCAB = COPY + ["^", "=", "|", "[", "?", "]", "#", "$"]
SRCS = ["LX", "RP", "RI", "RS", "CC", "OU"]                        # copy heads

NUMT = {"ISDIG": 1, "ISDOT": 1, "ISNEG": 1}                         # part of a number
NONNUM = {"ONE": 1, "ISDIG": -1, "ISDOT": -1, "ISNEG": -1, "ISBOS": -1}
OPEN = {"ISLP": 1, "ISSEP": 1, "ISEQ": 1, "ISCE": 1}                # '(' and row starts
ROWSTART = {"ISSEP": 1, "ISEQ": 1, "ISCE": 1}
MDF = {"ISSTAR": 1, "ISSLASH": 1}
ASF = {"ISPLUS": 1, "ISMINUS": 1}
UCTX = {"ISBOS": 1, "ISLP": 1, "ISPLUS": 1, "ISMINUS": 1, "ISSTAR": 1, "ISSLASH": 1}

FEATURES = (["ISDIG"] + sorted(set(FLAGS.values())) + [f"TK{c}" for c in COPY] +
            ["EQPOS", "OPN", "SEPPOS", "LMD", "LAS", "LCL", "LNN", "LNN2", "TPOS", "KPOS", "QPOS", "HPOS",
             "PMD", "PAS", "PCL", "PLNN", "PLNN2", "PSEP", "FMD", "FAS", "FCL", "XMD", "XAS", "FC",
             "CISRP", "CPOS", "GG", "XMDc", "XASc", "hasMD", "hasAS", "BIN", "OPP",
             "LB", "RB", "SIGNF", "NNEG", "R1", "DONE", "DROP2", "X1", "Y",
             "wS", "wX1", "wY", "wBIN", "wR1", "wDONE", "wGG", "wC", "wDROP2", "wLB", "wRB",
             "ROWS", "GEQ", "OUTP", "KAFTER", "START", "I0", "I1",
             "LEXf", "LEND", "inP", "inPI", "TERM", "RW", "C0", "CQ", "CCOPY", "D0", "OEND", "LXP_U"] +
            [f"{s}_{c}" for s in SRCS for c in COPY])


def _sel(m, a, flag, b, write):
    """write = flag ? a : b   (a, b non-negative position-sized expressions)."""
    m.unit(lin(a, {flag: 1e6}), -1e6, {write: 1})
    m.unit(lin(b, {flag: -1e6}), 0, {write: 1})


def build():
    L = Layout(VOCAB, FEATURES)
    embed = {}
    for c in COPY:
        e = {f"TK{c}": 1}
        if c in DIGITS:
            e["ISDIG"] = 1
        else:
            e[FLAGS[c]] = 1
        embed[c] = e
    for c in ["=", "|", "[", "?", "]", "#"]:
        embed[c] = {FLAGS[c]: 1}

    pos = lambda w: [({"POS": 1}, w)]
    # ---- markers (every position) ----------------------------------------------------
    a1 = [only(L, "ISEQ", "EQPOS"), last(L, OPEN, pos("OPN")), last(L, ROWSTART, pos("SEPPOS")),
          last(L, MDF, pos("LMD")), last(L, ASF, pos("LAS")), last(L, {"ISRP": 1}, pos("LCL")),
          last(L, NONNUM, [({"POS": 1}, "LNN"), ({"POS2": 1}, "LNN2")]),
          last(L, {"ISSEP": 1}, pos("TPOS")), last(L, {"ISCE": 1}, pos("KPOS")),
          last(L, {"ISQ": 1}, pos("QPOS")), only(L, "ISHASH", "HPOS")]
    prev = {"POS": 1, "ONE": -1}
    a2 = [lookup(L, prev, [({"LMD": 1}, "PMD"), ({"LAS": 1}, "PAS"), ({"LCL": 1}, "PCL"),
                           ({"LNN": 1}, "PLNN"), ({"LNN2": 1}, "PLNN2"), ({"SEPPOS": 1}, "PSEP")])]
    m2 = MLP(L)      # "first since my opener" flags, and "first ')' of my row"
    m2.ge({"OPN": 1, "PMD": -1}, 1, {"FMD": 1}, g=MDF)
    m2.ge({"OPN": 1, "PAS": -1}, 1, {"FAS": 1}, g=ASF)
    m2.ge({"SEPPOS": 1, "PCL": -1}, 1, {"FCL": 1}, g={"ISRP": 1})
    a3 = [last(L, {"FMD": 1}, pos("XMD")), last(L, {"FAS": 1}, pos("XAS")), last(L, {"FCL": 1}, pos("FC"))]

    # ---- row summary (meaningful at a row terminator '|') ------------------------------
    m3 = MLP(L)      # closer C = first ')' of the row if there is one, else the terminator
    m3.ge({"FC": 1, "PSEP": -1}, 1, {"CISRP": 1})
    m4a = MLP(L)
    _sel(m4a, {"FC": 1}, "CISRP", {"POS": 1}, "CPOS")
    a4 = [lookup(L, {"CPOS": 1, "ONE": -1}, [({"OPN": 1}, "GG"), ({"XMD": 1}, "XMDc"), ({"XAS": 1}, "XASc")])]
    m4 = MLP(L)
    m4.ge({"XMDc": 1, "GG": -1}, 1, {"hasMD": 1})
    m4.ge({"XASc": 1, "GG": -1}, 1, {"hasAS": 1})
    m5 = MLP(L)
    m5.ge({"hasMD": 1, "hasAS": 1}, 1, {"BIN": 1})
    _sel(m5, {"XMDc": 1}, "hasMD", {"XASc": 1}, "OPP")
    hR = Head(L, 3, 1)                       # right operand end: non-number whose PLNN == OPP
    R = L.R
    hR.K[R["PLNN2"], 0] = 1; hR.Q[R["ONE"], 0] = -LAM
    hR.K[R["PLNN"], 1] = 1;  hR.Q[R["OPP"], 1] = 2 * LAM
    for k, c in NONNUM.items():
        hR.K[R[k], 2] += c
    hR.Q[R["ONE"], 2] = BIG
    hR.V[R["POS"], 0] = 1;   hR.O[0, R["RB"]] = 1
    a5 = [lookup(L, {"OPP": 1, "ONE": -1}, [({"LNN": 1}, "LB")]), hR,
          lookup(L, {"GG": 1, "ONE": -1}, [({"ISNEG": 1}, "SIGNF")]),
          lookup(L, {"GG": 1, "ONE": 1}, [({"ISNEG": 1}, "NNEG")])]
    m6 = MLP(L)
    m6.when(gate("CISRP", neg=("BIN",)), {"R1": 1})
    m6.when(gate(neg=("CISRP", "BIN")), {"DONE": 1})
    m6.when(gate("CISRP", "SIGNF", "NNEG", neg=("BIN",)), {"DROP2": 1})
    m7 = MLP(L)      # next row = row[S+1 : X1] + insert + row[Y : T]
    _sel(m7, {"LB": 1, "ONE": 1}, "BIN", {"GG": 1}, "X1")
    m7.unit({"DROP2": 1}, 0, {"X1": -1})
    _sel(m7, {"RB": 1}, "BIN", {"CPOS": 1, "ONE": 1}, "Y")

    # ---- writer: read the summary from the last terminator ---------------------------
    a6 = [lookup(L, {"TPOS": 1}, [({"PSEP": 1}, "wS"), ({"X1": 1}, "wX1"), ({"Y": 1}, "wY"),
                                  ({"BIN": 1}, "wBIN"), ({"R1": 1}, "wR1"), ({"DONE": 1}, "wDONE"),
                                  ({"GG": 1}, "wGG"), ({"CPOS": 1}, "wC"), ({"DROP2": 1}, "wDROP2"),
                                  ({"LB": 1}, "wLB"), ({"RB": 1}, "wRB")])]
    m8 = MLP(L)
    m8.ge({"TPOS": 1}, 1, {"ROWS": 1})
    m8.ge({"POS": 1, "EQPOS": -1}, 0, {"GEQ": 1})
    m8.ge({"HPOS": 1}, 1, {"OUTP": 1})
    m8.ge({"KPOS": 1, "TPOS": -1}, 1, {"KAFTER": 1})
    _sel(m8, {"KPOS": 1}, "wBIN", {"TPOS": 1}, "START")              # next row starts after ']' or '|'
    _sel(m8, {"QPOS": 1, "ONE": 1}, "wBIN", {"wGG": 1, "ONE": 1, "wDROP2": 1}, "I0")
    _sel(m8, {"KPOS": 1}, "wBIN", {"wC": 1}, "I1")

    w = {"POS": 1, "START": -1}                                        # offset in the new row
    lenP = {"wX1": 1, "wS": -1, "ONE": -1}
    lenI = {"I1": 1, "I0": -1}
    cidx = {"POS": 1, "TPOS": -1}                                      # offset in the call block
    ncall = {"wRB": 1, "wLB": -1}                                      # |a op b| + 1
    cb = gate("ROWS", "wBIN", neg=("KAFTER", "OUTP"))
    m9 = MLP(L)
    m9.when(gate("GEQ", neg=("ROWS",)), {"LEXf": 1})
    m9.ge({"POS": 1, "EQPOS": -2, "ONE": 1}, 0, {"LEND": 1})
    m9.ge(lin(lenP, (-1, w)), 1, {"inP": 1})
    m9.ge(lin(lenP, lenI, (-1, w)), 1, {"inPI": 1})
    m9.ge(lin({"wY": 1}, w, (-1, lenP), (-1, lenI), {"TPOS": -1}), 0, {"TERM": 1})
    m9.when(gate("ROWS", "wBIN", "KAFTER", neg=("OUTP",)), {"RW": 1})
    m9.when(gate("ROWS", "wR1", neg=("OUTP",)), {"RW": 1})
    m9.eq(cidx, 0, {"C0": 1}, g=cb)
    m9.eq(lin(cidx, (-1, ncall)), 0, {"CQ": 1}, g=cb)
    m9.ge(cidx, 1, {"CCOPY": 1}, g=cb)
    m9.ge(lin(cidx, (-1, ncall)), 0, {"CCOPY": 1}, g=cb, coef=-1.0)
    m9.eq(cidx, 0, {"D0": 1}, g=gate("ROWS", "wDONE", neg=("OUTP",)))
    m9.ge({"wS": 1, "ONE": 1, "POS": 1, "HPOS": -1, "TPOS": -1}, 0, {"OEND": 1})

    def rd(src):
        return [({f"TK{c}": 1}, f"{src}_{c}") for c in COPY]
    a7 = [lookup(L, {"POS": 1, "ONE": 1, "EQPOS": -1}, rd("LX")),
          lookup(L, {"POS": 1, "EQPOS": -1}, [(UCTX, "LXP_U")]),
          lookup(L, lin({"wS": 1, "ONE": 1}, w), rd("RP")),
          lookup(L, lin({"I0": 1}, w, (-1, lenP)), rd("RI")),
          lookup(L, lin({"wY": 1}, w, (-1, lenP), (-1, lenI)), rd("RS")),
          lookup(L, lin({"wLB": 1}, cidx), rd("CC")),
          lookup(L, {"wS": 1, "ONE": 1, "POS": 1, "HPOS": -1}, rd("OU"))]

    m10 = MLP(L)
    for c in COPY:      # row 0: copy the input, turning unary '-' into '~'
        if c == "-":
            m10.when(gate("LEXf", "LX_-", neg=("LEND", "LXP_U")), out("-"))
            m10.when(gate("LEXf", "LX_-", "LXP_U", neg=("LEND",)), out("~"))
        else:
            m10.when(gate("LEXf", f"LX_{c}", neg=("LEND",)), out(c))
    m10.when(gate("LEXf", "LEND"), out("|"))
    for c in COPY:      # later rows: prefix, inserted span, suffix
        m10.when(gate("RW", "inP", f"RP_{c}"), out(c))
        m10.when(gate("RW", "inPI", f"RI_{c}", neg=("inP",)), out(c))
        m10.when(gate("RW", f"RS_{c}", neg=("inPI", "TERM")), out(c))
        m10.when(gate("CCOPY", f"CC_{c}"), out(c))
        m10.when(gate("OUTP", f"OU_{c}", neg=("OEND",)), out(c))
    m10.when(gate("RW", "TERM", neg=("inPI",)), out("|"))
    m10.when({"C0": 1}, out("["))
    m10.when({"CQ": 1}, out("?"))
    m10.when({"D0": 1}, out("#"))
    m10.when(gate("OUTP", "OEND"), out("$"))

    blocks = [("attn", a1), ("attn", a2), ("mlp", m2), ("attn", a3), ("mlp", m3), ("mlp", m4a),
              ("attn", a4), ("mlp", m4), ("mlp", m5), ("attn", a5), ("mlp", m6), ("mlp", m7),
              ("attn", a6), ("mlp", m8), ("mlp", m9), ("attn", a7), ("mlp", m10)]
    return Circuit(L, embed, blocks, name="parser")


# ---- orchestration -------------------------------------------------------------------
_PARSER = None


def lex(expr):
    """Character-level tokens (spaces dropped). Validation only; no rewriting."""
    toks = [c for c in expr if not c.isspace()]
    bad = [c for c in toks if c not in COPY or c == "~"]
    if bad:
        raise ValueError(f"unsupported characters: {''.join(bad)!r}")
    ops = "+-*/("
    for i in range(1, len(toks)):          # unary minus may not repeat: "--3", "*--3"
        if toks[i] == "-" and toks[i - 1] == "-" and (i < 2 or toks[i - 2] in ops):
            raise ValueError("repeated unary minus is not supported")
    if toks[:1] == ["-"] and toks[1:2] == ["-"]:
        raise ValueError("repeated unary minus is not supported")
    return toks


def _split_call(tokens):
    """'a op b' from the call block (the op is the first operator after position 0)."""
    for i in range(1, len(tokens)):
        if tokens[i] in "+-*/":
            a, op, b = "".join(tokens[:i]), tokens[i], "".join(tokens[i + 1:])
            return a.replace("~", "-"), op, b.replace("~", "-")
    raise RuntimeError(f"malformed call {tokens}")


def evaluate(expr, places=6, trace=False, max_steps=200000):
    """Evaluate an arithmetic expression with the parser atom and the arithmetic atoms.

    Returns the result string, or (result, trace_tokens, calls) if trace=True."""
    from . import arith
    global _PARSER
    if _PARSER is None:
        _PARSER = build()
    P = _PARSER
    st = Circuit.State(P)
    tok = P.L.tok
    stream = ["^"] + lex(expr) + ["="]
    for t in stream:
        logits = P.step(st, tok[t])
    gen, calls = [], []

    def feed(t):
        gen.append(t)
        return P.step(st, tok[t])

    while len(gen) < max_steps:
        i = int(logits.argmax())
        if logits[i] < EMIT / 2:
            raise RuntimeError(f"parser: no output after {''.join(gen[-40:])!r}")
        t = P.L.vocab[i]
        if t == "$":
            res = "".join(gen[gen.index("#") + 1:]).replace("~", "-")
            return (res, gen, calls) if trace else res
        logits = feed(t)
        if t == "?":                                    # latch onto the arithmetic atom
            start = len(gen) - 1 - gen[::-1].index("[")
            a, op, b = _split_call(gen[start + 1:-1])
            r = arith.evaluate(a, op, b, places=places)
            calls.append((a, op, b, r))
            for ch in r.replace("-", "~") + "]":
                logits = feed(ch)
    raise RuntimeError("parser: step limit reached")
