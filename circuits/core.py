"""
A small toolkit for writing transformer weights by hand.

A circuit is an ordinary decoder-only transformer: token embedding plus a fixed
(p, p^2) positional signal, a stack of residual blocks (multi-head causal softmax
attention, ReLU MLPs), and an unembedding. Every weight is set by a constructor in
this package; nothing is trained. The residual stream is a named set of features so
that constructions stay readable.

Three facts make exact programs possible:

* lookup:  a head whose score is -LAM * (p_k - t)^2, written as a bilinear form in
           (p_k^2, p_k) and (1, t). If the target t is a linear function of the
           residual stream, the head reads the token at exactly position t.
* select:  a head whose key score is a linear function of key features; with a large
           scale it attends to the maximiser, e.g. "the most recent '|'".
* ReLU arithmetic on small integers: [x >= c] = ReLU(x-c+1) - ReLU(x-c) and
           [x == c] = ReLU(x-c+1) - 2 ReLU(x-c) + ReLU(x-c-1) are exact on integers.

Gates are integer linear expressions that equal 1 when active and are <= 0 otherwise;
adding M*(g - 1) to a ReLU argument switches a unit off.

Generation is incremental (keys and values cached per head), so a forward step costs
O(sequence length) per head instead of re-running the whole prefix.
"""
import numpy as np

LAM = 200.0     # lookup sharpness: the right key beats a neighbour by e^-200
KAP = 200.0     # select sharpness per unit of key score
BIG = 1e12      # dominates any position-sized score
M = 1e6         # gate size for ReLU units; larger than any gated argument


# ---- linear expressions over residual features --------------------------------
def lin(*terms):
    """Sum of expressions. Each term is a dict {feature: coef} or (scale, dict)."""
    out = {}
    for t in terms:
        s, d = (1.0, t) if isinstance(t, dict) else t
        for k, c in d.items():
            out[k] = out.get(k, 0.0) + s * c
    return out


def gate(*flags, neg=()):
    """Expression equal to 1 iff all `flags` are 1 and all `neg` are 0 (flags in {0,1}).

    Each flag may itself be an expression that evaluates to 0/1."""
    parts = [f if isinstance(f, dict) else {f: 1.0} for f in flags]
    nparts = [f if isinstance(f, dict) else {f: 1.0} for f in neg]
    n = len(parts) + len(nparts)
    return lin(*parts, *[(-1.0, p) for p in nparts], {"ONE": float(len(nparts) - (n - 1))})


class Layout:
    """Names every residual dimension. Output logits get one feature per token."""
    BASE = ["ONE", "POS", "POS2", "ISBOS"]

    def __init__(self, vocab, features):
        self.vocab = list(vocab)
        self.tok = {c: i for i, c in enumerate(self.vocab)}
        names = list(self.BASE) + [f for f in features if f not in self.BASE]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature names")
        self.R = {n: i for i, n in enumerate(names)}
        for c in self.vocab:
            self.R[("out", c)] = len(self.R)
        self.D = len(self.R)
        self.V = len(self.vocab)

    def vec(self, expr):
        v = np.zeros(self.D)
        for k, c in expr.items():
            v[self.R[k]] += c
        return v


# ---- attention -----------------------------------------------------------------
class Head:
    def __init__(self, L, dk, dv):
        self.Q = np.zeros((L.D, dk))
        self.K = np.zeros((L.D, dk))
        self.V = np.zeros((L.D, dv))
        self.O = np.zeros((dv, L.D))


def _values(L, h, reads):
    for j, (rd, wr) in enumerate(reads):
        for k, c in rd.items():
            h.V[L.R[k], j] += c
        h.O[j, L.R[wr]] = 1.0


def lookup(L, target, reads):
    """Read the token at position `target` (a linear expression).

    reads: list of (value expression, feature written)."""
    h = Head(L, 2, len(reads))
    h.K[L.R["POS2"], 0] = 1.0
    h.K[L.R["POS"], 1] = 1.0
    h.Q[L.R["ONE"], 0] = -LAM
    for k, c in target.items():
        h.Q[L.R[k], 1] += 2.0 * LAM * c
    _values(L, h, reads)
    return h


def select(L, key, reads):
    """Attend to the key maximising the linear score `key` (query is constant)."""
    h = Head(L, 1, len(reads))
    h.Q[L.R["ONE"], 0] = 1.0
    for k, c in key.items():
        h.K[L.R[k], 0] += c
    _values(L, h, reads)
    return h


def only(L, flag, write):
    """Position of the unique token with `flag` (0, i.e. BOS, if there is none yet)."""
    return select(L, {flag: KAP, "ISBOS": KAP / 2}, [({"POS": 1.0}, write)])


def last(L, flag_expr, reads):
    """Read the most recent token whose flag expression is 1 (BOS if none)."""
    key = lin((BIG, flag_expr), {"POS": KAP, "ISBOS": BIG / 2})
    return select(L, key, reads)


# ---- MLP -----------------------------------------------------------------------
class MLP:
    def __init__(self, L):
        self.L, self._W, self._b, self._O = L, [], [], []

    def unit(self, inputs, bias, outputs):
        self._W.append(self.L.vec(inputs))
        self._b.append(float(bias))
        self._O.append(self.L.vec(outputs))

    def _gated(self, x, g):
        return x if g is None else lin(x, (M, g))

    def ge(self, x, c, out, g=None, coef=1.0):
        """out += coef * [x >= c] for integer x (when gate g is active)."""
        off = 0.0 if g is None else -M
        xi = self._gated(x, g)
        self.unit(xi, -c + 1 + off, {k: coef * v for k, v in out.items()})
        self.unit(xi, -c + off, {k: -coef * v for k, v in out.items()})

    def eq(self, x, c, out, g=None, coef=1.0):
        """out += coef * [x == c] for integer x (when gate g is active)."""
        off = 0.0 if g is None else -M
        xi = self._gated(x, g)
        for shift, w in ((1, 1.0), (0, -2.0), (-1, 1.0)):
            self.unit(xi, -c + shift + off, {k: coef * w * v for k, v in out.items()})

    def when(self, g, out, coef=1.0):
        """out += coef when integer gate g == 1 (g <= 1 by construction)."""
        self.unit(g, 0.0, {k: coef * v for k, v in out.items()})

    def build(self):
        self.Win = np.array(self._W).T
        self.bias = np.array(self._b)
        self.Wout = np.array(self._O)
        return self

    def __call__(self, x):
        return np.maximum(x @ self.Win + self.bias, 0.0) @ self.Wout


EMIT = 10.0     # logit written for the chosen output token


def out(token):
    return {("out", token): EMIT}


# ---- the model -----------------------------------------------------------------
class Circuit:
    """A decoder-only transformer with hand-written weights.

    blocks: list of ("attn", [Head, ...]) or ("mlp", MLP), applied residually in order.
    embed:  {token: expression} added to ONE (and ISBOS for '^').
    extra_unembed: list of (feature, token, coef) added to the default unembedding.
    """

    def __init__(self, L, embed, blocks, extra_unembed=(), name="circuit"):
        self.L, self.blocks, self.name = L, blocks, name
        self.E = np.zeros((L.V, L.D))
        for c, i in L.tok.items():
            e = lin({"ONE": 1.0}, embed.get(c, {}))
            if c == "^":
                e = lin(e, {"ISBOS": 1.0})
            self.E[i] = L.vec(e)
        self.W_U = np.zeros((L.D, L.V))
        for c in L.vocab:
            self.W_U[L.R[("out", c)], L.tok[c]] = 1.0
        for f, c, w in extra_unembed:
            self.W_U[L.R[f], L.tok[c]] += w
        for kind, b in blocks:
            if kind == "mlp":
                b.build()

    def n_params(self):
        n = self.E.size + self.W_U.size
        for kind, b in self.blocks:
            if kind == "attn":
                n += sum(h.Q.size + h.K.size + h.V.size + h.O.size for h in b)
            else:
                n += b.Win.size + b.bias.size + b.Wout.size
        return n

    class State:
        def __init__(self, circ, cap=256):
            self.n, self.cap, self.c = 0, cap, circ
            self.K, self.Vc = [], []
            for kind, b in circ.blocks:
                if kind == "attn":
                    self.K.append([np.zeros((cap, h.K.shape[1])) for h in b])
                    self.Vc.append([np.zeros((cap, h.V.shape[1])) for h in b])
                else:
                    self.K.append(None)
                    self.Vc.append(None)

        def grow(self):
            self.cap *= 2
            for i, ks in enumerate(self.K):
                if ks is None:
                    continue
                self.K[i] = [np.vstack([k, np.zeros_like(k)]) for k in ks]
                self.Vc[i] = [np.vstack([v, np.zeros_like(v)]) for v in self.Vc[i]]

    def step(self, st, tid):
        """Feed one token; return the logits for the next position."""
        if st.n == st.cap:
            st.grow()
        p = st.n
        R = self.L.R
        x = self.E[tid].copy()
        x[R["POS"]], x[R["POS2"]] = p, float(p) * p
        for bi, (kind, b) in enumerate(self.blocks):
            if kind == "attn":
                upd = np.zeros_like(x)
                for hi, h in enumerate(b):
                    q = x @ h.Q
                    st.K[bi][hi][p] = x @ h.K
                    st.Vc[bi][hi][p] = x @ h.V
                    s = st.K[bi][hi][:p + 1] @ q
                    a = np.exp(s - s.max())
                    a /= a.sum()
                    upd += (a @ st.Vc[bi][hi][:p + 1]) @ h.O
                x = x + upd
            else:
                x = x + b(x)
        st.n += 1
        return x @ self.W_U

    def run(self, prompt, stop="$", max_new=200000):
        """Feed `prompt` (list of tokens) and decode greedily until `stop`.

        Returns the generated tokens (without `stop`)."""
        st = Circuit.State(self)
        for t in prompt:
            logits = self.step(st, self.L.tok[t])
        gen = []
        while len(gen) < max_new:
            i = int(np.argmax(logits))
            if logits[i] < EMIT / 2:
                raise RuntimeError(f"{self.name}: no output written after {gen[-20:]}")
            t = self.L.vocab[i]
            if t == stop:
                return gen
            gen.append(t)
            logits = self.step(st, i)
        raise RuntimeError(f"{self.name}: no stop token after {max_new} steps")
