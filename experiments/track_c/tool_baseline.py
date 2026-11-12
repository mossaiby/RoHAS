"""Track C: a calculator-tool baseline (Toolformer-style) for the circuit under the learned gate.

    python experiments/track_c/tool_baseline.py [--only data train eval context diag]

  data     mathcalc_train and mixedcalc_train: math_train (first 40M tokens) and mixed_train
           with every equation "a op b=c" written as "a op b=" CALL "c", where CALL is the
           byte 0x01 (one BPE token; 4 occurrences in all training data, all in code). A
           parallel _labels.bin marks the answer tokens (tool output) with 255 and every other
           token with 2 ("math"), so train.py --own math drops exactly the tool output from the
           loss of these two sources; lang, facts and code have no label file and are unaffected.
  train    main_mono_s_calc: main_mono_s fine-tuned on mono_s's proportions with the annotated
           math and mixed data (50M tokens, lr 3e-4, 8k-token batches)
  eval     held-out arithmetic (eval seed 1000, 100 items per cell) and word problems for
           mono_s, mono_s+calc (with the tool) and the atom gates of main_ctxv20 (expert seeds
           0, 1, 2); val losses and fact recall of main_mono_s_calc (evaluate.py, tag
           tool_baseline, arithmetic skipped there since it needs the tool)
  context  arithmetic inside other text: a story, biography or code segment (validation
           pools), then a blank line or a space, then "a op b"; exact answers, + - *, 4 and 16
           digits (results/track_c/tool_context.json)
  diag     mixed-text loss of main_mono_s_calc on the four formats (tags <format>_tool_baseline)

--end: the variant main_mono_s_calc_end (data mathcalce_train, mixedcalce_train; diag tags
with "_end"). The tool inserts the answer and a closing marker (byte 0x02, one token), like
Toolformer's "-> result]", so the model resumes after a delimiter. Added after the first run,
whose format had no closing marker: there the model often kept writing digits after the
inserted answer on long numbers.

--varied (with --end): the variant main_mono_s_calcv_end. Its mixed data is mixedv_train,
mixed documents from mixed_train's pools in the formats of the experts' varied context
exposure (data_mixed_ood.VARIED), and the mixed share is raised to 0.2, the experts' exposure
fraction (data mathcalce_train, mixedcalcve_train; diag tags with "_vend"). Added to answer
the objection that the baseline is weak inside text only for lack of varied mixed text.

The tool. When the model emits CALL, the tool parses the tokens right before it with the
circuit expert's rule (<non-digit> a op b =, digits as single tokens), computes the answer
with Python integers and inserts its tokens; then the model continues. The circuit expert and
the tool therefore share the parser and the arithmetic; they differ in how the system decides
to use them: call labels and fine-tuning here, the label-free gate there.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_c import evaluate as E, synth  # noqa: E402
from experiments.track_c.data import OUT as DATA, EOT, segments  # noqa: E402
from experiments.track_c.data_mixed_ood import EVAL  # noqa: E402

PY = sys.executable
HERE = os.path.join(ROOT, "experiments", "track_c")
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
CALL = "\x01"
END = "\x02"      # --end: closing marker inserted by the tool after the answer (Toolformer's "]")
NAME = "main_mono_s_calc"
EQ = re.compile(r"(?<!\d)(\d+[+\-*]\d+=)(-?\d+)")
GATES = ["main_ctxv20_gate_atom", "main_ctxv20_e1_gate_atom", "main_ctxv20_e2_gate_atom"]


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))


# ---- data ----------------------------------------------------------------------------
def annotate(text, end=""):
    """Text with calls, and the character spans of the answers (tool output, with `end`)."""
    out, spans, last = [], [], 0
    for m in EQ.finditer(text):
        out.append(text[last:m.end(1)] + CALL)
        start = sum(map(len, out))
        out.append(m.group(2) + end)
        spans.append((start, start + len(m.group(2)) + len(end)))
        last = m.end()
    out.append(text[last:])
    return "".join(out), spans


def write_calc(tok, src, dst, max_tokens=None, end=""):
    arr = np.fromfile(os.path.join(DATA, f"{src}.bin"), dtype=np.uint16)
    eot = tok.token_to_id(EOT)
    if max_tokens:
        cut = int(np.nonzero(arr[:max_tokens] == eot)[0][-1]) + 1
        arr = arr[:cut]
    ends = np.nonzero(arr == eot)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    ids, labels, n_eq = [], [], 0
    B = 20000
    for s in range(0, len(starts), B):
        docs = tok.decode_batch([arr[a:b].tolist() for a, b in zip(starts[s:s + B], ends[s:s + B])],
                                skip_special_tokens=False)
        ann = [annotate(d, end) for d in docs]
        for (text, spans), enc in zip(ann, tok.encode_batch([t for t, _ in ann])):
            n_eq += len(spans)
            lab = np.full(len(enc.ids) + 1, 2, np.uint8)
            j = 0
            for k, (a, b) in enumerate(enc.offsets):
                while j < len(spans) and spans[j][1] <= a:
                    j += 1
                if j < len(spans) and a >= spans[j][0] and b <= spans[j][1]:
                    lab[k] = 255
            ids.append(np.array(enc.ids + [eot], np.uint16))
            labels.append(lab)
    ids, labels = np.concatenate(ids), np.concatenate(labels)
    ids.tofile(os.path.join(DATA, f"{dst}.bin"))
    labels.tofile(os.path.join(DATA, f"{dst}_labels.bin"))
    print(f"  {dst}: {len(ids):,} tokens, {n_eq:,} calls, {int((labels == 255).sum()):,} tool tokens "
          f"(from {len(arr):,} tokens of {src})", flush=True)


def build_mixedv(tok):
    """mixedv_train: mixed_train's pools (data.py), formats from data_mixed_ood.VARIED, 12M tokens."""
    from experiments.track_c.data import code_files, stories
    from experiments.track_c.data_mixed_ood import VARIED, build
    lang, n = [], 0
    for s in stories("train"):            # the story sample data.py used for mixed_train
        lang.append(s)
        n += len(s)
        if n > 25e6:
            break
    pools = {"lang": lang, "people": json.load(open(os.path.join(DATA, "people.json")))["old"],
             "code": [t for split, t in code_files() if split == "train"]}
    build(tok, pools, VARIED, 12_000_000, seed=500, name="mixedv_train")


# ---- the tool --------------------------------------------------------------------------
class Tool:
    """A model plus the calculator: after CALL, the answer's tokens are forced."""

    def __init__(self, name, tok, end=""):
        self.inner = E.Single(name)
        self.end = tok.encode(end).ids if end else []
        self.name = f"tool:{name}"
        self.call = tok.encode(CALL).ids[0]
        self.eq = tok.token_to_id("=")
        self.digit = {tok.token_to_id(str(d)): str(d) for d in range(10)}
        self.ops = {tok.token_to_id(o): o for o in "+-*"}
        self.tok = tok
        self.calls = self.bad = 0

    def parse(self, row):
        """Answer token ids for the equation right before the CALL that ends `row`, or None."""
        i = len(row) - 1
        if i < 1 or row[i - 1] != self.eq:
            return None
        j = i - 1
        b = []
        while j > 0 and row[j - 1] in self.digit:
            j -= 1
            b.append(self.digit[row[j]])
        if not b or j < 1 or row[j - 1] not in self.ops:
            return None
        op = self.ops[row[j - 1]]
        j -= 1
        a = []
        while j > 0 and row[j - 1] in self.digit:
            j -= 1
            a.append(self.digit[row[j]])
        if not a:
            return None
        x, y = int("".join(reversed(a))), int("".join(reversed(b)))
        return self.tok.encode(str(x + y if op == "+" else x - y if op == "-" else x * y)).ids + self.end

    def logprobs(self, x):
        return self.inner.logprobs(x)

    @torch.no_grad()
    def last_logprobs(self, x):
        lp = self.inner.last_logprobs(x)
        rows = x.tolist()
        for r, row in enumerate(rows):
            if self.call not in row:
                continue
            k = len(row) - 1 - row[::-1].index(self.call)
            if k == len(row) - 1:
                self.calls += 1
            ans = self.parse(row[:k + 1])
            if ans is None:
                self.bad += k == len(row) - 1
                continue
            done = row[k + 1:]
            if len(done) < len(ans) and done == ans[:len(done)]:
                lp[r] = -float("inf")
                lp[r, ans[len(done)]] = 0.0
        return lp


class Strip:
    """The tokenizer, with CALL removed from decoded text (outputs are compared as text)."""

    def __init__(self, tok):
        self.t = tok

    def decode(self, ids, **kw):
        return self.t.decode(ids, **kw).replace(CALL, "").replace(END, "")

    def __getattr__(self, k):
        return getattr(self.t, k)


def make(spec, tok):
    kind, name = spec.split(":", 1)
    return Tool(name, tok, END if name.endswith("_end") else "") if kind == "tool" else E.make_system(spec)


# ---- tests -------------------------------------------------------------------------------
def heldout(systems, tok):
    path = os.path.join(ROOT, "results", "track_c", "tool_heldout.json")
    res = json.load(open(path))["data"] if os.path.exists(path) else {}
    for spec in systems:
        if spec in res:
            continue
        s = make(spec, tok)
        line, number = E.arithmetic(s, Strip(tok), per_cell=100, seed=1000)
        word = E.word_problems(s, Strip(tok), seed=1001)
        res[spec] = {"arith": line, "arith_number": number, "word": word}
        if isinstance(s, Tool):
            res[spec]["tool_calls"], res[spec]["tool_unparsed"] = s.calls, s.bad
        print(spec, json.dumps({k: v for k, v in res[spec].items() if k != "arith"}), flush=True)
        save("track_c", "tool_heldout", res)
        del s
        torch.cuda.empty_cache()


def context_items(tok, n=50, seed=2000):
    rng = random.Random(seed)
    eot = tok.token_to_id(EOT)

    def docs(name, k):
        arr = np.fromfile(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16)[:3_000_000]
        ends = np.nonzero(arr == eot)[0]
        st = np.concatenate([[0], ends[:-1] + 1])
        pick = rng.sample(range(len(ends)), min(k, len(ends)))
        return [tok.decode(arr[st[i]:ends[i]].tolist()) for i in pick]
    people = json.load(open(os.path.join(DATA, "people.json")))["old"]
    pools = {"lang": docs("lang_val", 2000), "code": docs("code_val", 500), "people": people}
    items = []
    for kind in ["lang", "facts", "code"]:
        for join, jn in [("\n\n", "blank"), (" ", "inline")]:
            for op in "+-*":
                for d in (4, 16):
                    for _ in range(n):
                        seg = ""
                        while not seg:
                            seg = segments(kind, rng, pools)
                        # keep the context short and end it cleanly (no trailing digits)
                        seg = seg[-600:].rstrip()
                        if seg and seg[-1].isdigit():
                            seg += "."
                        q, a = synth.arith_line(rng, op, d, d)
                        items.append({"kind": kind, "join": jn, "op": op, "digits": d,
                                      "prompt": seg + join + q, "answer": a})
    return items


def context(systems, tok):
    path = os.path.join(ROOT, "results", "track_c", "tool_context.json")
    res = json.load(open(path))["data"] if os.path.exists(path) else {}
    items = context_items(tok)
    first = [tok.token_to_id("="), tok.token_to_id("=-")]
    for spec in systems:
        if spec in res:
            continue
        s = make(spec, tok)
        outs = E.greedy(s, Strip(tok), [it["prompt"] for it in items], max_new=2 * 16 + 4, first=first)
        cells = {}
        for it, o in zip(items, outs):
            o, want = o.strip(), "=" + it["answer"]
            ok = int(o.startswith(want) and not o[len(want):len(want) + 1].isdigit())
            cells.setdefault(f"{it['kind']}/{it['join']}/{it['op']}/{it['digits']}", []).append(ok)
        r = {k: float(np.mean(v)) for k, v in cells.items()}
        summ = {}
        for key in ("kind", "join", "digits"):
            for val in sorted({str(it[key]) for it in items}):
                summ[f"{key}={val}"] = float(np.mean([ok for k, v in cells.items() for ok in v
                                                      if dict(zip(("kind", "join", "op", "digits"),
                                                                  k.split("/")))[key] == val]))
        summ["all"] = float(np.mean([ok for v in cells.values() for ok in v]))
        res[spec] = {"cells": r, "summary": summ}
        if isinstance(s, Tool):
            res[spec]["tool_calls"], res[spec]["tool_unparsed"] = s.calls, s.bad
        print(spec, json.dumps(summ), flush=True)
        save("track_c", "tool_context", res)
        del s
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--end", action="store_true", help="the tool also inserts a closing marker (variant _end)")
    ap.add_argument("--varied", action="store_true", help="with --end: varied-format mixed data, share 0.2")
    a = ap.parse_args()
    assert a.end or not a.varied, "--varied is defined for the closing-marker variant"
    name, sfx, end = (NAME + "_end", "e", END) if a.end else (NAME, "", "")
    if a.varied:
        name = "main_mono_s_calcv_end"
    steps = a.only or ["data", "train", "eval", "context", "diag"]
    tok = tokenizer()
    systems = ["model:main_mono_s", f"tool:{name}"] + [f"gate:{g}:prob" for g in GATES]
    if "data" in steps:
        if not os.path.exists(os.path.join(DATA, f"mathcalc{sfx}_train.bin")):
            write_calc(tok, "math_train", f"mathcalc{sfx}_train", max_tokens=40_000_000, end=end)
        if not os.path.exists(os.path.join(DATA, f"mixedcalc{sfx}_train.bin")):
            write_calc(tok, "mixed_train", f"mixedcalc{sfx}_train", end=end)
        if a.varied and not os.path.exists(os.path.join(DATA, "mixedcalcve_train.bin")):
            if not os.path.exists(os.path.join(DATA, "mixedv_train.bin")):
                build_mixedv(tok)
            write_calc(tok, "mixedv_train", "mixedcalcve_train", end=end)
    if "train" in steps and not os.path.exists(os.path.join(CKPT, f"{name}.pt")):
        mono = json.load(open(os.path.join(ROOT, "results", "track_c", "train_main_mono_s.json")))["data"]["mix"]
        src = {"math": f"mathcalc{sfx}", "mixed": f"mixedcalc{sfx}"}
        mix = [f"{src.get(k, k)}={v}" for k, v in mono.items()]
        if a.varied:
            mix = ["lang=0.2", "facts=0.2", f"mathcalc{sfx}=0.2", "code=0.2", "mixedcalcve=0.2"]
        run([os.path.join(HERE, "train.py"), "--name", name, "--size", "s", "--init",
             os.path.join(CKPT, "main_mono_s.pt"), "--mix", *mix, "--own", "math", "--tokens", "50e6",
             "--micro", "16", "--batch-tokens", "8192", "--lr", "3e-4", "--evals", "4"])
    if "eval" in steps:
        heldout(systems, tok)
        path = os.path.join(ROOT, "results", "track_c", "eval_tool_baseline.json")
        if not os.path.exists(path) or f"model:{name}" not in json.load(open(path))["data"]["systems"]:
            run([os.path.join(HERE, "evaluate.py"), "--tag", "tool_baseline", "--skip", "math", "routing",
                 "--systems", f"model:{name}"])
    if "context" in steps:
        context(systems, tok)
    if "diag" in steps:
        for d in ["mixed_val"] + [f"mixed_{f}_val" for f in EVAL]:
            tag = ("tool_baseline" if d == "mixed_val" else f"{d[6:-4]}_tool_baseline") + \
                ("_vend" if a.varied else "_end" if a.end else "")
            run([os.path.join(HERE, "diag_mixed.py"), "--tag", tag, "--data", d, "--models", name])


if __name__ == "__main__":
    main()
