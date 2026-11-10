"""Track C: mixed validation text in formats no model was trained on.

    python experiments/track_c/data_mixed_ood.py

mixed_val, mixed_train (seen by the gates and mono_s) and the context-exposure files
(data_context.py) all come from data.build_mixed: 3-5 segments of 1-4 sentences / 1-3
attributes / 1-3 math lines / 5-20 code lines, joined by a blank line. To tell reading
foreign context apart from matching that format, two validation sets change the format and
keep the sources of mixed_val (validation stories, stored people with training templates,
fresh math, validation-split code):

  mixed_long_val    2-3 long segments (4-10 sentences, 3-6 attributes, 3-8 math lines,
                    20-60 code lines) joined by a single newline
  mixed_inline_val  4-8 short segments (1-2 sentences, 1-2 attributes, 1 math line, 2-6
                    code lines) joined by a space or a single newline, chosen per boundary,
                    so types change mid-paragraph
  mixed_marked_val  3-6 medium segments joined by a tab, " | " or " ; "

The varied context exposure (data_context.py --varied) draws its formats from VARIED: the
build_mixed format, "long" and a triple-newline format. Every join there is a newline, so
inline (half its joins are spaces) and marked (no newline joins) stay held out from all
training; long does not, for the varied-exposure experts.

The separator is encoded together with the segment that follows it, so tokens are those of
natural text, and it carries that segment's label. Writes <name>.bin and <name>_labels.bin.
"""
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.track_c import synth  # noqa: E402
from experiments.track_c.data import EOT, OUT, TYPES, code_files, stories  # noqa: E402

FORMATS = {
    # held-out evaluation formats (no model trains on them, except "long" for the varied exposure)
    "long": dict(n=(2, 3), sents=(4, 10), attrs=(3, 6), math=(3, 8), code=(20, 60), seps=["\n"]),
    "inline": dict(n=(4, 8), sents=(1, 2), attrs=(1, 2), math=(1, 1), code=(2, 6), seps=[" ", "\n"]),
    "marked": dict(n=(3, 6), sents=(2, 5), attrs=(2, 4), math=(2, 4), code=(8, 30), seps=["\t", " | ", " ; "]),
    # build_mixed's format, and a third newline format, for the varied exposure (data_context.py --varied)
    "blank": dict(n=(3, 5), sents=(1, 4), attrs=(1, 3), math=(1, 3), code=(5, 20), seps=["\n\n"]),
    "para3": dict(n=(3, 5), sents=(2, 6), attrs=(2, 4), math=(2, 5), code=(10, 40), seps=["\n\n\n"]),
}
EVAL = ["long", "inline", "marked"]
VARIED = ["blank", "long", "para3"]             # every join is newline-based: inline and marked stay held out


def segment(kind, rng, pools, f):
    if kind == "lang":
        sents = re.split(r"(?<=[.!?])\s+", rng.choice(pools["lang"]))
        k = min(rng.randint(*f["sents"]), len(sents))
        i = rng.randint(0, len(sents) - k)
        return " ".join(sents[i:i + k])
    if kind == "facts":
        p = rng.choice(pools["people"])
        attrs = rng.sample(list(synth.ATTRS), rng.randint(*f["attrs"]))
        return " ".join(synth.sentence(p, a, rng.choice(synth.TRAIN_T[a])) for a in attrs)
    if kind == "math":
        lines = []
        while len(lines) < f["math"][1]:
            lines += [l for l in synth.math_doc(rng).split("\n") if l.strip()]
        return "\n".join(lines[:rng.randint(*f["math"])])
    lines = rng.choice(pools["code"]).split("\n")           # blank lines kept, as in build_mixed
    k = rng.randint(*f["code"])
    i = rng.randint(0, max(0, len(lines) - k))
    return "\n".join(lines[i:i + k]).strip("\n")


def build(tok, pools, fmts, target, seed, own=None, name=None):
    """Documents in formats drawn uniformly from fmts, adjacent segments of different types.
    With `own` (the context-exposure data), each segment is that type with probability 1/2,
    every document has one, and same-type neighbours are allowed."""
    rng = random.Random(seed)
    eot = tok.token_to_id(EOT)
    ids, labels = [], []
    while len(ids) < target:
        f = FORMATS[fmts[0] if len(fmts) == 1 else rng.choice(fmts)]
        while True:
            n = rng.randint(*f["n"])
            if own is None:
                kinds = [rng.choice(TYPES) for _ in range(n)]
                ok = all(a != b for a, b in zip(kinds, kinds[1:]))
            else:
                others = [t for t in TYPES if t != own]
                kinds = [own if rng.random() < 0.5 else rng.choice(others) for _ in range(n)]
                ok = own in kinds
            if ok and len(set(kinds)) >= 2:
                break
        for j, k in enumerate(kinds):
            seg = segment(k, rng, pools, f)
            if not seg:
                continue
            t = tok.encode((rng.choice(f["seps"]) if j else "") + seg).ids
            ids += t
            labels += [TYPES.index(k)] * len(t)
        ids.append(eot)
        labels.append(labels[-1])
    name = name or f"mixed_{fmts[0]}_val"
    np.array(ids, np.uint16).tofile(os.path.join(OUT, f"{name}.bin"))
    np.array(labels, np.uint8).tofile(os.path.join(OUT, f"{name}_labels.bin"))
    share = {t: round(float(np.mean(np.array(labels) == i)), 3) for i, t in enumerate(TYPES)}
    print(f"  {name}: {len(ids):,} tokens, type share {share}", flush=True)
    return len(ids)


def main():
    import argparse
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="*", default=EVAL)
    a = ap.parse_args()
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    pools = {"lang": list(stories("val")),
             "people": json.load(open(os.path.join(OUT, "people.json")))["old"],
             "code": [t for split, t in code_files() if split == "val"]}
    for fmt in a.formats:
        build(tok, pools, [fmt], 500_000, seed=200 + EVAL.index(fmt))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
