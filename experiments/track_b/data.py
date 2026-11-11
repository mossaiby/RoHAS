"""Track B data: token files for pretrained experts that share the Qwen2.5 tokenizer.

    python experiments/track_b/data.py

Sources
  prose     fineweb-edu (HuggingFaceFW/fineweb-edu, sample/100BT shard 000): row groups 0-99
            for the training pool, 700-719 for validation
  code      codeparrot-clean-valid with the Track C filters and repository split (97 / 3)
  notebooks codeparrot/github-jupyter-text-code-pairs: each row is a markdown cell and the
            code cell after it; consecutive rows of one notebook form one document, each
            pair as "markdown\\n\\ncode", pairs joined by a blank line (held out from all training)

Files (data/track_b/, uint32 token ids, documents end with <|endoftext|>; mixed and notebook
files have a parallel _labels.bin, uint8, 0 = prose / markdown, 1 = code)
  mixed_train           2M tokens, alternating prose and code segments in the blank-line
                        format of Track C (the gate's training data)
  prose_val, code_val   300k tokens each, whole documents
  mixed_<fmt>_val       300k tokens each, fmt in blank, long, inline, marked (Track C formats;
                        long, inline and marked are held out from gate training)
  notebooks_val         300k tokens
"""
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT  # noqa: E402
from experiments.track_c.data import HUB, code_files  # noqa: E402
from experiments.track_c.data_mixed_ood import FORMATS  # noqa: E402

OUT = os.path.join(ROOT, "data", "track_b")
TOKENIZER = "Qwen/Qwen2.5-0.5B"
TYPES = ["prose", "code"]
VAL = 300_000
TRAIN = 2_000_000


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def prose(row_groups):
    import pyarrow.parquet as pq
    f = glob.glob(os.path.join(HUB, "datasets--HuggingFaceFW--fineweb-edu", "snapshots", "*", "sample", "100BT",
                               "000_00000.parquet"))[0]
    pf = pq.ParquetFile(f)
    out = []
    for g in row_groups:
        out += [t.strip() for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist() if t.strip()]
    return out


def notebooks():
    """Documents of (text, type) pieces, one per notebook, in file order."""
    import pyarrow.parquet as pq
    f = glob.glob(os.path.join(HUB, "datasets--codeparrot--github-jupyter-text-code-pairs", "snapshots", "*",
                               "data", "*.parquet"))[0]
    t = pq.read_table(f)
    cols = t.column_names
    key = [c for c in ("repo_name", "path") if c in cols]
    rows = t.select(key + ["markdown", "code"]).to_pylist()
    docs, cur, last = [], [], None
    for r in rows:
        k = tuple(r[c] for c in key)
        if k != last and cur:
            docs.append(cur)
            cur = []
        last = k
        md, code = (r["markdown"] or "").strip(), (r["code"] or "").strip("\n")
        if md and code:
            cur += [((("\n\n" if cur else "") + md), 0), ("\n\n" + code, 1)]
    if cur:
        docs.append(cur)
    return docs, cols


class Writer:
    def __init__(self, tok, name, labelled):
        self.tok, self.name, self.labelled = tok, name, labelled
        self.ids, self.labels = [], []

    def doc(self, pieces):
        """pieces: [(text, type)]; each piece is encoded on its own (it starts with its separator)."""
        for text, k in pieces:
            t = self.tok(text, add_special_tokens=False).input_ids
            self.ids += t
            self.labels += [k] * len(t)
        self.ids.append(self.tok.eos_token_id)
        self.labels.append(self.labels[-1] if self.labels else 0)

    def __len__(self):
        return len(self.ids)

    def close(self):
        np.array(self.ids, np.uint32).tofile(os.path.join(OUT, f"{self.name}.bin"))
        lab = np.array(self.labels, np.uint8)
        if self.labelled:
            lab.tofile(os.path.join(OUT, f"{self.name}_labels.bin"))
        share = round(float(np.mean(lab == 1)), 3) if self.labelled else None
        print(f"  {self.name}: {len(self.ids):,} tokens" + (f", code share {share}" if self.labelled else ""),
              flush=True)
        return len(self.ids)


def segment(kind, rng, pools, f):
    if kind == "prose":
        sents = re.split(r"(?<=[.!?])\s+", rng.choice(pools["prose"]))
        k = min(rng.randint(*f["sents"]), len(sents))
        i = rng.randint(0, len(sents) - k)
        return " ".join(sents[i:i + k])
    lines = rng.choice(pools["code"]).split("\n")
    k = rng.randint(*f["code"])
    i = rng.randint(0, max(0, len(lines) - k))
    return "\n".join(lines[i:i + k]).strip("\n")


def mixed(tok, pools, fmt, target, seed, name):
    rng = random.Random(seed)
    f = FORMATS[fmt]
    w = Writer(tok, name, True)
    while len(w) < target:
        n = rng.randint(*f["n"])
        first = rng.randrange(2)
        pieces = []
        for j in range(n):
            k = (first + j) % 2
            seg = segment(TYPES[k], rng, pools, f)
            if seg:
                pieces.append(((rng.choice(f["seps"]) if pieces else "") + seg, k))
        if len({k for _, k in pieces}) == 2:
            w.doc(pieces)
    return w.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tok = tokenizer()
    code = {"train": [], "val": []}
    for split, t in code_files():
        code[split].append(t)
    pools_train = {"prose": prose(range(0, 100)), "code": code["train"]}
    pools_val = {"prose": prose(range(700, 720)), "code": code["val"]}
    print(f"pools: prose {len(pools_train['prose'])}/{len(pools_val['prose'])}, "
          f"code {len(code['train'])}/{len(code['val'])} ({time.time() - t0:.0f}s)", flush=True)
    counts = {"mixed_train": mixed(tok, pools_train, "blank", TRAIN, seed=1, name="mixed_train")}
    for i, fmt in enumerate(["blank", "long", "inline", "marked"]):
        counts[f"mixed_{fmt}_val"] = mixed(tok, pools_val, fmt, VAL, seed=10 + i, name=f"mixed_{fmt}_val")
    for name, pool in [("prose_val", pools_val["prose"]), ("code_val", code["val"])]:
        w = Writer(tok, name, False)
        for d in random.Random(20).sample(pool, len(pool)):
            if len(w) >= VAL:
                break
            w.doc([(d, 0)])
        counts[name] = w.close()
    docs, cols = notebooks()
    w = Writer(tok, "notebooks_val", True)
    for d in random.Random(30).sample(docs, len(docs)):
        if len(w) >= VAL:
            break
        w.doc(d[:40])                                 # at most 20 cell pairs per notebook
    counts["notebooks_val"] = w.close()
    meta = {"tokenizer": TOKENIZER, "counts": counts, "notebook_columns": cols, "notebook_docs": len(docs),
            "seconds": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
