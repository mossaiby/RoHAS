"""Track C: context-exposure data, one file per expert type.

    python experiments/track_c/data_context.py [--tokens 12e6] [--varied] [--update]

On mixed text an expert reads segments of types it never saw, and most of the gated
system's loss gap to mono_s is in the first tokens of each segment (see diag_mixed.py).
In training, every 512-token window an expert sees comes from its own source, so it never
sees a change of type. ctx_<type>_train.bin holds mixed documents (as data.build_mixed) in
which each segment is <type> with probability 1/2, with a parallel label file; train.py
--own <type> computes the loss only on that type's tokens, so the expert learns to read
foreign context without being trained to predict it.

Provenance. An expert's OWN segments come from its own training data. FOREIGN segments come
from material no expert stores where possible, so that updating or deleting facts still
touches only the facts expert:
  facts  biographies of fictitious people: names 2,501 onward of the shuffle that gave the
         corpus its 2,500 people (synth.people), so no name is shared with old or new people;
         attributes drawn independently
  lang   stories from TinyStories train shard 3, which lang_train (shards 0-1) never reached
  math   fresh synthetic problems (math stores no items)
  code   training-split files: every one of them is already in code_train, so foreign code
         context reuses the code expert's data (no deletion claim is made for code)
Neither the gate's mixed_train nor mixed_val (validation stories and code) is used.

--varied writes ctxv_<type>_train instead: the same pools and own-type rule, but each
document's format (segment lengths and separator) is drawn from data_mixed_ood.VARIED (the
build_mixed format, long segments joined by one newline, triple-newline joins). The ctx_
files use only the build_mixed format, which is also mixed_val's, and overfit it.

--update writes ctxvu_facts_train only: as ctxv_facts_train, but the own people are the old
and the new ones, for an add-facts update that keeps the exposure (run_ctxv_followup.py).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.track_c import synth  # noqa: E402
from experiments.track_c.data import OUT, TYPES, build_mixed, code_files, hub_files, stories  # noqa: E402
from experiments.track_c.data_mixed_ood import VARIED, build  # noqa: E402


def story_sample(files, chars=25e6):
    import pyarrow.parquet as pq
    out, n = [], 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=10_000, columns=["text"]):
            for t in b.column(0).to_pylist():
                t = t.strip()
                if t:
                    out.append(t)
                    n += len(t)
                    if n > chars:
                        return out
    return out


def fictitious_people(n, seed=21):
    """People whose names are not among the corpus's old and new people."""
    rng = random.Random(0)                                   # the shuffle of synth.people(seed=0)
    names = [f"{f} {l}" for f in synth.FIRST for l in synth.LAST]
    rng.shuffle(names)
    stored = {p["name"] for p in json.load(open(os.path.join(OUT, "people.json")))["old"]} | \
             {p["name"] for p in json.load(open(os.path.join(OUT, "people.json")))["new"]}
    free = [nm for nm in names[2500:] if nm not in stored]
    rng = random.Random(seed)
    return [{"name": nm, **{a: rng.choice(v) for a, v in synth.ATTRS.items()}} for nm in free[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=12e6, help="tokens per file")
    ap.add_argument("--varied", action="store_true", help="formats from data_mixed_ood.VARIED")
    ap.add_argument("--update", action="store_true",
                    help="only ctxvu_facts_train: varied formats, own people = old and new (the add-facts update)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    old = json.load(open(os.path.join(OUT, "people.json")))["old"]
    own_lang, n = [], 0
    for s in stories("train"):            # the story sample data.py used (inside lang_train)
        own_lang.append(s)
        n += len(s)
        if n > 25e6:
            break
    shard3 = [f for f in hub_files("datasets--roneneldan--TinyStories", "data/train-*.parquet")
              if "train-00003-" in os.path.basename(f)]
    assert len(shard3) == 1
    foreign = {"lang": story_sample(shard3), "people": fictitious_people(2000),
               "code": [t for split, t in code_files() if split == "train"]}
    assert not {p["name"] for p in foreign["people"]} & {p["name"] for p in old}
    own = {"lang": ("lang", own_lang), "facts": ("people", old), "math": None, "code": None}
    if a.update:
        new = json.load(open(os.path.join(OUT, "people.json")))["new"]
        build(tok, {**foreign, "people": old + new}, VARIED, int(a.tokens), seed=400, own="facts",
              name="ctxvu_facts_train")
        print(f"done ({time.time() - t0:.0f}s)")
        return
    for i, t in enumerate(TYPES):
        pools = dict(foreign)
        if own[t]:
            pools[own[t][0]] = own[t][1]
        if a.varied:
            build(tok, pools, VARIED, int(a.tokens), seed=300 + i, own=t, name=f"ctxv_{t}_train")
        else:
            build_mixed(tok, "train", pools, int(a.tokens), seed=100 + i, own=t, name=f"ctx_{t}_train")
    print(f"foreign pools: {len(foreign['lang'])} stories (shard 3), {len(foreign['people'])} fictitious "
          f"people, {len(foreign['code'])} code files")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
