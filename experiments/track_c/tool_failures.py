"""Track C: why the calculator-tool models fail on arithmetic inside other text.

    python experiments/track_c/tool_failures.py [--models main_mono_s_calc_end main_mono_s_calcv_end]

Reruns the in-text items of tool_baseline.py (context_items, the same greedy decoding) and
classifies every wrong answer from the raw output, before CALL and END are stripped:
  no_call     the model never emits CALL after the equation and writes digits itself
  bad_call    it emits CALL, but where the tool cannot parse an equation (for example after
              "=-", since in training the call always follows "=") or the result is wrong
Writes results/track_c/tool_failures.json.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c import evaluate as E  # noqa: E402
from experiments.track_c.tool_baseline import CALL, END, context_items, make, tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["main_mono_s_calc_end", "main_mono_s_calcv_end"])
    a = ap.parse_args()
    tok = tokenizer()
    items = context_items(tok)
    first = [tok.token_to_id("="), tok.token_to_id("=-")]
    out = {}
    for m in a.models:
        s = make(f"tool:{m}", tok)
        raw = E.greedy(s, tok, [it["prompt"] for it in items], max_new=2 * 16 + 4, first=first)
        kinds, by_cell, examples = collections.Counter(), collections.Counter(), []
        for it, o in zip(items, raw):
            want = "=" + it["answer"]
            clean = o.replace(CALL, "").replace(END, "").strip()
            if clean.startswith(want) and not clean[len(want):len(want) + 1].isdigit():
                continue
            k = "no_call" if CALL not in o else "bad_call"
            kinds[k] += 1
            by_cell[f"{it['kind']}/{it['join']}/{k}"] += 1
            if len(examples) < 10:
                examples.append({"context_end": it["prompt"][-60:], "output": o[:40].replace(CALL, "<CALL>")
                                 .replace(END, "<END>"), "want": want, "kind": k})
        out[m] = {"items": len(items), "wrong": sum(kinds.values()), **kinds, "by_cell": dict(by_cell),
                  "examples": examples}
        print(m, dict(kinds), "wrong", sum(kinds.values()), "of", len(items), flush=True)
        del s
    save("track_c", "tool_failures", out)


if __name__ == "__main__":
    main()
