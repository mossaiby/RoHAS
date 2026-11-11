"""Track B, step 3 data: prose, code and arithmetic in mixed documents (Qwen tokenizer).

    python experiments/track_b/data_circuit.py

The arithmetic segments are 1-3 consecutive lines of Track C's synthetic math documents
(synth.math_doc: equations "a op b=c" with up to 5 digits for + and -, 3 for *, and templated
word problems); prose and code come from the step-1 pools. Label 2 marks arithmetic tokens.
  a4_train          1.5M tokens, blank-line format (trains the circuit's head)
  a4_blank_val      200k tokens, blank-line format
  a4_inline_val     200k tokens, inline format (held out)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.track_b.data import OUT, mixed, prose, tokenizer  # noqa: E402
from experiments.track_c import synth  # noqa: E402
from experiments.track_c.data import code_files  # noqa: E402


def math_pool(seed, n=20000):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        lines = synth.math_doc(rng).split("\n")
        k = rng.randint(1, 3)
        i = rng.randint(0, max(0, len(lines) - k))
        out.append("\n".join(lines[i:i + k]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = tokenizer()
    code = {"train": [], "val": []}
    for split, t in code_files():
        code[split].append(t)
    train = {"prose": prose(range(0, 100)), "code": code["train"], "math": math_pool(41)}
    val = {"prose": prose(range(700, 720)), "code": code["val"], "math": math_pool(42)}
    mixed(tok, train, "blank", 1_500_000, seed=201, name="a4_train", n_types=3)
    mixed(tok, val, "blank", 200_000, seed=210, name="a4_blank_val", n_types=3)
    mixed(tok, val, "inline", 200_000, seed=211, name="a4_inline_val", n_types=3)


if __name__ == "__main__":
    main()
