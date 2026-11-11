"""Track B: each expert's loss by segment type on the step-2 and step-3 mixed files (from the caches).

    python experiments/track_b/segment_losses.py

Shows the contamination found in step 2: on GSM8K train problems (m3_train) the experts,
and the math expert most, are far better than on GSM8K test problems (m3_blank_val); the
step-3 files use the two halves of GSM8K test instead. Writes results/track_b/segment_losses.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_b.data import TYPES  # noqa: E402
from experiments.track_b.cache import CACHE  # noqa: E402
from experiments.track_b.gate import labels  # noqa: E402

FILES = ["m3_train", "m3_blank_val", "m3c_train", "m3c_blank_val"]
EXPERTS = ["base", "coder", "math"]


def tlp(e, f):
    return np.load(os.path.join(CACHE, e, f"{f}.tlp.npy"), mmap_mode="r")


def main():
    out = {}
    for f in FILES:
        W = tlp(EXPERTS[0], f).shape[0]
        lab = labels(f, W)
        out[f] = {}
        for t, nm in enumerate(TYPES):
            m = lab == t
            out[f][nm] = {e: float(-np.asarray(tlp(e, f))[m].mean()) for e in EXPERTS}
            out[f][nm]["tokens"] = int(m.sum())
        print(f, {k: {e: round(x, 3) for e, x in v.items()} for k, v in out[f].items()})
    save("track_b", "segment_losses", out)


if __name__ == "__main__":
    main()
