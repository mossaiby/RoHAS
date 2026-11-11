"""Track B: cache the outputs of one frozen pretrained expert on every data file.

    python experiments/track_b/cache.py --expert base   # Qwen/Qwen2.5-0.5B
    python experiments/track_b/cache.py --expert coder  # Qwen/Qwen2.5-Coder-0.5B
    python experiments/track_b/cache.py --expert math --step 2 --bs 2   # Qwen/Qwen2.5-Math-1.5B

Every file (data.py) is cut into non-overlapping windows of 513 tokens (512 inputs and their
next tokens). For each window and input position the expert's
  h     final hidden state (after the last norm), float16, (W, 512, d)
  tlp   log-probability of the actual next token, float32, (W, 512)
  ent   entropy of the next-token distribution (nats), float32
  mx    largest log-probability, float32
go to data/track_b/cache/<expert>/<file>.<field>.npy; the windows' tokens go to
data/track_b/cache/<file>.x.npy (shared by the experts). Only one model is in memory at a time.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.track_b.data import OUT  # noqa: E402

EXPERTS = {"base": "Qwen/Qwen2.5-0.5B", "coder": "Qwen/Qwen2.5-Coder-0.5B", "math": "Qwen/Qwen2.5-Math-1.5B"}
FILES = ["mixed_train", "prose_val", "code_val", "mixed_blank_val", "mixed_long_val", "mixed_inline_val",
         "mixed_marked_val", "notebooks_val"]
FILES2 = ["m3_train", "prose_val", "code_val", "math_val", "m3_blank_val", "m3_long_val", "m3_inline_val",
          "m3_marked_val", "notebooks_val"]        # step 2 (+ math); the first file trains the gate
FILES3 = ["m3c_train", "prose_val", "code_val", "mathc_val", "m3c_blank_val", "m3c_long_val", "m3c_inline_val",
          "m3c_marked_val", "notebooks_val"]       # step 3 (+ math, GSM8K test halves)
CTX = 512
CACHE = os.path.join(OUT, "cache")


def windows(name):
    arr = np.fromfile(os.path.join(OUT, f"{name}.bin"), dtype=np.uint32)
    n = (len(arr) - 1) // CTX
    return np.stack([arr[i * CTX:(i + 1) * CTX + 1] for i in range(n)]).astype(np.int64)


@torch.no_grad()
def run(model, X, out, bs=4, chunk=128):
    W = len(X)
    d = model.config.hidden_size
    mm = {k: np.lib.format.open_memmap(os.path.join(out + f".{k}.npy"), mode="w+",
                                       dtype=np.float16 if k == "h" else np.float32,
                                       shape=(W, CTX, d) if k == "h" else (W, CTX)) for k in ("h", "tlp", "ent", "mx")}
    head = model.get_output_embeddings()
    for i in range(0, W, bs):
        xy = torch.from_numpy(X[i:i + bs]).cuda()
        x, y = xy[:, :-1], xy[:, 1:]
        h = model.model(input_ids=x).last_hidden_state              # after the final norm
        mm["h"][i:i + bs] = h.half().cpu().numpy()
        tl, en, mx = [], [], []
        for s in range(0, CTX, chunk):                                # logits over 152k tokens, in chunks
            lp = torch.log_softmax(head(h[:, s:s + chunk]).float(), dim=-1)
            tl.append(lp.gather(-1, y[:, s:s + chunk].unsqueeze(-1)).squeeze(-1))
            en.append(-(lp.exp() * lp).sum(-1))
            mx.append(lp.amax(-1))
        for k, v in (("tlp", tl), ("ent", en), ("mx", mx)):
            mm[k][i:i + bs] = torch.cat(v, dim=1).cpu().numpy()
    for v in mm.values():
        v.flush()
    return float(-mm["tlp"][:].mean())


def main():
    from transformers import AutoModelForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, choices=list(EXPERTS))
    ap.add_argument("--step", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    a.files = a.files or {1: FILES, 2: FILES2, 3: FILES3}[a.step]
    os.makedirs(os.path.join(CACHE, a.expert), exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(EXPERTS[a.expert], dtype=torch.bfloat16).cuda().eval()
    log = {}
    for f in a.files:
        out = os.path.join(CACHE, a.expert, f)
        if os.path.exists(out + ".done"):
            print(f"{f}: cached", flush=True)
            continue
        X = windows(f)
        xp = os.path.join(CACHE, f"{f}.x.npy")
        if not os.path.exists(xp):
            np.save(xp, X.astype(np.int32))
        t0 = time.time()
        loss = run(model, X, out, bs=a.bs)
        dt = time.time() - t0
        open(out + ".done", "w").write("ok")            # only complete files count as cached
        log[f] = {"windows": len(X), "loss": loss, "seconds": round(dt), "tok_per_s": round(len(X) * CTX / dt)}
        print(f"{f}: {json.dumps(log[f])}", flush=True)
    path = os.path.join(CACHE, a.expert, "log.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **log}, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
