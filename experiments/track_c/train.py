"""Train one track C model from scratch.

    python experiments/track_c/train.py --name math --size s --mix math=1 --tokens 100e6
    python experiments/track_c/train.py --name mono_s --size s \
        --mix lang=1 facts=1 math=1 code=1 mixed=0.03 --tokens 400e6
    python experiments/track_c/train.py --name mono_m --size m --mix ... --flops 4.1e16

--mix gives each data source's share of training tokens (normalised). --flops sets the
token budget from a FLOP budget (training FLOPs per token from model.flops_per_token).
--init continues from a checkpoint (used for the "add facts" test).
--own TYPE: for a source with a label file (<source>_labels.bin, e.g. the ctx_<type>_train
files of data_context.py), only tokens labelled TYPE count in the loss; other sources are
unaffected. Every token still costs a forward and backward pass, so the FLOPs are unchanged.
Writes checkpoints/track_c/<name>.pt and results/track_c/train_<name>.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT, save  # noqa: E402
from experiments.track_c import model as M  # noqa: E402

DATA = os.path.join(ROOT, "data", os.environ.get("TRACK_C_DATA", "track_c"))
CKPT = os.path.join(ROOT, "checkpoints", "track_c")
VAL_SETS = ["lang", "facts", "math", "code", "mixed"]


def tokens(name):
    return np.memmap(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16, mode="r")


def batch(arr, n, ctx, rng, labels=None):
    ix = rng.integers(0, len(arr) - ctx - 1, n)
    x = np.stack([arr[i:i + ctx + 1] for i in ix]).astype(np.int64)
    if labels is None:
        return torch.from_numpy(x)
    return torch.from_numpy(x), torch.from_numpy(np.stack([labels[i:i + ctx + 1] for i in ix]).astype(np.int64))


def labels(name):
    path = os.path.join(DATA, f"{name}_labels.bin")
    return np.memmap(path, dtype=np.uint8, mode="r") if os.path.exists(path) else None


@torch.no_grad()
def evaluate(model, ctx, device, n_batches=20, bs=16, sets=VAL_SETS):
    """Mean next-token loss (nats) on fixed windows of each validation set."""
    model.eval()
    out = {}
    for s in sets:
        arr = tokens(f"{s}_val")
        rng = np.random.default_rng(1234)
        tot = 0.0
        for _ in range(n_batches):
            xy = batch(arr, bs, ctx, rng).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xy[:, :-1])
            tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), xy[:, 1:].reshape(-1)).item()
        out[s] = tot / n_batches
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--mix", nargs="+", required=True, help="source=weight, e.g. lang=1 code=1")
    ap.add_argument("--tokens", type=float)
    ap.add_argument("--flops", type=float)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--micro", type=int, default=16, help="sequences per micro-batch (use 8 for size m)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--init", help="checkpoint to continue from")
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--own", choices=["lang", "facts", "math", "code"],
                    help="loss only on this type's tokens in labelled sources")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward pass")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    if args.init:
        model, _ck_init = M.load(args.init, device)
        model.train()
    else:
        model = M.make(args.size, meta["vocab"], args.ctx).to(device)
    if args.compile and hasattr(model, "aux_loss"):
        raise SystemExit("--compile is not used with BTX (the router loss is read from the modules)")
    fwd = torch.compile(model) if args.compile else model
    fpt = model.flops_per_token()
    budget = args.tokens if args.tokens else args.flops / fpt
    mix = {k: float(v) for k, v in (m.split("=") for m in args.mix)}
    z = sum(mix.values())
    mix = {k: v / z for k, v in mix.items()}
    sources = {k: tokens(f"{k}_train") for k in mix}
    TYPES = ["lang", "facts", "math", "code"]
    lab = {k: labels(f"{k}_train") for k in mix} if args.own else {}
    names, probs = list(mix), np.array([mix[k] for k in mix])

    seqs = args.batch_tokens // args.ctx
    accum = max(1, seqs // args.micro)
    steps = max(1, int(budget // (seqs * args.ctx)))
    lr = args.lr or (1e-3 if args.size in ("s", "xs") else 6e-4)
    if args.init:
        lr = args.lr or lr * 0.3
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    warm = max(1, int(args.warmup * steps))

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    print(f"{args.name}: {model.n_params() / 1e6:.1f}M params, {fpt / 1e6:.0f} MFLOP/token, "
          f"{steps} steps x {seqs * args.ctx} tokens = {steps * seqs * args.ctx / 1e6:.0f}M tokens, mix {mix}",
          flush=True)
    curve = []
    t0 = time.time()
    eval_every = max(1, steps // args.evals)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(accum):
            counts = rng.multinomial(args.micro, probs)
            xs, ys = [], []
            for n, c in zip(names, counts):
                if not c:
                    continue
                if lab.get(n) is not None:
                    x, lb = batch(sources[n], c, args.ctx, rng, lab[n])
                    y = x[:, 1:].clone()
                    y[lb[:, 1:] != TYPES.index(args.own)] = -100      # other types: context only
                else:
                    x = batch(sources[n], c, args.ctx, rng)
                    y = x[:, 1:]
                xs.append(x)
                ys.append(y)
            xy, tgt = torch.cat(xs).to(device), torch.cat(ys).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(xy[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                   ignore_index=-100)
            if hasattr(model, "aux_loss"):                    # BTX: load balancing of the routers
                loss = loss + model.aux_loss()
            loss = loss / accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % eval_every == 0 or step == steps - 1:
            ev = evaluate(model, args.ctx, device)
            tok_s = (step + 1) * seqs * args.ctx / (time.time() - t0)
            curve.append({"step": step + 1, "tokens": (step + 1) * seqs * args.ctx, "train": loss_acc, **ev})
            print(f"  step {step + 1}/{steps} train {loss_acc:.3f} | " +
                  " ".join(f"{k} {v:.3f}" for k, v in ev.items()) + f" | {tok_s / 1e3:.0f}k tok/s", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    trained = steps * seqs * args.ctx
    M.save(os.path.join(CKPT, f"{args.name}.pt"), model, mix=mix, tokens=trained, init=args.init,
           **({"experts": _ck_init.get("experts")} if args.init and "experts" in _ck_init else {}))
    elapsed = time.time() - t0
    save("track_c", f"train_{args.name}",
         {"name": args.name, "size": args.size, "params": model.n_params(), "flops_per_token": fpt,
          "tokens": trained, "flops": fpt * trained, "mix": mix, "lr": lr, "init": args.init, "own": args.own,
          "steps": steps, "batch_tokens": seqs * args.ctx,
          "data": os.path.basename(DATA), "compile": args.compile, "curve": curve, "final": curve[-1]},
         seed=args.seed, wall_time=elapsed)


if __name__ == "__main__":
    main()
