"""Track A: exact-match accuracy against operand length.

    python experiments/track_a/run_length_generalisation.py --systems compiled python
    python experiments/track_a/run_length_generalisation.py --systems qwen0.5b qwen1.5b

Systems
  compiled   the hand-weighted circuits (circuits/)
  python     Python integers (a tool call; the exactness reference)
  qwen0.5b   Qwen2.5-0.5B-Instruct, greedy, answer-only prompt (local HF cache)
  qwen1.5b   Qwen2.5-1.5B-Instruct, same
  (a trained arithmetic LM from track C is added later)

Tasks are integer +, -, *, and floor division; a and b have n digits (for division b has
max(1, n // 2)). The same items are used for every system (fixed seed).
Each run merges its systems into results/track_a/length_generalisation.json.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save, timer, pool_map, ROOT  # noqa: E402

OPS = ["add", "sub", "mul", "div"]
SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
DIGITS = [1, 2, 4, 8, 16, 32]
MODELS = {"qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}


def items(per_cell, seed):
    rng = random.Random(seed)
    out = []
    for op in OPS:
        for n in DIGITS:
            for _ in range(per_cell):
                a = rng.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1)
                m = max(1, n // 2) if op == "div" else n
                b = rng.randint(10 ** (m - 1) if m > 1 else 1, 10 ** m - 1)
                out.append((op, n, a, b))
    return out


def truth(op, a, b):
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a // b}[op]


def compiled_one(item):
    from experiments.track_a.run_correctness import _circ
    from circuits import addsub, mul, div
    op, n, a, b = item
    a, b = str(a), str(b)
    if op in ("add", "sub"):
        w = max(len(a), len(b))
        g = _circ(op).run(addsub.prompt(a.zfill(w), b.zfill(w), op))
        body = "".join(g[1:])
        if op == "add":
            return int(("1" if g[0] == "1" else "") + body)
        return -int(body) if g[0] == "NEG" else int(body)
    if op == "mul":
        return int(mul.answer(_circ("mul").run(mul.prompt(a, b))))
    return int(div.answer(_circ("div").run(div.prompt(a, b))))


def prompt_text(op, a, b):
    if op == "div":
        return f"Compute the integer part of {a} / {b} (floor division). Reply with only the final integer."
    return f"Compute {a} {SYM[op]} {b}. Reply with only the final integer."


def parse_int(text):
    t = text.replace(",", "").replace(" ", "")
    m = re.findall(r"-?\d+", t)
    return int(m[-1]) if m else None


def run_llm(model_key, its, batch=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.float16).to("cuda")
    model.eval()
    preds = []
    for i in range(0, len(its), batch):
        chunk = its[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt_text(op, a, b)}],
                                         tokenize=False, add_generation_prompt=True) for op, n, a, b in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        max_new = 2 * max(n for _, n, _, _ in chunk) + 16
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for j, seq in enumerate(out):
            preds.append(parse_int(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        print(f"{model_key}: {min(i + batch, len(its))}/{len(its)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["compiled", "python"])
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    its = items(args.per_cell, args.seed)

    path = os.path.join(ROOT, "results", "track_a", "length_generalisation.json")
    acc, systems = {}, []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)["data"]
        acc, systems = old["accuracy"], old["systems"]

    with timer() as t:
        for s in args.systems:
            if s == "compiled":
                preds = pool_map(compiled_one, its, chunksize=1)
            elif s == "python":
                preds = [truth(op, a, b) for op, n, a, b in its]
            else:
                preds = run_llm(s, its)
            cell = {}
            for (op, n, a, b), p in zip(its, preds):
                cell.setdefault(op, {}).setdefault(str(n), []).append(p == truth(op, a, b))
            acc[s] = {op: {d: sum(v) / len(v) for d, v in ds.items()} for op, ds in cell.items()}
            if s not in systems:
                systems.append(s)
            print(s, json.dumps(acc[s]))

    order = [s for s in ["compiled", "python", "qwen0.5b", "qwen1.5b"] if s in systems] + \
            [s for s in systems if s not in ("compiled", "python", "qwen0.5b", "qwen1.5b")]
    words = {"compiled": "Compiled", "python": "Python", "qwen0.5b": "QwenHalfB", "qwen1.5b": "QwenOneHalfB"}
    macros = {}
    for s in order:
        for op in OPS:
            v = acc[s].get(op, {}).get("32")
            if v is not None and s in words:
                macros[f"Len{words[s]}{op.capitalize()}ThirtyTwo"] = f"{100 * v:.0f}"
    save("track_a", "length_generalisation",
         {"systems": order, "ops": OPS, "digits": DIGITS, "per_cell": args.per_cell, "accuracy": acc},
         macros=macros, seed=args.seed, wall_time=t["elapsed"])


if __name__ == "__main__":
    main()
