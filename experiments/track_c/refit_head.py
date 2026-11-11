"""Track C: refit the circuit's head of trained two-level gates to convergence (follow-up).

    python experiments/track_c/refit_head.py [--gates ...] [--only fit eval]

The atom head (3 parameters: a linear map of the circuit's two confidence features in nats)
was trained jointly with the gate for a fixed number of steps, and in some gate seeds it had
not converged (weight on the active circuit 0.51 and 0.78 against 0.94-0.99 for other seeds).
Here every other part of each gate is frozen, and only the head is refitted, full batch, to
convergence on mixed_train windows, as in Track B step 3 (circuit_gate.py). The loss is the
mixture's log-likelihood without vetoes (vetoes only remove wrong tokens). Saved as
checkpoints/track_c/<gate>_conv.pt; then evaluated on held-out arithmetic (eval seed 1000,
100 items per cell), losses and routing (tag conv_heldout).
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import save  # noqa: E402
from experiments.track_c.gate import confidence, expert_outputs, load_experts, load_gate  # noqa: E402
from experiments.track_c.train import CKPT, batch, tokens  # noqa: E402

GATES = ["main_ctxv20_gate_atom", "main_ctxv20_gate_atom_s1", "main_ctxv20_gate_atom_s2",
         "main_seeded_gate_atom", "main_seeded_gate_atom_s1", "main_seeded_gate_atom_s2"]
HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def features(name, n_batches, micro=8, ctx=512):
    """Per token: the circuit's head input, log p(target) under the circuit and under the LM block."""
    gate, ck = load_gate(name)
    experts = load_experts(ck["experts"])
    assert ck["experts"][-1] == "circuit" and ck.get("atom_nats")
    arr, rng = tokens("mixed_train"), np.random.default_rng(0)
    own, lpc, lplm, act = [], [], [], []
    circ = experts[-1]
    for _ in range(n_batches):
        xy = batch(arr, micro, ctx, rng).cuda()
        x, y = xy[:, :-1], xy[:, 1:]
        logps, hs = expert_outputs(experts, x)
        ll = gate(hs, logps)                                        # (B, T, E), circuit last
        tl = logps.gather(-1, y[:, :, None, None].expand(-1, -1, logps.shape[2], 1)).squeeze(-1)  # (B, T, E)
        split = ll[..., :-1] - torch.logsumexp(ll[..., :-1], -1, keepdim=True)
        lplm.append(torch.logsumexp(split + tl[..., :-1], -1).flatten())
        lpc.append(tl[..., -1].flatten())
        cf = confidence(logps)
        E = logps.shape[2]
        own.append((torch.stack([cf[..., E - 1], cf[..., 2 * E - 1]], -1) * math.log(logps.shape[-1])).reshape(-1, 2))
        act.append(torch.from_numpy(np.stack([circ.targets(r.tolist()) for r in x.cpu().numpy()]) >= 0).cuda().flatten())
    return gate, ck, torch.cat(own), torch.cat(lpc), torch.cat(lplm), torch.cat(act)


def fit(head, own, lpc, lplm, steps, lr=3e-2):
    w = torch.cat([head.weight.detach().flatten(), head.bias.detach()]).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        a = own @ w[:2] + w[2]
        loss = -torch.logaddexp(torch.nn.functional.logsigmoid(a) + lpc,
                                torch.nn.functional.logsigmoid(-a) + lplm).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="+", default=GATES)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--batches", type=int, default=120, help="8 x 512-token windows each (about 0.5M tokens)")
    ap.add_argument("--steps", type=int, default=10000)
    a = ap.parse_args()
    steps = a.only or ["fit", "eval"]
    out = {}
    if "fit" in steps:
        for g in a.gates:
            gate, ck, own, lpc, lplm, act = features(g, a.batches)
            w0 = torch.cat([gate.atom.weight.flatten(), gate.atom.bias]).detach()
            lam = lambda w: torch.sigmoid(own @ w[:2] + w[2])  # noqa: E731
            before = (lam(w0)[act].mean().item(), lam(w0)[~act].mean().item())
            w, loss = fit(gate.atom, own, lpc, lplm, a.steps)
            w_half, _ = fit(gate.atom, own, lpc, lplm, a.steps // 2)
            after = (lam(w)[act].mean().item(), lam(w)[~act].mean().item())
            moved = (lam(w) - lam(w_half)).abs().max().item()
            sd = dict(ck["gate"])
            sd["atom.weight"], sd["atom.bias"] = w[:2].view(1, 2).cpu(), w[2:].cpu()
            torch.save({**ck, "gate": sd, "refit_from": g}, os.path.join(CKPT, f"{g}_conv.pt"))
            out[g] = {"head_before": w0.tolist(), "head_after": w.tolist(), "loss": loss,
                      "lambda_active_before": before[0], "lambda_inactive_before": before[1],
                      "lambda_active_after": after[0], "lambda_inactive_after": after[1],
                      "max_change_last_half": moved, "active_positions": int(act.sum()),
                      "tokens": int(act.numel())}
            print(g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[g].items()}, flush=True)
            del gate, own, lpc, lplm, act
            torch.cuda.empty_cache()
        save("track_c", "refit_head", {"gates": out, "steps": a.steps, "batches": a.batches})
    if "eval" in steps:
        subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--tag", "conv_heldout",
                        "--eval-seed", "1000", "--per-cell", "100", "--skip", "facts",
                        "--systems", *[f"gate:{g}_conv:prob" for g in a.gates]], check=True)


if __name__ == "__main__":
    main()
