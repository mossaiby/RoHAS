"""Track B, step 4: replace the prose expert under the frozen gate (pre-registered in docs/roadmap.md).

    python experiments/track_b/run_swap.py [--only cache gates]

  cache  Qwen2.5-0.5B-Instruct (base_inst) on FILES4; base and Coder again on the files whose
         hidden states were deleted to save disk (mixed_sub_train and two mixed validation sets)
  gates  for seeds 0-2:
           swap_orig_s<k>    frozen b_gate_s<k> over base + Coder (before the swap)
           swap_frozen_s<k>  frozen b_gate_s<k> over base_inst + Coder (the swap)
           swap_ref_s<k>     a gate retrained on mixed_sub_train over base_inst + Coder
           swap_check_s<k>   a gate retrained on mixed_sub_train over base + Coder
         retrained gates run 9 epochs on mixed_sub_train: as many updates as the step-1 gates
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    steps = a.only or ["cache", "gates"]
    cache, gate = os.path.join(HERE, "cache.py"), os.path.join(HERE, "gate.py")
    if "cache" in steps:
        run(cache, "--expert", "base_inst", "--step", "4")
        for e in ("base", "coder"):
            run(cache, "--expert", e, "--force", "--files", "mixed_sub_train", "mixed_blank_val", "mixed_inline_val")
    if "gates" in steps:
        for k in range(3):
            run(gate, "--name", f"swap_orig_s{k}", "--frozen", f"b_gate_s{k}", "--experts", "base", "coder",
                "--step", "4", "--seed", str(k))
            run(gate, "--name", f"swap_frozen_s{k}", "--frozen", f"b_gate_s{k}", "--experts", "base_inst", "coder",
                "--step", "4", "--seed", str(k))
            run(gate, "--name", f"swap_ref_s{k}", "--experts", "base_inst", "coder", "--step", "4",
                "--seed", str(k), "--epochs", "9")
            run(gate, "--name", f"swap_check_s{k}", "--experts", "base", "coder", "--step", "4",
                "--seed", str(k), "--epochs", "9")


if __name__ == "__main__":
    main()
