# Reproducing the paper

This file covers `paper/main_v2.tex` ("Different Data, Different Experts: Label-Free
Token-Level Routing over Trained, Pretrained, and Compiled Experts"). The rest of `README.md`
describes an earlier prototype.

Every number in the paper comes from `results/<track>/*.json`. `scripts/make_tables.py` turns
those files into `paper/results_macros.tex` (one `\res...` macro per number) and
`paper/tables/*.tex`; the manuscript contains no hand-typed results. The order in which the
experiments were run, their pre-registered criteria, and every deviation from them are in
`docs/roadmap.md`.

```bash
python scripts/make_tables.py
cd paper && latexmk -pdf main_v2.tex
```

## Environment

Python 3.11, PyTorch 2.6.0 (CUDA 12.4), transformers 5.x, numpy 2.x, plus `tokenizers`,
`datasets`, `pyarrow` and `huggingface_hub` for the data. Every result file records the exact
versions, the git commit and the hardware in its `meta` block.

All runs used one laptop: an RTX 3050 Ti with 4 GB, 16 GB RAM. Some runs were paused by
hibernation; their wall-clock times are marked invalid (`null`) in the result files.
Downloads from the Hugging Face Hub can hang with the Xet transfer backend; set
`HF_HUB_DISABLE_XET=1`, and `HF_HUB_OFFLINE=1` once the models are cached.

## Section 4 and Appendix A: compiled experts (Track A)

The experts are hand-weighted transformers in `circuits/`, with no training.

| Paper | Script | Result file |
| --- | --- | --- |
| Table 1, correctness | `experiments/track_a/run_correctness.py` (`--suite exhaustive3` takes about 2.5 h on 15 workers) | `results/track_a/correctness.json` |
| Table 2, length generalisation | `experiments/track_a/run_length_generalisation.py --systems compiled python`, then `--systems qwen0.5b qwen1.5b` | `results/track_a/length_generalisation.json` |
| Tables 13-14, size and cost | `experiments/track_a/run_cost.py` (on an idle machine) | `results/track_a/cost.json` |

## Section 5: experts trained from scratch (Track C)

Data (written to `data/track_c/`, not in git): `experiments/track_c/data.py`, then
`data_context.py` (context exposure; `--varied` for varied formats) and `data_mixed_ood.py`
(the held-out mixed formats).

| Paper | Script | Result files |
| --- | --- | --- |
| Experts, baselines, gates (Tables 3-4) | `run_track_c.py --scale main`; seeded experts and context exposure: `run_context.py --frac 0.2 [--varied]` | `train_*.json`, `eval_main*.json`, `diag_mixed_*.json` |
| Second seed of the baselines | `run_baseline_seed.py --seed 1` (about a day) | `*_main_r1*.json` |
| Expert seeds | `run_context.py` with `--seed` | `*_ctxv20_e1*`, `*_ctxv20_e2*` |
| Circuit under the gate (Table 5) | `run_circuit.py --prefix main [--variant seeded]`, `run_robustness.py`, `run_ctxv_followup.py` | `eval_*_circuit*.json`, `eval_*_heldout.json` |
| Circuit head refitted to convergence | `refit_head.py` | `refit_head.json`, `eval_conv_heldout.json` |
| Calculator baseline (Table 6) | `tool_baseline.py`, then `--end` and `--end --varied` for the two variants | `tool_heldout.json`, `tool_context.json` |
| False triggers and wrong arithmetic | `false_triggers.py` | `false_triggers.json` |
| Adding facts (Table 7) | `run_add_facts.py`, `run_ctxv_followup.py --only add eval` | `eval_*add_facts*.json` |
| Ablations (Table 12) | the gate variants in `run_circuit.py` | `eval_main_circuit*.json` |

All scripts are in `experiments/track_c/`; models train with `train.py`, gates with `gate.py`,
and are evaluated with `evaluate.py` and `diag_mixed.py`.

## Section 6: pretrained experts (Track B)

Experts: Qwen2.5-0.5B, Qwen2.5-Coder-0.5B, Qwen2.5-Math-1.5B, all frozen. Each model runs
once over all data and its outputs are cached (`data/track_b/cache/`, not in git, about 28 GB with all steps);
the gates train on the cache.

| Paper | Script | Result files |
| --- | --- | --- |
| Data | `experiments/track_b/data.py` (`--step 2`, `--step 3` for the math steps) | |
| Expert caches | `cache.py --expert base`, `--expert coder`, `--expert math --step 2 --bs 2` | |
| Table 8, prose and code | `gate.py --name b_gate_s<seed> --experts base coder --seed <seed>` | `results/track_b/b_gate_s*.json` |
| Table 9, adding a math expert | `gate.py --experts base coder math --step 2` (contaminated run) and `--step 3` (clean run); `segment_losses.py` | `b3_gate_s*.json`, `b3c_gate_s*.json`, `segment_losses.json` |
| Table 10, circuit under the pretrained gate | `data_circuit.py`, then `circuit_gate.py` (live decoding, about 1.5 h) and `veto_scan.py` | `circuit_gate.json`, `veto_scan.json` |
| Table 11, replacing an expert | `run_swap.py` (caches Qwen2.5-0.5B-Instruct, about 8 GB of hidden states in all) | `swap_{orig,frozen,ref,check}_s*.json` |
