# Roadmap: experiments and the next manuscript draft

Companion to [research_plan.md](research_plan.md), which says *what* to test. This
file sets the order, the go/no-go checks, and how the manuscript is assembled. It is
sized for one RTX 3050 Ti (4 GB), with free cloud GPUs (Kaggle T4, about 30 h/week)
where noted, at about 12 weeks of part-time work.

## Decisions (agreed)

1. **Scope.** The RoAS paper (under review) owns the formalism, the null atom, the
   anti-atom propositions, and the CounterFact results. This paper cites RoAS for them and
   does not restate them. Its contribution: heterogeneous experts under one calibrated
   gate (per-type experts, compiled experts that call each other, tools in the same read).
2. **Venue.** TMLR. Move to Artificial Intelligence or JAIR only if track C is strong.

## Phase 0: harness (week 1)

- `experiments/<track>/run_*.py` writes `results/<track>/*.json`, each with metadata (git
  commit, seed, package versions, hardware, wall-clock time).
- `scripts/make_tables.py` turns results into LaTeX macros (`paper/results_macros.tex`).
  Numbers not produced yet stay `\pending{...}`; no number is typed by hand.
- Cache expert outputs once (per-token top-k log-probs, target log-prob, last hidden
  state), so gate training and most evaluation need one model in memory at a time.

## Phase 1: track A, compiled experts (weeks 1–2, CPU)

- Exhaustive checks on small operands, random checks on long ones, parser tests at
  increasing nesting depth.
- Length-generalisation curve: accuracy against digit count for compiled circuits,
  prompted Qwen2.5-0.5B/1.5B, a trained arithmetic LM (from Phase 2), and a Python tool.
- Cost table: weights, nonzeros, scratchpad tokens, time.
- Deliverable: Section 4 of the draft, essentially complete.

## Phase 2: track C, from scratch (weeks 2–5, local GPU) — the main test of H1

- Data: TinyStories (language), synthetic biographies with held-out paraphrases (facts),
  synthetic arithmetic and templated word problems (math), the-stack-smol Python (code).
- One shared BPE tokenizer; four ~20M experts with their own training regimes; monolithic
  baselines at matched compute (4× parameters; 1× parameters with 4× steps).
- Gate trained on mixed documents without routing labels; probability mixture vs logit
  mixture vs hard routing.
- Metrics: per-type loss, fact recall on paraphrases, arithmetic beyond training length,
  mixed-text perplexity, the cost of adding facts without forgetting.
- **Go/no-go:** experts + gate ≥ monolithic at equal compute → headline result.
  Otherwise reframe around modularity and cost and report the gap honestly.

## Phase 3: track B, pretrained experts with one tokenizer (weeks 4–8)

- Qwen2.5-0.5B (language), Qwen2.5-Coder-0.5B (code), Qwen2.5-Math-1.5B in 4-bit (math),
  the compiled circuits, corpus atoms. Gate training from cached outputs. Generation
  benchmarks (HumanEval and MBPP pass@1, GSM8K) locally if they fit, else on Kaggle.
- Build it up gradually: language + code → + math → + circuits → + facts; each addition is
  a table row and modularity evidence.
- Swap test: Coder-0.5B → 1.5B; Math-1.5B → compiled circuits. Report the change on the
  swapped type, on all other types, and the gate retraining steps needed.
- **Go/no-go:** language + code must beat the best single expert on mixed text before more
  experts are added.

## Phase 4: frozen backbone and tools (weeks 7–10)

- Track D: TOFU forget/retain with logit- and hidden-level anti-atoms (paraphrase leakage
  is the key number); MQuAKE for updates.
- Track D2: tools through the learned gate; API-Bank for tool choice; BFCL irrelevance for
  abstention; anti-atom vs allowlist revocation; tool → record → abstain fallback.
  AgentDojo only if a security claim is made.

## Phase 5: ablations and cost (weeks 9–11)

- Gate input, window, latching, consumption guard, C and temperature, null atom, soft vs
  hard mixing: mostly on track C, a subset confirmed on track B.
- Training FLOPs (experts + gate vs monolithic), the cost of adding one capability,
  inference latency per token.

## Writing (in parallel)

| Weeks | Writing |
|---|---|
| 1–2 | New outline and abstract skeleton; Related Work (BTX, Co-LLM, ToolkenGPT/ToolGen, Tracr) |
| 2–3 | Method, trimmed: read operator by citation to RoAS, the gate, latching, mixing rules, compiled experts, call blocks |
| 3–4 | Section 4 (compiled experts, parser) from Phase 1 |
| 6–9 | Sections 5–7 as each track's results arrive |
| 11–12 | Discussion, limitations, abstract last; internal review pass; figures |
| 12–13 | Polish, reproducibility appendix, submit |

Draft structure:

1. Introduction: data types need different training; hypotheses H1–H3 as testable claims.
2. Related work.
3. Method: read and gate (citing RoAS), mixing rules, latching and call blocks; the veto
   theorem as a one-line remark.
4. Compiled experts: the toolkit, + − × ÷, the parser, atoms calling atoms (track A).
5. Training from scratch (track C, H1).
6. Pretrained experts (track B, H1–H2).
7. Frozen backbone and tools (tracks D and D2, H3).
8. Ablations and cost.
9. Discussion and limitations.

Cut from the current draft: measure-theory framing beyond what RoAS set up, "Production vN"
wording, claims such as "eliminates hallucination" or "zero leakage" without a benchmark,
and the mock barometer experiment.

## Status

| Item | Status |
|---|---|
| Phase 0 harness | done: `experiments/common.py` (results JSON with metadata), `scripts/make_tables.py` (macros and tables), `paper/main_v2.tex` builds cleanly |
| Track A correctness | done: 4,044,460 cases (exhaustive below 10², 10³; long; decimals; parser), 0 errors |
| Track A length generalisation | done: compiled and Python 100% at 1–32 digits; Qwen2.5-0.5B/1.5B reach 0% by 16–32 digits (×, ÷ by 4 digits) |
| Track A cost | done: sizes (11k–310k weights, 0.8–5.5% nonzero) and token/time scaling |
| Track A, trained arithmetic expert row | waits for Track C |
| Track C harness | done: `experiments/track_c/` (data, model, train, gate, evaluate, `run_track_c.py`, `run_add_facts.py`); 8k-token BPE with single-digit tokens; 150M story, 116M code, 101M math, 2.6M biography, 3M mixed tokens |
| Track C pilot (T = 10M per expert) | done, see below; mono_s wins at this budget |
| Track C main (T = 100M per expert) | done, see main notes: experts win on their own types and fact recall, gated system loses on mixed text |
| Track C add-facts test | done for both variants, including the fair `replay_all` baseline for the monolithic models |
| Tracks B, D, D2, ablations | not started |
| Draft | outline, abstract skeleton, Related Work and Section 4 written; Sections 5–9 are TODO skeletons |

Reproduce Track A: `python experiments/track_a/run_correctness.py` (plus `--suite exhaustive3`, roughly 2.5 h on 15 workers; the recorded time includes a 2 h hibernation),
`python experiments/track_a/run_length_generalisation.py --systems compiled python qwen0.5b qwen1.5b`,
`python experiments/track_a/run_cost.py`, then `python scripts/make_tables.py` and `latexmk -pdf main_v2.tex` in `paper/`.

### Track C pilot notes (T = 10M tokens per expert, 5.9e15 FLOPs per system)

- **Batch size matters more than anything else at this budget.** With 32k-token batches an
  expert gets 305 steps and its loss is far behind the baseline (lang 3.42 vs 2.46). With
  8k-token batches (1,220 steps) the gap mostly closes (lang 2.48 vs 2.29). All runs now use
  8k batches for every system; the 32k pilot (`eval_pilot.json`) and the sweep
  (`train_sweep_*.json`) are kept as the record of that choice.
- **Independent experts lose to mono_s on every type** (lang 2.48/2.29, facts 0.69/0.61,
  math 0.91/0.79, code 3.89/2.75; mixed text 3.63/2.38 for the prob gate). Code gains most
  from the shared model, which suggests transfer from language.
- **Seeding (BTM style, half the budget on a shared seed) helps every type:** lang 2.32,
  facts 0.68, math 0.84, code 3.29; gated mixed text 3.18; 1-digit addition 36% vs 14%. It
  still trails mono_s.
- **The gate routes well** (argmax on the right expert for 92-99% of story and code tokens,
  82-90% of facts, 66-74% of math), but on mixed text every expert reads context of types it
  never saw, and mixing costs 0.18 nats on facts against the facts expert alone.
- mono_m (70M parameters, 13M tokens) is badly undertrained at this budget.
- Fact recall is at chance for every system at this scale (about 4 passes over the
  biographies); arithmetic beyond 3 digits is 0 everywhere.
- Ideas if the main run also loses: expose experts to a small share of foreign context,
  feed the gate expert confidence (entropy, max log-prob), seed fraction ablation.

### Track C main notes (T = 100M tokens per expert, 5.0e16 FLOPs per system)

- **Experts beat both monolithic models on their own data** (seeded experts: lang 1.61 vs
  mono_s 1.74, math 0.70 vs 0.71, code 1.99 vs 2.15) and on fact recall (training templates
  1.00 vs 0.70; held-out templates 0.32 vs 0.06).
- **The gated system loses on mixed text:** 2.16 (seeded, prob) and 2.51 (independent) vs
  mono_s 1.99. Seeding closes about two thirds of the gap and improves routing (facts 0.89,
  math 0.92 argmax accuracy, from 0.72 and 0.75). Go/no-go: split result, so the paper
  leads with modularity and update cost and reports the mixed-text gap.
- Arithmetic is length-bound for every system (near 0 from 6 digits; the independent math
  expert reaches 0.36 at 6-digit +). The facts expert has a worse facts loss than mono_s
  (0.80 vs 0.44) despite far better recall: check before writing.
- **Adding 500 people** (25M new-fact tokens; replay doubles it):
  - Expert-only update with replay: lang, math, code unchanged; old recall 0.96, new 0.99.
  - mono_s fine-tuned on new facts only, or with facts-only replay: lang/math/code losses
    rise to 7-12 nats. This baseline is too weak and is reported only as a lower bound.
  - Fair baseline, `replay_all` (half new facts, half all four sources, same FLOPs as the
    expert update for mono_s): lang 1.74 → 1.82, code 2.15 → 2.31, mixed 1.99 → 2.22, old
    recall 0.70 → 0.60, new 0.98. mono_m does no better at 4x the cost.
  - After one fair update the seeded gated system (mixed 2.18) and mono_s (2.22) are level
    on mixed text; the gated system loses nothing elsewhere, mono_s loses on every type.
- Hibernation: timings of `train_main_seeded_lang.json` and
  `eval_main_seeded_add_facts.json` are marked invalid.
