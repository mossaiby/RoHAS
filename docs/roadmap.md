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
| Track C + compiled circuit as a 5th expert | done (seeded experts), see circuit notes: exact answers at every length to 32 digits, other types unchanged |
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

### Track C circuit notes (compiled circuits as a fifth expert, seeded experts, 6M gate tokens)

`run_circuit.py`; results in `eval_main_circuit.json` (ablations) and `eval_main_circuit_atom.json`
(final). No routing labels; the circuit sees token ids only and abstains with the null atom
elsewhere. Arithmetic is scored two ways: `arith` (the generated line is exactly "=" + answer)
and `arith_number` (the answer is exact and not followed by another digit).

- **Final system (`gate_atom`, prob):** answer number exact in every cell, + and − at 1-32
  digits, * at 1-8 digits (50 items each); without the circuit, the same experts reach 0 by 8
  digits (+) and 4 digits (*). Losses unchanged (lang 1.685, facts 0.879, math 0.713, code 1.994,
  mixed 2.140 vs 2.162 for the confidence-only control), fact recall unchanged, word problems
  0.995 vs 0.905. Circuit weight 0.993 where it is active, 0.000 elsewhere.
- **Line match** fails from 16 digits (+ 0.12, 0.06, 0.00 at 16/24/32): the answer is right and
  complete, then the LM experts write "=" instead of a newline on lines longer than any they saw.
  The circuit leaves the next non-digit to the other experts by design.
- **What each piece does** (ablations, all prob unless noted):
  - joint gate on hidden states + confidence (`gate_circ`): trust in the circuit fades after
    about 6 digits (+ 0.16 at 12).
  - separate confidence head on all experts' confidence (`gate_split`): exact to 8 digits, fails
    from 12 (the LM experts' confidence rises, wrongly, on long numbers).
  - two-level gate, atom weighted by its own confidence (`gate_atom_mlp`, `gate_atom_lin`): with
    the confidence divided by log V, the active and abstaining states differ by about 1 per
    feature, and the head settles at weight 0.40 where the circuit is active, although the data
    optimum is near 1 (checked: the circuit is right at all 6,940 active positions sampled).
    In nats the weight is 0.993.
  - vetoes off (`gate_atom:prob_noveto`): + 0.20 at 12 digits, 0.04 at 32; the LMs append digits
    to a complete answer, and pick the wrong sign token for subtraction.
  - hard routing (`gate_atom:hard`) is also exact on the number, but costs 0.13 nats on mixed
    text and 0.8 on facts.
- The circuit expert was checked on 2M tokens each of mixed, math, code and story text: no wrong
  active predictions and no vetoed true token (after the window-start fix: an equation whose
  first operand starts at the first position of the window is ignored).
- Not counted in FLOPs: the circuits run once per equation on the CPU (`circuit_calls` in the
  train json). Arithmetic in all tag files was re-scored with both metrics; original wall
  times are kept, the re-score time is `wall_time_rescore_arith_s`.

### Track C circuit robustness (`run_robustness.py`)

The final gate design was chosen on the development items (eval seed 0), so it is re-tested on
fresh items (eval seed 1000, 100 per cell; `eval_main_heldout*.json`), with two more gate seeds
and on the independent experts (`eval_main_circuit_seeds.json`). Experts are fixed across gate
seeds, so the spread is gate variance only.

- **Answer number exact in every held-out cell** (+ and − at 1-32 digits, * at 1-8) for all three
  seeds of the seeded atom gate and for the atom gate on the independent experts. Without the
  circuit the same experts reach 0 by 8 digits (+, −) and 3 digits (*); mono_s 0.01 at 6-digit +.
- Circuit weight where active: 0.993, 0.785, 0.981 (seeds 0-2), 0.996 (independent); about 0 elsewhere.
  Seed 1's lower weight still gives exact answers (prob mixing: weight > 0.5 decides the argmax).
- Losses, seeded, control vs atom gate (3 seeds each): mixed 2.162/2.162/2.162 vs 2.140/2.141/2.141;
  facts 0.882/0.895/0.892 vs 0.879/0.903/0.902; lang and code within 0.002. Independent experts:
  mixed 2.504 vs 2.438, facts 0.957 vs 0.978 (a small facts cost to watch).
- Word problems (held-out): 0.91 for the seeded control, 1.00 for every atom gate.
- Vetoes off: + 0.29 at 12 digits, 0.02 at 32 (seeded); independent 0.26 at 8 digits.
- Line match (answer then newline) from 16 digits is erratic across seeds (+ 0.18-0.46 at 16,
  ≤ 0.08 at 24-32): the LM experts, not the circuit, end the line.
- Eval wall times vary 4x between identical workloads in this run (other load on the machine);
  gate training times are consistent (330 s control, 795 s atom gate).


### Track C mixed-text gap (`diag_mixed.py`, `data_context.py`, `run_context.py`)

Where the seeded gate loses to mono_s on mixed text (2.162 vs 1.990; `diag_mixed_main.json`):
- The per-token oracle over the four seeded experts (best expert per token, with hindsight) is
  2.027, still above mono_s: better routing alone cannot close the gap.
- The experts are bad inside foreign context, both in the first 16 tokens of a segment and
  later (facts late: seeded gate 1.46, mono_s 0.62). Each expert only ever saw windows of its
  own type, so it never learned to read past a change of type.

Context exposure, f = 0.2 (`main_ctx20_*`, same seed model, recipe and FLOPs as the seeded
experts). 20% of each branch's tokens are mixed documents where each segment is the expert's
own type with probability 1/2. The loss is only on own-type tokens, so an expert learns to
read foreign context but is never trained to predict it. Foreign segments come from material
no expert stores: fictitious people (no stored name), TinyStories shard 3, and fresh math.
Code reuses code_train files, so no deletion claim is made for code.
- **Mixed: 1.828 (conf gate), 1.824 (atom gate), vs 2.162 seeded and 1.990 mono_s.** The
  exposed system now beats the compute-matched monolithic model on mixed text. Label oracle
  1.786, token oracle 1.755.
- Early / late (first 16 tokens / rest), lang facts math code:
  - Early: 2.51 1.10 1.55 3.30, vs seeded 3.55 2.35 2.21 4.05 and mono_s 2.77 1.34 1.62 3.56.
  - Late: 2.06 0.44 0.52 2.14, vs seeded 2.57 1.46 0.66 2.22 and mono_s 2.28 0.62 0.54 2.32.
- Pure-domain losses, expert alone (seeded → exposed):
  - lang 1.612 → 1.633
  - facts 0.614 → 0.458
  - math 0.699 → 0.700
  - code 1.987 → 1.995
- Pure-domain losses, conf gate: lang 1.685 → 1.667, facts 0.882 → 0.498, math 0.734 → 0.741,
  code 1.994 → 1.998.
- The facts gain is a surprise, given fewer supervised facts tokens. A likely cause is that
  short segments of 1-3 attributes in random order make the expert robust to the new orderings
  in facts_val. This is not verified.
- Fact recall: train templates 1.00. Held-out templates: expert 0.323 → 0.269, gate
  0.246 → 0.227 (a small cost).
- Purity holds. The lang, math and code experts recall stored facts at chance (0.013-0.023,
  as before).
- Circuit (atom gate): answer number exact in every development cell, word problems 1.00,
  circuit weight 0.983 where active.
- **Caveat: the exposure documents come from the same generator as mixed_val** (same segment
  lengths and "\n\n" separators). Part of the gain may be format matching. mono_s and the gate
  saw that format too (mixed_train), but the experts now train on it directly. A mixed
  evaluation in a different format is needed before this goes in the paper.

Format check (`data_mixed_ood.py`; `diag_mixed_long_main.json`, `diag_mixed_inline_main.json`).
Two validation sets keep mixed_val's sources but change the format, which no model trained on:
- **long**: 2-3 long segments joined by one newline.
- **inline**: 4-8 short segments joined by a space or a newline, so the type changes
  mid-paragraph.

| system | mixed_val | long | inline |
|---|---|---|---|
| mono_s | 1.990 | 1.858 | 2.680 |
| mono_m | 2.052 | 1.916 | 2.655 |
| seeded gate (conf) | 2.162 | **1.788** | **2.592** |
| exposed gate f=0.2 (conf) | **1.828** | **1.760** | 2.631 |
| independent gate (conf) | 2.504 | 2.066 | 3.080 |

- On formats nobody trained on, the seeded gate already beats mono_s **without exposure**: by
  0.07 on long and 0.09 on inline. mono_s's lead on mixed_val is mostly familiarity with that
  format, since mixed_train is 1.5% of its data. The seeded gate also saw mixed_train, but only
  the gate learned from it; the experts never did.
- **Exposure overfits the format.** It helps a little on long (−0.028) and hurts on inline
  (+0.039). The worst cell is inline math early in a segment: label oracle 4.71 exposed vs 2.97
  seeded. The exposed math expert expects foreign text to end with a blank line.
- What exposure does help on inline is late in a segment (lang 2.27 vs 2.56, code 2.43 vs 2.55).
  The early-token losses got worse.
- The independent experts lose to mono_s everywhere. Their facts expert cannot read facts
  inside other text (facts 3.6-4.3), so seeding is what makes mixed text work.
- For the paper: report mixed loss on formats no expert or baseline was trained on. Exposure,
  if kept, needs varied formats and a further held-out format family to test on.

Varied-format exposure. Pre-registered 2026-11-10, before training.

Setup:
- Same recipe as main_ctx20 (f = 0.2, own-type loss, same pools), prefix `main_ctxv20`.
- Each exposure document's format is drawn from `data_mixed_ood.VARIED`: blank-line joins with
  short segments (build_mixed), single-newline joins with long segments (= the long eval
  format), and triple-newline joins with medium segments.
- Every join is a newline. That keeps two families held out from all training: inline, where
  half the joins are spaces, and marked (`mixed_marked_val`: tab, " | " or " ; " joins).
- The gate is still trained on mixed_train (build_mixed format), as for every other gate.

Criteria (gate with confidence features, prob mixing):
- Held-out families (inline, marked): main_ctxv20 at most the seeded gate's loss.
- Training families (mixed_val, long): below the seeded gate's loss.
- Purity: stored-fact recall of the lang, math and code experts stays at chance.
- Own-type token share of the ctxv files: lang 0.316, facts 0.193, math 0.430, code 0.809
  (ctx: 0.338, 0.228, 0.441, 0.789).

Varied-format exposure results (`main_ctxv20`; `eval_main_ctxv20.json`, `diag_mixed_*main_ctxv20.json`,
`diag_mixed_marked_main.json`). Mixed loss, gates with confidence features, prob mixing:

| system | mixed_val (A) | long (A for ctxv) | inline (held out) | marked (held out) |
|---|---|---|---|---|
| mono_s | 1.990 | 1.858 | 2.680 | 2.258 |
| mono_m | 2.052 | 1.916 | 2.655 | 2.285 |
| seeded gate | 2.162 | 1.788 | 2.592 | 2.118 |
| exposed, one format (ctx20) | **1.828** | 1.760 | 2.631 | 2.121 |
| **exposed, varied (ctxv20)** | 1.920 | **1.657** | **2.359** | **2.025** |
| independent gate | 2.504 | 2.066 | 3.080 | 2.470 |

- **All pre-registered criteria are met:**
  - Held out: inline −0.233 and marked −0.093 against the seeded gate.
  - Training formats: mixed_val −0.242 and long −0.131.
  - Purity: the lang, math and code experts recall stored facts at 0.019, 0.018 and 0.017 (chance).
- ctxv20 beats mono_s and mono_m on all four families. ctx20 (one format) does no better than
  the seeded gate on the held-out families, so the gain needs varied formats.
- On mixed_val, ctxv20 is behind ctx20 (1.920 vs 1.828). That is the cost of not matching
  that one format.
- Pure-domain losses of the ctxv20 conf gate (seeded): lang 1.696 (1.685), facts 0.534
  (0.882), math 0.724 (0.734), code 2.004 (1.994).
- Fact recall: training templates 0.998. Held-out templates: expert 0.337 (seeded 0.323),
  gate 0.270 (0.246).
- Atom gate: answer number exact in every development cell, word problems 1.00. Circuit weight
  0.973 where active and 0.0024 where inactive (seeded 1.7e-6). Losses are unaffected.
- Limits:
  - One run: one exposure draw and one gate seed.
  - The held-out families are only two, and I designed them.
  - The gate itself only ever trains on the build_mixed format.

Varied exposure, follow-up (`run_ctxv_followup.py`; `eval_main_ctxv20_{seeds,add_facts,heldout}.json`,
`diag_mixed_*_main_ctxv20_followup.json`, `diag_mixed_*_main_add_replay_all.json`).
- **Gate seeds**, experts fixed, conf gate, seeds 0/1/2:
  - mixed_val 1.920/1.919/1.920
  - long 1.657/1.658/1.659
  - inline 2.359/2.360/2.361
  - marked 2.025/2.027/2.029

  The gate adds no meaningful spread. Expert seeds are still untested.
- **Atom gate seeds:** answer number exact in every development and held-out cell for all
  three seeds; word problems 1.00. The circuit weight where active is 0.973/0.512/0.940. Seed 1
  sits just above the 0.5 that decides the argmax, so it is exact but fragile. The
  confidence-only gate gets 0 from 7 digits.
- **Adding 500 people** (seeded recipe, no exposure in the update, gate weights unchanged).
  With replay: old-fact recall 0.994 (training templates) and 0.269 (held-out templates); new
  people 0.993 and 0.191. The seeded updated expert gets 0.988/0.249 old and 0.986/0.208 new.
  lang, math and code are unchanged, since those experts are not touched.
- Mixed loss after the update with replay:

  | system | mixed_val | long | inline | marked |
  |---|---|---|---|---|
  | ctxv20 conf gate, before | 1.920 | 1.657 | 2.359 | 2.025 |
  | ctxv20 conf gate, after | 1.960 | 1.672 | 2.422 | 2.042 |
  | seeded conf gate, after | 2.177 | 1.796 | 2.615 | 2.130 |
  | mono_s, fair update (replay_all) | 2.223 | 2.009 | 2.855 | 2.430 |
  | mono_m, fair update (replay_all) | 2.234 | 2.014 | 2.773 | 2.384 |

  - After one update the varied-exposure system beats every updated baseline on every format,
    by 0.26-0.43 against mono_s.
  - The update gives back 0.015-0.063 of the gain, all of it in facts segments (mixed_val
    facts late 0.59 → 1.04). An update without exposure partly unlearns reading facts inside
    other text.
  - Next: an update that keeps the exposure (ctxv-style documents with old and new people as
    the own type).
- **An update that keeps the exposure** (`replay_ctx`; `data_context.py --update`):
  - Same 25M tokens and same new-fact share (1/2) as replay; the rest is 0.3 old facts and
    0.2 varied-format exposure documents, with old and new people as the own type.
  - **Mixed loss is fully recovered:** 1.921 / 1.657 / 2.356 / 2.024 (mixed_val / long /
    inline / marked), vs 1.920 / 1.657 / 2.359 / 2.025 before the update. mixed_val facts
    late: 0.60.
  - **Recall is the best of any update:** old facts 0.998 (training templates) and 0.297
    (held-out templates); new people 0.996 and 0.234. Plain replay gives 0.994/0.269 and
    0.993/0.191; seeded replay 0.988/0.249 and 0.986/0.208.
  - Pure-domain losses of the conf gate: lang 1.706 (1.696 before the update), facts 0.607,
    math 0.723, code 2.002. The lang expert is unchanged, so the +0.01 on lang comes from the
    updated facts expert's share in the mixture.
  - Atom gate: the circuit weight where active (0.973) is unchanged. Arithmetic was not re-run,
    since the update does not touch the math expert or the circuit.
  - Recipe for the paper: an expert update keeps the expert's exposure mix.
- **Expert seeds** (`run_context.py --varied --seed 1|2`; prefixes main_ctxv20_e1, _e2):
  - Each seed retrains the four experts with another data order from the same seed model
    (main_seeded_seed), then the conf and atom gates with gate seed 0.
  - Mixed loss of the conf gate, mean ± sd over 3 seeds: 1.920 ± 0.001 (mixed_val),
    1.658 ± 0.002 (long), 2.353 ± 0.005 (inline), 2.027 ± 0.005 (marked). The worst seed
    (1.921 / 1.660 / 2.359 / 2.033) still beats every other system on every format, by at
    least 0.07 nats.
  - Pure-domain losses vary by ≤ 0.014 (facts 0.526–0.540). Recall on training templates
    0.998–1.000 and on held-out templates 0.27–0.30.
  - Purity holds for every seed: the lang, math and code experts recall facts at 0.015–0.022.
  - Atom gate is exact on every arithmetic cell and on all word problems for every seed; its
    circuit weight where active is 0.973–0.980.
  - Not covered: the seed model and the baselines are single runs.
  - The run was paused by hibernation during the e2 atom-gate eval. Only losses and
    accuracies come from it, so no numbers are affected.
