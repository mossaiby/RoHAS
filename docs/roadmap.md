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

### Track C calculator-tool baseline (`tool_baseline.py`; pre-registered 2026-11-10, before training)

Question: is the circuit under the label-free gate better than, or just equal to, the
standard alternative of a model trained to call a calculator (Toolformer-style)?

- **Baseline (mono_s+calc):** main_mono_s fine-tuned on its own data mix with every
  equation in the math and mixed data annotated as a call: `a op b=` then the call token
  (byte 0x01, one BPE token, 4 occurrences in all training data, all in code), then the
  answer. The answer tokens are tool output: masked from the loss, and inserted by the tool at
  inference. The tool parses the equation right before the call with the same rule as the
  circuit expert, so both systems get the same parser and differ only in how they decide to
  use it: labels and fine-tuning versus the label-free gate. 50M tokens, lr 3e-4, 8k-token
  batches; the same proportions as mono_s (lang, facts, math, code at 0.246 each, mixed
  0.015). The fine-tune costs 50M tokens more than mono_s; the gate system is not retrained.
- **Systems compared:** mono_s (no tool), mono_s+calc, and the atom gate main_ctxv20_gate_atom
  with its two expert seeds (e1, e2).
- **Tests:**
  1. Held-out arithmetic (eval seed 1000, 100 items per cell, up to 32 digits) and word
     problems.
  2. Arithmetic inside other text (new): the bare equation follows a story, biography or
     code segment, joined by a blank line (the training format of mixed documents) or by a
     space (inline; held out for both systems). Exact-answer accuracy, 4- and 16-digit
     operands, + − ×.
  3. Cost to the other data: val losses on lang, facts and code, and mixed-text loss on
     the four formats split by segment type. On math segments the baseline's loss is not
     comparable (it expects the call token after "="), so it is reported apart.
- **Predictions:**
  - P1: mono_s+calc is exact (≥ 0.99) at every length on bare prompts and word problems.
    If so, the circuit result is matched by a labelled tool, and the paper must say so.
  - P2: mono_s+calc changes mono_s's lang, facts and code val losses by ≤ 0.02 nats.
  - P3: the atom gate is ≥ 0.95 exact in every context of test 2.
  - No prediction for mono_s+calc in test 2; that is the open question.
- **How the paper uses it:** if the tool matches the gate everywhere, the claim narrows
  to: no call labels, no retraining of any expert, and the system stays a calibrated
  language model (it can score text). An advantage is claimed only where test 2 shows one.
- **Results, pre-registered format (main_mono_s_calc; no closing marker):**
  - Fine-tune: 50M tokens, about 19 min.
  - **P1 fails.** Held-out arithmetic is exact up to 8 digits (+, −) and 3 digits (×). Then
    + falls to 0.65 / 0.65 / 0.01 / 0.00 at 12 / 16 / 24 / 32 digits, − to 0.48 / 0.17 / 0 / 0,
    and × to 0.98 / 0.29 / 0.28 at 4 / 6 / 8. Word problems 1.00. mono_s without a tool: 0 from
    6 digits (+, −) and from 3 (×), word problems 0.82.
  - Cause, checked on raw outputs: the tool always inserts the right answer, but on numbers
    longer than in training the model keeps writing digits after it (the line-ending
    failure that the circuit's veto prevents).
  - **Test 2 (arithmetic inside other text):** mono_s+calc 0.425 overall (blank-line join 0.56,
    inline 0.29; 4 digits 0.77, 16 digits 0.08). Besides the line ending, the model sometimes
    never calls the tool after "=" (about 15% of items in a 144-item check). mono_s 0.114. The
    atom gate is exact on every item, for all three expert seeds (**P3 holds**).
  - The atom gate is exact on all held-out arithmetic and word problems for all three
    expert seeds.
  - **P2, literally, fails, in the good direction:** val losses lang 1.710 (mono_s 1.739),
    facts 0.437 (0.438), code 2.113 (2.146). The extra 50M tokens help. Fact recall 0.841 /
    0.073 (0.702 / 0.062).
  - Mixed text by segment type (non-math segments): better than mono_s on long, inline and
    marked; worse on the blank-line format (lang 2.54 vs 2.45, facts 1.12 vs 0.99). Math
    segments are not comparable (plain equations, while the model expects the call token).
- **Deviation from the pre-registration:** the pre-registered format lacks the closing
  delimiter that Toolformer's format has ("-> result]"), which handicaps the baseline on
  exactly the line ending. So we add a variant, main_mono_s_calc_end, where the tool also
  inserts a closing marker (byte 0x02). Both are reported; the paper uses the stronger one.
- **Results, closing-marker variant (main_mono_s_calc_end):**
  - Held-out arithmetic is exact at every length up to 24 digits for + and −, and for × up to
    8. At 32 digits: + 0.94, − 0.80. Word problems 1.00.
  - Test 2: 0.703 overall. By context: facts 0.935, lang 0.81, code 0.365; blank-line join
    0.874, inline 0.532; 4 digits 0.848, 16 digits 0.559. Code inline: 0.00–0.10.
  - Every remaining failure, checked on raw outputs, is a missing call: after "=" the model
    writes digits itself (7 of 40 32-digit subtractions; most code-context items). Once it
    calls, answer and ending are always right.
  - Val losses: lang 1.712, facts 0.439, code 2.115; fact recall 0.840 / 0.067. Mixed text
    by segment type matches the first variant within 0.03.
- **Conclusion for the paper:**
  - On bare arithmetic, a labelled calculator matches the circuit up to 24 digits and then
    falls short only because the model stops calling the tool on longer numbers than it was
    trained on.
  - Inside other text, the label-free gate is exact everywhere (3 expert seeds), while
    the tool model is 0.70 and fails mostly after code and inline. Its trigger is learned
    from where calls appear in training; the circuit's applicability is decided by the
    circuit itself and the gate learns to trust it from confidence alone.
  - Caveat: the circuit applies wherever its pattern appears, so an equation-like string in
    code (where "=" may be assignment) would also trigger it. Measured below (false
    triggers): none on held-out text, but wrong stated arithmetic is penalised.

### Track C circuit false triggers (`false_triggers.py`)

- **Held-out text, 12.9M tokens** (lang, facts, math, code val; four mixed formats), in
  512-token windows:
  - No false prediction and no violated veto anywhere.
  - Code (3.8M tokens of Python): no prediction at all ("==" is one token, and digit
    equations followed by "=" do not occur). 2,694 vetoes, all respected.
  - Lang and facts: the circuit never fires. Math: 262k predictions, all correct.
- **Stress test**, 200 items per kind, "<kid> wrote <eq>=<result> on the board.", nats over
  "=" and the stated result:

  | System | Correct | Wrong value | Missing minus |
  |---|---|---|---|
  | atom gate | 0.8 | 15.9 | 52.1 |
  | atom gate, vetoes off | 0.8 | 15.5 | 29.2 |
  | conf gate (no circuit) | 2.8 | 14.1 | 31.5 |
  | mono_s | 6.2 | 11.3 | 16.8 |

  - The circuit's prediction costs little on wrong text (+1.8 over the conf gate), because
    the gate's math expert already expects the right answer.
  - The veto is the expensive part: on a missing minus, "=" is vetoed (-30 nats), +20.6 over
    the conf gate.
- **For the paper:** zero false triggers on natural held-out text, including code; the
  system encodes that stated arithmetic is correct, so modelling erroneous text would need
  softer vetoes.

### Track B, step 1: language + code (pre-registered 2026-11-11, before any expert outputs were computed)

The roadmap's go/no-go for Track B: two pretrained experts that share a tokenizer, combined
by the Track C gate, must beat the better single expert on mixed text before more experts
are added.

- **Experts:** frozen and run in bf16: Qwen2.5-0.5B (base, "prose") and Qwen2.5-Coder-0.5B
  ("code"). Same tokenizer and vocabulary (151,936 rows); Coder is a continued pre-training
  of the base model on mostly code.
- **Data** (`experiments/track_b/data.py`):
  - prose: fineweb-edu (sample/100BT shard 000; row groups 0–99 train, 700–719 val);
  - code: codeparrot-clean-valid with the Track C filters and repository split (97 / 3);
  - mixed: documents of alternating prose and code segments in the Track C formats. The gate
    trains on 2M tokens in the blank-line format only. Validation uses all four formats
    (blank, long, inline, marked), from the validation pools;
  - natural mixed (held out from all training): GitHub notebooks
    (codeparrot/github-jupyter-text-code-pairs), markdown and the code cell after it,
    joined by a blank line, consecutive pairs of one notebook in one document; tokens
    labelled markdown or code.
  - Each validation set is 300k tokens in 512-token windows.
  - Contamination: both experts were likely pre-trained on web text and GitHub code like
    these, so absolute losses are optimistic; the comparison is between systems.
- **Gate:** the Track C confidence gate (hidden states and their 32-token causal mean, plus
  entropy and largest log-probability per expert; probability mixing), trained without
  routing labels on cached expert outputs. 3 epochs over the 2M tokens, lr 1e-3, seed 0.
- **Systems:** base alone, Coder alone, the gate, a label oracle (the expert of each token's
  segment type; mixed and notebook sets) and the per-token oracle (best expert per token
  in hindsight).
- **Criteria (go if all hold):**
  - G1: the gate's loss is at least 0.01 nats below the better single expert on each of the
    four synthetic mixed formats;
  - G2: the same on the natural notebooks;
  - G3: on pure prose and on pure code, the gate is within 0.02 nats of the better expert.
- If Coder is better than the base model on prose too, G1 and G2 can still hold through
  per-token mixing, but the margin is expected to be small. We report it whatever it is.
- **Results** (`gate.py`, `results/track_b/b_gate_s{0,1,2}.json`):
  - Caching: about 3,900 tokens/s per 0.5B expert on the RTX 3050 Ti (4.1M tokens each, about
    17 min); cache 14 GB.
  - Gate: 0.92M parameters, 67 s of training. Loss in nats per token, gate mean over 3 seeds
    (sd ≤ 0.0013):

    | Set | Base | Coder | Fixed 50/50 | Gate | Label oracle | Token oracle |
    |---|---|---|---|---|---|---|
    | prose | 2.766 | 2.837 | 2.747 | 2.743 | – | 2.583 |
    | code | 1.538 | 1.480 | 1.464 | 1.461 | – | 1.348 |
    | mixed blank | 2.743 | 2.744 | 2.673 | 2.663 | 2.706 | 2.509 |
    | mixed long | 2.268 | 2.235 | 2.181 | 2.179 | 2.216 | 2.033 |
    | mixed inline | 3.562 | 3.480 | 3.380 | 3.380 | 3.461 | 3.181 |
    | mixed marked | 2.545 | 2.538 | 2.475 | 2.469 | 2.505 | 2.318 |
    | notebooks | 1.986 | 1.891 | 1.876 | 1.873 | 1.938 | 1.735 |

  - **Pre-registered criteria: all hold (go).** Margins over the better single expert:
    G1 0.056–0.100 on the four mixed formats; G2 0.019 on notebooks (worst seed 0.018); G3
    the gate is *better* than either expert on pure prose (by 0.023) and pure code (0.019).
  - The gate also beats the label oracle everywhere: on notebooks Coder is better than the
    base model on the markdown too (2.536 vs 2.653), and the gate puts 0.55 on Coder there.
  - **Control (added after the criteria were evaluated): almost all of the gain is
    ensembling.** A fixed 50/50 mixture (the best constant weights on mixed_train) is within
    0.001–0.010 of the gate on every set. The gate's own contribution is 0.010 (blank),
    0.007 (marked), 0.004 (prose), 0.003 (code, long, notebooks), 0.001 (inline). Its weights
    are soft: about 0.65 / 0.35 toward the matching expert on mixed text, 0.66 / 0.34 on
    prose and 0.28 / 0.72 on code.
  - Reading: the two experts overlap heavily (Coder continues the base model's training), so
    either expert is a fair fallback for the other and plain averaging already helps. With
    such experts, token-level routing adds little. The Track C result, where routing
    mattered, came from experts that are poor outside their own data.
- **Next:** the routing claim needs experts that differ more. The planned next step adds
  math (Qwen2.5-Math-1.5B in 4-bit) and the circuits, where one expert is far better on
  some tokens; the constant mixture is now a standard control row.

### Track B, step 2: + math (pre-registered 2026-11-11, before any math-expert outputs were computed)

Step 1 passed its criteria, but a fixed mixture got almost all of the gain. This step asks
the routing question directly: with an expert that is far better than the others on some
text, does the gate beat the best *fixed* mixture of the same experts?

- **Experts:** Qwen2.5-0.5B (prose), Qwen2.5-Coder-0.5B (code), Qwen2.5-Math-1.5B (math),
  frozen, bf16 (4-bit only if 1.5B does not fit the 4 GB GPU; then this note is amended
  before caching). The math expert must share the tokenizer (checked on download).
  Confound: the math expert is three times larger, so it may be better on prose and code
  too; the per-type losses and gate weights show whether it is.
- **Data** (data.py --step 2):
  - math: GSM8K (openai/gsm8k, main): question, newline, the solution with the calculator
    annotations "<<...>>" removed (the final "#### n" line kept). Train split for the
    gate's documents, test split for validation;
  - m3_train: 1.5M tokens of documents mixing prose, code and math segments (adjacent
    segments differ), blank-line format; a math segment is one whole GSM8K item;
  - validation: math_val (GSM8K test, all 1,319 items), m3 mixed sets in the four formats
    (200k tokens each), and the step-1 prose, code and notebook sets.
- **Gate:** the step-1 gate with three experts (hidden states of different widths
  concatenated: 896 + 896 + 1536); same training recipe; seeds 0, 1, 2.
- **Control:** the best constant mixture, with weights fitted on m3_train (simplex grid,
  step 0.02).
- **Criteria:**
  - R1 (routing beyond ensembling): on each of the four m3 formats, the gate's loss is at
    least 0.02 nats below the best constant mixture.
  - R2: on math_val, the gate is within 0.02 of the math expert; on prose and code, within
    0.02 of the better single expert.
  - R3: on each m3 format and on notebooks, the gate is at least 0.01 below the best
    single expert.
- **If R1 fails,** the paper's Track B claim is limited to "mixing pretrained experts
  helps, mostly as an ensemble", and the routing evidence rests on Track C and the circuit.
- **Results, as pre-registered** (`results/track_b/b3_gate_s{0,1,2}.json`):
  - Qwen2.5-Math-1.5B fits in bf16 (peak 3.4 GB at batch 2), so no quantization; it shares
    the tokenizer exactly. Caching: about 2,000 tokens/s.
  - Fixed mixture (fitted on m3_train): base 0.32, Coder 0.22, math 0.46.
  - Loss, gate mean over 3 seeds (sd ≤ 0.0023):

    | Set | Best single | Fixed mix | Gate | Gate vs fixed |
    |---|---|---|---|---|
    | prose | 2.766 | 2.740 | 2.719 | +0.022 |
    | code | 1.480 | 1.500 | 1.449 | +0.052 |
    | math (GSM8K test) | 1.476 | 1.304 | 1.352 | −0.048 |
    | m3 blank | 2.168 | 2.060 | 2.064 | −0.004 |
    | m3 long | 2.038 | 1.963 | 1.941 | +0.022 |
    | m3 inline | 2.223 | 2.060 | 2.060 | +0.001 |
    | m3 marked | 2.225 | 2.117 | 2.093 | +0.024 |
    | notebooks | 1.891 | 1.806 | 1.786 | +0.020 |

  - **R1 fails** (holds on long and marked, not on blank and inline). **R2 holds** (math
    1.352 vs the math expert's 1.476; prose and code better than either expert). **R3
    holds** everywhere, by 0.10–0.16 on mixed text.
- **Diagnosis, after the fact: the gate's math training data is contaminated.** GSM8K's
  train split is memorised by the experts, the math expert most:

  | Math segments | Base | Coder | Math |
  |---|---|---|---|
  | m3_train (GSM8K train) | 1.265 | 1.389 | 0.965 |
  | m3_blank_val (GSM8K test) | 1.557 | 1.605 | 1.621 |

  - On unseen problems the math expert is no better than the base model, but on the
    gate's training problems it is far better. So the gate learns to trust it on math (weight
    0.63, Coder 0.04), and on test problems that trust is wrong: there the fixed mixture,
    which keeps Coder at 0.22, wins by 0.065 on math segments.
  - Consistent with this: the gate beats the fixed mixture by 0.10 on m3_train itself.
  - The gate is not the problem; the training data does not match the test distribution
    for one expert.
- **Deviation (decided after the results above):** re-run with uncontaminated math for
  the gate. GSM8K test is split in two halves by item index (even / odd): the even half
  builds the gate's training documents (m3c_train), the odd half the validation sets (m3c_*,
  mathc_val). Same criteria, same recipe. Both runs are reported. Caveat: the even half
  repeats about 7 times in the 1.5M training tokens.
- Also learnt: Qwen2.5-Math-1.5B is not a strong specialist on GSM8K-style text in loss
  terms (1.621 vs the base model's 1.557 on unseen problems). Its training style differs
  (LaTeX, chain-of-thought formats). A loss-based routing test needs an expert that is
  clearly better on some unseen text.
- **Results, uncontaminated re-run** (`results/track_b/b3c_gate_s{0,1,2}.json`; GSM8K test
  halves):
  - The math expert is no longer special on the gate's training math (m3c_train math
    segments: base 1.564, Coder 1.614, math 1.613), as on validation.
  - Fixed mixture (fitted on m3c_train): base 0.40, Coder 0.32, math 0.28.
  - Loss, gate mean over 3 seeds (sd ≤ 0.0016):

    | Set | Base | Coder | Math | Fixed mix | Gate | Gate vs fixed (worst seed) |
    |---|---|---|---|---|---|---|
    | prose | 2.766 | 2.837 | 2.983 | 2.726 | 2.720 | +0.006 (+0.005) |
    | code | 1.538 | 1.480 | 1.803 | 1.474 | 1.449 | +0.025 (+0.025) |
    | math (GSM8K odd half) | 1.506 | 1.507 | 1.480 | 1.308 | 1.294 | +0.014 (+0.013) |
    | m3c blank | 2.166 | 2.188 | 2.337 | 2.051 | 2.023 | +0.029 (+0.029) |
    | m3c long | 2.048 | 2.036 | 2.255 | 1.946 | 1.925 | +0.021 (+0.021) |
    | m3c inline | 2.200 | 2.201 | 2.269 | 2.035 | 2.012 | +0.023 (+0.023) |
    | m3c marked | 2.222 | 2.229 | 2.378 | 2.105 | 2.077 | +0.029 (+0.029) |
    | notebooks | 1.986 | 1.891 | 1.937 | 1.811 | 1.786 | +0.026 (+0.024) |

  - **R1 holds** on all four mixed formats for every seed (smallest margin 0.0207, long).
    **R2 holds:** math 1.294 vs the math expert's 1.480; prose and code better than any
    single expert. **R3 holds**, by 0.11–0.19.
  - Gate weights on blank-format mixed text, by segment (base / Coder / math): prose 0.47 /
    0.24 / 0.29, code 0.29 / 0.56 / 0.15, math 0.32 / 0.14 / 0.54. On math segments the math
    expert is the worst alone (1.630), yet weighted highest. Mixing gives 1.405 there, so the
    experts' errors are complementary.
- **Reading, with caveats:**
  - **The routing claim holds, modestly.** Weights that depend on the token beat any fixed
    weights by 0.02–0.03 nats on mixed text, and on notebooks no system was trained on.
    Ensembling of the same experts gives a larger share: the fixed mixture beats the best
    single expert by 0.09–0.17. So routing is about 15–25% of the total gain.
  - **The margins are just above the threshold**, and this run is a deviation chosen after
    the contaminated run failed. It deserves less weight than a result pre-registered in
    full; the paper must report both runs.
  - The training math repeats about 7 times. The gate trained on it does not overfit
    measurably: it is better than the fixed mixture on held-out math.
  - **Lesson for the paper:** a gate trained on data that an expert has memorised learns
    to trust that expert where it will not help. Gate training data must be unseen by
    every expert, which pretrained models make hard to guarantee.

### Track B, step 3: the compiled circuit under the pretrained gate (pre-registered 2026-11-11, before any data or outputs)

The routing claim at its sharpest: an exact specialist joins frozen pretrained experts.

- **Experts:** Qwen2.5-0.5B and Qwen2.5-Coder-0.5B (step 1, frozen), plus the compiled
  arithmetic circuit (circuit_expert.py, unchanged; Qwen splits digits into single tokens and
  has "=" and "=-" tokens, as the Track C tokenizer does). The math model is left out:
  generation runs all experts live, and three models do not fit 4 GB.
- **Gate:** the step-1 gate b_gate_s0, *frozen*, plus the two-level atom head of Track C: the
  circuit's weight comes from its own two confidence features in nats through a linear head,
  and the LM experts share the rest as before. Only the head's 3 parameters are trained. So
  the test also asks H2 on pretrained models: an expert is added without retraining the gate.
  Vetoes are applied at generation; the loss evaluation is without vetoes (for correct text
  they can only lower the loss, since false vetoes were never seen in Track C).
- **Data** (Qwen tokenizer): documents mixing prose, code and arithmetic segments (Track C
  synth.math_doc lines: equations of up to 5 digits for + and −, 3 for ×, and templated word
  problems). 1.5M tokens in the blank-line format train the head; validation has the blank
  and inline formats (200k tokens each).
- **Evaluation:**
  - Live greedy generation on held-out equations ("a op b", the next token restricted to
    "=" or "=-" as in Track C): + and − at 1, 4, 8, 16, 32 digits, × at 1, 2, 4, 8; 40
    items per cell. Systems: base alone, Coder alone, the gate, the gate with the circuit.
  - Arithmetic inside other text: a prose or code segment, then a blank line or a space,
    then the equation; +, −, × at 4 and 16 digits; 20 per cell.
  - Loss: on prose_val, code_val, notebooks_val (no equations) and on the new mixed sets.
- **Criteria:**
  - C1: the gate with the circuit is exact (≥ 0.99) in every cell of the held-out
    equations.
  - C2: ≥ 0.95 exact inside other text, in every context and join.
  - C3: on prose, code and notebooks, its loss is within 0.005 of the gate alone.
  - C4: the circuit's weight is ≥ 0.9 where it is active and ≤ 0.01 where it is not.
- **Correction before any criterion was read off (step count):** the head was first fitted for
  500 Adam steps from zero, which had not converged: weight 0.72 where the circuit is active,
  loss still falling. At 2,000 steps the weight is 0.97, at 10,000 steps 0.9998 (0.0015 where
  inactive), and the loss then changes by < 1e-5. The pre-registration fixed no step count;
  the head is now trained to convergence (10,000 steps). The circuit is never wrong on this
  data (0 mismatches in 30,956 active positions), so the optimum is a weight near 1.

**Result (2026-11-11; results/track_b/circuit_gate.json): all four criteria hold.**

- C4: the head's weight is 0.9998 where the circuit is active and 0.0015 where it is not.
- C1: the gate with the circuit is exact in all 14 equation cells (1.000, up to 32 digits for
  + and −, 8 for ×). The pretrained experts alone, and the gate over them, reach 0.95–1.0 at
  one digit and fall to 0.000–0.025 from 8 digits on (− fails from 4 digits).
- C2: 1.000 in all 24 in-text cells (prose and code contexts, blank and inline joins). Without
  the circuit: base 0.06, Coder 0.01, gate 0.06.
- C3: prose 2.7441 vs 2.7429 (+0.0013), code 1.4620 vs 1.4606 (+0.0014), notebooks 1.8739 vs
  1.8733 (+0.0006). The circuit is never active on these sets (0 active positions); the cost is
  the 0.0015 weight it keeps where it is inactive.
- On the mixed sets the answer tokens fall from 0.596 to 0.0002 nats (blank) and 0.512 to
  0.0002 (inline); the overall loss falls by 0.020 and 0.032.

### Track C: a second seed of the monolithic baselines (decided 2026-11-11, before training)

mono_s and mono_m were trained once each. Both are retrained with seed 1 (initialisation and
data order) and exactly the settings of train_main_mono_*.json
(experiments/track_c/run_baseline_seed.py), and evaluated as before: losses and fact recall,
held-out arithmetic for mono_s, and mixed text in all four formats. There is no criterion; the
second seed is reported next to the first in every table where the baselines appear, and any
claim of the paper that the second seed reverses is withdrawn.

**Result (2026-11-12; eval_main_r1, eval_main_r1_heldout, diag_mixed_*_main_r1):** no claim is
reversed. Mixed-text loss moves by at most 0.024 nats between seeds (mono_s 1.990/1.981 blank,
1.858/1.856 long, 2.680/2.704 inline, 2.258/2.255 marked; mono_m 2.052, 1.925, 2.648, 2.280
with seed 1). The worst varied-exposure expert seed beats the best of the four monolithic runs
on every format, by at least 0.061 nats. Per-type losses agree within 0.04; mono_s's fact
recall with training templates varies more (0.70 / 0.86), held-out templates 0.06 / 0.03.
mono_s seed 1 fails arithmetic from 8 digits (+, −) and 4 digits (×), as seed 0. The run was
paused by hibernation, so its wall-clock times are invalid.

### Track C follow-up: the circuit's head refitted to convergence (decided 2026-11-12, before running)

Under the from-scratch gate the circuit's weight where it is active varies with the gate seed
(main_ctxv20: 0.97 / 0.51 / 0.94; seeded: 0.99 / 0.78 / 0.98). Its 3-parameter head was trained
jointly with the gate for a fixed number of steps, and the low seeds look unconverged, as the
Track B head was at 500 steps. experiments/track_c/refit_head.py freezes every other part of
these six gates and refits only the head, full batch (about 0.5M mixed_train tokens, 10,000
Adam steps from the trained values, loss without vetoes), then evaluates the refitted gates on
held-out arithmetic (eval seed 1000, 100 per cell), losses and routing (tag conv_heldout).
Expected: weight ≥ 0.9 where active and ≤ 0.01 where inactive for every seed, arithmetic still
exact, losses within 0.005 of the jointly trained gates. The paper keeps the original runs and
reports the refit as a follow-up, whichever way it comes out.

**Result (2026-11-12; refit_head.json, eval_conv_heldout.json): as expected, for all six gates.**
The weight where the circuit is active goes from 0.97 / 0.51 / 0.94 (main_ctxv20) and
0.99 / 0.78 / 0.98 (seeded) to ≥ 0.9999; where it is inactive it stays at 0.0025 (main_ctxv20)
and 0.0000 (seeded). Held-out arithmetic (all cells, + and − to 32 digits, × to 8) and word
problems are 100% for every refitted gate. Losses change by at most 0.0047 nats (a decrease,
on math for main_ctxv20 seed 1); all other changes are ≤ 0.002. The head's output changes by
≤ 0.0001 over the last 5,000 steps. The paper keeps the jointly trained gates in its tables and
reports the refit as a follow-up (Section 5.4).

### Track B, step 4: replacing a pretrained expert under the frozen gate (pre-registered 2026-11-12, before any outputs)

H2 with pretrained experts: replace one expert by another and leave the gate as it is.

- **Swap:** the prose expert Qwen2.5-0.5B is replaced by Qwen2.5-0.5B-Instruct (same
  architecture and width, further trained for chat), as an expert would be replaced by a newer
  version. Coder is unchanged. The step-1 gates b_gate_s0..s2 are used frozen, with the new
  expert's outputs in the old one's slot.
- **Data:** prose_val, code_val, notebooks_val, mixed_blank_val (the gate's training format)
  and mixed_inline_val (held out). Disk does not allow hidden-state caches of all of
  mixed_train for three experts, so the reference gates train on its first 1,302 windows (one
  third, mixed_sub_train) for 9 epochs, the same number of updates as the step-1 gates.
- **Systems** (3 seeds each):
  - orig: frozen step-1 gate over base + Coder (the system before the swap);
  - swap: the same frozen gate over Instruct + Coder;
  - ref: a gate retrained on mixed_sub_train over Instruct + Coder (what retraining would buy);
  - check: a gate retrained on mixed_sub_train over base + Coder, to show that the shorter
    training file gives a gate as good as the step-1 gate.
  Plus each expert alone and the fixed mixture.
- **Criteria:**
  - S1 (no retraining needed): on every evaluation set, swap is within 0.01 nats of ref
    (means over seeds).
  - S2 (only the replaced type changes): on code_val, swap is within 0.01 nats of orig.
  - S3 (the new expert is used): on every set, swap beats both of its experts alone.
  - Validity: check within 0.01 of orig on every set; otherwise ref is not a fair reference
    and S1 is reported as not tested.
- **Caveat stated in advance:** if Instruct and base differ on prose_val by less than 0.02
  nats, the swap is too small to say much, and we will say so.

**Result (2026-11-12; results/track_b/swap_*.json).** The test ran under the original package
versions (torch 2.6.0, transformers 5.22.1; they had been downgraded on this machine in the
meantime and were reinstalled). The recomputed base and Coder outputs are bit-identical to the
earlier caches. The swap is not small: Instruct is 0.060 nats worse than base on prose_val.

- Validity check **fails**: gates retrained on the third of mixed_train (check) are 0.009–0.016
  worse than the step-1 gates on the same pair (> 0.01 on four of five sets). So, as
  pre-registered, ref is not a fair reference and **S1 is not tested**. For the record, the
  frozen gate is better than ref on four sets (by 0.004–0.009) and worse by 0.0035 on inline
  (worst seed 0.0046).
- S2 **holds**: code_val 1.4696 after the swap vs 1.4608 before (+0.0088 < 0.01).
- S3 **holds**: after the swap the gate beats both of its experts alone on every set, by
  0.004 (notebooks) to 0.064 (inline).
- Not pre-registered, reported: the fixed mixture refitted for the new pair is within
  −0.002…+0.013 of the frozen gate (better than it on prose, code, notebooks and inline), as
  in step 1, where the two experts' gain was almost all ensembling. The gate's weights follow
  the replacement: on prose segments of the mixed sets the prose expert's weight is 0.67 / 0.66
  after the swap vs 0.65 / 0.63 before.

### Track C follow-up: the calculator baseline with varied-format mixed text (pre-registered 2026-11-12, before any data)

Possible objection: the calculator baseline (main_mono_s_calc_end) is weak inside other text
only because its mixed training data has one format (blank-line joins, 1.5% of its tokens),
while the gated experts had varied-format context exposure (20% of their branch tokens).

- **Data:** mixedv_train, 12M tokens of mixed documents built like mixed_train (same pools:
  the training story sample, the stored people, training-split code, fresh math) but with each
  document's format drawn from data_mixed_ood.VARIED (blank, long, para3: the formats of the
  experts' exposure), annotated with calls and closing markers (mixedcalcve_train).
- **Model:** main_mono_s_calcv_end: main_mono_s fine-tuned as main_mono_s_calc_end (50M
  tokens, lr 3e-4, 8k-token batches, tool output out of the loss), but with the mixed share
  raised to 0.2 (lang, facts, math, code 0.2 each), the experts' exposure fraction.
- **Tests:** as before: held-out arithmetic and word problems, arithmetic inside other text
  (tool_context: stories, biographies, code; blank and inline joins; + − ×; 4 and 16 digits),
  mixed-text loss in four formats.
- **No pass/fail for our system.** We report the result as it comes out. If the varied baseline
  is exact (≥ 0.95) in every in-text cell, the paper's claim that the circuit under the gate is
  exact where a calculator-calling model is not is withdrawn for text, and kept only where the
  data still support it.

**Result (2026-11-12; tool_heldout.json, tool_context.json, diag_mixed_*_tool_baseline_vend.json).**
The objection is partly right. Varied mixed text makes the calculator baseline much stronger:
held-out arithmetic 100% in every cell (up to 32 digits; the original _end variant 94% / 80% at
32 digits), word problems 100%, in-text 89% overall (was 70%): 100% after stories and
biographies and on blank-line joins. It still fails after code on the same line (code/inline
cells 10–76%; in-text code 68%, inline 79%); 6 of 36 cells are below 0.95. In 189 of 192
failures there (inspected in the raw outputs) the model makes no call and writes digits itself;
in 3 it calls but the answer is wrong. Not every cell is ≥ 0.95, so the claim stays, narrowed:
the circuit under the gate is exact where this baseline fails, on equations that follow code on
the same line (a join held out from both systems). Mixed-text loss of the variant: blank 2.164,
long 1.882, inline 2.516, marked 2.226 (all above the varied-exposure gated experts).

**Correction (2026-11-12; tool_failures.py, results/track_c/tool_failures.json).** The paper said
that in every inspected failure of the closing-marker calculator model the call is missing. A
full classification of its 534 wrong in-text answers (of 1,800) finds no call in 505 and a call
that still gives a wrong answer in 29. The varied variant: 190 of 192 without a call. The paper
now gives these counts from the result file. The 500-step weight of the Track B circuit head
(0.72) is now stored in circuit_gate.json (head trajectory; the refit is bit-identical).

### Scale study, first step: corpus v2 at 5e16 and 5e17 FLOPs (pre-registered 2026-11-22, before any v2 data or runs)

Goal: test whether the from-scratch results hold at ten times the compute, on real-world prose
instead of children's stories. Runs on the RTX 3090 machine (`run_scale.py`); the 5e16 runs on
corpus v1 stay as they are.

- **Data (corpus v2, `data/track_c_v2/`, `data.py --corpus v2`).** Prose: FineWeb-Edu
  (sample/10BT files 000–001; every 100th document of file 001 is validation, the next the
  foreign-context pool). Code: codeparrot-clean-train files 1–10 plus codeparrot-clean-valid,
  the v1 filters and repository split. Biographies (2,000 + 500 people) and synthetic math
  unchanged. Tokenizer retrained on the v2 sources by the v1 recipe (8,192 tokens, single-digit
  tokens). About 1.5B training tokens each of prose, code and math, so no budget up to 5e18
  repeats them. Gate data mixed_train 30M tokens; context-exposure files ctxv_<type>_train
  42M tokens each (`data_context.py --varied --tokens 42e6`); held-out formats long, inline,
  marked (`data_mixed_ood.py`).
- **Two points, the same recipe.** The recipe is that of the 5e16 runs whose results are in the
  paper: seeded experts (seed on 4fT tokens of the four types, f = 1/2; four branches on (1−f)T
  tokens, 20% from ctxv_<type> with the loss on their own type only), mono_s (expert size, all
  four types plus mixed text, at the systems' total FLOPs), mono_m (four times the parameters,
  same FLOPs), 8,192-token batches, learning rate 1e-3 for expert-size models and 6e-4 for the
  larger, cosine schedule, one seed. The gate trains on G tokens of mixed_train, and mono_s and
  mono_m see exactly G tokens of mixed text, as before. The confidence gate (prob) and the
  two-level atom gate with the compiled circuit and vetoes are trained.
  - **sc16:** sizes s (17.5M) and m (70M), T = 100M, G = 6M: the 5e16 point on corpus v2.
  - **sc17:** sizes s17 (52.8M; d 576, 12 layers) and m17 (210M; d 1024, 16 layers),
    T = 350M, G = 60M: 5.0e17 FLOPs per system. G grows with the budget, so that the gate's and
    the mixed text's shares of each system stay those of 5e16.
  - No learning-rate or batch tuning at either point. `torch.compile` is used for training
    (same arithmetic; it is faster). Independent (unseeded) experts are not run at this step.
- **Measurements (as in the paper).** evaluate.py: loss per type, fact recall with training and
  held-out templates, arithmetic and word problems; diag_mixed.py: mixed-text loss on mixed_val
  (the gate's training format) and on the held-out formats long, inline and marked, with the
  oracles.
- **Criteria, tested at sc17 (and reported at sc16):**
  - S1: the gated ctxv experts (confidence gate, prob) have a lower loss than mono_s on the
    inline and on the marked held-out formats. This is the paper's main mixed-text claim.
  - S2: the experts (each on its own type) beat mono_s on prose and on code loss.
  - S3: fact recall with held-out templates: gated experts at least as high as mono_s.
  - S4: under the atom gate with vetoes, the arithmetic of evaluate.py is exact (100%) at every
    digit count.
  - Whatever the outcome, we report the margins at sc16 and sc17 side by side; the direction of
    the change is the result of this step, not a pass or fail. If a criterion fails at sc17, the
    paper says that the claim holds at the smaller scale only.

**Results, 20 biographies per person (`eval_sc16.json`, `eval_sc17.json`, `diag_mixed_*sc1[67].json`;
finished 2026-11-23).** All four criteria hold at both points. Loss in nats; "mono_s" is the
expert-size monolithic model.

| | sc16 | sc17 |
|---|---|---|
| S1 inline: gate vs mono_s (mono_m) | 2.785 vs 3.045 (3.070) | 2.678 vs 2.794 (2.819) |
| S1 marked: gate vs mono_s (mono_m) | 2.426 vs 2.601 (2.649) | 2.242 vs 2.361 (2.376) |
| S2 prose: lang expert vs mono_s | 3.435 vs 3.553 | 3.065 vs 3.225 |
| S2 code: code expert vs mono_s | 1.836 vs 1.918 | 1.547 vs 1.667 |
| S3 held-out recall: gate_conf / gate_atom vs mono_s | 0.274 / 0.279 vs 0.042 | 0.238 / 0.267 vs 0.096 |
| S4 atom gate, answer number exact (`arith_number`) | 100% in every cell | 100% in every cell |

The S1 margins shrink with scale (inline −0.260 → −0.116, marked −0.175 → −0.119); the S2
margins grow. Under the stricter `arith` score (the generated line is exactly "=" + answer) the
atom gate is 89% (sc16) and 96% (sc17) over all cells.

**The facts set overfits at sc17 (found after the run).** facts_train is a fixed 2.57M tokens
(20 biographies per person), so at sc17 the facts expert's path sees it about 120 times and
mono_s about 140 times. The facts validation loss (fresh biographies of the same people: same
templates, new orderings) reaches its minimum and then rises in every sc17 run: seed 0.43 → 0.77,
mono_s 0.41 → 0.81, facts expert 0.48 (first evaluation) → 1.05. The gated system then has the
worst facts loss at sc17 (1.42, mono_s 0.82, mono_m 0.66), while training-template recall is 100%.
At sc16 the effect is small (facts expert 0.38 → 0.45). This is a defect of the design: the facts
set did not grow with the budget. Other deviations: the ctxv own-type shares on v2 differ from v1
(lang 0.524 instead of 0.316), and the pipeline check's G was cut to 0.02M after its first
launch, because the larger G gave mono_m a negative mixed weight (`run_scale.py` now asserts
that G is under half of each monolithic run).

### Scale study, step 1b: a facts set that does not repeat (pre-registered 2026-11-23, before the new data or runs)

- **Data (`data/track_c_v2f/`).** Corpus v2 unchanged (same tokenizer, the same files
  hard-linked), except facts_train: 600 biographies per person instead of 20 (the same 2,000
  people, attributes and training templates; new orderings and template choices; about 77M
  tokens). Built by `data.py --grow-facts track_c_v2 --bios 600`. The facts expert's path then
  sees each biography of the set about 4 times at sc17 and once at sc16; mono_s about 5 and 1.3
  times. Evaluation (people, prompts, facts_val) is unchanged.
- **Runs.** Points 17f and 16f: the sc17 and sc16 configurations exactly, on track_c_v2f, in
  that order (`run_scale_v2f.sh`). Nothing else changes.
- **Criteria.** S1–S4 as above, tested at sc17f and reported at sc16f, side by side with sc17
  and sc16. Added check F: the facts validation loss of the sc17f seed, facts expert and mono_s
  does not rise by more than 0.05 nats from its minimum to the end of training. If F fails, the
  overfitting is not only repetition, and we say so.
- The 20-biography results above stay in the record; the paper uses 16f and 17f.

**Results, 600 biographies per person (`eval_sc1[67]f.json`, `diag_mixed_*sc1[67]f.json`; finished
2026-11-24).** All four criteria and check F hold at both points.

| | sc16f | sc17f |
|---|---|---|
| S1 inline: gate vs mono_s (mono_m) | 2.749 vs 3.010 (3.099) | 2.668 vs 2.794 (2.802) |
| S1 marked: gate vs mono_s (mono_m) | 2.411 vs 2.581 (2.679) | 2.235 vs 2.352 (2.347) |
| S2 prose: lang expert vs mono_s | 3.434 vs 3.544 | 3.039 vs 3.212 |
| S2 code: code expert vs mono_s | 1.833 vs 1.920 | 1.534 vs 1.658 |
| S3 held-out recall: gate_conf / gate_atom vs mono_s (mono_m) | 0.208 / 0.208 vs 0.026 (0.022) | 0.162 / 0.150 vs 0.074 (0.112) |
| S4 atom gate, `arith_number` | 100% in every cell | 100% in every cell |
| Strict `arith`, atom gate: worst cell / mean | 0.78 / 0.99 | 0.96 / 1.00 |
| F: facts val loss, minimum → end (seed, facts expert, mono_s) | 0.692 → 0.699, 0.376 → 0.376, 0.615 → 0.615 | 0.428 → 0.431, 0.392 → 0.393, 0.409 → 0.416 |
| Facts loss: facts expert / gate_atom / mono_s / mono_m | 0.375 / 0.530 / 0.606 / 0.828 | 0.390 / 0.856 / 0.417 / 0.384 |

The S1 margins shrink with scale as before (inline −0.261 → −0.126, marked −0.170 → −0.117);
the S2 margins grow (prose −0.110 → −0.173, code −0.087 → −0.124). Held-out recall is lower than
with 20 biographies for every system (part of the old recall came from memorising the small set),
but the gated systems keep about twice mono_s at sc17f and eight times at sc16f. Two results
against the expert system, to be reported: at sc17f the gate's facts loss (0.856) is twice
mono_s's (0.417) although the facts expert alone reaches 0.390, so the gate, trained on mixed
text only, does not hand plain facts documents to the facts expert; and its math loss is
slightly above mono_s's at both points (0.711 vs 0.706, 0.706 vs 0.694). With the larger facts
set, sc16f's mono_s and mono_m have higher facts loss than at sc16 (0.606 vs 0.421, 0.828 vs
0.582): at 5e16 they see each biography about once.

### Track B, step 5: GSM8K accuracy with 1.5B experts and the compiled circuit (pre-registered 2026-11-25, before any 1.5B outputs, code or data on the 3090)

The question: when frozen pretrained experts solve GSM8K under the label-free gate, does adding
the compiled circuit raise answer accuracy, by making the arithmetic inside the solutions exact?
This is a generation benchmark, not a loss.

- **Experts (frozen, bf16, all on the RTX 3090):** Qwen2.5-1.5B (base15), Qwen2.5-Coder-1.5B
  (coder15), Qwen2.5-Math-1.5B (math). One tokenizer (checked: identical vocabularies of 151,936).
- **Gate:** the Track B gate (gate.py, unchanged recipe: 3 epochs on m3c_train, AdamW 1e-3,
  cosine, batch 8, seed 0) over the three, trained on cached outputs. The data files are those
  of step 2's uncontaminated re-run: GSM8K test even items (without the "<<...>>" annotations)
  build m3c_train, so **GSM8K is evaluated only on the 659 odd test items**, none seen by any
  trained part.
- **The circuit for GSM8K text (`GsmCircuit`).** The Track C circuit expert with one stricter
  trigger and the other operations of circuits/arith.py: it is active when the context ends with
  `<<a op b=` [answer prefix], with a and b non-negative decimals (digits, at most one "."),
  op one of + − × ÷ (`*`, `/`), "<<" immediately before a and "=" immediately after b; so a
  chained annotation such as `<<3*4+2=` never triggers it (the Track C rule would have computed
  4+2). The answer is arith.evaluate (÷ to 6 places), with trailing fractional zeros and a
  trailing "." removed; division by zero abstains. Vetoes as in Track C (after a complete answer,
  no digit and no "."). The head (3 parameters) is fitted as in step 3, on a4_train with the
  Track C trigger, against the new gate, to convergence (10,000 steps); its features do not
  depend on the trigger, so the same head serves the GSM8K trigger.
- **Evaluation:** greedy decoding, 8-shot: the prompt is 8 GSM8K train items (random.Random(0),
  fixed) as `Question: q\nAnswer: a` with the original answers (annotations and "#### n"
  kept), joined by blank lines, then `Question: q\nAnswer:`. Up to 384 new tokens; a row stops
  at a blank line, "Question:" or end of text. Accuracy = the number after "####" equals the
  reference (commas, "$" removed; numeric comparison); the last number in the output is a
  second, reported score (flexible). KV-cached decoding with left padding.
- **Systems:** base15, coder15, math alone; gate; gate + circuit; base15 + circuit (same head
  and vetoes); and a calculator tool, as in the original GSM8K paper, under the gate and under
  base15: when the text ends with `=` inside an open `<<`, the expression since `<<` (numbers,
  + − × ÷, parentheses) is evaluated exactly, and its value (an integer, or 6 places with
  zeros removed) followed by `>>` is written into the text.
- **Criteria:**
  - Q1 (main): gate + circuit is more accurate than the gate; the paired 95% bootstrap interval
    of the difference (10,000 resamples of the 659 items, seed 0) lies above zero.
  - Q2: in the outputs of gate + circuit, every complete annotation the circuit covers (one
    operation on two non-negative decimals) has the right value (written value within one unit
    of its last written place of the exact value). Reported beside the same rate for the gate,
    with the counts of covered and uncovered annotations.
  - Q3: on prose_val, code_val and notebooks_val the gate with the circuit is within 0.005 nats
    of the gate (as step 3's C3).
  - No pass/fail against the calculator: it covers chained expressions the circuit leaves to the
    LMs, so we expect it to be at least as accurate; the paper reports the gap and the share of
    annotations the circuit does not cover.
  - Also reported: the gate against the best single expert (accuracy and loss), and the error
    types of the items that gate + circuit gets right and the gate wrong (and the reverse).
- Failure of Q1 is reported as the result: most GSM8K errors at this size may be reasoning, not
  arithmetic. If a criterion cannot be read off as written, the deviation is recorded here
  before the number is used.

**Why the gate's facts loss is high at sc17f (diagnostic added after the results;
`diag_facts_routing.py`, `results/track_c/diag_facts_routing.json`).** The gate does route plain
biographies to the facts expert: on facts_val its weight there is 0.90 at both points (gate_conf
and gate_atom), the facts expert has the largest weight on 91% of tokens, and the weight is the
same at every position in the window (0.88–0.92); on the biography segments of mixed text it is
also 0.90. So the gate, trained on mixed text only, never commits more than about 0.9 to one
expert: that alone costs at least −log 0.9 ≈ 0.105 nats, and the remaining weight goes to experts
that are very bad on biographies (lang 6.7, code 9.1, math 15.9 nats at sc17f). The gap to the
facts expert alone is +0.155 at sc16f (0.530 vs 0.375) and +0.466 at sc17f (0.856 vs 0.390): the
same weights cost more at sc17f. A calibration problem of the gate on unmixed text, not wrong
routing; the paper says so.

### Scale study: a Branch-Train-MiX baseline (pre-registered 2026-11-25, before any BTX code or runs)

Goal: compare the gate with the closest prior way of combining branched experts. BTX
(Sukhbaatar et al., 2024) merges experts branched from one seed into a mixture-of-experts model
and fine-tunes all of it; our experts at sc16f and sc17f are branched from a common seed, so
they are BTX's "branch-train" stage already, and only the "MiX" stage is new.

- **Merge (`btx.py`).** From sc<pt>_{lang,facts,math,code} (no circuit: BTX has no place for
  it). In every layer the four feed-forward blocks become the experts of a mixture-of-experts
  layer with a new linear router (d → 4, initialised N(0, 0.02²)); top-2 routing, the two
  weights a softmax over the two chosen router logits (as in BTX and Mixtral). All other
  weights (token and position embeddings, attention, layer norms) are the element-wise mean of
  the four experts'. Load-balancing loss (Switch Transformer form, E·Σ f_i P_i) with
  coefficient 0.01, as in BTX.
- **Fine-tune (`train.py --init`).** All parameters, the expert-size recipe without tuning:
  learning rate 1e-3, 2% warm-up, cosine to 0.1×, 8,192-token batches. Data: half mixed_train,
  half the four types in equal shares (BTX fine-tunes on its experts' domains; the mixed half
  is the data our gate sees). FLOPs per token count the active parameters only (attention,
  embeddings, two of the four feed-forward blocks, router) by `flops_per_token`'s formula.
- **Two budgets per point.**
  - btx: the FLOPs of the gate's training as recorded in train_sc<pt>f_gate_conf.json
    (1.01e15 at sc16f, 2.90e16 at sc17f). The system's total is then that of the gated system
    (experts + gate), so this is the compute-matched comparison.
  - btx10: ten times that. Its total exceeds the budget of the other systems (by about 18% at
    sc16f and 52% at sc17f; the exact figures are computed from the train files and reported);
    a generous reference, not compute-matched.
- **Measurements.** The same as for the other systems: evaluate.py (loss per type, recall,
  arithmetic, word problems) and diag_mixed.py on mixed_val and the held-out formats.
- **Comparisons, stated in advance; no pass/fail.** At each point, btx and btx10 against
  gate_conf (prob) and mono_s:
  - B1: loss on the inline and marked held-out formats (S1's quantity).
  - B2: loss on prose, code, facts and math.
  - B3: held-out-template recall.
  - B4: arithmetic (`arith_number`), also against gate_atom, which has the circuit.
  - Whatever the outcome, the paper reports it. If compute-matched BTX beats the gate on B1,
    the paper says so, and the case for the gate rests on what BTX cannot do without
    retraining (frozen experts that can be replaced, as in Track B step 4, and compiled
    circuits as experts).
- One seed. No router, learning-rate or data-mix tuning; if a run diverges, the deviation is
  recorded here before any rerun.
- **Checked before any fine-tuning (2026-11-25, CPU, 4 windows per set).** Sizes: sc16f BTX
  45.8M parameters, 27.0M active, 179 MFLOP/token (expert 123), so btx trains on 5.63M tokens
  and btx10 on 56.3M; sc17f 148.4M, 84.7M active, 549 MFLOP/token (expert 358): 52.9M and 529M
  tokens. The merged sc16f model before fine-tuning has a loss near that of a uniform guess
  (ln 8192 = 9.01): prose 8.93, code 9.59, mixed 9.38, against 3.59 (lang expert), 1.71 (code
  expert) and 3.81 (best single expert on mixed). The branches have moved too far from the seed
  for their attention weights to be averaged, so the fine-tuning has to rebuild the shared
  layers. This is part of the result (BTX's paper merges experts that stay closer to the seed),
  and the design is not changed for it.
- **Deviation (2026-11-25, before the first fine-tune step).** Micro-batch 4 sequences instead of 8
  (gradient accumulation 4 instead of 2; the same 8,192-token batch, schedule and FLOPs), so the
  fine-tune fits beside the other jobs on the shared GPU. The load-balancing loss is computed per
  micro-batch of 2,048 tokens. Wall-clock times of these runs are not comparable to the others.
