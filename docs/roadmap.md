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
| Tracks B, C, D, D2, ablations | not started |
| Draft | outline, abstract skeleton, Related Work and Section 4 written; Sections 5–9 are TODO skeletons |

Reproduce Track A: `python experiments/track_a/run_correctness.py` (plus `--suite exhaustive3`, about 4.5 h on 15 workers),
`python experiments/track_a/run_length_generalisation.py --systems compiled python qwen0.5b qwen1.5b`,
`python experiments/track_a/run_cost.py`, then `python scripts/make_tables.py` and `latexmk -pdf main_v2.tex` in `paper/`.
