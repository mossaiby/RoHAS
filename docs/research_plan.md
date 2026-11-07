# Research plan: heterogeneous experts under a learned token-level gate

## 1. The claim to test

> Data types differ in what training should do to them: facts should be stored
> verbatim and be removable; algorithmic skills should generalise beyond the training
> lengths, and memorising hurts them; language and code need distribution matching.
> Training one expert per data type, each with its own regime, and combining them with
> a learned per-token gate λ(x) (with a null atom for abstention), matches or beats one
> monolithic model at equal training compute. Any expert can also be replaced, retrained
> or added without touching the others.

The paper needs evidence for three separable statements:

- **H1 (specialisation):** at matched total compute, per-type experts + gate ≥ monolithic
  model, per type and on mixed text.
- **H2 (modularity):** replacing or upgrading one expert changes performance only on its
  own type, and the gate needs zero or a few hundred steps to adapt.
- **H3 (frozen backbone):** when the backbone cannot be changed (weights in silicon),
  external atoms and anti-atoms provide insertion, update and removal of facts, at a
  measured cost.

The arithmetic circuits (`circuits/`) are the extreme case of H1: an expert with zero
training cost and exact length generalisation.

## 2. Prior work to position against

| Line | Representative work | How this differs |
|---|---|---|
| Train domain experts separately, then merge | BTM (Li et al. 2022), c-BTM (Gururangan et al. 2023), BTX (Sukhbaatar et al. 2024), DEMix (Gururangan et al. 2022) | Those split by *domain* with one training regime; here the split is by *data type* with different regimes (memorisation vs anti-memorisation), plus compiled experts and abstention. BTX already uses math/code/wiki experts, so it is the baseline to beat. |
| Token-level collaboration between separate LMs | Co-LLM (Shen et al. 2024), CALM (Bansal et al. 2024), DExperts (Liu et al. 2021), proxy-tuning (Liu et al. 2024) | Co-LLM learns per-token deferral between a base and an expert model, which is the closest mechanism. Compare against it directly. |
| Routing whole queries to models | Zooter, LLM-Blender, Mixture-of-Agents | These route per query, not per token. |
| Compiled / hand-written transformers | RASP (Weiss et al. 2021), Tracr (Lindner et al. 2023), weight-space programming (Takehi et al. 2026) | Here compiled circuits run as experts *inside* an LLM's decoding loop. |
| Arithmetic in transformers | Abacus embeddings (McLeish et al. 2024), scratchpads, PAL / PoT, Toolformer | A compiled expert vs a trained arithmetic expert vs a Python tool. |
| Tools as first-class output elements | ToolkenGPT (Hao et al. 2023), ToolGen (Wang et al. 2024); Toolformer, ReAct, Gorilla, ToolLLM; grammar/JSON-constrained function calling | One-token-per-tool is not new, so do not claim "tools as atoms" itself. The claim is that tools sit in the same calibrated read as memory and parameter experts: one gate across spaces, a null atom for "no tool fits", anti-atom revocation, fallback to memory under the margin certificate, and atoms calling atoms (the parser atom). |
| Editing / unlearning | ROME, MEMIT, GRACE, SERAC, WISE; TOFU (Maini et al. 2024) | The frozen-hardware setting; logit- and hidden-level anti-atoms. |

## 3. Experiments

Hardware assumed: one RTX 3050 Ti (4 GB). Track A runs on CPU; B and C fit with 4-bit
models of 1.5B parameters or less; D is sized for it.

### A. Compiled arithmetic experts (done in part)

- **Correctness:** `tests/test_circuits.py`, random signed decimals vs `Decimal`. For the
  paper, report exhaustive checks up to 3 digits and random checks up to 200 digits per
  operation, with zero errors.
- **Length generalisation curve:** accuracy against operand length (1–100 digits) for the
  compiled expert, the base LLM alone, a fine-tuned arithmetic LM (Abacus-style, trained
  up to 20 digits), and a Python tool. The compiled curve is flat at 100% by construction;
  the point is the comparison.
- **Cost:** parameter count (dense entries / nonzeros: add 11k / 0.5k, sub 18k / 1k,
  mul 310k / 7.3k, div 100k / 4.2k),
  scratchpad tokens per operation (mul ≈ m·(2n+3m+1), div ≈ 11n·(2n+m+4) at worst), and
  wall-clock time.

- **Experts calling experts:** the parser atom (`circuits/parser.py`) is itself a compiled
  transformer. It finds the innermost group and the operator with the highest precedence
  without counting brackets, then emits a call block that the latch routes to the
  arithmetic atoms. Report rows and tokens against expression size and nesting depth
  (depth-5 balanced trees: 31 calls, about 5.5k tokens), and test against a reference
  evaluator with identical truncating-division semantics (`tests/test_parser.py`). This
  is the first step from "arithmetic expert" toward the "compiled program" direction.

### B. Pretrained experts that share a tokenizer (main experiment for H1 and H2)

Qwen2.5 checkpoints share one tokenizer, so their logits can be mixed token by token:

- language: Qwen2.5-0.5B or 1.5B (base)
- code: Qwen2.5-Coder-0.5B or 1.5B
- math: Qwen2.5-Math-1.5B, plus the compiled circuits
- facts: corpus atoms (RoAS read) and, as a parametric alternative, a 0.5B model fine-tuned
  verbatim on the fact set

Pitfall: the padded vocabulary size differs between checkpoints. Slice logits to the
shared tokenizer range before mixing.

**Gate.** Input: hidden states of the smallest model (cheap, and it allows skipping
experts that receive zero mass). Output: λ over {language, code, math, circuits,
corpus, null}. Train it without routing labels by minimising the negative log-likelihood
of the mixture on mixed text. Compare three mixture forms: probability mixture
(log-sum-exp, the proper latent-variable model), logit mixture (product of experts, the
paper's Level B), and hard top-1 routing.

**Evaluation data** (all public):

| Type | Benchmarks | Metric |
|---|---|---|
| language | WikiText-103 and a C4 slice | perplexity; gate firing rate on prose (false triggers) |
| code | HumanEval, MBPP (sanitised) | pass@1 |
| math | GSM8K, GSM-Hard, and a templated set with arithmetic embedded in text | exact match |
| facts | CounterFact, zsRE | efficacy / paraphrase / neighbourhood (reuse the RoAS scorer) |
| mixed | documents that interleave prose, code and numbers (build ~1k by concatenation; also natural ones such as StackExchange posts with code) | perplexity, token-level routing accuracy against oracle routing |

**Baselines:** each expert alone; the largest single model at equal total parameters
(e.g. Qwen2.5-3B against three 1B experts); uniform mixture; oracle routing (upper
bound); query-level routing; Co-LLM-style learned deferral; and BTX if a merged
checkpoint is affordable (otherwise cite its numbers).

**Modularity (H2):** swap Qwen2.5-Coder-0.5B for 1.5B, and Qwen2.5-Math-1.5B for the
compiled circuits. Report (i) the change on the swapped type, (ii) the change on all other
types (should be ≈0), and (iii) gate retraining steps needed (0, 100, 1000).

### C. Training from scratch (the cleanest test of H1)

Four small models of the same size (≈10–30M parameters, shared BPE tokenizer), trained
separately on the same token budget per type:

| Expert | Data | Regime |
|---|---|---|
| language | TinyStories or a FineWeb-Edu slice | standard LM; 1 epoch; dropout |
| facts | synthetic biographies (in the style of Allen-Zhu & Li, *Physics of LMs 3.1*), with paraphrased augmentations | many epochs, no dropout, train to memorisation; test with held-out paraphrases |
| math | synthetic arithmetic and templated word problems | abacus or index-hint positions, length curriculum, anti-memorisation (fresh samples every step, no repetition) |
| code | the-stack-smol (Python) | standard LM |

Monolithic baseline: one model with 4× parameters (and, separately, 1× parameters with
4× steps), trained on the shuffled mixture with the same total compute. Report per-type
loss and accuracy, mixed-text perplexity, fact recall vs paraphrase generality, and
arithmetic accuracy beyond the training length. This is the experiment that can show
the data-type argument rather than assert it. It also measures forgetting directly:
append new facts by training only the fact expert, and compare with continued training
of the monolithic model.

### D. Frozen backbone: insertion, update and removal (H3)

Setting: the backbone's weights cannot change (weights in silicon). It is either
observable at the last hidden state or only at the logits.

- **Removal:** TOFU forget/retain splits. Anti-atoms at the logit level (when a forget-set
  query fires, push the fact's answer tokens down) and at the hidden level (RoAS read with
  negative mass). Metrics: TOFU forget quality, model utility, paraphrase leakage
  (the hard case, since the base still holds the fact), and neighbour damage. Baselines:
  gradient-ascent unlearning and NPO (Zhang et al. 2024). These change weights, so they
  are upper references, not competitors.
- **Update:** CounterFact and MQuAKE (multi-hop: does the update propagate?).
  Compare append-plus-cancel against append-only.
- Report the inference overhead per query and the storage per removed fact.

### D2. Tools in the same read (Ω_tools)

The current engine routes tools with an if/else on similarity and a constant mock
barometer, so it is an illustration, not evidence. To make it evidence:

- **Routing:** tool selection goes through the learned gate λ(x), together with corpus,
  parameter and null atoms; tool keys are unit-normalised embeddings of tool
  descriptions, as for corpus records.
- **Choosing a tool:** API-Bank or ToolBench (pick the right tool and arguments among many).
  Baselines: prompt-based function calling with the same base model, a
  ToolkenGPT-style learned tool-token head, and retrieve-then-call (Gorilla/ToolLLM-style
  retriever).
- **Abstention:** the BFCL irrelevance-detection category (no listed tool fits). It tests
  the null atom directly; report the calibrated θ from held-out data, as in RoAS.
- **Revocation:** revoke k of N tools by anti-atom vs by removal from an allowlist.
  Measure calls to revoked tools (should be 0), what near-miss queries receive instead
  (abstention vs a wrong neighbouring tool, the anti-atom vs deletion difference of
  RoAS Proposition 3), and reversibility.
- **Fallback chain:** tool → stored record → abstain. Measure how often the margin
  certificate covers the fallback and how often the fallback answer is correct.
- **Security:** if claimed, AgentDojo (prompt injections delivered through tool outputs),
  comparing revocation and gating against prompt-level defences.
- **Atoms calling atoms:** the parser atom's calls to the arithmetic atoms are a small,
  exactly checkable instance of nested tool use. Report call counts and correctness
  against the reference evaluator, and use it as the controlled case before nested calls
  between real tools.

### E. Ablations (for the paper's mechanism claims)

Gate input (last hidden state / mean / max / all three), window size W, session latching
on or off, consumption guard on or off, expert logit scale C and gate temperature, null
atom on or off (abstention vs forced answer on out-of-scope prompts), and soft vs hard
mixing.

### F. Cost accounting (the motivation is training cost, so measure it)

Training FLOPs: experts plus gate vs monolithic; the cost of adding one new capability
(train one expert + gate steps vs retrain or fine-tune the monolith); inference latency
and memory per token with k experts (dense, or sparse with skipping when λ = 0).

## 4. Suggested paper order

1. Track C (from scratch). It is cheap, controlled, and directly tests the data-type claim.
2. Track B with two experts (language + code) before adding math, the circuits and facts.
3. Track D on TOFU, then the tools track D2 (BFCL irrelevance, revocation, fallback).
4. Track A as a short section on compiled experts plus length generalisation.

## 5. Ideas from weight-space programming (Takehi et al., 2026)

- **LLM-authored experts.** The post shows a frontier LLM writing working network weights
  zero-shot. The hand-written circuits here are a small instance of that. A study could
  have an LLM write circuits for new closed-world skills (dates, unit conversion,
  checksums, parsing), verify them exhaustively, and register them as atoms: a
  "zero-training expert" pipeline.
- **Direction subtraction as an anti-atom.** Their Qwen3.5-0.8B edit subtracts one
  direction from two MLP outputs. For a backbone in silicon with an exposed residual
  stream, a stored direction is a natural hidden-level anti-atom (like a steering vector).
  Measuring it against the logit-level anti-atom on TOFU is a clean comparison.
- **Reading weights to build the gate.** If an LLM can identify what a network does from
  its weights, an expert registry can describe new experts automatically, and the gate's
  prior over λ can be set before any routing data exists.
