# Reads over Heterogeneous Atom Spaces: Modular Logit-Level Integration of Algorithmic Circuits into Language Models

This repository provides an empirical and mathematical reference implementation of the **Valued Atom Space framework** for Large Language Models.

---

## 1. The Core Problem: The Heterogeneity Conflict in LLMs

Standard foundation models operate on the assumption that all information should be mapped into a single, uniform parametric tensor $\theta \in \mathbb{R}^P$. However, language processing encompasses three fundamentally incompatible inductive biases:

1. **Factual Data:** Demands localized, verbatim recall and instantaneous updates/deletions without retraining (*Right to be Forgotten*). Storing facts in dense weights causes parameter drift, catastrophic forgetting, and hallucinations.
2. **Mathematical & Algorithmic Skills:** Demands step-invariant state transitions, exact carry lookahead, and symbolic fidelity. Statistical memorization actively impairs length generalization.
3. **Natural Language Syntax:** Demands high-entropy, contextual distribution matching over discourse tokens.

### The Atom Space Resolution
Instead of forcing algorithmic logic and factual records into the dense weights of a base LLM, our framework structures the system as an **integration of measurable spaces (Valued Atom Spaces)**:
$$\Omega = \Omega_{\mathrm{ctx}} \cup \Omega_{\mathrm{corpus}} \cup \Omega_{\mathrm{params}} \cup \{\omega_\varnothing\}$$

Integration occurs at **Level B (Token-Level Logit Space)** through the Bochner Read Operator:
$$\ell_{\mathrm{read}}(x_{<t}) = \sum_{\omega \in \Omega} \lambda_\omega(x_{<t}) \, v_{x_{<t}}(\omega)$$
where $\lambda_\omega(x) \in \Delta^{|\Omega|-1}$ is a learned gate distribution and $v_{x}(\omega) \in \mathbb{R}^{|\mathcal{V}|}$ is the query-dependent value map.

---

## 2. Who Computes What? (Architecture of the Arithmetic Unit)

All mathematical reasoning is modularized into the parameter expert space $\Omega_{\mathrm{params}}$, requiring **zero fine-tuning of the base language model (GPT-2)**:

```
                          Prompt Context x_{<t}
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                                         ▼
      Base LM (GPT-2)                           Dual-Scale Gate
    Emits \ell_{base}(x)                     [h_last || h_win_mean]
              │                                         │
              │                                         ▼
              │                           \lambda(x_{<t}) in \Delta^{K-1}
              │                                         │
              │       ┌────────────────┬────────────────┴───────────────┐
              │       │                │                                │
              │   \lambda_base    \lambda_null                   \lambda_{op}
              │       │                │                                │
              │       ▼                ▼                                ▼
              │   [Base LM]       [Null Atom]              Unified Arithmetic Bank
              │                    v(\omega) = 0             (\Omega_params)
              │                                                         │
              │                                      ┌──────────┬───────┴──┬──────────┐
              │                                      ▼          ▼          ▼          ▼
              │                                  \omega_+    \omega_-   \omega_*   \omega_cmp
              │                                  (Adder)    (Subtr.)    (Mult.)    (Comp.)
              │                                      └──────────┼──────────┘          │
              │                                                 ▼                     ▼
              │                                        \ell_{math} \in R^{|V|}        │
              │                                        (+100 on certified token,      │
              │                                         -100 on all other tokens)     │
              │                                                 │                     │
              └────────────────────────┬────────────────────────┘                     │
                                       ▼                                              │
                            Level B Logit Blending:                                   │
           \ell_{comb} = \lambda_base \ell_base + \lambda_op \ell_{math}              │
                                       │                                              │
                                       ▼                                              │
                     Next Token: y_t = argmax(\ell_{comb})                            │
```

### Operational Breakdown

| Operator | Active Atom | Mechanism | Description |
| :--- | :---: | :--- | :--- |
| **Addition (`+`)** | $\omega_{\mathrm{add}}$ | **Causal Carry-Lookahead Transformer** | Uses an embedded transformer circuit to perform prefix-sum lookahead over decimals of arbitrary precision. It emits next-digit logits and ends with the sentinel `'$'`. |
| **Subtraction (`-`)** | $\omega_{\mathrm{sub}}$ | **Ten's Complement Reduction** | Directly reduces subtraction to addition: $a - b = a + \mathrm{comp}_{10}(b)$. It evaluates differences using the exact same adder circuit without dedicated subtraction weights. |
| **Multiplication (`*`)** | $\omega_{\mathrm{mul}}$ | **Partial Product Accumulation** | Decomposes multiplication into scalar-digit partial products and accumulates them iteratively through the adder engine. |
| **Comparison (`cmp`)** | $\omega_{\mathrm{cmp}}$ | **First-Mismatch Evaluator** | Locates the most significant digit mismatch across $>, <, ==, \ge, \le$ and emits certified booleans (`" True"` or `" False"`). |
| **Abstention** | $\omega_\varnothing$ | **Null Atom (Proposition 1)** | When no arithmetic is active or when an equation finishes, mass routes to $\omega_\varnothing$ ($v(\omega_\varnothing) = \mathbf{0}$). This drops expert logits and returns full control to the Base LM. |

---

## 3. Mathematical Foundations

### A. Level B Logit Blending (Continuous Product of Experts)
When expert values are expressed as logits with sharp certainty ($+C$ on certified tokens, $-C$ on all other vocabulary elements), the read operator acts as an **algebraic veto**:
$$p(y_t \mid x_{<t}) \propto p_{\mathrm{base}}(y_t)^{\lambda_{\mathrm{base}}} \cdot \prod_{\omega \in \Omega_{\mathrm{math}}} p_{\omega}(y_t)^{\lambda_\omega}$$
If the base model hallucinates an incorrect digit, the expert atom's logit of $-100.0$ imposes an absolute veto, guaranteeing mathematically certified outputs without vocabulary corruption.

### B. Proposition 1: Calibrated Null-Atom Abstention
A null atom $\omega_\varnothing$ carries value $v(\omega_\varnothing) = \mathbf{0}$ and reference kernel $K_x(\omega_\varnothing) = e^{\theta / \tau}$. The effective activation multiplier satisfies:
$$M_x = \frac{Z_x}{Z_x + m_\varnothing e^{\theta/\tau}}$$
* If arithmetic similarity exceeds threshold $\theta$, $M_x \to 1$ and the specialist atom executes.
* If similarity drops below threshold $\theta$, $M_x \to 0$ and the specialist atom completely abstains.

### C. Dual-Scale Context Windowing
At prompt boundaries ending in `'='`, single-token representations $h_t$ diffuse the operator identity (`+` vs `-`). The router uses a **dual-scale concatenation**:
$$h_{\mathrm{input}} = \left[ h_t \;\Big\|\; \frac{1}{W} \sum_{k=0}^{W-1} h_{t-k} \right] \in \mathbb{R}^{2 \cdot d_{\mathrm{model}}}$$
where $h_t$ provides the boundary trigger and $\bar{h}_{\mathrm{win}}$ preserves the operator identity.

### D. Session Latching & Consumption Guards
1. **Atomic Invariance:** Once the learned gate activates an operator at the boundary `'='`, it latches the session. The atom retains control until emitting its termination sentinel (`'$'`), preventing the sliding context window from dropping the operator mid-number.
2. **Consumption Guard:** Upon sentinel emission, the completed equation is flagged as consumed. The gate returns to $\lambda_{\mathrm{base}} = 1.0$, preventing completed expressions in the history from re-triggering.

---

## 4. Execution & Empirical Verification

### Setup
```bash
pip install torch transformers numpy
python atom_space_engine.py
```

### Empirical Trace Analysis

#### Test 1: Addition (`1.234 + 2.345 =`)
```text
Step 01 | Emitted: ' '    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 02 | Emitted: '3'    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 03 | Emitted: '.'    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 04 | Emitted: '5'    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 05 | Emitted: '7'    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 06 | Emitted: '9'    | Learned Gate: [Base: 0.000 | +: 1.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 07 | Emitted: '.'    | Learned Gate: [Base: 1.000 | +: 0.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
Step 08 | Emitted: '\n'   | Learned Gate: [Base: 1.000 | +: 0.000 | -: 0.000 | cmp: 0.000 | *: 0.000]
FINAL OUTPUT: "The telemetry distance is 1.234 + 2.345 = 3.579."
```
* **Boundary:** $\lambda_+ = 1.000$ fires at the prompt edge.
* **Execution:** Carry lookahead produces exact digits `3.579`.
* **Handover:** The consumption guard marks the equation satisfied, and GPT-2 naturally continues without re-triggering.

#### Test 2: Subtraction (`14.25 - 7.90 =`)
```text
Step 01–05: Emitted ' 6.35' | Learned Gate: -: 1.000
Step 06:    Emitted '%'     | Learned Gate: Base: 1.000
Step 07–10: Emitted ' of the total tank' | Learned Gate: Base: 1.000
FINAL OUTPUT: "The tank depletion is 14.25 - 7.90 = 6.35% of the total tank"
```
* **Semantic Continuity:** Evaluates `6.35` via Ten's Complement, after which GPT-2 supplies the contextual suffix `"% of the total tank"`.

#### Test 3: Multiplication (`12.5 * 2.0 =`)
```text
Step 01–06: Emitted ' 25.00' | Learned Gate: *: 1.000
Step 07–10: Emitted '\n\n\nThe' | Learned Gate: Base: 1.000
FINAL OUTPUT: "Calculated power value is 12.5 * 2.0 = 25.00"
```

#### Test 4: Pure Natural Language (Guaranteed Protection)
```text
PROMPT: "The solar system consists of eight planets and"
Steps 01–08: Emitted " a core of about 1,000 solar" | Learned Gate: Base: 1.000 throughout
```
* **Provable Non-Interference:** When arithmetic is absent, math gates remain at $0.000$, allowing unconstrained text generation.
