"""
Prototype: Token-Level Logit Blending (Level B) over Atom Spaces
Integrating a Hand-Weighted Transformer Adder (input_file_1.py) into a Base LM
via the Valued Atom Space Read Operator and Null-Atom Abstention.
"""

import numpy as np
import re
import adder as adder

# -----------------------------------------------------------------------------
# 1. Global Shared Vocabulary
# -----------------------------------------------------------------------------
# Standard alphanumeric vocabulary for natural language + arithmetic symbols
SHARED_VOCAB = list(
    "0123456789.+= ^$"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " ,:;()-\n"
)
SHARED_TOK = {c: i for i, c in enumerate(SHARED_VOCAB)}
V_SHARED = len(SHARED_VOCAB)


# -----------------------------------------------------------------------------
# 2. Expert Atom: Exact Symbolic Transformer Adder (input_file_1.py)
# -----------------------------------------------------------------------------
class ExactAdderAtom:
    """
    Parametric Expert Atom: omega_add in Omega_params.
    Value map v_x(omega_add) evaluated by the hand-crafted attention heads.
    """
    def __init__(self, sharpness=10.0):
        self.name = "ExactAdder (input_file_1)"
        self.sharpness = sharpness

    def detect_trigger(self, context: str):
        """Detects if context ends with an expression ready for evaluation."""
        pattern = r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*$"
        m = re.search(pattern, context)
        if m:
            a, b = m.group(1), m.group(2)
            pa, pb = adder.pad(a, b)
            prompt = f"^{pa} + {pb} ="
            ids = [adder.tok[c] for c in prompt]
            return {"active": True, "ids": ids, "digits_emitted": ""}
        return None

    def evaluate(self, context: str, session: dict):
        """
        Returns:
            similarity s_add: float (cosine / activation score)
            logits: np.ndarray of shape (V_SHARED,)
            session: updated session state
        """
        # If not already calculating, check if context triggers a new addition
        if session is None or not session.get("active", False):
            session = self.detect_trigger(context)

        # Inactive state: Abstain (Null Atom state, Section 3.4)
        if session is None or not session["active"]:
            return 0.0, np.zeros(V_SHARED), None

        # Active state: run forward pass through the hand-crafted transformer
        raw_logits = adder.forward(session["ids"])[-1]  # Shape (16,)
        best_char = adder.VOCAB[int(raw_logits.argmax())]

        # Check for end-of-calculation sentinel '$'
        if best_char == "$":
            # Hand calculation back to base LM: abstain immediately
            return 0.0, np.zeros(V_SHARED), None

        # Build projected logits into SHARED_VOCAB
        # Non-arithmetic tokens receive an algebraic veto (-50.0)
        projected_logits = np.full(V_SHARED, -50.0)
        for c, logit_val in zip(adder.VOCAB, raw_logits):
            if c in SHARED_TOK and c != "$":
                projected_logits[SHARED_TOK[c]] = logit_val * self.sharpness

        # High activation similarity (1.0) when actively generating digits
        return 1.0, projected_logits, session

    def advance(self, session: dict, char_emitted: str):
        """Appends the emitted character to the adder's internal transformer sequence."""
        if session and session.get("active", False):
            if char_emitted in adder.tok:
                session["ids"].append(adder.tok[char_emitted])
                session["digits_emitted"] += char_emitted
        return session


# -----------------------------------------------------------------------------
# 3. Base Language Model Atom (Context / Syntax)
# -----------------------------------------------------------------------------
class BaseLMAtom:
    """
    Base Atom: omega_base in Omega_ctx / Language Model.
    Generates syntax and discourse. Tends to approximate or hallucinate math.
    """
    def __init__(self):
        self.name = "BaseLM (Generative)"

    def evaluate(self, context: str):
        similarity = 0.7  # Baseline text generation affinity
        logits = np.zeros(V_SHARED)

        # Baseline language prior (space, common letters, punctuation)
        for c in " eatorinhsd":
            if c in SHARED_TOK:
                logits[SHARED_TOK[c]] = 2.0

        # Simulate base LM behavior after '=': it tries to write numbers, but fuzzy
        if context.strip().endswith("="):
            for d in "0123456789 ":
                logits[SHARED_TOK[d]] = 4.0
        elif any(context.endswith(s) for s in ["3.579", "100.000", "22.00"]):
            # Once numbers end, base LM produces continuation
            for c in " kg, which was expected.":
                if c in SHARED_TOK:
                    logits[SHARED_TOK[c]] = 6.0

        return similarity, logits


# -----------------------------------------------------------------------------
# 4. Atom Space Read Operator with Null-Atom Gating
# -----------------------------------------------------------------------------
class AtomSpaceRouter:
    """
    Implements:
      z(x) = sum_i lambda_i(x) * v(omega_i)
    with Proposition 1 null-atom abstention:
      lambda(x) = softmax([s_base, s_add, s_null] / tau)
    """
    def __init__(self, theta_null=0.5, tau=0.1):
        self.theta_null = theta_null  # Calibrated abstention threshold
        self.tau = tau                # Router temperature
        self.adder_atom = ExactAdderAtom()
        self.base_atom = BaseLMAtom()

    def generate(self, prompt: str, max_new_tokens: int = 15):
        context = prompt
        session = None
        print(f"\n{'='*75}\nPROMPT: \"{prompt}\"\n{'='*75}")

        for step in range(max_new_tokens):
            # 1. Query Expert Atom (omega_add)
            s_add, logits_add, session = self.adder_atom.evaluate(context, session)

            # 2. Query Base LM Atom (omega_base)
            s_base, logits_base = self.base_atom.evaluate(context)

            # 3. Query Null Atom (omega_null) - Proposition 1
            s_null = self.theta_null
            logits_null = np.zeros(V_SHARED)

            # 4. Gating distribution lambda in Delta^2
            scores = np.array([s_base, s_add, s_null]) / self.tau
            exp_scores = np.exp(scores - np.max(scores))
            lambdas = exp_scores / np.sum(exp_scores)
            lam_base, lam_add, lam_null = lambdas[0], lambdas[1], lambdas[2]

            # 5. Bochner Read Operator in logit space (Level B)
            blended_logits = (
                lam_base * logits_base +
                lam_add  * logits_add +
                lam_null * logits_null
            )

            # 6. Select token (argmax read)
            next_char_id = int(blended_logits.argmax())
            next_char = SHARED_VOCAB[next_char_id]

            # Advance internal transformer state
            session = self.adder_atom.advance(session, next_char)
            context += next_char

            # Display token step and gating distribution
            gate_info = f"[Base: {lam_base:5.3f} | Add: {lam_add:5.3f} | Null: {lam_null:5.3f}]"
            print(f"Step {step+1:02d} | Emitted: {repr(next_char):4s} | Gate: {gate_info} | Suffix: ...{repr(context[-20:])}")

            # Natural termination when base model finishes sentence
            if next_char == "." and (session is None or not session.get("active", False)):
                break

        print(f"\nFINAL OUTPUT:\n\"{context}\"")
        return context


# -----------------------------------------------------------------------------
# 5. Verification Scenarios
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    router = AtomSpaceRouter(theta_null=0.5, tau=0.1)

    # Test 1: Decimal addition with carry inside a natural language prompt
    router.generate("The recorded payload is 1.234 + 2.345 =", max_new_tokens=15)

    # Test 2: Stress-test long carry chain (99.999 + 0.001)
    router.generate("Experiment 42: Mass = 99.999 + 0.001 =", max_new_tokens=15)

    # Test 3: Natural language prompt without arithmetic (Adder must fully abstain)
    router.generate("The capital of France is Paris.", max_new_tokens=5)