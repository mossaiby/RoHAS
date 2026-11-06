"""
Modular Atom Bank with GPT-2 as omega_base (Base Language Model).
Combines:
  - omega_base: Hugging Face GPT-2 (124M)
  - omega_add : Exact hand-crafted transformer adder from input_file_1.py
  - omega_sub : Exact subtraction oracle
  - omega_cmp : Exact comparison evaluator
  - omega_mul : Exact multiplication oracle
  - omega_null: Proposition 1 Abstention Atom (Section 3.4)
"""

import re
import numpy as np
from decimal import Decimal, getcontext
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import the hand-crafted addition transformer from input_file_1.py
import adder as adder

getcontext().prec = 100

# -----------------------------------------------------------------------------
# 1. Initialize GPT-2 Backbone and Tokenizer
# -----------------------------------------------------------------------------
print("Loading GPT-2 backbone...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
base_model.eval()

V_SIZE = tokenizer.vocab_size  # 50,257 for standard GPT-2


def get_token_id_for_char(char: str) -> int:
    """Finds the exact single-token BPE ID in GPT-2 for a given character."""
    tokens = tokenizer.encode(char, add_special_tokens=False)
    if len(tokens) == 1:
        return tokens[0]
    # Fallback to byte encoding for raw characters
    encoded = tokenizer.convert_tokens_to_ids(char)
    return encoded if encoded is not None else tokens[0]


def make_sharp_logits(target_char: str, confidence: float = 100.0, veto: float = -100.0) -> np.ndarray:
    """Builds a logit vector across all 50,257 GPT-2 tokens, imposing an algebraic veto."""
    logits = np.full(V_SIZE, veto, dtype=np.float32)
    tok_id = get_token_id_for_char(target_char)
    logits[tok_id] = confidence
    return logits


# -----------------------------------------------------------------------------
# 2. Specialist Atoms in Omega_math
# -----------------------------------------------------------------------------
class AdditionAtom:
    """omega_add: Exact 6-layer attention transformer from input_file_1.py"""
    def __init__(self):
        self.symbol = "+"
        self.name = "ExactAdder (input_file_1)"

    def detect_trigger(self, context: str):
        pattern = r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*$"
        m = re.search(pattern, context)
        if m:
            a, b = m.group(1), m.group(2)
            pa, pb = adder.pad(a, b)
            prompt = f"^{pa} + {pb} ="
            ids = [adder.tok[c] for c in prompt]
            return {"active": True, "ids": ids, "output_emitted": ""}
        return None

    def step(self, context: str, session: dict):
        if session is None or not session.get("active", False):
            session = self.detect_trigger(context)

        if session is None or not session["active"]:
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        # Run forward pass through input_file_1 hand-crafted transformer
        raw_logits = adder.forward(session["ids"])[-1]
        best_char = adder.VOCAB[int(raw_logits.argmax())]

        # Check for end-of-calculation sentinel '$'
        if best_char == "$":
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        # Build projected logit vector over the 50,257 GPT-2 vocabulary
        logits = np.full(V_SIZE, -100.0, dtype=np.float32)
        target_id = get_token_id_for_char(best_char)
        logits[target_id] = 100.0

        return 1.0, logits, session

    def advance(self, session: dict, char_emitted: str):
        if session and session.get("active", False):
            if char_emitted in adder.tok:
                session["ids"].append(adder.tok[char_emitted])
                session["output_emitted"] += char_emitted
        return session


class SubtractionAtom:
    """omega_sub: Exact subtraction oracle with borrow resolution."""
    def __init__(self):
        self.symbol = "-"

    def detect_trigger(self, context: str):
        pattern = r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=\s*$"
        m = re.search(pattern, context)
        if m:
            diff = Decimal(m.group(1)) - Decimal(m.group(2))
            diff_str = f" {diff:f}"
            return {"active": True, "target_str": diff_str, "step_idx": 0}
        return None

    def step(self, context: str, session: dict):
        if session is None or not session.get("active", False):
            session = self.detect_trigger(context)

        if session is None or not session["active"]:
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        idx = session["step_idx"]
        target = session["target_str"]
        if idx >= len(target):
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        target_char = target[idx]
        return 1.0, make_sharp_logits(target_char), session

    def advance(self, session: dict, char_emitted: str):
        if session and session.get("active", False):
            session["step_idx"] += 1
        return session


class ComparisonAtom:
    """omega_cmp: Exact numerical comparison evaluator."""
    def __init__(self):
        self.symbol = "cmp"

    def detect_trigger(self, context: str):
        pattern = r"(\d+\.?\d*)\s*(==|>=|<=|>|<)\s*(\d+\.?\d*)\s*(=\s*|\?\s*)$"
        m = re.search(pattern, context)
        if m:
            a_val, op, b_val = Decimal(m.group(1)), m.group(2), Decimal(m.group(3))
            ops = {
                ">": a_val > b_val,
                "<": a_val < b_val,
                "==": a_val == b_val,
                ">=": a_val >= b_val,
                "<=": a_val <= b_val,
            }
            res_str = f" {str(ops[op])}"
            return {"active": True, "target_str": res_str, "step_idx": 0}
        return None

    def step(self, context: str, session: dict):
        if session is None or not session.get("active", False):
            session = self.detect_trigger(context)

        if session is None or not session["active"]:
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        idx = session["step_idx"]
        target = session["target_str"]
        if idx >= len(target):
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        return 1.0, make_sharp_logits(target[idx]), session

    def advance(self, session: dict, char_emitted: str):
        if session and session.get("active", False):
            session["step_idx"] += 1
        return session


class MultiplicationAtom:
    """omega_mul: Exact arbitrary-precision multiplication oracle."""
    def __init__(self):
        self.symbol = "*"

    def detect_trigger(self, context: str):
        pattern = r"(\d+\.?\d*)\s*[\*x]\s*(\d+\.?\d*)\s*=\s*$"
        m = re.search(pattern, context)
        if m:
            prod = Decimal(m.group(1)) * Decimal(m.group(2))
            prod_str = f" {prod:f}"
            return {"active": True, "target_str": prod_str, "step_idx": 0}
        return None

    def step(self, context: str, session: dict):
        if session is None or not session.get("active", False):
            session = self.detect_trigger(context)

        if session is None or not session["active"]:
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        idx = session["step_idx"]
        target = session["target_str"]
        if idx >= len(target):
            return 0.0, np.zeros(V_SIZE, dtype=np.float32), None

        return 1.0, make_sharp_logits(target[idx]), session

    def advance(self, session: dict, char_emitted: str):
        if session and session.get("active", False):
            session["step_idx"] += 1
        return session


# -----------------------------------------------------------------------------
# 3. Base Language Model Atom (Hugging Face GPT-2)
# -----------------------------------------------------------------------------
class GPT2BaseAtom:
    """omega_base in Omega_ctx: Real causal LM generating vocabulary logits."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, context: str):
        similarity = 0.7  # Baseline text continuation affinity
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :].cpu().numpy().astype(np.float32)

        return similarity, logits


# -----------------------------------------------------------------------------
# 4. Atom Space Read Router (Level B Token-Level Blending)
# -----------------------------------------------------------------------------
class GPT2AtomSpaceRouter:
    """
    R(x; Omega) = lambda_base * logits_base + sum_m lambda_m * logits_m + lambda_null * 0
    """
    def __init__(self, theta_null: float = 0.5, tau: float = 0.1):
        self.theta_null = theta_null
        self.tau = tau
        self.base_atom = GPT2BaseAtom(base_model, tokenizer)
        self.math_bank = [
            AdditionAtom(),
            SubtractionAtom(),
            ComparisonAtom(),
            MultiplicationAtom()
        ]

    def generate(self, prompt: str, max_new_tokens: int = 15):
        context = prompt
        sessions = [None] * len(self.math_bank)

        print(f"\n{'='*80}\nINPUT PROMPT: \"{prompt}\"\n{'='*80}")

        for step in range(max_new_tokens):
            atom_names = ["GPT2", "Null"]
            scores = []
            logit_vectors = []

            # 1. Compute Base LM Logits from GPT-2
            s_base, l_base = self.base_atom.evaluate(context)
            scores.append(s_base)
            logit_vectors.append(l_base)

            # 2. Null Atom (Proposition 1 Abstention: value = 0)
            scores.append(self.theta_null)
            logit_vectors.append(np.zeros(V_SIZE, dtype=np.float32))

            # 3. Query all Specialist Atoms
            for i, atom in enumerate(self.math_bank):
                s_i, l_i, sessions[i] = atom.step(context, sessions[i])
                scores.append(s_i)
                logit_vectors.append(l_i)
                atom_names.append(atom.symbol)

            # 4. Learned Gate Distribution lambda
            scores_arr = np.array(scores) / self.tau
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            lambdas = exp_scores / np.sum(exp_scores)

            # 5. Level B Bochner Read in Logit Space
            blended_logits = np.zeros(V_SIZE, dtype=np.float32)
            for lam, l_vec in zip(lambdas, logit_vectors):
                blended_logits += lam * l_vec

            # 6. Sample argmax token from the blended distribution
            next_token_id = int(blended_logits.argmax())
            next_text = tokenizer.decode([next_token_id])

            # Advance state in all active atoms
            for i, atom in enumerate(self.math_bank):
                sessions[i] = atom.advance(sessions[i], next_text)

            context += next_text

            # Print gate distribution trace
            gate_str = " | ".join([f"{name}: {lam:5.3f}" for name, lam in zip(atom_names, lambdas)])
            print(f"Token {step+1:02d} | Emitted: {repr(next_text):6s} | Gate: [{gate_str}]")

            # Check if GPT-2 produced EOS or sentence end when math is inactive
            any_active = any(s is not None and s.get("active", False) for s in sessions)
            if not any_active and next_token_id == tokenizer.eos_token_id:
                break

        print(f"\nFINAL GENERATED TEXT:\n\"{context}\"")
        return context


# -----------------------------------------------------------------------------
# 5. Execution Test Suite
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    router = GPT2AtomSpaceRouter(theta_null=0.5, tau=0.1)

    # Test 1: Addition with carry through the 6-layer hand-crafted transformer
    router.generate("The mass calculation is 1.234 + 2.345 =", max_new_tokens=15)

    # Test 2: Subtraction with borrow
    router.generate("The remaining capacity is 14.25 - 7.90 =", max_new_tokens=15)

    # Test 3: Comparison logic
    router.generate("Is threshold valid? 99.5 > 100.2 =", max_new_tokens=15)

    # Test 4: Pure natural language prose (Math atoms must completely abstain)
    router.generate("The theory of general relativity describes gravity as", max_new_tokens=12)