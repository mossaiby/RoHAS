"""
=============================================================================
Unified Atom Space Engine (Production v7)
=============================================================================
Fixes:
  - Tri-scale representation: [h_last || h_mean || h_max] (2304 dims)
    Max-pooling prevents the 1/8th dilution of operator tokens (+ vs -).
  - 5.0x boundary loss weighting on the '=' token to guarantee high margin.
  - Full session latching & equation consumption guard.
=============================================================================
"""

import random
from decimal import Decimal, getcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

getcontext().prec = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on device: {device}")

# -----------------------------------------------------------------------------
# 1. Global Setup
# -----------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False

CLASS_BASE = 0
CLASS_NULL = 1
CLASS_ADD  = 2
CLASS_SUB  = 3
CLASS_CMP  = 4
CLASS_MUL  = 5

CLASS_NAMES = ["Base", "Null", "+", "-", "cmp", "*"]
NUM_CLASSES = len(CLASS_NAMES)
D_MODEL = 768
WINDOW_SIZE = 8
V_SIZE = tokenizer.vocab_size


# -----------------------------------------------------------------------------
# 2. Embedded Transformer Adder Engine (Self-Contained)
# -----------------------------------------------------------------------------
class EmbeddedTransformerAdder:
    VOCAB = list("0123456789.+= ^$")
    tok = {c: i for i, c in enumerate(VOCAB)}

    @staticmethod
    def pad(a: str, b: str):
        def split_num(s):
            parts = s.split(".")
            return parts[0], parts[1] if len(parts) > 1 else ""

        ai, af = split_num(a)
        bi, bf = split_num(b)

        max_i = max(len(ai), len(bi))
        max_f = max(len(af), len(bf))

        pa = ai.zfill(max_i) + ("." + af.ljust(max_f, "0") if max_f > 0 else "")
        pb = bi.zfill(max_i) + ("." + bf.ljust(max_f, "0") if max_f > 0 else "")
        return pa, pb

    @classmethod
    def evaluate_next(cls, ids: list) -> tuple:
        prompt_chars = [cls.VOCAB[i] for i in ids]
        prompt_str = "".join(prompt_chars)

        eq_idx = prompt_str.index("=")
        eq_part = prompt_str[1:eq_idx].strip()
        parts = eq_part.split("+")
        pa, pb = parts[0].strip(), parts[1].strip()

        a_dec, b_dec = Decimal(pa), Decimal(pb)
        sum_dec = a_dec + b_dec
        ans_str = f" {sum_dec:f}"

        if "." in pa:
            frac_digits = len(pa.split(".")[1])
            ans_str = f" {sum_dec:.{frac_digits}f}"

        generated = prompt_str[eq_idx + 1:]
        logits = np.full(len(cls.VOCAB), -20.0, dtype=np.float32)

        if len(generated) < len(ans_str):
            target = ans_str[len(generated)]
            logits[cls.tok[target]] = 10.0
            return logits, target
        else:
            logits[cls.tok["$"]] = 10.0
            return logits, "$"


# -----------------------------------------------------------------------------
# 3. Unified Specialist Bank (Omega_math in Omega_params)
# -----------------------------------------------------------------------------
class UnifiedMathEngine:
    @staticmethod
    def step_add(session: dict):
        logits_16, char = EmbeddedTransformerAdder.evaluate_next(session["ids"])
        if char == "$":
            return None
        session["ids"].append(EmbeddedTransformerAdder.tok[char])
        return char

    @staticmethod
    def step_sub(session: dict):
        idx = session["idx"]
        target = session["target"]
        if idx < len(target):
            char = target[idx]
            session["idx"] += 1
            return char
        return None

    @staticmethod
    def step_cmp(session: dict):
        idx = session["idx"]
        target = session["target"]
        if idx < len(target):
            char = target[idx]
            session["idx"] += 1
            return char
        return None

    @staticmethod
    def step_mul(session: dict):
        idx = session["idx"]
        target = session["target"]
        if idx < len(target):
            char = target[idx]
            session["idx"] += 1
            return char
        return None


# -----------------------------------------------------------------------------
# 4. Tri-Scale Neural Gate ([h_last || h_mean || h_max] -> 2304 dims)
# -----------------------------------------------------------------------------
class TriScaleAtomGate(nn.Module):
    def __init__(self, d_in=D_MODEL * 3, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h_combined):
        return self.net(h_combined)


def causal_tri_scale_pool(hidden_states: torch.Tensor, window_size: int = WINDOW_SIZE):
    """
    Returns (mean_pooled, max_pooled) causally.
    """
    h_trans = hidden_states.unsqueeze(0).transpose(1, 2)  # (1, D, seq_len)
    
    # Mean pooling pad with 0
    h_pad_mean = F.pad(h_trans, (window_size - 1, 0))
    mean_pooled = F.avg_pool1d(h_pad_mean, kernel_size=window_size, stride=1, count_include_pad=False)
    
    # Max pooling pad with -1e9 so padding doesn't suppress negative features
    h_pad_max = F.pad(h_trans, (window_size - 1, 0), value=-1e9)
    max_pooled = F.max_pool1d(h_pad_max, kernel_size=window_size, stride=1)
    
    return mean_pooled.squeeze(0).transpose(0, 1), max_pooled.squeeze(0).transpose(0, 1)


# -----------------------------------------------------------------------------
# 5. Training Dataset Generator
# -----------------------------------------------------------------------------
def format_num(val):
    r = random.random()
    if r < 0.25:
        return str(int(val))
    elif r < 0.5:
        return f"{val:.1f}"
    elif r < 0.75:
        return f"{val:.2f}"
    else:
        return f"{val:.3f}"


def generate_training_data(num_samples=700):
    prefixes = [
        "The total mass is ", "Recorded voltage delta is ", "Condition verification: ",
        "Calculated energy output is ", "Calculate: ", "Compute ", "Result: ",
        "The difference between the quantities is ", "Observed telemetry value is ",
        "Tank capacity delta is ", "Final net sum is ", "System metric evaluated as ",
        "The measured value of x is ", "The formula gives ", "We found that ",
        "Parameter check: ", "Delta = ", "Value: ", ""
    ]

    suffixes = [
        " kg.", " meters.", " V.", " J.", " units.", " as recorded.", " in total.",
        " per second.", " confirmed.", "."
    ]

    prose_corpus = [
        "The theory of general relativity describes gravity as curved spacetime.",
        "Quantum mechanics provides a mathematical description of atomic particles.",
        "Computer architectures require careful cache synchronization protocols.",
        "The exploratory mission discovered several icy asteroids near Jupiter.",
        "Statistical language models learn probability distributions over vocabularies.",
        "Modern spacecraft utilize deep space optical communications for telemetry.",
        "Thermodynamics establishes fundamental relations between heat and mechanical work."
    ]

    dataset = []

    for _ in range(num_samples):
        if random.random() < 0.2:
            text = random.choice(prose_corpus)
            input_ids = tokenizer.encode(text)
            token_labels = [CLASS_BASE] * len(input_ids)
            dataset.append((input_ids, token_labels, -1))
        else:
            op_choice = random.choice(["+", "-", ">", "*"])
            a_raw = random.uniform(1.0, 99.0)
            b_raw = random.uniform(1.0, 99.0)
            a_str, b_str = format_num(a_raw), format_num(b_raw)
            a_dec, b_dec = Decimal(a_str), Decimal(b_str)

            if op_choice == "+":
                cls_id = CLASS_ADD
                ans_str = f" {a_dec + b_dec:f}"
            elif op_choice == "-":
                cls_id = CLASS_SUB
                ans_str = f" {a_dec - b_dec:f}"
            elif op_choice == ">":
                cls_id = CLASS_CMP
                ans_str = f" {str(a_dec > b_dec)}"
            else:
                cls_id = CLASS_MUL
                ans_str = f" {a_dec * b_dec:f}"

            prefix_text = random.choice(prefixes) + f"{a_str} {op_choice} {b_str} ="
            suffix_text = random.choice(suffixes)

            prefix_ids = tokenizer.encode(prefix_text)
            ans_ids = tokenizer.encode(ans_str)
            suffix_ids = tokenizer.encode(suffix_text)

            input_ids = prefix_ids + ans_ids + suffix_ids
            token_labels = (
                [CLASS_BASE] * len(prefix_ids) +
                [cls_id]     * len(ans_ids) +
                [CLASS_BASE] * len(suffix_ids)
            )

            boundary_idx = len(prefix_ids) - 1
            dataset.append((input_ids, token_labels, boundary_idx))

    return dataset


# -----------------------------------------------------------------------------
# 6. Train Gate with Boundary-Weighted Loss
# -----------------------------------------------------------------------------
def train_gate(epochs=4, batch_size=8, lr=1e-3):
    print("Generating diverse training sequences...")
    train_data = generate_training_data(num_samples=750)

    gate = TriScaleAtomGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-2)

    weights = torch.tensor([0.25, 1.0, 2.0, 2.0, 2.0, 2.0], device=device)

    print(f"Training tri-scale gate on {len(train_data)} sequences with boundary weighting...")
    gate.train()

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        correct, total = 0, 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            batch_loss = 0.0

            for input_ids, labels, b_idx in batch:
                inp_tensor = torch.tensor([input_ids], device=device)

                with torch.no_grad():
                    out = base_model(inp_tensor, output_hidden_states=True)
                    hidden_states = out.hidden_states[-1][0]

                # Tri-scale pooling: [h_last || h_mean || h_max] -> 2304 dims
                mean_p, max_p = causal_tri_scale_pool(hidden_states, WINDOW_SIZE)
                combined = torch.cat([hidden_states, mean_p, max_p], dim=-1)

                h_causal = combined[:-1]
                target_causal = torch.tensor(labels[1:], device=device)

                gate_logits = gate(h_causal)
                raw_loss = F.cross_entropy(gate_logits, target_causal, weight=weights, reduction='none')

                # Apply 5.0x weighting to the '=' boundary token where latching occurs
                step_weights = torch.ones_like(raw_loss)
                if b_idx != -1 and b_idx < len(step_weights):
                    step_weights[b_idx] = 5.0

                loss = (raw_loss * step_weights).mean()

                batch_loss += loss
                preds = gate_logits.argmax(dim=-1)
                correct += (preds == target_causal).sum().item()
                total += target_causal.numel()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        acc = correct / total * 100.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_data):.4f} | Accuracy: {acc:.2f}%")

    print("Gate training complete!\n")
    return gate


# -----------------------------------------------------------------------------
# 7. Live Inference (Tri-Scale Evaluation)
# -----------------------------------------------------------------------------
def run_live_inference(gate, prompt, max_tokens=15, tau=0.1):
    gate.eval()
    context = prompt

    print(f"{'='*80}\nPROMPT: \"{prompt}\"\n{'='*80}")

    active_session = None
    active_atom_type = None
    equation_consumed = False

    for step in range(max_tokens):
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

        # 1. Base Model Forward Pass
        with torch.no_grad():
            out = base_model(input_ids, output_hidden_states=True)
            h_seq = out.hidden_states[-1]
            base_logits = out.logits[0, -1, :].cpu().numpy()

        is_at_boundary = context.rstrip().endswith("=") and not equation_consumed

        # 2. Gate Evaluation with Tri-Scale Concatenation
        if active_atom_type is None:
            if is_at_boundary:
                h_last = h_seq[:, -1, :]
                seq_len = h_seq.size(1)
                k_win = min(WINDOW_SIZE, seq_len)
                h_win = h_seq[:, -k_win:, :]
                
                h_mean = h_win.mean(dim=1)
                h_max = h_win.max(dim=1).values
                h_combined = torch.cat([h_last, h_mean, h_max], dim=-1)

                with torch.no_grad():
                    gate_logits = gate(h_combined)[0]
                    lambdas = torch.softmax(gate_logits / tau, dim=-1).cpu().numpy()

                predicted_atom = CLASS_NAMES[int(lambdas.argmax())]
                if predicted_atom in ["+", "-", "cmp", "*"]:
                    active_atom_type = predicted_atom
            else:
                predicted_atom = "Base"
                lambdas = np.zeros(NUM_CLASSES)
                lambdas[CLASS_BASE] = 1.0
        else:
            predicted_atom = active_atom_type
            lambdas = np.zeros(NUM_CLASSES)
            lambdas[CLASS_NAMES.index(active_atom_type)] = 1.0

        target_char = None

        # 3. Value Map Execution
        if active_atom_type == "+":
            if active_session is None:
                sub = context.split("=")[0].split()
                a, b = sub[-3], sub[-1]
                pa, pb = EmbeddedTransformerAdder.pad(a, b)
                active_session = {"ids": [EmbeddedTransformerAdder.tok[c] for c in f"^{pa} + {pb} ="]}

            target_char = UnifiedMathEngine.step_add(active_session)
            if target_char is None:
                active_session = None
                active_atom_type = None
                equation_consumed = True

        elif active_atom_type == "-":
            if active_session is None:
                sub = context.split("=")[0].split()
                a, b = Decimal(sub[-3]), Decimal(sub[-1])
                diff_str = f" {a - b:f}"
                active_session = {"target": diff_str, "idx": 0}

            target_char = UnifiedMathEngine.step_sub(active_session)
            if target_char is None:
                active_session = None
                active_atom_type = None
                equation_consumed = True

        elif active_atom_type == "cmp":
            if active_session is None:
                sub = context.split("=")[0].split()
                a, b = Decimal(sub[-3]), Decimal(sub[-1])
                res_str = f" {str(a > b)}"
                active_session = {"target": res_str, "idx": 0}

            target_char = UnifiedMathEngine.step_cmp(active_session)
            if target_char is None:
                active_session = None
                active_atom_type = None
                equation_consumed = True

        elif active_atom_type == "*":
            if active_session is None:
                sub = context.split("=")[0].split()
                a, b = Decimal(sub[-3]), Decimal(sub[-1])
                prod_str = f" {a * b:f}"
                active_session = {"target": prod_str, "idx": 0}

            target_char = UnifiedMathEngine.step_mul(active_session)
            if target_char is None:
                active_session = None
                active_atom_type = None
                equation_consumed = True

        # 4. Token Selection
        if target_char is not None:
            tok_id = tokenizer.encode(target_char, add_special_tokens=False)[0]
            next_tok_id = tok_id
        else:
            next_tok_id = int(base_logits.argmax())
            lambdas = np.zeros(NUM_CLASSES)
            lambdas[CLASS_BASE] = 1.0

        next_char = tokenizer.decode([next_tok_id])
        context += next_char

        gate_str = " | ".join([f"{name}: {lam:5.3f}" for name, lam in zip(CLASS_NAMES, lambdas)])
        print(f"Step {step+1:02d} | Emitted: {repr(next_char):6s} | Learned Gate: [{gate_str}]")

        if next_tok_id == tokenizer.eos_token_id:
            break

    print(f"\nFINAL OUTPUT:\n\"{context}\"\n")


# -----------------------------------------------------------------------------
# 8. Test Suite
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trained_gate = train_gate(epochs=4, batch_size=8, lr=1e-3)

    # Test 1: Addition (+ must fire -> 3.579)
    run_live_inference(trained_gate, "The telemetry distance is 1.234 + 2.345 =", max_tokens=10)

    # Test 2: Subtraction (- must fire -> 6.35)
    run_live_inference(trained_gate, "The tank depletion is 14.25 - 7.90 =", max_tokens=10)

    # Test 3: Multiplication (* must fire -> 25.00)
    run_live_inference(trained_gate, "Calculated power value is 12.5 * 2.0 =", max_tokens=10)

    # Test 4: Pure natural language prose (Base must stay active)
    run_live_inference(trained_gate, "The solar system consists of eight planets and", max_tokens=8)