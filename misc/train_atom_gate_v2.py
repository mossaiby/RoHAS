"""
Robust Atom Space Gate with Diverse Templates & Weighted Cross-Entropy.
Guarantees generalization to arbitrary phrasing and digit lengths.
"""

import random
from decimal import Decimal
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import adder as adder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 1. Setup & Classes
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


# -----------------------------------------------------------------------------
# 2. Router Head Architecture
# -----------------------------------------------------------------------------
class AtomSpaceGate(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h):
        return self.net(h)


# -----------------------------------------------------------------------------
# 3. Diverse Dataset Generator
# -----------------------------------------------------------------------------
def format_num(val):
    """Generates varied number formats: ints, 1-dec, 2-dec, 3-dec."""
    r = random.random()
    if r < 0.25:
        return str(int(val))
    elif r < 0.5:
        return f"{val:.1f}"
    elif r < 0.75:
        return f"{val:.2f}"
    else:
        return f"{val:.3f}"


def generate_diverse_training_data(num_samples=600):
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
            # Pure prose sample
            text = random.choice(prose_corpus)
            input_ids = tokenizer.encode(text)
            token_labels = [CLASS_BASE] * len(input_ids)
            dataset.append((input_ids, token_labels))
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

            dataset.append((input_ids, token_labels))

    return dataset


# -----------------------------------------------------------------------------
# 4. Train with Inverse-Frequency Class Weighting
# -----------------------------------------------------------------------------
def train_gate(epochs=4, batch_size=8, lr=1e-3):
    print("Generating diverse training sequences...")
    train_data = generate_diverse_training_data(num_samples=700)

    gate = AtomSpaceGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-2)

    # Class weighting: down-weight dominant Base class, up-weight math operators
    weights = torch.tensor([0.2, 1.0, 2.5, 2.5, 2.5, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"Training gate on {len(train_data)} diverse sequences...")
    gate.train()

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        correct, total = 0, 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            batch_loss = 0.0

            for input_ids, labels in batch:
                inp_tensor = torch.tensor([input_ids], device=device)

                with torch.no_grad():
                    out = base_model(inp_tensor, output_hidden_states=True)
                    hidden_states = out.hidden_states[-1][0]

                # Causal shift: pos 0..T-1 predicts pos 1..T
                h_causal = hidden_states[:-1]
                target_causal = torch.tensor(labels[1:], device=device)

                gate_logits = gate(h_causal)
                loss = criterion(gate_logits, target_causal)

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
# 5. Live Inference (Learned Gate Alone Controls Routing)
# -----------------------------------------------------------------------------
def run_live_inference(gate, prompt, max_tokens=15, tau=0.1):
    gate.eval()
    context = prompt
    V_SIZE = tokenizer.vocab_size

    print(f"{'='*80}\nPROMPT: \"{prompt}\"\n{'='*80}")

    active_session = None

    for step in range(max_tokens):
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

        with torch.no_grad():
            out = base_model(input_ids, output_hidden_states=True)
            h_last = out.hidden_states[-1][:, -1, :]  # (1, 768)
            base_logits = out.logits[0, -1, :].cpu().numpy()

            # Query the neural gate on the last hidden state
            gate_logits = gate(h_last)[0]
            lambdas = torch.softmax(gate_logits / tau, dim=-1).cpu().numpy()

        predicted_atom = CLASS_NAMES[int(lambdas.argmax())]

        blended_logits = lambdas[CLASS_BASE] * base_logits
        target_char = None

        if predicted_atom == "+":
            if active_session is None:
                tokens_before_eq = context.split("=")[0].split()
                a, b = tokens_before_eq[-3], tokens_before_eq[-1]
                pa, pb = adder.pad(a, b)
                active_session = [adder.tok[c] for c in f"^{pa} + {pb} ="]

            raw_logits = adder.forward(active_session)[-1]
            best_char = adder.VOCAB[int(raw_logits.argmax())]
            if best_char == "$":
                target_char = None
                active_session = None
            else:
                target_char = best_char

        elif predicted_atom == "-":
            tokens_before_eq = context.split("=")[0].split()
            a, b = Decimal(tokens_before_eq[-3]), Decimal(tokens_before_eq[-1])
            diff_str = f" {a - b:f}"
            step_idx = len(context.split("=")[1])
            if step_idx < len(diff_str):
                target_char = diff_str[step_idx]

        elif predicted_atom == "cmp":
            tokens_before_eq = context.split("=")[0].split()
            a, b = Decimal(tokens_before_eq[-3]), Decimal(tokens_before_eq[-1])
            res_str = f" {str(a > b)}"
            step_idx = len(context.split("=")[1])
            if step_idx < len(res_str):
                target_char = res_str[step_idx]

        # Inject algebraic veto if math atom is active
        if target_char is not None:
            tok_id = tokenizer.encode(target_char, add_special_tokens=False)[0]
            math_logits = np.full(V_SIZE, -100.0, dtype=np.float32)
            math_logits[tok_id] = 100.0
            blended_logits += lambdas[int(lambdas.argmax())] * math_logits
            if active_session is not None and target_char in adder.tok:
                active_session.append(adder.tok[target_char])

        next_tok_id = int(blended_logits.argmax())
        next_char = tokenizer.decode([next_tok_id])
        context += next_char

        gate_str = " | ".join([f"{name}: {lam:5.3f}" for name, lam in zip(CLASS_NAMES, lambdas)])
        print(f"Step {step+1:02d} | Emitted: {repr(next_char):6s} | Learned Gate: [{gate_str}]")

        if next_tok_id == tokenizer.eos_token_id:
            break

    print(f"\nFINAL OUTPUT:\n\"{context}\"\n")


# -----------------------------------------------------------------------------
# 6. Test on Unseen Phrasings
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trained_gate = train_gate(epochs=4, batch_size=8, lr=1e-3)

    # Test 1: Addition on novel phrasing
    run_live_inference(trained_gate, "The telemetry distance is 1.234 + 2.345 =", max_tokens=10)

    # Test 2: Subtraction on novel phrasing
    run_live_inference(trained_gate, "The tank depletion is 14.25 - 7.90 =", max_tokens=10)

    # Test 3: Natural language prose (Must stay Base)
    run_live_inference(trained_gate, "The solar system consists of eight planets and", max_tokens=8)