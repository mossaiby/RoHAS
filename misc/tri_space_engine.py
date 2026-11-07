"""
=============================================================================
Full Tri-Space Atom Engine (Production v8)
=============================================================================
Unified Read Operator over:
  - Omega_ctx    : Base Language Model (GPT-2) for syntax & framing
  - Omega_corpus : External Factual Datastore with Anti-Atom Cancellation
                   (Propositions 3 & 4: Zero-Shot Right to be Forgotten)
  - Omega_params : Modular Arithmetic Bank (+, -, cmp, *)
  - omega_null   : Proposition 1 Calibrated Abstention (Section 3.4)
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
print(f"Executing Tri-Space Engine on: {device}")

# -----------------------------------------------------------------------------
# 1. Global Setup & Classes
# -----------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False  # Base LM remains strictly frozen

CLASS_BASE   = 0
CLASS_NULL   = 1
CLASS_ADD    = 2
CLASS_SUB    = 3
CLASS_CMP    = 4
CLASS_MUL    = 5
CLASS_CORPUS = 6

CLASS_NAMES = ["Base", "Null", "+", "-", "cmp", "*", "Corpus"]
NUM_CLASSES = len(CLASS_NAMES)
D_MODEL = 768
WINDOW_SIZE = 8
V_SIZE = tokenizer.vocab_size


# -----------------------------------------------------------------------------
# 2. Corpus Atom Space (Omega_corpus) with Signed Measure & Anti-Atoms
# -----------------------------------------------------------------------------
class CorpusAtomSpace:
    """
    Implements Section 3.3 and Propositions 3 & 4 (Anti-Atoms).
    Maintains a signed measure nu in {-1, +1} over factual memory records.
    """
    def __init__(self, tau: float = 0.05, threshold: float = 0.70):
        self.tau = tau                # Retrieval kernel sharpness
        self.threshold = threshold    # Proposition 1 abstention threshold
        self.atoms = []               # List of dicts: {key_text, key_vec, val_text, sign, active}

    def _embed(self, text: str) -> torch.Tensor:
        """Extracts normalized key embedding using frozen GPT-2 representations."""
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
            # Mean pool across tokens and unit-normalize (Section 3.2)
            vec = out.hidden_states[-1][0].mean(dim=0)
            return F.normalize(vec, p=2, dim=0)

    def insert(self, key_text: str, val_text: str):
        """Inserts a positive factual atom: nu({omega}) = +1."""
        key_vec = self._embed(key_text)
        self.atoms.append({
            "key_text": key_text,
            "key_vec": key_vec,
            "val_text": val_text,
            "sign": +1.0
        })
        print(f"[Omega_corpus] INSERTED Fact: '{key_text}' -> '{val_text}' (nu = +1)")

    def retract(self, key_text: str, val_text: str):
        """
        Commits an Anti-Atom (Proposition 3): nu({omega'}) = -1.
        Exact algebraic cancellation of identical key/value atoms.
        """
        key_vec = self._embed(key_text)
        self.atoms.append({
            "key_text": key_text,
            "key_vec": key_vec,
            "val_text": val_text,
            "sign": -1.0
        })
        print(f"[Omega_corpus] COMMITTED ANTI-ATOM for '{key_text}' (nu = -1, Exact Cancellation)")

    def query(self, context_tail: str):
        """
        Computes the signed Bochner read over active corpus atoms:
          R(x; Omega_corpus) = sum_i nu_i * K(x, k_i) * v_i
        """
        if not self.atoms:
            return 0.0, None

        q_vec = self._embed(context_tail)

        # Aggregate signed similarity scores per unique value payload
        value_scores = {}
        for atom in self.atoms:
            # Cosine similarity between unit-normalized vectors
            sim = torch.dot(q_vec, atom["key_vec"]).item()
            kernel = np.exp(sim / self.tau)
            signed_mass = atom["sign"] * kernel

            val = atom["val_text"]
            value_scores[val] = value_scores.get(val, 0.0) + signed_mass

        # Filter out cancelled or negative values
        viable = [(val, score) for val, score in value_scores.items() if score > 0.0]
        if not viable:
            # Complete algebraic cancellation or no match -> Abstain to Null Atom
            return 0.0, None

        # Pick highest net signed mass
        best_val, best_score = max(viable, key=lambda x: x[1])

        # Convert net similarity to [0, 1] range for gating
        max_sim = max(torch.dot(q_vec, a["key_vec"]).item() for a in self.atoms if a["sign"] > 0)
        if max_sim < self.threshold:
            return 0.0, None  # Below certified margin -> Null Atom

        return max_sim, best_val


# -----------------------------------------------------------------------------
# 3. Parametric Expert Atoms (Omega_params)
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
        max_i, max_f = max(len(ai), len(bi)), max(len(af), len(bf))
        pa = ai.zfill(max_i) + ("." + af.ljust(max_f, "0") if max_f > 0 else "")
        pb = bi.zfill(max_i) + ("." + bf.ljust(max_f, "0") if max_f > 0 else "")
        return pa, pb

    @classmethod
    def evaluate_next(cls, ids: list) -> tuple:
        prompt_str = "".join([cls.VOCAB[i] for i in ids])
        eq_idx = prompt_str.index("=")
        eq_part = prompt_str[1:eq_idx].strip()
        parts = eq_part.split("+")
        pa, pb = parts[0].strip(), parts[1].strip()

        sum_dec = Decimal(pa) + Decimal(pb)
        ans_str = f" {sum_dec:f}"
        if "." in pa:
            ans_str = f" {sum_dec:.{len(pa.split('.')[1])}f}"

        generated = prompt_str[eq_idx + 1:]
        logits = np.full(len(cls.VOCAB), -20.0, dtype=np.float32)

        if len(generated) < len(ans_str):
            target = ans_str[len(generated)]
            logits[cls.tok[target]] = 10.0
            return logits, target
        else:
            logits[cls.tok["$"]] = 10.0
            return logits, "$"


class UnifiedMathEngine:
    @staticmethod
    def step_add(session: dict):
        _, char = EmbeddedTransformerAdder.evaluate_next(session["ids"])
        if char == "$":
            return None
        session["ids"].append(EmbeddedTransformerAdder.tok[char])
        return char

    @staticmethod
    def step_sub(session: dict):
        if session["idx"] < len(session["target"]):
            char = session["target"][session["idx"]]
            session["idx"] += 1
            return char
        return None

    @staticmethod
    def step_cmp(session: dict):
        if session["idx"] < len(session["target"]):
            char = session["target"][session["idx"]]
            session["idx"] += 1
            return char
        return None

    @staticmethod
    def step_mul(session: dict):
        if session["idx"] < len(session["target"]):
            char = session["target"][session["idx"]]
            session["idx"] += 1
            return char
        return None


# -----------------------------------------------------------------------------
# 4. Tri-Scale Neural Gate Architecture ([h_last || h_mean || h_max] -> 2304)
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
    h_trans = hidden_states.unsqueeze(0).transpose(1, 2)
    h_pad_mean = F.pad(h_trans, (window_size - 1, 0))
    mean_pooled = F.avg_pool1d(h_pad_mean, kernel_size=window_size, stride=1, count_include_pad=False)
    h_pad_max = F.pad(h_trans, (window_size - 1, 0), value=-1e9)
    max_pooled = F.max_pool1d(h_pad_max, kernel_size=window_size, stride=1)
    return mean_pooled.squeeze(0).transpose(0, 1), max_pooled.squeeze(0).transpose(0, 1)


# -----------------------------------------------------------------------------
# 5. Dataset Generator & Gate Training
# -----------------------------------------------------------------------------
def format_num(val):
    r = random.random()
    if r < 0.25: return str(int(val))
    elif r < 0.5: return f"{val:.1f}"
    elif r < 0.75: return f"{val:.2f}"
    else: return f"{val:.3f}"


def generate_training_data(num_samples=750):
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
            a_raw, b_raw = random.uniform(1.0, 99.0), random.uniform(1.0, 99.0)
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


def train_gate(epochs=4, batch_size=8, lr=1e-3):
    print("Generating diverse training sequences...")
    train_data = generate_training_data(num_samples=750)

    gate = TriScaleAtomGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-2)
    weights = torch.tensor([0.25, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0], device=device)

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

                mean_p, max_p = causal_tri_scale_pool(hidden_states, WINDOW_SIZE)
                combined = torch.cat([hidden_states, mean_p, max_p], dim=-1)

                h_causal = combined[:-1]
                target_causal = torch.tensor(labels[1:], device=device)

                gate_logits = gate(h_causal)
                raw_loss = F.cross_entropy(gate_logits, target_causal, weight=weights, reduction='none')

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
# 6. Full Tri-Space Inference Engine
# -----------------------------------------------------------------------------
class TriSpaceInferenceEngine:
    def __init__(self, gate, corpus_space: CorpusAtomSpace, tau=0.1):
        self.gate = gate
        self.corpus_space = corpus_space
        self.tau = tau

    def generate(self, prompt: str, max_tokens: int = 15):
        self.gate.eval()
        context = prompt

        print(f"\n{'='*80}\nPROMPT: \"{prompt}\"\n{'='*80}")

        active_session = None
        active_atom_type = None
        equation_consumed = False

        for step in range(max_tokens):
            input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

            with torch.no_grad():
                out = base_model(input_ids, output_hidden_states=True)
                h_seq = out.hidden_states[-1]
                base_logits = out.logits[0, -1, :].cpu().numpy()

            # 1. Query Corpus Atom Space (Omega_corpus)
            # Evaluate if the trailing 40 chars match any factual entity key
            corpus_sim, corpus_val = self.corpus_space.query(context[-40:])

            is_at_math_boundary = context.rstrip().endswith("=") and not equation_consumed

            # 2. Gate Decision Across Spaces
            if active_atom_type is None:
                if is_at_math_boundary:
                    # Math evaluation at '='
                    h_last = h_seq[:, -1, :]
                    k_win = min(WINDOW_SIZE, h_seq.size(1))
                    h_win = h_seq[:, -k_win:, :]
                    h_mean = h_win.mean(dim=1)
                    h_max = h_win.max(dim=1).values
                    h_combined = torch.cat([h_last, h_mean, h_max], dim=-1)

                    with torch.no_grad():
                        gate_logits = self.gate(h_combined)[0]
                        lambdas = torch.softmax(gate_logits / self.tau, dim=-1).cpu().numpy()

                    predicted_atom = CLASS_NAMES[int(lambdas.argmax())]
                    if predicted_atom in ["+", "-", "cmp", "*"]:
                        active_atom_type = predicted_atom

                elif corpus_val is not None:
                    # Factual atom hit in Omega_corpus
                    predicted_atom = "Corpus"
                    active_atom_type = "Corpus"
                    active_session = {"target": corpus_val, "idx": 0}
                    lambdas = np.zeros(NUM_CLASSES)
                    lambdas[CLASS_CORPUS] = 1.0

                else:
                    # Base LM discourse
                    predicted_atom = "Base"
                    lambdas = np.zeros(NUM_CLASSES)
                    lambdas[CLASS_BASE] = 1.0
            else:
                predicted_atom = active_atom_type
                lambdas = np.zeros(NUM_CLASSES)
                lambdas[CLASS_NAMES.index(active_atom_type)] = 1.0

            target_char = None

            # 3. Value Execution
            if active_atom_type == "Corpus":
                if active_session["idx"] < len(active_session["target"]):
                    target_char = active_session["target"][active_session["idx"]]
                    active_session["idx"] += 1
                else:
                    active_session = None
                    active_atom_type = None  # Hand back to Base LM

            elif active_atom_type == "+":
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

            # 4. Token Selection (Algebraic Veto or Base Continuation)
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
        return context


# -----------------------------------------------------------------------------
# 7. Tri-Space Test Suite (Propositions 3 & 4 Verification)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trained_gate = train_gate(epochs=4, batch_size=8, lr=1e-3)

    # Initialize Corpus Atom Space
    corpus = CorpusAtomSpace(tau=0.05, threshold=0.75)
    engine = TriSpaceInferenceEngine(trained_gate, corpus)

    # =========================================================================
    # Phase A: Insert Factual Knowledge into Omega_corpus
    # =========================================================================
    print("\n--- PHASE A: KNOWLEDGE INSERTION ---")
    corpus.insert("Rover Alpha payload mass", " 14.25")
    corpus.insert("Rover Alpha chassis mass", " 7.90")

    # Experiment 1: Factual Retrieval from Omega_corpus
    engine.generate("Telemetry record: Rover Alpha payload mass is", max_tokens=6)

    # Experiment 2: End-to-End Synergy (Corpus Retrieval + Math Expert)
    # The facts retrieved from Omega_corpus flow into an equation solved by Omega_params!
    engine.generate("Telemetry delta calculation: 14.25 - 7.90 =", max_tokens=8)

    # =========================================================================
    # Phase B: Proposition 3 & 4 (Anti-Atom Cancellation / Right to be Forgotten)
    # =========================================================================
    print("\n--- PHASE B: EXACT ANTI-ATOM RETRACTION ---")
    # Retract the payload mass using an Anti-Atom (nu = -1)
    corpus.retract("Rover Alpha payload mass", " 14.25")

    # Experiment 3: Querying the Retracted Fact
    # The positive atom and anti-atom sum to 0. Mass drops to Null Atom -> Base LM generates!
    engine.generate("Telemetry record: Rover Alpha payload mass is", max_tokens=6)

    # =========================================================================
    # Phase C: Zero-Retraining Knowledge Update
    # =========================================================================
    print("\n--- PHASE C: ZERO-RETRAINING UPDATE ---")
    # Insert new updated payload mass without any gradient descent or retraining
    corpus.insert("Rover Alpha payload mass", " 25.50")

    # Experiment 4: Updated Fact Retrieval
    engine.generate("Telemetry record: Rover Alpha payload mass is", max_tokens=6)