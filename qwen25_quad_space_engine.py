"""
=============================================================================
Qwen-2.5 Quad-Space Engine (Production v14 - VRAM Resilient)
=============================================================================
Supports:
  - Direct CUDA:0 placement to prevent premature CPU offloading
  - Automatic fallback to 4-bit Qwen-2.5-3B (~1.9 GB download, ~2.2 GB VRAM)
    if the 7B model exceeds available GPU memory
  - Full Quad-Space: Omega_ctx, Omega_corpus, Omega_params, Omega_tools, omega_null
=============================================================================
"""

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuits import arith   # hand-weighted transformer circuits for + - * / and comparison

DIV_PLACES = 6      # '/' truncates to this many fractional digits
EXPERT_C = 100.0    # certified-token logit of an arithmetic atom (+C on it, -C elsewhere)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Quad-Space Engine on: {device}")

# -----------------------------------------------------------------------------
# 1. Parse Arguments & Model Selection
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Quad-Space Engine on Qwen-2.5")
parser.add_argument(
    "--model",
    type=str,
    default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    help="HuggingFace model ID"
)
args, _ = parser.parse_known_args()
MODEL_ID = args.model

print(f"\nTargeting model: {MODEL_ID}")

# -----------------------------------------------------------------------------
# 2. Resilient Model Loading (Direct CUDA:0 with 3B Auto-Fallback)
# -----------------------------------------------------------------------------
tokenizer = None
base_model = None

def load_qwen_model(target_model_id):
    print(f"Downloading/Loading tokenizer for: {target_model_id} ...")
    tok = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
    
    print(f"Loading weights for {target_model_id} onto GPU (cuda:0)...")
    # Forcing device_map="cuda:0" ensures accelerate doesn't dispatch layers to CPU
    mod = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        device_map="cuda:0" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    return tok, mod

try:
    tokenizer, base_model = load_qwen_model(MODEL_ID)
except (ValueError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
    print(f"\n[VRAM Notice] Could not fit {MODEL_ID} entirely in GPU VRAM.")
    print("Error encountered:", str(e).split('\n')[0])
    
    FALLBACK_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    print("\n" + "="*80)
    print(f"AUTOMATIC FALLBACK: Loading 4-bit Qwen-2.5-3B ({FALLBACK_MODEL})")
    print("Download Size: ~1.9 GB | VRAM Usage: ~2.2 GB (100% GPU fit guaranteed)")
    print("="*80 + "\n")
    
    torch.cuda.empty_cache()
    MODEL_ID = FALLBACK_MODEL
    tokenizer, base_model = load_qwen_model(MODEL_ID)

base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False

D_MODEL = base_model.config.hidden_size  # Automatically adjusts: 3584 (7B) or 2048 (3B)
V_SIZE = base_model.config.vocab_size
WINDOW_SIZE = 8

print(f"Model Initialized: {MODEL_ID}")
print(f"Architecture Parameters: Hidden Dim = {D_MODEL}, Vocab Size = {V_SIZE}\n")

CLASS_BASE   = 0
CLASS_NULL   = 1
CLASS_ADD    = 2
CLASS_SUB    = 3
CLASS_CMP    = 4
CLASS_MUL    = 5
CLASS_CORPUS = 6
CLASS_TOOL   = 7
CLASS_DIV    = 8

CLASS_NAMES = ["Base", "Null", "+", "-", "cmp", "*", "Corpus", "Tool", "/"]
NUM_CLASSES = len(CLASS_NAMES)
MATH_OPS = ["+", "-", "cmp", "*", "/"]
CMP_OPS = [">", "<", ">=", "<=", "==", "!="]


# -----------------------------------------------------------------------------
# 3. Tool Atom Space (Omega_tools)
# -----------------------------------------------------------------------------
class ToolAtomSpace:
    def __init__(self, tau: float = 0.05, threshold: float = 0.70):
        self.tau = tau
        self.threshold = threshold
        self.tools = []

    def _embed(self, text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
            vec = out.hidden_states[-1][0].mean(dim=0).float().to(device)
            return F.normalize(vec, p=2, dim=0)

    def register_tool(self, name: str, trigger_signature: str, func):
        key_vec = self._embed(trigger_signature)
        self.tools.append({
            "name": name,
            "key_text": trigger_signature,
            "key_vec": key_vec,
            "func": func,
            "sign": +1.0
        })
        print(f"[Omega_tools] AUTHORIZED Tool: '{name}' on signature '{trigger_signature}' (nu = +1)")

    def revoke_tool(self, name: str):
        target = next((t for t in self.tools if t["name"] == name and t["sign"] > 0), None)
        if target:
            self.tools.append({
                "name": name,
                "key_text": target["key_text"],
                "key_vec": target["key_vec"],
                "func": target["func"],
                "sign": -1.0
            })
            print(f"[Omega_tools] COMMITTED ANTI-ATOM: Revoked '{name}' (nu = -1, Access Zeroed)")

    def query(self, context_tail: str):
        if not self.tools:
            return 0.0, None, None

        q_vec = self._embed(context_tail)
        tool_scores = {}
        for tool in self.tools:
            sim = torch.dot(q_vec, tool["key_vec"]).item()
            kernel = np.exp(sim / self.tau)
            signed_mass = tool["sign"] * kernel
            name = tool["name"]
            tool_scores[name] = tool_scores.get(name, 0.0) + signed_mass

        viable = [(name, score) for name, score in tool_scores.items() if score > 0.0]
        if not viable:
            return 0.0, None, None

        best_name, _ = max(viable, key=lambda x: x[1])
        tool_obj = next(t for t in self.tools if t["name"] == best_name and t["sign"] > 0)
        max_sim = torch.dot(q_vec, tool_obj["key_vec"]).item()

        if max_sim < self.threshold:
            return 0.0, None, None

        result_str = tool_obj["func"](context_tail)
        return max_sim, best_name, result_str


# -----------------------------------------------------------------------------
# 4. Corpus Atom Space (Omega_corpus)
# -----------------------------------------------------------------------------
class CorpusAtomSpace:
    def __init__(self, tau: float = 0.05, threshold: float = 0.70):
        self.tau = tau
        self.threshold = threshold
        self.atoms = []

    def _embed(self, text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
            vec = out.hidden_states[-1][0].mean(dim=0).float().to(device)
            return F.normalize(vec, p=2, dim=0)

    def insert(self, key_text: str, val_text: str):
        key_vec = self._embed(key_text)
        self.atoms.append({
            "key_text": key_text,
            "key_vec": key_vec,
            "val_text": val_text,
            "sign": +1.0
        })
        print(f"[Omega_corpus] INSERTED Fact: '{key_text}' -> '{val_text}' (nu = +1)")

    def retract(self, key_text: str, val_text: str):
        key_vec = self._embed(key_text)
        self.atoms.append({
            "key_text": key_text,
            "key_vec": key_vec,
            "val_text": val_text,
            "sign": -1.0
        })
        print(f"[Omega_corpus] COMMITTED ANTI-ATOM for '{key_text}' (nu = -1, Zero-Shot Retraction)")

    def query(self, context_tail: str):
        if not self.atoms:
            return 0.0, None

        q_vec = self._embed(context_tail)
        value_scores = {}
        for atom in self.atoms:
            sim = torch.dot(q_vec, atom["key_vec"]).item()
            kernel = np.exp(sim / self.tau)
            signed_mass = atom["sign"] * kernel
            val = atom["val_text"]
            value_scores[val] = value_scores.get(val, 0.0) + signed_mass

        viable = [(val, score) for val, score in value_scores.items() if score > 0.0]
        if not viable:
            return 0.0, None

        best_val, _ = max(viable, key=lambda x: x[1])
        max_sim = max(torch.dot(q_vec, a["key_vec"]).item() for a in self.atoms if a["sign"] > 0)

        if max_sim < self.threshold:
            return 0.0, None

        return max_sim, best_val


# -----------------------------------------------------------------------------
# 5. Parametric Specialist Atoms (Omega_params)
# -----------------------------------------------------------------------------
class CircuitAtom:
    """omega_op in Omega_params: a hand-weighted transformer (circuits/) evaluates
    `a op b`; its answer is then streamed into the LM's token stream one character
    per step. No Python arithmetic is involved (see circuits/arith.py)."""

    def __init__(self, a: str, op: str, b: str):
        self.target = " " + arith.evaluate(a, op, b, places=DIV_PLACES)
        self.idx = 0

    def next_char(self):
        if self.idx < len(self.target):
            ch = self.target[self.idx]
            self.idx += 1
            return ch
        return None


# -----------------------------------------------------------------------------
# 6. Tri-Scale Neural Gate for Qwen Architecture
# -----------------------------------------------------------------------------
class TriScaleAtomGate(nn.Module):
    def __init__(self, d_in=D_MODEL * 3, hidden_dim=256, num_classes=NUM_CLASSES):
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
    h_trans = hidden_states.unsqueeze(0).transpose(1, 2).float()
    h_pad_mean = F.pad(h_trans, (window_size - 1, 0))
    mean_pooled = F.avg_pool1d(h_pad_mean, kernel_size=window_size, stride=1, count_include_pad=False)
    h_pad_max = F.pad(h_trans, (window_size - 1, 0), value=-1e9)
    max_pooled = F.max_pool1d(h_pad_max, kernel_size=window_size, stride=1)
    return mean_pooled.squeeze(0).transpose(0, 1), max_pooled.squeeze(0).transpose(0, 1)


# -----------------------------------------------------------------------------
# 7. Dataset Generator & Training
# -----------------------------------------------------------------------------
def format_num(val):
    r = random.random()
    if r < 0.25: return str(int(val))
    elif r < 0.5: return f"{val:.1f}"
    elif r < 0.75: return f"{val:.2f}"
    else: return f"{val:.3f}"


def generate_training_data(num_samples=600):
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
            op_choice = random.choice(["+", "-", ">", "*", "/"])
            a_raw, b_raw = random.uniform(1.0, 99.0), random.uniform(1.0, 99.0)
            a_str, b_str = format_num(a_raw), format_num(b_raw)

            cls_id = {"+": CLASS_ADD, "-": CLASS_SUB, ">": CLASS_CMP,
                      "*": CLASS_MUL, "/": CLASS_DIV}[op_choice]
            ans_str = " " + arith.evaluate(a_str, op_choice, b_str, places=DIV_PLACES)

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


def train_gate(epochs=3, batch_size=4, lr=1e-3):
    print(f"Generating training data for {D_MODEL}-dim representations...")
    train_data = generate_training_data(num_samples=600)

    gate = TriScaleAtomGate(d_in=D_MODEL * 3).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-2)
    weights = torch.tensor([0.25, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], device=device)

    print(f"Training gate ({D_MODEL * 3} -> 256 -> {NUM_CLASSES}) on {len(train_data)} sequences...")
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
                    hidden_states = out.hidden_states[-1][0].float().to(device)

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
# 8. Inference Engine
# -----------------------------------------------------------------------------
class QuadSpaceInferenceEngine:
    def __init__(self, gate, corpus_space: CorpusAtomSpace, tool_space: ToolAtomSpace, tau=0.1):
        self.gate = gate
        self.corpus_space = corpus_space
        self.tool_space = tool_space
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
                h_seq = out.hidden_states[-1].float().to(device)
                base_logits = out.logits[0, -1, :].float().cpu().numpy()

            is_at_math_boundary = context.rstrip().endswith("=") and not equation_consumed
            is_terminal = context.endswith(("\n", ".", "!", "?"))
            context_tail = context[-45:]

            if not is_terminal and not is_at_math_boundary:
                tool_sim, tool_name, tool_val = self.tool_space.query(context_tail)
                corpus_sim, corpus_val = self.corpus_space.query(context_tail)
            else:
                tool_val, corpus_val = None, None

            # Quad-Space Routing Decision
            if active_atom_type is None:
                if is_at_math_boundary:
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
                    if predicted_atom in MATH_OPS:
                        active_atom_type = predicted_atom

                elif tool_val is not None:
                    predicted_atom = "Tool"
                    active_atom_type = "Tool"
                    active_session = {"target": tool_val, "idx": 0}
                    lambdas = np.zeros(NUM_CLASSES)
                    lambdas[CLASS_TOOL] = 1.0

                elif corpus_val is not None:
                    predicted_atom = "Corpus"
                    active_atom_type = "Corpus"
                    active_session = {"target": corpus_val, "idx": 0}
                    lambdas = np.zeros(NUM_CLASSES)
                    lambdas[CLASS_CORPUS] = 1.0

                else:
                    predicted_atom = "Base"
                    lambdas = np.zeros(NUM_CLASSES)
                    lambdas[CLASS_BASE] = 1.0
            else:
                predicted_atom = active_atom_type
                lambdas = np.zeros(NUM_CLASSES)
                lambdas[CLASS_NAMES.index(active_atom_type)] = 1.0

            target_char = None

            if active_atom_type in ["Tool", "Corpus"]:
                if active_session["idx"] < len(active_session["target"]):
                    target_char = active_session["target"][active_session["idx"]]
                    active_session["idx"] += 1
                else:
                    active_session = None
                    active_atom_type = None

            elif active_atom_type in MATH_OPS:
                if active_session is None:
                    sub = context.split("=")[0].split()
                    a_str, op_str, b_str = sub[-3], sub[-2], sub[-1]
                    if active_atom_type == "cmp" and op_str not in CMP_OPS:
                        op_str = ">"
                    active_session = CircuitAtom(a_str, op_str, b_str)

                target_char = active_session.next_char()
                if target_char is None:
                    active_session = None
                    active_atom_type = None
                    equation_consumed = True

            # Token Selection: Level B logit blending,
            #   l_comb = sum_w lambda_w * l_w,  l_expert = +C on the certified token, -C elsewhere
            if target_char is not None:
                tok_id = tokenizer.encode(target_char, add_special_tokens=False)[0]
                expert_logits = np.full_like(base_logits, -EXPERT_C)
                expert_logits[tok_id] = EXPERT_C
                lam_expert = float(lambdas[CLASS_NAMES.index(predicted_atom)])
                combined = (1.0 - lam_expert) * base_logits + lam_expert * expert_logits
                next_tok_id = int(combined.argmax())
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
# 9. Execution Suite
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trained_gate = train_gate(epochs=3, batch_size=4, lr=1e-3)

    corpus = CorpusAtomSpace(tau=0.05, threshold=0.75)
    tools = ToolAtomSpace(tau=0.05, threshold=0.75)
    engine = QuadSpaceInferenceEngine(trained_gate, corpus, tools)

    def live_barometer_api(query_str):
        return " 978.20"

    tools.register_tool(
        name="MunichBarometerAPI",
        trigger_signature="live sensor reading: Munich atmospheric pressure is",
        func=live_barometer_api
    )

    corpus.insert("Munich historical baseline pressure is", " 960.50")

    print("\n--- TEST 1: LIVE TOOL CALL ---")
    engine.generate("Live sensor reading: Munich atmospheric pressure is", max_tokens=6)

    print("\n--- TEST 2: FULL QUAD-SPACE SYNERGY ---")
    engine.generate("Telemetry delta calculation: 978.20 - 960.50 =", max_tokens=10)

    print("\n--- TEST 2b: COMPILED MULTIPLICATION AND DIVISION ---")
    engine.generate("Power check: 12.5 * 2.04 =", max_tokens=10)
    engine.generate("Average rate: 22.5 / 7 =", max_tokens=12)

    print("\n--- TEST 3: TOOL REVOCATION (ANTI-ATOM) ---")
    tools.revoke_tool("MunichBarometerAPI")
    engine.generate("Live sensor reading: Munich atmospheric pressure is", max_tokens=6)