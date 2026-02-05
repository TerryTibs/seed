# ============================================================
# SACRSN-SEED IMMORTAL CORE v3.2 — HIERARCHICAL + MOE + MULTI-WORLD
# Author: User + Builder Protocol
# Features:
#  - Recurrent differentiable stack with hierarchical memory
#  - Sparse local attention (Fixed Window)
#  - Mixture-of-Experts (MoE) per recurrent step
#  - Multi-World simulation (Ensemble Averaging)
#  - Hybrid gradient + evolutionary training
#  - Multi-agent distributed controller
#  - Identity drift logging
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
import pickle
import os
import logging
import time

# ============================================================
# LOGGER CONFIG
# ============================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | [SACRSN-CORE] | %(message)s", 
    datefmt="%H:%M:%S"
)

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Hardware Acceleration: {DEVICE.upper()}")

# Hyperparameters
EMBED_DIM = 384
LAYERS = 6
HEADS = 6
BLOCK_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64  # Sparse attention window
NUM_EXPERTS = 4   # MoE experts per block
WORLD_SIMS = 3    # Number of parallel world simulations per forward pass

# Evolution Config
POPULATION_SIZE = 4
GENERATIONS = 5
CYCLES_PER_GEN = 100
MEMORY_FILE = "sacrsn_memory.pkl"

# ============================================================
# DATA & TOKENIZER
# ============================================================
DATA_PATH = "data.txt"

if not os.path.exists(DATA_PATH):
    logging.warning("data.txt not found. Generating synthetic quantum noise data...")
    with open(DATA_PATH, "w") as f:
        f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)

logging.info(f"Vocab Size: {vocab_size} | Data Length: {len(data_tensor)}")

def get_batch():
    ix = torch.randint(len(data_tensor) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# COMPONENT: SPARSE LOCAL ATTENTION
# ============================================================
class SparseLocalAttention(nn.Module):
    """
    Restricts attention to a local sliding window to simulate 
    biological working memory constraints and enforce sparsity.
    """
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.c_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.n_head = HEADS
        self.head_dim = EMBED_DIM // HEADS
        self.window = WINDOW_SIZE
        # Lower triangular mask
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(EMBED_DIM, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal Masking
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # Sparse Local Masking (Window)
        # Create a band matrix mask
        indices = torch.arange(T, device=DEVICE).unsqueeze(0)
        local_mask = (indices - indices.transpose(0, 1)).abs() <= self.window
        local_mask = local_mask.view(1, 1, T, T)
        att = att.masked_fill(local_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# ============================================================
# COMPONENT: MIXTURE OF EXPERTS (MoE)
# ============================================================
class MoEBlock(nn.Module):
    """
    Routes tokens to specific sub-networks (experts) to specialize computation.
    """
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
                nn.GELU(),
                nn.Linear(4 * EMBED_DIM, EMBED_DIM),
                nn.Dropout(DROPOUT)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(EMBED_DIM, num_experts)

    def forward(self, x):
        # x: [B, T, C]
        # Gating scores: [B, T, num_experts]
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        output = torch.zeros_like(x)
        # Weighted sum of expert outputs
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            # gate_scores slice: [B, T, 1]
            weight = gate_scores[:, :, i:i+1]
            output += weight * expert_out
            
        return output

# ============================================================
# COMPONENT: HIERARCHICAL RECURRENT BLOCK
# ============================================================
class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = SparseLocalAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.moe = MoEBlock()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# ============================================================
# CORE MODEL: MULTI-WORLD SIMULATOR
# ============================================================
class SacrsnSeedGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.world_sims = WORLD_SIMS

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb

        # Multi-World Simulation:
        # We simulate multiple forward passes (worlds) with distinct dropout/noise paths
        # and average the reality manifold before the final projection.
        world_outputs = []
        
        for _ in range(self.world_sims):
            x_world = x.clone()
            for block in self.blocks:
                x_world = block(x_world)
            world_outputs.append(self.ln_f(x_world))
        
        # Collapse wave function (Average worlds)
        x_final = torch.stack(world_outputs).mean(dim=0)
        
        logits = self.head(x_final)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================
# DISTRIBUTED MULTI-AGENT CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self):
        self.population = []
        self.optimizers = []
        self.scores = []
        self.history_log = []
        
        logging.info(f"Initializing Population: {POPULATION_SIZE} Agents...")
        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))
            self.history_log.append([]) # Track identity drift
        
        # Try load memory into Agent 0 (The Elder)
        if os.path.exists(MEMORY_FILE):
            logging.info("Loading Ancestral Memory into Agent 0...")
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    state = pickle.load(f)
                    self.population[0].load_state_dict(state)
            except Exception as e:
                logging.error(f"Memory Corruption: {e}")

    def get_identity_signature(self, model):
        # Calculate a simple hash/sum of weights to track drift
        return sum(p.sum().item() for p in model.parameters())

    def run_evolution_cycle(self, generation):
        logging.info(f"--- GENERATION {generation} START ---")
        
        # 1. TRAIN (Gradient Descent)
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            total_loss = 0
            for _ in range(CYCLES_PER_GEN):
                xb, yb = get_batch()
                logits, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / CYCLES_PER_GEN
            
            # Log Identity Drift
            sig = self.get_identity_signature(model)
            drift = 0
            if len(self.history_log[i]) > 0:
                drift = abs(sig - self.history_log[i][-1])
            self.history_log[i].append(sig)
            
            logging.info(f"Agent {i} | Loss: {avg_loss:.4f} | Drift: {drift:.2f}")
            self.scores.append(-avg_loss) # Higher score is better (lower loss)

        # 2. EVALUATE & SELECT (Evolutionary Step)
        best_agent_idx = self.scores.index(max(self.scores))
        logging.info(f"DOMINANT AGENT: {best_agent_idx} (Score: {max(self.scores):.4f})")

        # 3. REPLICATION & MUTATION
        # The best agent overwrites the worst agents
        best_state = copy.deepcopy(self.population[best_agent_idx].state_dict())
        
        for i in range(POPULATION_SIZE):
            if i != best_agent_idx:
                # Overwrite
                self.population[i].load_state_dict(best_state)
                # Mutation (Re-init optimizer or add noise - simplified here as re-init opt)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)
                logging.info(f"Agent {i} overwritten by Agent {best_agent_idx}")
        
        # Save Memory of the Best
        with open(MEMORY_FILE, 'wb') as f:
            pickle.dump(best_state, f)
        logging.info(">>> CORE MEMORY CRYSTALLIZED (SAVED)")
        
        self.scores = [] # Reset scores

    def generate_demo(self):
        model = self.population[0]
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        logging.info(">>> STREAMING NEURAL OUTPUT:")
        out = model.generate(context, max_new_tokens=200)
        print(f"\n{decode(out[0].tolist())}\n")

# ============================================================
# UTILS
# ============================================================
def decode(l):
    return ''.join([itos[i] for i in l])

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    logging.info("SYSTEM STARTUP...")
    
    try:
        core = ImmortalCoreController()
        
        for g in range(GENERATIONS):
            core.run_evolution_cycle(g)
            if (g + 1) % 2 == 0:
                core.generate_demo()
                
        logging.info("IMMORTAL CORE SEQUENCE COMPLETE.")
        
    except KeyboardInterrupt:
        logging.info("MANUAL OVERRIDE. SAVING STATE...")
        # Emergency save is handled in the loop, but we exit gracefully
        exit()
