# ============================================================
# SACRSN-SEED IMMORTAL CORE (ULTIMATE EDITION)
# Integrated Version: v3.2 (Logic) + v3.3 (Distributed/Optimized)
# ============================================================
#
# FEATURES INCLUDED:
# 1. ARCHITECTURE:
#    - Hierarchical Recurrent Stack
#    - Vectorized Sparse Local Attention (Fixed Window)
#    - Mixture-of-Experts (MoE)
#    - Multi-World Simulation (Ensemble Averaging)
#
# 2. EVOLUTIONARY CONTROLLER:
#    - Distributed Multi-Agent System (DDP)
#    - 4-Stage Life Cycle: Train -> Regenerate -> Evaluate -> Evolve
#    - Identity Drift Tracking
#    - "Split-Brain" Prevention (Synced Scoring)
#
# 3. UTILITIES:
#    - Safe Memory Persistence (Backups)
#    - Auto-Data Generation
#    - Gradient Clipping
#    - Hybrid CPU/GPU/Multi-GPU Support
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import math
import copy
import pickle
import os
import logging
import time

# ============================================================
# 1. CONFIGURATION & LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [SACRSN-CORE] | %(message)s",
    datefmt="%H:%M:%S"
)

# Hardware Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Model Hyperparameters
EMBED_DIM = 384
LAYERS = 6
HEADS = 6
BLOCK_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64   # Sparse Attention Window
NUM_EXPERTS = 4    # MoE Experts
WORLD_SIMS = 3     # Multi-World Ensemble Count

# Evolution Config
POPULATION_SIZE = 4
GENERATIONS = 10
CYCLES_PER_GEN = 100
REGENERATE_STEPS = 50
EVAL_BATCHES = 4
GRAD_CLIP = 1.0

# Persistence
MEMORY_FILE = "sacrsn_memory.pkl"
DATA_PATH = "data.txt"

# ============================================================
# 2. DATA INFRASTRUCTURE
# ============================================================
def setup_data(rank):
    """
    Ensures data exists. Rank 0 creates it; others wait.
    Returns: data_tensor, vocab_size, itos, stoi
    """
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            logging.warning("data.txt not found. Generating synthetic quantum noise...")
            with open(DATA_PATH, "w") as f:
                # Synthetic data for demonstration
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    
    # Synchronization Barrier: Wait for Rank 0 to finish writing
    if NUM_GPUS > 1:
        dist.barrier()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    
    if rank == 0:
        logging.info(f"Vocab: {vocab_size} | Data Tokens: {len(data_tensor)}")
        
    return data_tensor, vocab_size, itos, stoi

# Globals to be populated in 'run'
data_tensor = None
vocab_size = 0
itos = {}
stoi = {}

def get_batch():
    ix = torch.randint(len(data_tensor) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 3. MODEL ARCHITECTURE (OPTIMIZED)
# ============================================================

class SparseLocalAttention(nn.Module):
    """Vectorized Sparse Attention with Fixed Window and Causal Masking"""
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(EMBED_DIM, 3*EMBED_DIM)
        self.c_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.n_head = HEADS
        self.head_dim = EMBED_DIM // HEADS
        self.window = WINDOW_SIZE
        # Causal Mask Buffer
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                             .view(1,1,BLOCK_SIZE,BLOCK_SIZE))

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(EMBED_DIM, dim=2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        
        # Standard Self-Attention Score
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 1. Apply Causal Mask
        mask = self.bias[:,:, :T, :T]
        att = att.masked_fill(mask==0, float('-inf'))
        
        # 2. Apply Sparse Local Window Mask (Vectorized)
        indices = torch.arange(T, device=x.device).unsqueeze(0)
        local_mask = (indices - indices.transpose(0,1)).abs() <= self.window
        local_mask = local_mask.view(1,1,T,T)
        att = att.masked_fill(local_mask==0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)

class MoEBlock(nn.Module):
    """Mixture-of-Experts Layer"""
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBED_DIM, 4*EMBED_DIM),
                nn.GELU(),
                nn.Linear(4*EMBED_DIM, EMBED_DIM),
                nn.Dropout(DROPOUT)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(EMBED_DIM, num_experts)

    def forward(self, x):
        # Gating
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        # Weighted Sum of Experts
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            output += gate_scores[:,:,i:i+1] * expert(x)
        return output

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

class SacrsnSeedGPT(nn.Module):
    """Multi-World Simulator Wrapper"""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.world_sims = WORLD_SIMS

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        
        # Multi-World Simulation (Ensemble Averaging)
        world_outputs = []
        for _ in range(self.world_sims):
            x_world = x.clone()
            for block in self.blocks:
                x_world = block(x_world)
            world_outputs.append(self.ln_f(x_world))
        
        # Collapse Reality Manifold
        x_final = torch.stack(world_outputs).mean(dim=0)
        logits = self.head(x_final)
        
        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab_size)
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
# 4. DISTRIBUTED IMMORTAL CORE CONTROLLER
# ============================================================

class ImmortalCoreController:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.population = []
        self.optimizers = []
        self.scores = []
        self.identity_log = []

        logging.info(f"[RANK {rank}] Initializing Population: {POPULATION_SIZE} Agents...")
        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            
            # DDP Wrapping
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, 
                    device_ids=[rank] if DEVICE=="cuda" else None,
                    find_unused_parameters=True # Needed for MoE/Multi-World
                )
            
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))
            self.identity_log.append([])

        # Safe Memory Loading
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    state = pickle.load(f)
                    
                # Fix keys for DDP compatibility
                if world_size > 1:
                    new_state = {}
                    for k, v in state.items():
                        key = f"module.{k}" if not k.startswith("module.") else k
                        new_state[key] = v
                    state = new_state
                
                self.population[0].load_state_dict(state, strict=False)
                logging.info(f"[RANK {rank}] Ancestral Memory Restored.")
            except Exception as e:
                logging.warning(f"[RANK {rank}] Memory load failed: {e}")

    def unwrap(self, model):
        """Helper to get underlying model from DDP wrapper"""
        return model.module if hasattr(model, "module") else model

    def get_identity(self, model):
        """Simple hash of weights to track drift"""
        return sum(p.mean().item() for p in model.parameters())

    # --- PHASE 1: TRAIN (Gradient Descent) ---
    def phase_train(self):
        for model, opt in zip(self.population, self.optimizers):
            model.train()
            for _ in range(CYCLES_PER_GEN):
                xb, yb = get_batch()
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

    # --- PHASE 2: REGENERATE (Self-Consolidation) ---
    def phase_regenerate(self):
        """Secondary training phase to consolidate memory"""
        for model, opt in zip(self.population, self.optimizers):
            model.train()
            for _ in range(REGENERATE_STEPS):
                xb, yb = get_batch()
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

    # --- PHASE 3: EVALUATE (Accuracy + Sync) ---
    def phase_evaluate(self):
        self.scores = []
        for i, model in enumerate(self.population):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    xb, yb = get_batch()
                    logits, _ = model(xb)
                    preds = logits.argmax(dim=-1)
                    mask = (yb != -1)
                    correct += (preds == yb).float().sum()
                    total += mask.sum()
            
            # Sync Scores across GPUs
            local_acc = correct / total
            if self.world_size > 1:
                dist.all_reduce(local_acc, op=dist.ReduceOp.SUM)
                local_acc /= self.world_size
            
            self.scores.append(local_acc.item())

            # Log Drift
            sig = self.get_identity(model)
            if self.rank == 0 and len(self.identity_log[i]) > 0:
                drift = sig - self.identity_log[i][-1]
                logging.info(f"  [Agent {i}] Acc: {local_acc:.4f} | Drift: {drift:.6f}")
            self.identity_log[i].append(sig)

    # --- PHASE 4: EVOLVE (Selection & Mutation) ---
    def phase_evolve(self):
        # Select
        best_idx = self.scores.index(max(self.scores))
        if self.rank == 0:
            logging.info(f">>> EVOLUTION: Dominant Agent {best_idx} (Score: {self.scores[best_idx]:.4f})")

        # Copy Best State
        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())

        # Overwrite Others
        for i in range(POPULATION_SIZE):
            if i != best_idx:
                self.unwrap(self.population[i]).load_state_dict(best_state)
                # Reset Optimizer (Mutation via fresh momentum)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

        # Save
        if self.rank == 0:
            self.save_memory(best_state)

    def save_memory(self, state):
        try:
            if os.path.exists(MEMORY_FILE):
                os.rename(MEMORY_FILE, f"{MEMORY_FILE}.backup")
            with open(MEMORY_FILE, 'wb') as f:
                pickle.dump(state, f)
            logging.info("  [System] Memory Crystallized & Backed Up.")
        except Exception as e:
            logging.error(f"  [System] Save Failed: {e}")

    # --- MASTER LIFECYCLE ---
    def run_cycle(self, generation):
        if self.rank == 0:
            logging.info(f"\n=== CYCLE {generation} ===")
        
        self.phase_train()
        if self.rank == 0: logging.info("  [Phase] Regenerating...")
        self.phase_regenerate()
        if self.rank == 0: logging.info("  [Phase] Evaluating...")
        self.phase_evaluate()
        self.phase_evolve()

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.population[0])
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(context, max_new_tokens=200, temperature=0.9)
            text = "".join([itos[i] for i in out[0].tolist()])
            print(f"\n>>> AGENT OUTPUT:\n{text}\n")

# ============================================================
# 5. EXECUTION & MULTIPROCESSING
# ============================================================
def run(rank, world_size):
    global data_tensor, vocab_size, itos, stoi 
    
    # 1. Setup DDP Environment
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 2. Load Data
    data_tensor, vocab_size, itos, stoi = setup_data(rank)

    # 3. Start Controller
    core = ImmortalCoreController(rank, world_size)
    
    # 4. Evolution Loop
    for g in range(GENERATIONS):
        core.run_cycle(g)
        if (g+1) % 2 == 0:
            core.generate_demo()
            
    # 5. Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        print(f"SACRSN-CORE: Spawning across {NUM_GPUS} GPUs.")
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        print("SACRSN-CORE: Running in Single-Process Mode.")
        run(0, 1)
