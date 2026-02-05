# ============================================================
# SACRSN-SEED IMMORTAL CORE v5.1 — THE OMEGA BUILD
# Combines: v3.8 Features + 7 Critical Stability Fixes
# ============================================================
#
# CRITICAL FIXES IMPLEMENTED:
# 1. MoE Balance Loss: Now computed, returned, and added to training loss.
# 2. Sparse Mask Cache: Pre-computed buffers to reduce O(T^2) overhead.
# 3. Parallel Worlds: Uses "Super-Batching" (B*W) instead of serial loops.
# 4. Identity Metric: Uses Vector Norm instead of Mean for high-signal drift.
# 5. Optimizer State: Momentum is copied during evolution, not reset.
# 6. Pad Token: Explicit <PAD> at index 0 to avoid collision.
# 7. True Destruction: Weight wipe mechanism added before regeneration.
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
import random
import sys

# ============================================================
# 1. CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Hyperparameters
EMBED_DIM = 384
LAYERS = 6
HEADS = 6
BLOCK_SIZE = 256
BATCH_SIZE = 16         # Effective batch size = 16 * 5 (Worlds) = 80 per GPU
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64
NUM_EXPERTS = 4
WORLD_SIM = 5           # Parallel Simulation Count
AUX_LOSS_WEIGHT = 0.01  # Weight for MoE load balancing

# Lifecycle
POPULATION_SIZE = 4
GENERATIONS = 20
CYCLES_PER_GEN = 250    # Training steps
REGENERATE_STEPS = 50   # Dreaming steps
EVAL_BATCHES = 4
GRAD_CLIP = 1.0

# Curriculum & Destruction
CURRICULUM_STEPS = [0.25, 0.5, 0.75, 1.0]
WIPE_RATIO = 0.1        # Percentage of weights to destroy during regeneration

# Paths
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data.txt"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ============================================================
# 2. DATA INFRASTRUCTURE (FIX #6: Explicit Pad Token)
# ============================================================
def setup_data(rank):
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            logging.warning(">>> Data not found. Generating synthetic quantum noise...")
            with open(DATA_PATH, "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    
    if NUM_GPUS > 1:
        dist.barrier()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # FIX #6: Reserve index 0 for Padding
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars) + 1 
    stoi = {ch: i+1 for i, ch in enumerate(chars)} # Start at 1
    itos = {i+1: ch for i, ch in enumerate(chars)}
    itos[0] = "<PAD>"
    stoi["<PAD>"] = 0
    
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    
    if rank == 0:
        logging.info(f">>> VOCAB: {vocab_size} (Inc. PAD) | TOKENS: {len(data_tensor)}")
        
    return data_tensor, vocab_size, itos, stoi

# Globals
data_tensor = None
vocab_size = 0
itos = {}
stoi = {}

def decode(tokens): 
    return "".join([itos.get(t, "") for t in tokens if t != 0])

def get_batch(difficulty=1.0):
    """Curriculum batching with padding support."""
    seq_len = max(16, int(BLOCK_SIZE * difficulty))
    if len(data_tensor) < seq_len + 1:
        seq_len = len(data_tensor) - 2
        
    ix = torch.randint(len(data_tensor) - seq_len, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    
    # Pad if curriculum length < BLOCK_SIZE
    if seq_len < BLOCK_SIZE:
        pad = torch.zeros(BATCH_SIZE, BLOCK_SIZE - seq_len, dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
        y = torch.cat([y, pad], dim=1)
        
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 3. MEMORY & IDENTITY (FIX #4: Stronger Metric)
# ============================================================
def get_identity(model):
    """FIX #4: Uses Vector Norm of first 512 params for high-signal fingerprint."""
    # Collect flattened parameters from the first few layers
    params = []
    for p in model.parameters():
        params.append(p.flatten())
        if sum(len(x) for x in params) > 512:
            break
    vec = torch.cat(params)[:512]
    return torch.norm(vec).item()

def save_memory(model):
    try:
        m = model.module if hasattr(model, "module") else model
        state = m.state_dict()
        if os.path.exists(MEMORY_FILE):
            os.rename(MEMORY_FILE, MEMORY_BACKUP)
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump(state, f)
        logging.info(">>> MEMORY SAVED (.pkl)")
    except Exception as e:
        logging.error(f"Memory save failed: {e}")

def save_checkpoint(model, optimizer, generation, tag=""):
    try:
        m = model.module if hasattr(model, "module") else model
        checkpoint = {
            "model_state": m.state_dict(),
            "optim_state": optimizer.state_dict(),
            "generation": generation,
            "timestamp": time.time()
        }
        filename = f"checkpoint_gen{generation}{tag}.pt"
        path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)
        logging.info(f">>> FULL CHECKPOINT SAVED: {path}")
    except Exception as e:
        logging.error(f"Checkpoint failed: {e}")

# ============================================================
# 4. NEURAL ARCHITECTURE
# ============================================================
class SparseAttention(nn.Module):
    """FIX #2: Vectorized Attention with Cached Masks"""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM*3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.window = WINDOW_SIZE
        self.head_dim = EMBED_DIM // HEADS
        self.num_heads = HEADS
        self.gate = nn.Parameter(torch.ones(EMBED_DIM))
        
        # FIX #2: Cache Causal Mask
        causal_mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))
        
        # FIX #2: Cache Sparse Window Mask
        indices = torch.arange(BLOCK_SIZE).unsqueeze(0)
        local_mask = (indices - indices.transpose(0, 1)).abs() <= self.window
        self.register_buffer("local_mask", local_mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply Cached Masks (Sliced to current T)
        att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(self.local_mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out * self.gate)

class MoEBlock(nn.Module):
    """FIX #1: Exports Balance Loss"""
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBED_DIM, EMBED_DIM*4),
                nn.GELU(),
                nn.Linear(EMBED_DIM*4, EMBED_DIM),
                nn.Dropout(DROPOUT)
            ) for _ in range(NUM_EXPERTS)
        ])
        self.gate = nn.Linear(EMBED_DIM, NUM_EXPERTS)

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        # FIX #1: Store loss for retrieval
        self.balance_loss = (scores.mean(dim=(0,1)) ** 2).sum() * NUM_EXPERTS 
        
        out = 0
        for i, expert in enumerate(self.experts):
            # Weighting experts
            out += scores[:,:,i:i+1] * expert(x)
        return out

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.moe = MoEBlock()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class SacrsnSeedGPT(nn.Module):
    """FIX #3: Parallel Multi-World Simulation"""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.world_sims = WORLD_SIM

    def forward(self, idx, targets=None, noise_scale=0.0):
        B,T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok + pos
        
        # FIX #3: Super-Batching (Parallel Worlds)
        # Reshape [B, T, C] -> [Worlds*B, T, C]
        x_expanded = x.repeat(self.world_sims, 1, 1)
        
        # Add Independent Noise per World
        if noise_scale > 0:
            noise = torch.randn_like(x_expanded) * noise_scale
            x_expanded = x_expanded + noise
            
        # Parallel Pass
        for block in self.blocks:
            x_expanded = block(x_expanded)
            
        x_expanded = self.ln_f(x_expanded)
        
        # Reshape and Mean Pool: [Worlds*B, T, C] -> [Worlds, B, T, C] -> [B, T, C]
        x_final = x_expanded.view(self.world_sims, B, T, EMBED_DIM).mean(dim=0)
        
        logits = self.head(x_final)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)
            
            # FIX #1: Gather MoE Aux Loss
            aux_loss = 0
            for block in self.blocks:
                aux_loss += block.moe.balance_loss
            loss = loss + AUX_LOSS_WEIGHT * aux_loss
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================
# 5. IMMORTAL CORE CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.population = []
        self.optimizers = []
        self.scores = []
        
        if rank == 0:
            logging.info(f">>> SPAWNING {POPULATION_SIZE} AGENTS ON {DEVICE.upper()} (Worlds={WORLD_SIM})")

        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[rank], find_unused_parameters=True)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))

        # Memory Load
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f: state = pickle.load(f)
                if world_size > 1:
                    new_state = {f"module.{k}" if not k.startswith("module.") else k: v for k,v in state.items()}
                    state = new_state
                self.population[0].load_state_dict(state, strict=False)
                if rank == 0: logging.info(">>> ANCESTRAL MEMORY RESTORED")
            except:
                if rank == 0: logging.warning(">>> MEMORY LOAD FAILED (Fresh Start)")

    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    # --- FIX #7: CATASTROPHIC DESTRUCTION ---
    def destroy_weights(self, model, ratio=WIPE_RATIO):
        """Randomly zeroes out weights to force re-learning."""
        with torch.no_grad():
            for p in model.parameters():
                mask = (torch.rand_like(p) > ratio).float()
                p.mul_(mask)

    # --- PHASE 1: TRAIN ---
    def phase_train(self, generation):
        # Noise Annealing: High to Low
        noise_level = max(0.0, 0.01 * (1.0 - generation / GENERATIONS))
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            log_agent = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                diff = random.choice(CURRICULUM_STEPS) # Curriculum
                xb, yb = get_batch(difficulty=diff)
                
                _, loss = model(xb, yb, noise_scale=noise_level)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                
                if log_agent and (step % 50 == 0 or step == CYCLES_PER_GEN - 1):
                    logging.info(f"[Agent 0] Step {step+1}/{CYCLES_PER_GEN} | Loss: {loss.item():.4f} | Diff: {diff}")

    # --- PHASE 2: REGENERATE (Dreaming + Destruction) ---
    def phase_regenerate(self):
        if self.rank == 0: logging.info("  [PHASE 2] DESTRUCTION & REGENERATION...")
        
        for model, opt in zip(self.population, self.optimizers):
            # Apply destruction to simulate trauma/learning
            self.destroy_weights(model)
            model.train()
            
            for _ in range(REGENERATE_STEPS):
                xb, yb = get_batch(difficulty=1.0) # Full difficulty for recovery
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

    # --- PHASE 3: EVALUATE ---
    def phase_evaluate(self):
        self.scores = []
        for model in self.population:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    xb, yb = get_batch(difficulty=1.0)
                    _, loss = model(xb, yb)
                    total_loss += loss.item()
            
            avg_loss = total_loss / EVAL_BATCHES
            loss_tensor = torch.tensor(avg_loss).to(DEVICE)
            
            # Split-Brain Fix
            if self.world_size > 1:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= self.world_size
            
            self.scores.append(-loss_tensor.item())

    # --- PHASE 4: EVOLVE (FIX #5: Optimizer Copy) ---
    def phase_evolve(self, generation):
        best_idx = self.scores.index(max(self.scores))
        if self.rank == 0:
            logging.info(f"  [PHASE 4] DOMINANT AGENT: {best_idx} | Score: {self.scores[best_idx]:.4f}")

        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())
        best_optim_state = copy.deepcopy(self.optimizers[best_idx].state_dict())

        for i in range(POPULATION_SIZE):
            if i != best_idx:
                self.unwrap(self.population[i]).load_state_dict(best_state)
                # FIX #5: Load best optimizer state (Momentum transfer)
                self.optimizers[i].load_state_dict(best_optim_state)

        if self.rank == 0:
            save_memory(best_agent)
            save_checkpoint(self.population[best_idx], self.optimizers[best_idx], generation)

    def run_cycle(self, generation):
        if self.rank == 0: logging.info(f"\n=== EVOLUTION CYCLE {generation} ===")
        
        ids_before = [get_identity(m) for m in self.population]
        
        self.phase_train(generation)
        self.phase_regenerate()
        self.phase_evaluate()
        self.phase_evolve(generation)
        
        if self.rank == 0:
            ids_after = [get_identity(m) for m in self.population]
            for i, (b, a) in enumerate(zip(ids_before, ids_after)):
                logging.info(f"    Agent {i} Drift: {a - b:.6f}")

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.population[0])
            model.eval()
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(ctx, max_new_tokens=300, temperature=0.8, top_k=50)
            text = decode(out[0].tolist())
            print(f"\n[DEMO OUTPUT]\n{text}\n")

# ============================================================
# 6. EXECUTION
# ============================================================
def run(rank, world_size):
    global data_tensor, vocab_size, itos, stoi 
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    data_tensor, vocab_size, itos, stoi = setup_data(rank)
    core = ImmortalCoreController(rank, world_size)
    
    try:
        for g in range(GENERATIONS):
            core.run_cycle(g)
            if (g+1) % 2 == 0:
                core.generate_demo()
                
    except KeyboardInterrupt:
        if rank == 0:
            logging.info("\n>>> INTERRUPT DETECTED. SAVING...")
            best_agent = core.unwrap(core.population[0])
            save_memory(best_agent)
            save_checkpoint(core.population[0], core.optimizers[0], 999, tag="_interrupt")
            
    finally:
        if world_size > 1:
            dist.destroy_process_group()
        if rank == 0:
            logging.info(">>> SYSTEM SHUTDOWN.")

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        run(0, 1)
