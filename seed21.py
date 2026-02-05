# ============================================================
# SACRSN-SEED IMMORTAL CORE v5.0 — THE OMEGA BUILD
# Combines every feature from seed1 through seed20
# ============================================================
# Features:
# 1. ARCHITECTURE:
#    - Recurrent Stack + Vectorized Sparse Attention (seed5/20)
#    - Mixture-of-Experts with Load Balancing (seed14)
#    - Deep Multi-World Simulation with Noise Injection (seed19)
#
# 2. TRAINING DYNAMICS:
#    - Curriculum Learning (Variable Sequence Lengths) (seed6)
#    - Noise Annealing (High variance -> Low variance) (seed19)
#    - 4-Phase Lifecycle: Train -> Regenerate -> Evaluate -> Evolve (seed10)
#
# 3. INFRASTRUCTURE:
#    - DDP Distributed Training (seed1/9)
#    - "Split-Brain" Score Synchronization (seed9)
#    - Real-time Console Logging (seed1/7)
#    - Dual Saving: .pkl (Memory) + .pt (Full Checkpoint) (seed15/16)
#    - Graceful Shutdown on KeyboardInterrupt (seed13)
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
import sys
import time
import random

# ============================================================
# 1. CONFIGURATION & LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Hyperparameters (Merged from seed.py and seed20)
EMBED_DIM = 384         # Increased from 256
LAYERS = 6              # Increased from 4
HEADS = 8
BLOCK_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64        # Sparse Window
NUM_EXPERTS = 4
WORLD_SIM = 5           # Deep Simulation

# Training & Evolution
POPULATION_SIZE = 4
GENERATIONS = 20
CYCLES_PER_GEN = 500    # Training steps
REGENERATE_STEPS = 100  # Dreaming steps
EVAL_BATCHES = 8
GRAD_CLIP = 1.0

# Curriculum (from seed6)
CURRICULUM_STEPS = [0.25, 0.5, 0.75, 1.0]

# File Paths
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data.txt"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ============================================================
# 2. DATA INFRASTRUCTURE (With Curriculum)
# ============================================================
def setup_data(rank):
    """Auto-generates data if missing, ensures safe concurrent access."""
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            logging.warning(">>> Data not found. Generating synthetic quantum noise...")
            with open(DATA_PATH, "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    
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
        logging.info(f">>> VOCAB: {vocab_size} | TOKENS: {len(data_tensor)}")
        
    return data_tensor, vocab_size, itos, stoi

# Globals
data_tensor = None
vocab_size = 0
itos = {}
stoi = {}

def decode(tokens): return "".join([itos[t] for t in tokens])

def get_batch(difficulty=1.0):
    """
    Curriculum Batch Generator (Restored from seed6).
    difficulty: float 0.0 to 1.0, determines sequence length.
    """
    seq_len = max(16, int(BLOCK_SIZE * difficulty))
    if len(data_tensor) < seq_len + 1:
        seq_len = len(data_tensor) - 2
        
    ix = torch.randint(len(data_tensor) - seq_len, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    
    # Pad if curriculum length < BLOCK_SIZE (for static graph compatibility)
    if seq_len < BLOCK_SIZE:
        pad = torch.zeros(BATCH_SIZE, BLOCK_SIZE - seq_len, dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
        y = torch.cat([y, pad], dim=1)
        
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 3. IDENTITY, MEMORY & CHECKPOINTING
# ============================================================
def identity_signature(model):
    """Restored: sum(mean())"""
    return sum(p.mean().item() for p in model.parameters())

def save_memory(model):
    """Saves lightweight weights only (.pkl)"""
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
    """Saves full training state (.pt)"""
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
    """Vectorized Sparse Attention + Cached Mask (seed20)"""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM*3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.window = WINDOW_SIZE
        self.head_dim = EMBED_DIM // HEADS
        self.num_heads = HEADS
        self.gate = nn.Parameter(torch.ones(EMBED_DIM))
        
        # Cache causal mask
        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal Masking
        causal = self.causal_mask[:T,:T].view(1,1,T,T)
        att = att.masked_fill(causal==0, float('-inf'))
        
        # Sparse Window Masking
        idx = torch.arange(T, device=x.device)
        local = (idx[None,:] - idx[:,None]).abs() <= self.window
        local = local.view(1,1,T,T)
        att = att.masked_fill(local==0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out * self.gate)

class MoEBlock(nn.Module):
    """Mixture of Experts + Load Balancing (seed14)"""
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
        scores = torch.softmax(self.gate(x), dim=-1)
        self.balance_loss = scores.mean() # For auxiliary loss if needed
        out = 0
        for i, expert in enumerate(self.experts):
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
    """Multi-World Simulator + Noise Injection (seed19)"""
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
        
        # Multi-World Simulation with Noise Annealing
        world_outputs = []
        for _ in range(self.world_sims):
            wx = x.clone()
            if noise_scale > 0:
                wx = wx + torch.randn_like(wx) * noise_scale
            for block in self.blocks:
                wx = block(wx)
            world_outputs.append(self.ln_f(wx))
        
        # Collapse Reality
        x_final = torch.stack(world_outputs).mean(dim=0)
        logits = self.head(x_final)
        
        loss = None
        if targets is not None:
            # Flatten for CE Loss
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling (Restored from seed1/12)
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
        self.identity_log = []

        if rank == 0:
            logging.info(f">>> SPAWNING {POPULATION_SIZE} AGENTS ON {DEVICE.upper()} (World={world_size})")

        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[rank], find_unused_parameters=True)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))
            self.identity_log.append([])

        # Safe Loading
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    state = pickle.load(f)
                # DDP Key Fix
                if world_size > 1:
                    new_state = {}
                    for k, v in state.items():
                        key = f"module.{k}" if not k.startswith("module.") else k
                        new_state[key] = v
                    state = new_state
                self.population[0].load_state_dict(state, strict=False)
                if rank == 0: logging.info(">>> ANCESTRAL MEMORY RESTORED")
            except Exception as e:
                if rank == 0: logging.warning(f">>> MEMORY LOAD FAILED: {e}")

    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def get_identity(self, model):
        return sum(p.mean().item() for p in model.parameters())

    # --- PHASE 1: TRAIN (With Logging & Curriculum) ---
    def phase_train(self, generation):
        # Noise Annealing (seed19): High noise early, low noise later
        noise_level = max(0.0, 0.005 * (1.0 - generation / GENERATIONS))
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            log_agent = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                # Curriculum (seed6): Random difficulty selection
                diff = random.choice(CURRICULUM_STEPS)
                xb, yb = get_batch(difficulty=diff)
                
                _, loss = model(xb, yb, noise_scale=noise_level)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                
                # VERBOSE LOGGING (Restored from seed1)
                if log_agent and (step % 50 == 0 or step == CYCLES_PER_GEN - 1):
                    logging.info(f"[Agent 0] Step {step+1}/{CYCLES_PER_GEN} | Loss: {loss.item():.4f} | Diff: {diff}")

    # --- PHASE 2: REGENERATE (Dreaming) ---
    def phase_regenerate(self):
        if self.rank == 0: logging.info("  [PHASE 2] REGENERATION (DREAMING)...")
        for model, opt in zip(self.population, self.optimizers):
            model.train()
            for _ in range(REGENERATE_STEPS):
                xb, yb = get_batch(difficulty=1.0)
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

    # --- PHASE 3: EVALUATE (Split-Brain Safe) ---
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
            # Sync scores across GPUs (seed9 fix)
            loss_tensor = torch.tensor(avg_loss).to(DEVICE)
            if self.world_size > 1:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= self.world_size
            
            self.scores.append(-loss_tensor.item()) # Score is negative loss

    # --- PHASE 4: EVOLVE (Selection + Checkpointing) ---
    def phase_evolve(self, generation):
        best_idx = self.scores.index(max(self.scores))
        if self.rank == 0:
            logging.info(f"  [PHASE 4] DOMINANT AGENT: {best_idx} | Score: {self.scores[best_idx]:.4f}")

        # Deep Copy Best
        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())

        # Overwrite Others
        for i in range(POPULATION_SIZE):
            if i != best_idx:
                self.unwrap(self.population[i]).load_state_dict(best_state)
                # Mutation via Optimizer Reset
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

        # Dual Saving (seed15/16)
        if self.rank == 0:
            save_memory(best_agent) # .pkl
            save_checkpoint(self.population[best_idx], self.optimizers[best_idx], generation) # .pt

    def run_cycle(self, generation):
        if self.rank == 0: logging.info(f"\n=== EVOLUTION CYCLE {generation} ===")
        
        # Snapshot Identity Before
        ids_before = [self.get_identity(m) for m in self.population]
        
        self.phase_train(generation)
        self.phase_regenerate()
        self.phase_evaluate()
        self.phase_evolve(generation)
        
        # Snapshot Identity After & Log Drift
        if self.rank == 0:
            ids_after = [self.get_identity(m) for m in self.population]
            for i, (b, a) in enumerate(zip(ids_before, ids_after)):
                logging.info(f"    Agent {i} Drift: {a - b:.6f}")

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.population[0])
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=50)
            text = decode(out[0].tolist())
            print(f"\n[DEMO OUTPUT]\n{text}\n")

# ============================================================
# 6. EXECUTION
# ============================================================
def run(rank, world_size):
    global data_tensor, vocab_size, itos, stoi 
    
    # Setup DDP
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # Load Data
    data_tensor, vocab_size, itos, stoi = setup_data(rank)
    core = ImmortalCoreController(rank, world_size)
    
    try:
        for g in range(GENERATIONS):
            core.run_cycle(g)
            if (g+1) % 2 == 0:
                core.generate_demo()
                
    except KeyboardInterrupt:
        if rank == 0:
            logging.info("\n>>> INTERRUPT DETECTED. EMERGENCY SAVE...")
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
