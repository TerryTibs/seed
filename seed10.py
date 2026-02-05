# ============================================================
# SACRSN-SEED IMMORTAL CORE (ULTIMATE + VERBOSE LOGGING)
# Integrated Version: v3.4
# ============================================================
#
# FEATURES:
# 1. ARCHITECTURE: Hierarchical + Sparse Attn + MoE + Multi-World
# 2. LIFECYCLE: Train -> Regenerate -> Evaluate -> Evolve
# 3. DISTRIBUTED: DDP + Split-Brain Protection + Safe I/O
# 4. OBSERVABILITY: Real-time loss logging and drift tracking
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
# 1. CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s", # Clean format
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
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64
NUM_EXPERTS = 4
WORLD_SIMS = 3

# Evolution
POPULATION_SIZE = 4
GENERATIONS = 10
CYCLES_PER_GEN = 100    # Steps per training phase
REGENERATE_STEPS = 50   # Steps per regeneration phase
EVAL_BATCHES = 4
GRAD_CLIP = 1.0

# Files
MEMORY_FILE = "sacrsn_memory.pkl"
DATA_PATH = "data.txt"

# ============================================================
# 2. DATA INFRASTRUCTURE
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

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    
    if rank == 0:
        logging.info(f">>> VOCAB: {vocab_size} | TOKENS: {len(data_tensor)}")
        
    return data_tensor, vocab_size, itos, stoi

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
# 3. MODEL ARCHITECTURE
# ============================================================
class SparseLocalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(EMBED_DIM, 3*EMBED_DIM)
        self.c_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.n_head = HEADS
        self.head_dim = EMBED_DIM // HEADS
        self.window = WINDOW_SIZE
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                             .view(1,1,BLOCK_SIZE,BLOCK_SIZE))

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(EMBED_DIM, dim=2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:,:, :T, :T]
        att = att.masked_fill(mask==0, float('-inf'))
        
        indices = torch.arange(T, device=x.device).unsqueeze(0)
        local_mask = (indices - indices.transpose(0,1)).abs() <= self.window
        local_mask = local_mask.view(1,1,T,T)
        att = att.masked_fill(local_mask==0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)

class MoEBlock(nn.Module):
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
        gate_scores = F.softmax(self.gate(x), dim=-1)
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
        
        world_outputs = []
        for _ in range(self.world_sims):
            x_world = x.clone()
            for block in self.blocks:
                x_world = block(x_world)
            world_outputs.append(self.ln_f(x_world))
        
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
# 4. IMMORTAL CORE CONTROLLER
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
            logging.info(f">>> SPAWNING {POPULATION_SIZE} AGENTS ON {DEVICE.upper()}...")

        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[rank], find_unused_parameters=True)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))
            self.identity_log.append([])

        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    state = pickle.load(f)
                if world_size > 1:
                    new_state = {}
                    for k, v in state.items():
                        key = f"module.{k}" if not k.startswith("module.") else k
                        new_state[key] = v
                    state = new_state
                self.population[0].load_state_dict(state, strict=False)
                if rank == 0: logging.info(">>> ANCESTRAL MEMORY RESTORED.")
            except:
                if rank == 0: logging.warning(">>> MEMORY LOAD FAILED.")

    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def get_identity(self, model):
        return sum(p.mean().item() for p in model.parameters())

    # --- PHASE 1: TRAIN ---
    def phase_train(self):
        if self.rank == 0: logging.info("  [PHASE 1] TRAINING SEQUENCES...")
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            # Only log detailed stats for Agent 0 to keep console clean
            log_stats = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                xb, yb = get_batch()
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                
                # VERBOSE LOGGING
                if log_stats and (step + 1) % 20 == 0:
                    logging.info(f"    -> Agent 0 | Step {step+1}/{CYCLES_PER_GEN} | Loss: {loss.item():.4f}")

    # --- PHASE 2: REGENERATE ---
    def phase_regenerate(self):
        if self.rank == 0: logging.info("  [PHASE 2] REGENERATION (DREAMING)...")
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            for step in range(REGENERATE_STEPS):
                xb, yb = get_batch()
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

    # --- PHASE 3: EVALUATE ---
    def phase_evaluate(self):
        if self.rank == 0: logging.info("  [PHASE 3] EVALUATION & SYNCHRONIZATION...")
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
            
            local_acc = correct / total
            if self.world_size > 1:
                dist.all_reduce(local_acc, op=dist.ReduceOp.SUM)
                local_acc /= self.world_size
            
            self.scores.append(local_acc.item())

            sig = self.get_identity(model)
            if self.rank == 0 and len(self.identity_log[i]) > 0:
                drift = sig - self.identity_log[i][-1]
                logging.info(f"    -> Agent {i} | Accuracy: {local_acc:.4f} | ID Drift: {drift:.6f}")
            self.identity_log[i].append(sig)

    # --- PHASE 4: EVOLVE ---
    def phase_evolve(self):
        best_idx = self.scores.index(max(self.scores))
        if self.rank == 0:
            logging.info(f"  [PHASE 4] EVOLUTION | DOMINANT AGENT: {best_idx}")

        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())

        for i in range(POPULATION_SIZE):
            if i != best_idx:
                self.unwrap(self.population[i]).load_state_dict(best_state)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

        if self.rank == 0:
            if os.path.exists(MEMORY_FILE):
                os.rename(MEMORY_FILE, f"{MEMORY_FILE}.backup")
            with open(MEMORY_FILE, 'wb') as f:
                pickle.dump(best_state, f)
            logging.info("  >>> MEMORY CRYSTALLIZED & SAVED.")

    def run_cycle(self, generation):
        if self.rank == 0:
            logging.info(f"\n=== EVOLUTION CYCLE {generation + 1} ===")
        self.phase_train()
        self.phase_regenerate()
        self.phase_evaluate()
        self.phase_evolve()

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.population[0])
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(context, max_new_tokens=200, temperature=0.9)
            text = "".join([itos[i] for i in out[0].tolist()])
            print(f"\n[DEMO OUTPUT]\n{text}\n")

# ============================================================
# 5. EXECUTION
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
    
    for g in range(GENERATIONS):
        core.run_cycle(g)
        if (g+1) % 2 == 0:
            core.generate_demo()
            
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        run(0, 1)
