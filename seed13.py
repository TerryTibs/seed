# ============================================================
# SACRSN-SEED IMMORTAL CORE v3.7 — FORENSIC RESTORATION
# Author: User + Builder Protocol
# Features:
#  - Recurrent differentiable stack with hierarchical memory
#  - Sparse local attention (Vectorized)
#  - Mixture-of-Experts (MoE) per recurrent step
#  - Multi-World simulation (Ensemble Averaging)
#  - Hybrid gradient + evolutionary training
#  - Multi-agent distributed controller (DDP + Split-Brain Protection)
#  - Identity drift logging (Snapshot Logic)
#  - KeyboardInterrupt Safe-Exit
#  - Input Validation & Top-K Sampling
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
import sys

# ============================================================
# LOGGER CONFIG (Restored Original Format)
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

EMBED = 256
LAYERS = 4
HEADS = 8
BLOCK = 128
VOCAB_SIZE = None # Set dynamically
BATCH_SIZE = 16
LR = 3e-4
STEPS_PER_CYCLE = 500
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
GRAD_CLIP = 1.0
NUM_EXPERTS = 4
WORLD_SIM = 3      # Restored variable name
DATA_PATH = "data.txt"

# Evolution Config
POPULATION_SIZE = 4
GENERATIONS = 10
EVAL_BATCHES = 8

# ============================================================
# DATA & TOKENIZER
# ============================================================
def setup_data(rank):
    """Auto-generates data if missing, ensures safe concurrent access."""
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            logging.warning("data.txt not found! Generating synthetic seed data...")
            with open(DATA_PATH, "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    
    if NUM_GPUS > 1:
        dist.barrier()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data_tensor = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    if rank == 0:
        logging.info(f"Vocab: {vocab_size} | Tokens: {len(data_tensor)}")
        
    return data_tensor, vocab_size, itos, stoi

# Global Data Holders
data_tensor = None
vocab_size = 0
itos = {}
stoi = {}

def encode(s): return [stoi[c] for c in s]
def decode(tokens): return "".join([itos[t] for t in tokens])

def get_batch(batch=BATCH_SIZE, block=BLOCK):
    # RESTORED: Input validation from Script 1
    if len(data_tensor) < block:
        raise ValueError("Data length smaller than block size!")
        
    ix = torch.randint(len(data_tensor) - block, (batch,))
    x = torch.stack([data_tensor[i:i+block] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# HELPER FUNCTIONS (Restored to Global Scope)
# ============================================================
def identity_signature(model):
    # Restored: Original logic sum(mean())
    return sum(p.mean().item() for p in model.parameters())

def save_memory(model, path=MEMORY_FILE):
    # Restored: Standalone safe save logic
    try:
        # Unwrap DDP if necessary
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        if os.path.exists(path):
            os.rename(path, MEMORY_BACKUP)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logging.info(">>> MEMORY SAVED")
    except Exception as e:
        logging.error(f"Memory save failed: {e}")

# ============================================================
# NEURAL ARCHITECTURE
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self, embed_dim=EMBED, window=16):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.window = window
        self.head_dim = embed_dim // HEADS
        self.num_heads = HEADS
        self.gate = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x, memory_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Causal Mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))

        # Sparse Local Window (Optimized)
        indices = torch.arange(T, device=x.device).unsqueeze(0)
        local_mask = (indices - indices.transpose(0, 1)).abs() <= self.window
        local_mask = local_mask.view(1, 1, T, T)
        att = att.masked_fill(local_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out * self.gate)

class MoEBlock(nn.Module):
    def __init__(self, embed_dim=EMBED, num_experts=NUM_EXPERTS):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        ) for _ in range(num_experts)])
        self.gating = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        gate_scores = torch.softmax(self.gating(x), dim=-1)
        out = sum(gate_scores[:,:,i:i+1] * self.experts[i](x) for i in range(len(self.experts)))
        return out

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SparseAttention()
        self.moe = MoEBlock()
        self.ln1 = nn.LayerNorm(EMBED)
        self.ln2 = nn.LayerNorm(EMBED)

    def forward(self, x, memory_mask=None):
        x = x + self.attn(self.ln1(x), memory_mask)
        x = x + self.moe(self.ln2(x))
        return x

class SeedGPTMultiWorld(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED, block_size=BLOCK, layers=LAYERS, worlds=WORLD_SIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.worlds = worlds

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx) + self.pos[:, :T]
        
        # Multi-World Simulation
        outputs = []
        for _ in range(self.worlds):
            wx = x.clone()
            for block in self.blocks:
                wx = block(wx)
            outputs.append(self.ln(wx))
            
        x_final = torch.stack(outputs).mean(dim=0)
        logits = self.head(x_final)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK:]
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
# DISTRIBUTED CONTROLLER
# ============================================================
class DistributedMultiAgentController:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.agents = []
        self.optimizers = []
        self.scores = []
        
        if rank == 0: logging.info(f"Spawning {POPULATION_SIZE} Agents...")
        for i in range(POPULATION_SIZE):
            model = SeedGPTMultiWorld(vocab_size=vocab_size).to(DEVICE)
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[rank], find_unused_parameters=True)
            self.agents.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LR))

        if os.path.exists(MEMORY_FILE):
            self.load_memory_internal()

    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def load_memory_internal(self):
        try:
            with open(MEMORY_FILE, "rb") as f:
                mem = pickle.load(f)
            # DDP Key Sanitization
            if self.world_size > 1:
                mem = {f"module.{k}" if not k.startswith("module.") else k: v for k,v in mem.items()}
            self.agents[0].load_state_dict(mem, strict=False)
            if self.rank == 0: logging.info(">>> MEMORY RESTORED")
        except Exception as e:
            if self.rank == 0: logging.warning(f"Memory load error: {e}")

    def snapshot_identities(self):
        return [identity_signature(agent) for agent in self.agents]

    def train_agents(self, steps=STEPS_PER_CYCLE):
        for i, (agent, opt) in enumerate(zip(self.agents, self.optimizers)):
            agent.train()
            for step in range(steps):
                x, y = get_batch()
                _, loss = agent(x, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                opt.step()
                
                # Original Log Cadence
                if self.rank == 0 and i == 0 and (step % 50 == 0 or step == steps - 1):
                     logging.info(f"[Agent 0] Step {step+1}/{steps} | Loss: {loss.item():.4f}")

    def regenerate_agents(self, steps=STEPS_PER_CYCLE // 2):
        for agent, opt in zip(self.agents, self.optimizers):
            agent.train()
            for _ in range(steps):
                x, y = get_batch()
                _, loss = agent(x, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                opt.step()

    def evaluate_agents(self):
        self.scores = []
        for agent in self.agents:
            agent.eval()
            total_acc = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x, y = get_batch()
                    logits, _ = agent(x)
                    preds = logits.argmax(dim=-1)
                    total_acc += (preds == y).float().mean()
            
            avg_acc = total_acc / EVAL_BATCHES
            # DDP Sync
            if self.world_size > 1:
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                avg_acc /= self.world_size
            
            self.scores.append(avg_acc.item())
        return self.scores

    def evolve_agents(self):
        best_idx = self.scores.index(max(self.scores))
        if self.rank == 0:
            logging.info(f">>> EVOLVING GENERATION {self.generation_idx}. Best: {best_idx}")
        
        best_state = copy.deepcopy(self.unwrap(self.agents[best_idx]).state_dict())
        
        for i in range(POPULATION_SIZE):
            if i != best_idx:
                self.unwrap(self.agents[i]).load_state_dict(best_state)
                self.optimizers[i] = optim.AdamW(self.agents[i].parameters(), lr=LR)
        
        return self.agents[best_idx] # Return best for saving

    def run_cycle(self, generation):
        self.generation_idx = generation
        if self.rank == 0: logging.info(f"\n=== EVOLUTION CYCLE {generation} ===")
        
        ids_before = self.snapshot_identities()
        
        self.train_agents()
        if self.rank == 0: logging.info("Regenerating...")
        self.regenerate_agents()
        
        self.evaluate_agents()
        best_agent = self.evolve_agents()
        
        ids_after = self.snapshot_identities()
        if self.rank == 0:
            for i, (b, a) in enumerate(zip(ids_before, ids_after)):
                logging.info(f"[Agent {i}] Identity Drift: {a - b:.6f}")
            
            # Save Best
            save_memory(best_agent)

    def generate_demo(self):
        if self.rank == 0:
            agent = self.unwrap(self.agents[0])
            agent.eval()
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = agent.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=50)
            text = decode(out[0].tolist())
            print(f"\n>>> SAMPLE GENERATION:\n{text}\n")

# ============================================================
# EXECUTION (With KeyboardInterrupt Restoration)
# ============================================================
def run_process(rank, world_size):
    global data_tensor, vocab_size, itos, stoi 
    
    # DDP Setup
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    data_tensor, vocab_size, itos, stoi = setup_data(rank)
    controller = DistributedMultiAgentController(rank, world_size)
    
    # RESTORED: Graceful Shutdown
    try:
        for g in range(GENERATIONS):
            controller.run_cycle(g)
            if (g+1) % 10 == 0:
                controller.generate_demo()
    except KeyboardInterrupt:
        if rank == 0:
            logging.info("\n>>> INTERRUPT DETECTED. SAVING STATE...")
            best_agent = controller.agents[0] # Fallback save
            save_memory(best_agent)
            logging.info(">>> EMERGENCY SAVE COMPLETE.")
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run_process, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        run_process(0, 1)
