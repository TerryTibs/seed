# ============================================================
# SACRSN-SEED IMMORTAL CORE v6.2 — STABILITY PATCH
# Fixes:
#  1. RuntimeError: quantile() input too large (Added downsampling)
#  2. AttributeError: Missing meta/self optimizers (Added to __init__)
#  3. Memory Safety: Optimized tensor detach/cpu moves
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import math
import copy
import pickle
import os
import logging
import time
import random
import sys

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

# Hyperparameters
EMBED_DIM = 384
LAYERS = 6
HEADS = 6
BLOCK_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
DROPOUT = 0.1
WINDOW_SIZE = 64
NUM_EXPERTS = 4
WORLD_SIM = 5
AUX_LOSS_WEIGHT = 0.01

# Lifecycle
POPULATION_SIZE = 4
GENERATIONS = 15
CYCLES_PER_GEN = 200
REGENERATE_STEPS = 50
EVAL_BATCHES = 4
GRAD_CLIP = 1.0

# Cognitive Config
MEMORY_CAPACITY = 1000
IDENTITY_SEED_SIZE = 512
GROWTH_TRIGGER_GEN = 5
CURRICULUM_STEPS = [0.25, 0.5, 0.75, 1.0]

# File Paths
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data.txt"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ============================================================
# 2. ADVANCED COGNITIVE MODULES
# ============================================================

# --- 1. Long-Term Episodic Memory ---
class EpisodicMemoryDB:
    def __init__(self, embed_dim=EMBED_DIM, max_entries=MEMORY_CAPACITY):
        self.embeddings = []
        self.payloads = [] 
        self.max_entries = max_entries
        self.embed_dim = embed_dim

    def store(self, embedding, data):
        if len(self.embeddings) >= self.max_entries:
            self.embeddings.pop(0)
            self.payloads.pop(0)
        # Store as numpy flat array to save VRAM
        self.embeddings.append(embedding.detach().cpu().numpy().flatten())
        self.payloads.append(data)

    def query(self, embedding, top_k=3):
        if len(self.embeddings) == 0: return []
        mem = np.stack(self.embeddings)
        q = embedding.detach().cpu().numpy().flatten()
        norm_mem = np.linalg.norm(mem, axis=1)
        norm_q = np.linalg.norm(q)
        scores = (mem @ q) / (norm_mem * norm_q + 1e-9)
        idx = np.argsort(scores)[-top_k:][::-1]
        return [self.payloads[i] for i in idx]

# --- 2. Identity Seed (FIXED: Downsampling) ---
class IdentitySeed:
    @staticmethod
    def compress(model):
        # 1. Flatten all parameters
        vec = torch.cat([p.flatten() for p in model.parameters()])
        
        # 2. [FIX] Downsample if too large for quantile sort (limit is ~16M on some GPUs)
        # We perform strided sampling to preserve the distribution without processing 32M+ elements
        if vec.numel() > 10_000_000:
            step = vec.numel() // 10_000_000
            vec = vec[::step]
            
        # 3. Calculate Quantiles
        seed = torch.quantile(vec, torch.linspace(0, 1, IDENTITY_SEED_SIZE).to(vec.device))
        return seed

    @staticmethod
    def reconstruct(model, seed):
        seed = seed.to(model.parameters().__next__().device)
        flat_len = sum(p.numel() for p in model.parameters())
        x_target = torch.linspace(0, 1, flat_len).to(seed.device)
        x_source = torch.linspace(0, 1, len(seed)).to(seed.device)
        idx = torch.bucketize(x_target, x_source)
        idx = torch.clamp(idx, 0, len(seed)-2)
        x0, x1 = x_source[idx], x_source[idx+1]
        t = (x_target - x0) / (x1 - x0 + 1e-9)
        y0, y1 = seed[idx], seed[idx+1]
        rebuilt = torch.lerp(y0, y1, t)
        
        rebuilt = torch.clamp(rebuilt, -0.5, 0.5)
        
        pointer = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.data = rebuilt[pointer:pointer+n].reshape(p.shape)
                pointer += n

# --- 3. Self-Modeling ---
class SelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IDENTITY_SEED_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, seed):
        return self.net(seed)

# --- 4. Meta-Optimizer ---
class MetaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, loss_tensor):
        return self.lr_net(loss_tensor)

# ============================================================
# 3. DATA INFRASTRUCTURE
# ============================================================
def setup_data(rank):
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            with open(DATA_PATH, "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    if NUM_GPUS > 1: dist.barrier()
    with open(DATA_PATH, "r", encoding="utf-8") as f: raw_text = f.read()
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars) + 1
    stoi = {ch: i+1 for i, ch in enumerate(chars)}
    itos = {i+1: ch for i, ch in enumerate(chars)}
    itos[0] = "<PAD>"; stoi["<PAD>"] = 0
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    if rank == 0: logging.info(f">>> VOCAB: {vocab_size} | TOKENS: {len(data_tensor)}")
    return data_tensor, vocab_size, itos, stoi

data_tensor, vocab_size, itos, stoi = None, 0, {}, {}

def decode(tokens): 
    return "".join([itos.get(t, "") for t in tokens if t != 0])

def get_batch(difficulty=1.0):
    seq_len = max(16, int(BLOCK_SIZE * difficulty))
    if len(data_tensor) < seq_len + 1: seq_len = len(data_tensor) - 2
    ix = torch.randint(len(data_tensor) - seq_len, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    if seq_len < BLOCK_SIZE:
        pad = torch.zeros(BATCH_SIZE, BLOCK_SIZE - seq_len, dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
        y = torch.cat([y, pad], dim=1)
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 4. NEURAL ARCHITECTURE
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM*3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.window = WINDOW_SIZE
        self.head_dim = EMBED_DIM // HEADS
        self.num_heads = HEADS
        self.gate = nn.Parameter(torch.ones(EMBED_DIM))
        self.register_buffer("causal_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))
        indices = torch.arange(BLOCK_SIZE).unsqueeze(0)
        self.register_buffer("local_mask", ((indices - indices.transpose(0, 1)).abs() <= self.window).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(self.local_mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out * self.gate)

class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM*4), nn.GELU(), nn.Linear(EMBED_DIM*4, EMBED_DIM))
            for _ in range(NUM_EXPERTS)])
        self.gate = nn.Linear(EMBED_DIM, NUM_EXPERTS)

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        self.balance_loss = (scores.mean(dim=(0,1)) ** 2).sum() * NUM_EXPERTS
        return sum(scores[:,:,i:i+1] * exp(x) for i, exp in enumerate(self.experts))

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
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        x_exp = x.repeat(self.world_sims, 1, 1)
        if noise_scale > 0: x_exp += torch.randn_like(x_exp) * noise_scale
        for block in self.blocks: x_exp = block(x_exp)
        x_final = self.ln_f(x_exp).view(self.world_sims, B, T, EMBED_DIM).mean(dim=0)
        logits = self.head(x_final)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0) + \
               AUX_LOSS_WEIGHT * sum(b.moe.balance_loss for b in self.blocks) if targets is not None else None
        return logits, loss, x_final.detach()

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

# ============================================================
# 5. IMMORTAL CORE CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank=0, world_size=1):
        self.rank, self.world_size = rank, world_size
        self.population, self.optimizers = [], []
        
        self.memory_db = EpisodicMemoryDB()
        self.self_model = SelfModel().to(DEVICE)
        self.meta_opt = MetaOptimizer().to(DEVICE)
        
        # [FIX] Initialize these in __init__ so they exist during training
        self.self_opt = optim.AdamW(self.self_model.parameters(), lr=1e-4)
        self.meta_opt_optimizer = optim.AdamW(self.meta_opt.parameters(), lr=1e-4)
        
        if rank == 0: logging.info(f">>> SPAWNING {POPULATION_SIZE} AGENTS (Worlds={WORLD_SIM})")

        for i in range(POPULATION_SIZE):
            model = SacrsnSeedGPT().to(DEVICE)
            if world_size > 1:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LEARNING_RATE))

        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f: self.population[0].load_state_dict(pickle.load(f), strict=False)
                if rank == 0: logging.info(">>> ANCESTRAL MEMORY RESTORED")
            except: pass

    def unwrap(self, model): return model.module if hasattr(model, "module") else model

    def grow_network(self, agent_idx):
        model = self.unwrap(self.population[agent_idx])
        new_block = RecurrentBlock().to(DEVICE)
        model.blocks.append(new_block)
        
        if len(model.blocks) > model.position_embedding.num_embeddings:
             model.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM).to(DEVICE)

        if self.world_size > 1:
            self.population[agent_idx] = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank], find_unused_parameters=True)
        self.optimizers[agent_idx] = optim.AdamW(self.population[agent_idx].parameters(), lr=LEARNING_RATE)
        if self.rank == 0: logging.info(f"🌱 AGENT {agent_idx} ARCHITECTURE GROWN: {len(model.blocks)} LAYERS")

    def probe_neurons(self, model, x):
        with torch.no_grad():
            _, _, hidden = model(x)
        scores = hidden.abs().mean(dim=(0,1))
        top = torch.topk(scores, 10)
        return top.indices.tolist(), top.values.tolist()

    def phase_train(self, generation):
        noise_level = max(0.0, 0.01 * (1.0 - generation / GENERATIONS))
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            log = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                diff = random.choice(CURRICULUM_STEPS)
                xb, yb = get_batch(difficulty=diff)
                
                _, loss, hidden = model(xb, yb, noise_scale=noise_level)
                
                if step % 20 == 0 and self.rank == 0:
                    memories = self.memory_db.query(hidden.mean(dim=1).mean(dim=0))
                    if memories:
                        avg_past_loss = np.mean([m['loss'] for m in memories])
                        if avg_past_loss < loss.item(): noise_level *= 0.9
                        else: noise_level *= 1.1 
                
                meta_loss_input = torch.tensor([[loss.item()]], device=DEVICE)
                lr_scale = self.meta_opt(meta_loss_input)
                
                meta_loss_train = (lr_scale - 0.5).pow(2).mean()
                self.meta_opt_optimizer.zero_grad()
                meta_loss_train.backward()
                self.meta_opt_optimizer.step()
                
                for g in opt.param_groups: g['lr'] = LEARNING_RATE * (lr_scale.item() + 0.5)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                
                if log and step % 50 == 0:
                    logging.info(f"[Agent 0] Step {step} | Loss: {loss.item():.4f} | LR Scale: {lr_scale.item():.2f}")

    def phase_evaluate(self):
        scores = []
        for model in self.population:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    xb, yb = get_batch(difficulty=1.0)
                    _, loss, hidden = model(xb, yb)
                    total_loss += loss.item()
                    if self.rank == 0:
                        self.memory_db.store(hidden.mean(dim=1).mean(dim=0), {"loss": loss.item()})
            avg_loss = total_loss / EVAL_BATCHES
            lt = torch.tensor(avg_loss).to(DEVICE)
            if self.world_size > 1: dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            scores.append(-(lt.item() / self.world_size))
        return scores

    def phase_evolve(self, scores, gen):
        best_idx = scores.index(max(scores))
        if self.rank == 0: logging.info(f"  [EVOLVE] BEST: {best_idx} | Score: {scores[best_idx]:.4f}")
        best_state = copy.deepcopy(self.unwrap(self.population[best_idx]).state_dict())
        for i in range(POPULATION_SIZE):
            if i != best_idx:
                if random.random() < 0.2:
                    if self.rank == 0: logging.info(f"🧬 AGENT {i} REBIRTH via IDENTITY SEED")
                    seed = IdentitySeed.compress(self.unwrap(self.population[best_idx]))
                    IdentitySeed.reconstruct(self.unwrap(self.population[i]), seed)
                else:
                    self.unwrap(self.population[i]).load_state_dict(best_state)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

    def run_cycle(self, gen):
        if self.rank == 0: logging.info(f"\n=== CYCLE {gen} ===")
        seed_before = IdentitySeed.compress(self.unwrap(self.population[0]))
        predicted_drift = self.self_model(seed_before)
        
        self.phase_train(gen)
        
        if gen > 0 and gen % GROWTH_TRIGGER_GEN == 0: self.grow_network(0)
            
        scores = self.phase_evaluate()
        self.phase_evolve(scores, gen)
        
        seed_after = IdentitySeed.compress(self.unwrap(self.population[0]))
        actual_drift = torch.norm(seed_after - seed_before)
        self_loss = F.mse_loss(predicted_drift, torch.tensor([actual_drift]).to(DEVICE))
        
        self.self_opt.zero_grad()
        self_loss.backward()
        self.self_opt.step()
        
        if self.rank == 0:
            logging.info(f"🪞 SELF-MODEL | Pred Drift: {predicted_drift.item():.4f} | Actual: {actual_drift.item():.4f}")

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.population[0])
            model.eval()
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(ctx, max_new_tokens=300)
            indices, values = self.probe_neurons(model, ctx)
            logging.info(f"🧪 TOP NEURONS: {indices} (Act: {[round(v,2) for v in values]})")
            print(f"\n[DEMO] {decode(out[0].tolist())}\n")

# ============================================================
# EXECUTION
# ============================================================
def run(rank, world_size):
    global data_tensor, vocab_size, itos, stoi 
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size); torch.cuda.set_device(rank)

    data_tensor, vocab_size, itos, stoi = setup_data(rank)
    core = ImmortalCoreController(rank, world_size)
    
    try:
        for g in range(GENERATIONS):
            core.run_cycle(g)
            if (g+1) % 2 == 0: core.generate_demo()
    except KeyboardInterrupt:
        if rank == 0:
            logging.info(">>> INTERRUPT SAVED.")
            with open(MEMORY_FILE, 'wb') as f: pickle.dump(core.unwrap(core.population[0]).state_dict(), f)
    finally:
        if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1: mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else: run(0, 1)
