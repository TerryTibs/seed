# ============================================================
# SACRSN-SEED IMMORTAL CORE v8.0 — THE FINAL CONVERGENCE
# ============================================================
# A superset of all previous versions (seed.py, seed1-seed24).
# Contains all logic, memory systems, safety hooks, and logging.
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

# --- HYPERPARAMETERS (ALL ALIASES DEFINED) ---
# Stable Defaults
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

# Legacy Aliases (For backward compatibility)
EMBED = EMBED_DIM
BLOCK = BLOCK_SIZE
LR = LEARNING_RATE
EXPERT_COUNT = NUM_EXPERTS
WORLD_COUNT = WORLD_SIM
WORLD_BRANCHES = WORLD_SIM
ATTN_WINDOW = WINDOW_SIZE

# Lifecycle Config
POPULATION_SIZE = 4
GENERATIONS = 20
CYCLES_PER_GEN = 200
REGENERATE_STEPS = 50
EVAL_BATCHES = 4
GRAD_CLIP = 1.0

# Legacy Lifecycle Aliases
STEPS_PER_CYCLE = CYCLES_PER_GEN

# Cognitive Config
MEMORY_CAPACITY = 1000
IDENTITY_SEED_SIZE = 512
CURRICULUM_STEPS = [0.25, 0.5, 0.75, 1.0]
SELECTIVE_THRESHOLD = 0.10
WIPE_RATIO_DEFAULT = 0.1

# File Paths
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
PT_FILE = "seed_model.pt"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data.txt"
SYNTH_DATA_PATH = "data_recursive.txt"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ============================================================
# 2. DATA INFRASTRUCTURE
# ============================================================
def setup_data(rank):
    if rank == 0:
        if not os.path.exists(DATA_PATH):
            with open(DATA_PATH, "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
    if NUM_GPUS > 1: dist.barrier()
    
    raw_text = ""
    with open(DATA_PATH, "r", encoding="utf-8") as f: raw_text += f.read()
    if os.path.exists(SYNTH_DATA_PATH):
        with open(SYNTH_DATA_PATH, "r", encoding="utf-8") as f: raw_text += f.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars) + 1
    stoi = {ch: i+1 for i, ch in enumerate(chars)}
    itos = {i+1: ch for i, ch in enumerate(chars)}
    itos[0] = "<PAD>"; stoi["<PAD>"] = 0
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    
    if rank == 0: logging.info(f">>> VOCAB: {vocab_size} | TOKENS: {len(data_tensor)}")
    return data_tensor, vocab_size, itos, stoi

# Globals
data_tensor, vocab_size, itos, stoi = None, 0, {}, {}

def encode(s): 
    return [stoi.get(c, 0) for c in s]

def decode(tokens): 
    return "".join([itos.get(t, "") for t in tokens if t != 0])

def get_batch(difficulty=1.0, block=BLOCK):
    if data_tensor is None: return None, None
    if len(data_tensor) < block:
        raise ValueError("Data length too small for block size!")

    seq_len = max(16, int(block * difficulty))
    if len(data_tensor) < seq_len + 1: seq_len = len(data_tensor) - 2
    
    ix = torch.randint(len(data_tensor) - seq_len, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    
    # Pad if curriculum length < BLOCK
    if seq_len < block:
        pad = torch.zeros(BATCH_SIZE, block - seq_len, dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
        y = torch.cat([y, pad], dim=1)
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 3. GLOBAL HELPER FUNCTIONS (Legacy API Support)
# ============================================================
def identity_signature(model):
    return sum(p.mean().item() for p in model.parameters())

def compress_identity(model):
    return IdentitySeed.compress(model)

def restore_identity(model, seed):
    IdentitySeed.reconstruct(model, seed)

def destroy_weights(model, wipe_ratio=WIPE_RATIO_DEFAULT):
    with torch.no_grad():
        for p in model.parameters():
            mask = (torch.rand_like(p) > wipe_ratio).float()
            p.mul_(mask)

def spawn_agents(base_model, count=3):
    return [copy.deepcopy(base_model) for _ in range(count)]

def evaluate_agent(agent, batch_size=8):
    x, y = get_batch() 
    with torch.no_grad():
        logits, _, _, _ = agent(x)
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()

def generate(model, prompt, steps=200, temperature=1.0, top_k=None):
    tokens = torch.tensor([encode(prompt)], device=DEVICE)
    model.eval()
    return decode(model.generate(tokens, steps, temperature, top_k)[0].tolist())

def save_memory(model, path=MEMORY_FILE):
    try:
        state = model.state_dict()
        if os.path.exists(path): os.rename(path, MEMORY_BACKUP)
        with open(path, "wb") as f: pickle.dump(state, f)
        logging.info(">>> MEMORY SAVED (Global .pkl)")
        torch.save(state, PT_FILE) # seed18 requirement
    except Exception as e: logging.error(f"Save Error: {e}")

def load_memory(model, path=MEMORY_FILE):
    try:
        with open(path, "rb") as f: model.load_state_dict(pickle.load(f), strict=False)
        logging.info(">>> MEMORY RESTORED (Global)")
    except: logging.info(">>> FRESH MIND")

# ============================================================
# 4. ADVANCED COGNITIVE MODULES
# ============================================================

# --- Hierarchical Memory (seed3) ---
class HierarchicalMemory(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, short_len=32, medium_len=16, long_len=8):
        super().__init__()
        self.short_mem = nn.Parameter(torch.randn(short_len, embed_dim))
        self.medium_mem = nn.Parameter(torch.randn(medium_len, embed_dim))
        self.long_mem = nn.Parameter(torch.randn(long_len, embed_dim))
    def read(self): 
        return torch.cat([self.short_mem, self.medium_mem, self.long_mem], dim=0)

# --- Persistent Episodic Memory (seed21) ---
class PersistentEpisodicMemory:
    def __init__(self, embed_dim=EMBED_DIM, max_entries=MEMORY_CAPACITY):
        self.embeddings, self.payloads = [], []
        self.max_entries = max_entries
    def store(self, embedding, data):
        if len(self.embeddings) >= self.max_entries: self.embeddings.pop(0); self.payloads.pop(0)
        self.embeddings.append(embedding.detach().cpu().numpy().flatten())
        self.payloads.append(data)
    def query(self, embedding, top_k=3):
        if len(self.embeddings) == 0: return []
        mem = np.stack(self.embeddings)
        q = embedding.detach().cpu().numpy().flatten()
        norm_mem = np.linalg.norm(mem, axis=1); norm_q = np.linalg.norm(q)
        scores = (mem @ q) / (norm_mem * norm_q + 1e-9)
        idx = np.argsort(scores)[-top_k:][::-1]
        return [self.payloads[i] for i in idx]
    def save(self):
        try:
            with open("episodic_memory.pkl", "wb") as f: pickle.dump((self.embeddings, self.payloads), f)
        except Exception as e: logging.error(f"DB Save Error: {e}")
    def load(self):
        if os.path.exists("episodic_memory.pkl"):
            try:
                with open("episodic_memory.pkl", "rb") as f: self.embeddings, self.payloads = pickle.load(f)
                logging.info(f"🧠 MEMORY DB LOADED ({len(self.embeddings)} entries)")
            except: pass

# --- Identity Seed (seed19/22) ---
class IdentitySeed:
    @staticmethod
    def compress(model):
        vec = torch.cat([p.flatten() for p in model.parameters()])
        # [FIX] Downsample large models to avoid quantile crash
        if vec.numel() > 10_000_000:
            step = vec.numel() // 10_000_000
            vec = vec[::step]
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
                n = p.numel(); p.data = rebuilt[pointer:pointer+n].reshape(p.shape); pointer += n

# --- Predictive Self-Model (seed21) ---
class PredictiveSelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IDENTITY_SEED_SIZE, 256), nn.ReLU(), nn.Linear(256, IDENTITY_SEED_SIZE))
    def forward(self, seed): return self.net(seed)

# --- Goal Engine (seed21) ---
class GoalEngine:
    def __init__(self):
        self.goals = ["minimize_loss"]
    def evolve_goals(self, memory_db):
        if len(memory_db.payloads) > 10:
            losses = [m['loss'] for m in memory_db.payloads[-10:]]
            if np.std(losses) < 0.05 and "increase_creativity" not in self.goals:
                self.goals.append("increase_creativity")
                logging.info("🧠 GOAL EVOLVED: Added 'increase_creativity'")
    def reward_modifier(self, loss):
        if "increase_creativity" in self.goals: return loss * random.uniform(0.9, 1.1)
        return loss

# --- Meta-Optimizer (seed19) ---
class MetaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, loss_tensor): return self.lr_net(loss_tensor)

# ============================================================
# 5. NEURAL ARCHITECTURE
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

    def forward(self, x, memory=None, loss_mask=None):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        
        # --- Memory Integration (seed3) ---
        if memory is not None:
            mem_exp = memory.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem_exp, k], dim=1)
            v = torch.cat([mem_exp, v], dim=1)
            T_total = k.size(1) 
        else:
            T_total = T

        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T_total,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T_total,self.num_heads,self.head_dim).transpose(1,2)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        
        self_att_start = T_total - T
        att_self = att[:,:,:,self_att_start:]
        mask_slice = self.causal_mask[:,:,:T,:T]
        att_self = att_self.masked_fill(mask_slice == 0, float('-inf'))
        local_slice = self.local_mask[:,:,:T,:T]
        att_self = att_self.masked_fill(local_slice == 0, float('-inf'))
        att[:,:,:,self_att_start:] = att_self

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.proj(out * self.gate)
        
        # --- Selective Masking (seed6) ---
        if loss_mask is not None: 
            out = out * loss_mask.unsqueeze(-1)
        return out

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

    def forward(self, x, memory=None, loss_mask=None):
        x = x + self.attn(self.ln1(x), memory=memory, loss_mask=loss_mask)
        x = x + self.moe(self.ln2(x))
        return x

class SacrsnSeedGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.global_memory = HierarchicalMemory(EMBED_DIM) # seed3
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.world_sims = WORLD_SIM
        self.meta_memory = None # seed6

    def forward(self, idx, targets=None, noise_scale=0.0, loss_mask=None):
        B,T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        
        # Inject Memory (seed3)
        mem_ctx = self.global_memory.read()
        x = x + mem_ctx.mean(dim=0).unsqueeze(0).unsqueeze(0)

        # Multi-World + Noise
        x_exp = x.repeat(self.world_sims, 1, 1)
        if noise_scale > 0: x_exp += torch.randn_like(x_exp) * noise_scale
        
        # [FIX] Reshape loss_mask for Multi-World Broadcast
        expanded_mask = None
        if loss_mask is not None:
            if loss_mask.dim() == 1: loss_mask = loss_mask.view(B, T)
            expanded_mask = loss_mask.repeat(self.world_sims, 1)

        for block in self.blocks: 
            x_exp = block(x_exp, memory=mem_ctx, loss_mask=expanded_mask)
        
        x_final = self.ln_f(x_exp).view(self.world_sims, B, T, EMBED_DIM).mean(dim=0)
        logits = self.head(x_final)
        
        self.meta_memory = logits.mean(dim=1).detach() # seed6
        
        raw_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0, reduction='none') if targets is not None else None
        loss = raw_loss.mean() + AUX_LOSS_WEIGHT * sum(b.moe.balance_loss for b in self.blocks) if raw_loss is not None else None
        
        return logits, loss, x_final.detach(), raw_loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits, _, _, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

# Class Aliases
SeedGPT = SacrsnSeedGPT
RecurrentWorld = SacrsnSeedGPT

# ============================================================
# 6. IMMORTAL CORE CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank=0, world_size=1):
        self.rank, self.world_size = rank, world_size
        self.population, self.optimizers = [], []
        self.score_history = []
        
        self.memory_db = PersistentEpisodicMemory(EMBED_DIM)
        self.memory_db.load()
        self.self_model = PredictiveSelfModel().to(DEVICE)
        self.meta_opt = MetaOptimizer().to(DEVICE)
        self.goal_engine = GoalEngine()
        
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

    # --- Saving (seed15/17/18) ---
    def save_memory(self, model):
        try:
            state = model.state_dict()
            if os.path.exists(MEMORY_FILE): os.rename(MEMORY_FILE, MEMORY_BACKUP)
            with open(MEMORY_FILE, "wb") as f: pickle.dump(state, f)
            logging.info(">>> MEMORY .pkl SAVED")
        except Exception as e: logging.error(f"PKL Save Error: {e}")

    def save_checkpoint(self, model, optimizer, generation, tag=""):
        try:
            checkpoint = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "generation": generation,
                "timestamp": time.time()
            }
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_gen{generation}_{tag}_{ts}.pt"
            path = os.path.join(CHECKPOINT_DIR, filename)
            torch.save(checkpoint, path)
            logging.info(f">>> CHECKPOINT .pt SAVED: {path}")
        except Exception as e: logging.error(f"PT Save Error: {e}")

    # --- Logic ---
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

    def auto_grow(self):
        if len(self.score_history) > 3:
            recent = self.score_history[-3:]
            if recent[-1] <= recent[-2]:
                self.grow_network(0)
                if self.rank == 0: logging.info("🌱 NAS: PERFORMANCE STAGNATED. ADDING LAYER.")

    def simulate_future(self, model):
        with torch.no_grad():
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(ctx, 100)
        return decode(out[0].tolist())

    def destroy_weights(self, model, ratio=0.1):
        with torch.no_grad():
            for p in model.parameters():
                mask = (torch.rand_like(p) > ratio).float()
                p.mul_(mask)

    def probe_neurons(self, model, x):
        with torch.no_grad():
            _, _, hidden, _ = model(x)
        scores = hidden.abs().mean(dim=(0,1))
        top = torch.topk(scores, 10)
        return top.indices.tolist(), top.values.tolist()

    # --- Legacy Distributed Training Wrapper (seed1/seed2 compatibility) ---
    def train_agents_distributed(self, steps_per_cycle=CYCLES_PER_GEN):
        """Legacy wrapper"""
        self.phase_train(0)

    def phase_train(self, generation):
        noise_level = max(0.0, 0.01 * (1.0 - generation / GENERATIONS))
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            log = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                diff = random.choice(CURRICULUM_STEPS)
                xb, yb = get_batch(difficulty=diff)
                
                _, loss, hidden, _ = model(xb, yb, noise_scale=noise_level)
                loss = self.goal_engine.reward_modifier(loss)
                
                if step % 20 == 0 and self.rank == 0:
                    memories = self.memory_db.query(hidden.mean(dim=1).mean(dim=0))
                    if memories:
                        avg_past_loss = np.mean([m['loss'] for m in memories])
                        if avg_past_loss < loss.item(): noise_level *= 0.9
                        else: noise_level *= 1.1 
                
                meta_in = torch.tensor([[loss.item()]], device=DEVICE)
                lr_scale = self.meta_opt(meta_in)
                meta_loss = (lr_scale - 0.5).pow(2).mean()
                self.meta_opt_optimizer.zero_grad(); meta_loss.backward(); self.meta_opt_optimizer.step()
                for g in opt.param_groups: g['lr'] = LEARNING_RATE * (lr_scale.item() + 0.5)
                
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
                
                if log and step % 50 == 0:
                    logging.info(f"[Agent 0] Step {step} | Loss: {loss.item():.4f} | LR: {lr_scale.item():.2f}")

    def phase_regenerate(self):
        if self.rank == 0: logging.info("  [PHASE 2] DESTRUCTION & REGENERATION (SELECTIVE)...")
        for model, opt in zip(self.population, self.optimizers):
            self.destroy_weights(model, ratio=WIPE_RATIO_DEFAULT)
            model.train()
            for _ in range(REGENERATE_STEPS):
                xb, yb = get_batch(difficulty=1.0)
                _, _, _, raw_loss = model(xb, yb)
                threshold = torch.quantile(raw_loss, 1 - SELECTIVE_THRESHOLD)
                loss_mask = (raw_loss > threshold).float()
                _, loss, _, _ = model(xb, yb, loss_mask=loss_mask)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()

    def phase_evaluate(self):
        scores = []
        for model in self.population:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    xb, yb = get_batch(difficulty=1.0)
                    _, loss, hidden, _ = model(xb, yb)
                    total_loss += loss.item()
                    if self.rank == 0:
                        self.memory_db.store(hidden.mean(dim=1).mean(dim=0), {"loss": loss.item()})
            avg_loss = total_loss / EVAL_BATCHES
            lt = torch.tensor(avg_loss).to(DEVICE)
            if self.world_size > 1: dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            scores.append(-(lt.item() / self.world_size))
        if self.rank == 0: self.goal_engine.evolve_goals(self.memory_db)
        return scores

    def agent_council_vote(self, scores):
        return np.argsort(scores)[-1]

    def phase_evolve(self, scores, gen):
        best_idx = self.agent_council_vote(scores)
        self.score_history.append(scores[best_idx])
        if self.rank == 0: logging.info(f"  [EVOLVE] BEST AGENT: {best_idx} | Score: {scores[best_idx]:.4f}")
        
        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())
        
        for i in range(POPULATION_SIZE):
            if i != best_idx:
                if random.random() < 0.2:
                    if self.rank == 0: logging.info(f"🧬 AGENT {i} REBIRTH via IDENTITY SEED")
                    seed = IdentitySeed.compress(best_agent)
                    IdentitySeed.reconstruct(self.unwrap(self.population[i]), seed)
                    with torch.no_grad():
                        for p in self.population[i].parameters(): p.add_(torch.randn_like(p) * 0.01)
                else:
                    self.unwrap(self.population[i]).load_state_dict(best_state)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

        if self.rank == 0:
            self.memory_db.save()
            self.save_memory(best_agent)
            self.save_checkpoint(best_agent, self.optimizers[best_idx], gen, tag="best")

    def run_cycle(self, gen):
        if self.rank == 0: logging.info(f"\n=== CYCLE {gen} ===")
        seed_before = IdentitySeed.compress(self.unwrap(self.population[0]))
        pred_next_seed = self.self_model(seed_before)
        
        self.phase_train(gen)
        scores = self.phase_evaluate()
        
        self.auto_grow() 
        self.phase_evolve(scores, gen)
        self.phase_regenerate()
        
        seed_after = IdentitySeed.compress(self.unwrap(self.population[0]))
        self_loss = F.mse_loss(pred_next_seed, seed_after.detach())
        self.self_opt.zero_grad(); self_loss.backward(); self.self_opt.step()
        
        if self.rank == 0:
            logging.info(f"🪞 SELF-MODEL LOSS: {self_loss.item():.6f}")
            synth_text = self.simulate_future(self.unwrap(self.population[0]))
            with open(SYNTH_DATA_PATH, "a") as f: f.write(synth_text + " ")
            if gen % 2 == 0:
                global data_tensor
                data_tensor, _, _, _ = setup_data(self.rank)

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
            logging.info("\n>>> INTERRUPT DETECTED. SAVING FULL STATE...")
            best_agent = core.unwrap(core.population[0])
            best_opt = core.optimizers[0]
            core.save_memory(best_agent) 
            core.save_checkpoint(best_agent, best_opt, 999, tag="interrupt")
            
    finally:
        if world_size > 1: dist.destroy_process_group()
        if rank == 0: logging.info(">>> SYSTEM SHUTDOWN.")

if __name__ == "__main__":
    if NUM_GPUS > 1: mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else: run(0, 1)
