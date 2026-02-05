# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v40.0 — THE OMEGA SINGULARITY
# ==============================================================================
# "The seed that remembers, the mind that builds, the core that cannot die."
#
# UNIFIED FEATURE STACK:
# 1. ARCHITECTURE: Sparse Attn (Cached), MoE (Balanced), Multi-World (Noise), Hierarchical Memory
# 2. COGNITION: Agency (RL), World Model (GRU), Self-Model (Predictive), Vector DB (Memmap)
# 3. EVOLUTION: Civilization Roles, Sharded Merge, Identity Genome, Curiosity
# 4. INFRASTRUCTURE: DDP (Sync), Atomic I/O, Telemetry, Synthetic Data Guard
# ==============================================================================

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
import hashlib
import json
import shutil
from collections import deque

# ============================================================
# 1. CONFIGURATION & LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

CONFIG = {
    # Neural Architecture
    "EMBED_DIM": 384,
    "LAYERS": 6,
    "HEADS": 6,
    "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4,
    "WINDOW_SIZE": 64,
    "WORLD_SIM": 5,        # Parallel realities per forward pass
    
    # Training Dynamics
    "BATCH_SIZE": 16,      # Effective batch = 16 * WORLD_SIM
    "LR": 3e-4,
    "DROPOUT": 0.1,
    "GRAD_CLIP": 1.0,
    "AUX_LOSS_WEIGHT": 0.01,
    
    # Evolutionary Lifecycle
    "POPULATION_SIZE": 4,
    "GENERATIONS": 50,
    "CYCLES_PER_GEN": 200,
    "REGENERATE_STEPS": 50,
    "EVAL_BATCHES": 4,
    
    # Cognitive / Meta
    "MEMORY_CAPACITY": 500_000, # Disk-backed limit
    "IDENTITY_SEED_SIZE": 512,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0],
    "SYNTH_RATIO_CAP": 0.2,     # Max % of training data that can be synthetic
    "WIPE_RATIO": 0.1           # Synaptic pruning ratio
}

PATHS = {
    "MEM_PKL": "seed_memory.pkl",
    "MEM_BAK": "seed_memory_backup.pkl",
    "CHECKPOINT": "checkpoints/seed_ckpt.pt",
    "ARCHIVE": "IMMORTAL_ARCHIVE.pt",
    "TELEMETRY": "telemetry.jsonl",
    "DIR_CKPT": "checkpoints",
    "DIR_SANDBOX": "rewrite_sandbox",
    "DATA": "data.txt",
    "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat",
    "MEM_META": "memory_meta.pkl"
}

for d in [PATHS["DIR_CKPT"], PATHS["DIR_SANDBOX"]]:
    if not os.path.exists(d): os.makedirs(d)

# ============================================================
# 2. DATA INFRASTRUCTURE (Robust)
# ============================================================
class DataManager:
    def __init__(self, rank):
        self.rank = rank
        self.data_tensor = None
        self.synth_tensor = torch.tensor([], dtype=torch.long)
        self.vocab_size = 0
        self.itos = {}
        self.stoi = {}
        self._load_data()

    def _load_data(self):
        # Auto-Genesis
        if self.rank == 0 and not os.path.exists(PATHS["DATA"]):
            logging.warning("Data missing. Generating genesis block.")
            with open(PATHS["DATA"], "w") as f:
                f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)
        
        if NUM_GPUS > 1: dist.barrier()
            
        with open(PATHS["DATA"], "r", encoding="utf-8") as f: raw_text = f.read()
        
        synth_text = ""
        if os.path.exists(PATHS["SYNTH"]):
            with open(PATHS["SYNTH"], "r", encoding="utf-8") as f: synth_text = f.read()

        chars = sorted(list(set(raw_text + synth_text)))
        self.vocab_size = len(chars) + 1
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i+1: ch for i, ch in enumerate(chars)}
        self.itos[0] = "<PAD>"; self.stoi["<PAD>"] = 0
        
        self.data_tensor = torch.tensor([self.stoi[c] for c in raw_text], dtype=torch.long)
        if synth_text:
            self.synth_tensor = torch.tensor([self.stoi[c] for c in synth_text], dtype=torch.long)

        if self.rank == 0:
            logging.info(f"DATA | Vocab: {self.vocab_size} | Real: {len(self.data_tensor)} | Synth: {len(self.synth_tensor)}")

    def get_batch(self, difficulty=1.0):
        if self.data_tensor is None: raise RuntimeError("Data not initialized!")
        
        # Synthetic Mixing Gate
        use_synth = (len(self.synth_tensor) > CONFIG["BLOCK_SIZE"]) and \
                    (random.random() < CONFIG["SYNTH_RATIO_CAP"])
        source = self.synth_tensor if use_synth else self.data_tensor
        
        # Fallbacks
        if len(source) < CONFIG["BLOCK_SIZE"]: source = self.data_tensor
        if len(source) < CONFIG["BLOCK_SIZE"]: 
            # Critical fallback padding
            return torch.zeros((CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"]), dtype=torch.long).to(DEVICE), \
                   torch.zeros((CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"]), dtype=torch.long).to(DEVICE)

        seq_len = max(16, int(CONFIG["BLOCK_SIZE"] * difficulty))
        if len(source) < seq_len + 5: seq_len = len(source) - 2
            
        ix = torch.randint(len(source) - seq_len, (CONFIG["BATCH_SIZE"],))
        x = torch.stack([source[i:i+seq_len] for i in ix])
        y = torch.stack([source[i+1:i+seq_len+1] for i in ix])
        
        # Curriculum Padding
        if seq_len < CONFIG["BLOCK_SIZE"]:
            pad = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"] - seq_len, dtype=torch.long)
            x = torch.cat([x, pad], dim=1); y = torch.cat([y, pad], dim=1)
            
        return x.to(DEVICE), y.to(DEVICE)

    def decode(self, tokens):
        return "".join([self.itos.get(t, "") for t in tokens if t != 0])

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

# ============================================================
# 3. UTILITIES & I/O
# ============================================================
def atomic_save(obj, path, use_torch=False):
    """Prevents file corruption on interrupt"""
    tmp_path = path + ".tmp"
    try:
        if use_torch: torch.save(obj, tmp_path)
        else:
            with open(tmp_path, "wb") as f: pickle.dump(obj, f)
        
        if os.path.exists(path):
            try: os.replace(tmp_path, path)
            except OSError: os.remove(path); os.rename(tmp_path, path)
        else: os.rename(tmp_path, path)
    except Exception as e:
        logging.error(f"Atomic Save Failed: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def identity_hash(model):
    vec = torch.cat([p.flatten() for p in model.parameters()])
    if vec.numel() > 1_000_000: vec = vec[::100] # Sampling for speed
    return hashlib.sha256(vec.detach().cpu().numpy().tobytes()).hexdigest()[:12]

# ============================================================
# 4. COGNITIVE LAYER (The Mind)
# ============================================================

# --- Agency Core (RL Decision Making) ---
class AgencyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONFIG["EMBED_DIM"], 256), nn.ReLU(), nn.Linear(256, 5), nn.Softmax(-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs, self.rewards = [], []
        self.return_buffer = deque(maxlen=100) # For stable baseline

    def decide(self, state_embed):
        actions = ["train", "evolve", "explore", "reflect", "rest"]
        if state_embed is None: return "train"
        
        state = torch.nan_to_num(state_embed, nan=0.0).detach()
        probs = self.net(state)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action_idx))
        return actions[action_idx.item()]

    def update_policy(self):
        if not self.rewards: return
        R = 0
        policy_loss = []
        returns = []
        # Discounted Return
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(DEVICE)
        
        # Stable Baseline Normalization
        self.return_buffer.extend(returns.tolist())
        mean = np.mean(self.return_buffer) if self.return_buffer else 0.0
        std = np.std(self.return_buffer) if self.return_buffer and len(self.return_buffer)>1 else 1.0
        returns = (returns - mean) / (std + 1e-9)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()
        del self.saved_log_probs[:]; del self.rewards[:]

# --- Infinite Memory (Disk-Backed) ---
class DiskEpisodicMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.count = 0
        self.payloads = []
        self.centroids = []
        
        if os.path.exists(PATHS["MEM_VECS"]):
            self.embeddings = np.memmap(PATHS["MEM_VECS"], dtype='float32', mode='r+', shape=(self.max, self.dim))
            self.load_meta()
        else:
            self.embeddings = np.memmap(PATHS["MEM_VECS"], dtype='float32', mode='w+', shape=(self.max, self.dim))

    def store(self, embedding, data):
        emb = embedding.detach().cpu().numpy().flatten()
        
        # Centroid-based density pruning (Novelty check)
        if len(self.centroids) > 0 and self.count % 10 != 0:
            dists = np.linalg.norm(np.stack(self.centroids) - emb, axis=1)
            if dists.min() < 0.05: return # Skip redundant memory
            
        idx = self.count % self.max
        self.embeddings[idx] = emb
        if idx < len(self.payloads): self.payloads[idx] = data
        else: self.payloads.append(data)
        
        # Safety cap
        if len(self.payloads) > self.max: self.payloads = self.payloads[-self.max:]
        
        self.count += 1
        if self.count % 500 == 0: 
            self.embeddings.flush()
            # Update centroids
            valid = min(self.count, self.max)
            sample_idx = np.random.choice(valid, min(valid, 20), replace=False)
            self.centroids = self.embeddings[sample_idx]

    def query(self, embedding, top_k=5):
        if self.count == 0: return []
        valid = min(self.count, self.max)
        # Monte Carlo Search for speed
        idx_pool = np.random.choice(valid, min(valid, 5000), replace=False)
        mem = self.embeddings[idx_pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top_local = np.argsort(sim)[-top_k:][::-1]
        real_indices = idx_pool[top_local]
        return [self.payloads[i] for i in real_indices if i < len(self.payloads)]

    def save_meta(self):
        with open(PATHS["MEM_META"] + ".tmp", "wb") as f:
            pickle.dump({"count": self.count, "payloads": self.payloads}, f)
        os.replace(PATHS["MEM_META"] + ".tmp", PATHS["MEM_META"])

    def load_meta(self):
        if os.path.exists(PATHS["MEM_META"]):
            try:
                with open(PATHS["MEM_META"], "rb") as f:
                    meta = pickle.load(f)
                    self.count = meta.get("count", 0)
                    self.payloads = meta.get("payloads", [])
            except: pass

# --- Meta-Learning & Self-Modeling ---
class MetaLearningEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def get_lr_scale(self, loss, grad_norm):
        inp = torch.tensor([loss, grad_norm], device=DEVICE).float()
        return self.net(inp) * 2.0 # Scale 0.0 to 2.0

class IdentitySeed:
    @staticmethod
    def compress(model):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // CONFIG["SEED_SIZE"])
        # Multi-Anchor
        anchors = [flat[i::step][:CONFIG["SEED_SIZE"]] for i in range(3)]
        for i in range(3):
            if len(anchors[i]) < CONFIG["SEED_SIZE"]: 
                anchors[i] = F.pad(anchors[i], (0, CONFIG["SEED_SIZE"] - len(anchors[i])))
        sampled = torch.cat(anchors).detach().cpu()
        hash_sig = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
        return {"weights": sampled, "meta": {"layers": len(model.blocks), "hash": hash_sig}}

    @staticmethod
    def reconstruct(model, seed):
        weights = seed["weights"].to(DEVICE)
        # Blend Anchors
        if weights.numel() == 3 * CONFIG["SEED_SIZE"]:
            weights = weights.view(3, CONFIG["SEED_SIZE"]).mean(dim=0)
        
        target_layers = seed["meta"].get("layers", CONFIG["LAYERS"])
        # Arch Sync
        while len(model.blocks) < target_layers: model.blocks.append(RecurrentBlock().to(DEVICE))
        while len(model.blocks) > target_layers: del model.blocks[-1]

        total = sum(p.numel() for p in model.parameters())
        x_t = torch.linspace(0, 1, total, device=DEVICE)
        x_s = torch.linspace(0, 1, len(weights), device=DEVICE)
        idx = torch.bucketize(x_t, x_s).clamp(0, len(weights)-2)
        # Interpolate
        val = torch.lerp(weights[idx], weights[idx+1], (x_t - x_s[idx]) / (x_s[idx+1] - x_s[idx] + 1e-9))
        
        ptr = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(val[ptr:ptr+n].reshape(p.shape))
                ptr += n

# ============================================================
# 5. NEURAL ARCHITECTURE (The Body)
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.window = CONFIG["WINDOW_SIZE"]
        self.num_heads = CONFIG["HEADS"]
        self.head_dim = dim // self.num_heads
        self.gate = nn.Parameter(torch.ones(dim))
        
        b = CONFIG["BLOCK_SIZE"]
        self.register_buffer("causal_mask", torch.tril(torch.ones(b,b)).view(1,1,b,b))
        idx = torch.arange(b).unsqueeze(0)
        self.register_buffer("local_mask", ((idx - idx.T).abs() <= self.window).view(1,1,b,b))

    def forward(self, x, memory=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)

        # Hierarchical Memory Injection
        if memory is not None:
            mem_exp = memory.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem_exp, k], dim=1)
            v = torch.cat([mem_exp, v], dim=1)
            T_total = k.size(1)
        else:
            T_total = T

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_total, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_total, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        start = T_total - T
        
        # Safe Masking
        att_self = att[:, :, :, start:]
        if T <= self.causal_mask.size(2):
            mask = self.causal_mask[:,:,:T,:T]
            local = self.local_mask[:,:,:T,:T]
            att_self = att_self.masked_fill(mask==0, float('-inf'))
            att_self = att_self.masked_fill(local==0, float('-inf'))
        att[:, :, :, start:] = att_self

        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y * self.gate)

class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            for _ in range(CONFIG["NUM_EXPERTS"])
        ])
        self.gate = nn.Linear(dim, CONFIG["NUM_EXPERTS"])
        self.register_buffer("balance_loss", torch.tensor(0.0))

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        self.balance_loss = (scores.mean(dim=(0,1))**2).sum() * CONFIG["NUM_EXPERTS"]
        return sum(scores[:,:,i:i+1] * e(x) for i,e in enumerate(self.experts))

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoEBlock()
    def forward(self, x, memory=None):
        x = x + self.attn(self.ln1(x), memory)
        x = x + self.moe(self.ln2(x))
        return x

class SacrsnSeedGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(CONFIG["BLOCK_SIZE"], dim)
        self.memory_bank = nn.Parameter(torch.randn(32, dim)) # Hierarchical
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.world_sims = CONFIG["WORLD_SIM"]

    def forward(self, idx, targets=None, noise_scale=0.0):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        
        # Multi-World + Noise
        x_exp = x.repeat_interleave(self.world_sims, dim=0)
        if noise_scale > 0:
            x_exp += torch.randn_like(x_exp) * noise_scale
        
        for block in self.blocks:
            x_exp = block(x_exp, memory=self.memory_bank)
            
        x_final = self.ln_f(x_exp).view(self.world_sims, B, T, -1).mean(dim=0)
        logits = self.head(x_final)
        meta_memory = x_final.mean(dim=1).detach()

        loss = None
        if targets is not None:
            raw = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            aux = sum(b.moe.balance_loss for b in self.blocks)
            loss = raw + CONFIG["AUX_LOSS_WEIGHT"] * aux

        return logits, loss, meta_memory

    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -CONFIG["BLOCK_SIZE"]:])
            logits = logits[:, -1, :]
            probs = F.softmax(torch.nan_to_num(logits, nan=-1e9), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

# ============================================================
# 6. IMMORTAL CONTROLLER (The Brain)
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data = DataManager(rank)
        self.memory = DiskEpisodicMemory()
        
        # Modules
        self.agency = AgencyCore().to(DEVICE)
        self.meta_opt = MetaLearningEngine().to(DEVICE)
        self.telemetry = TelemetryLogger() # (Inferred class)
        
        self.population = []
        self.optimizers = []
        self.meta_optimizer = optim.AdamW(self.meta_opt.parameters(), lr=1e-4)
        
        self._spawn_population()
        self._load_state()

    def _spawn_population(self):
        for _ in range(CONFIG["POPULATION_SIZE"]):
            model = SacrsnSeedGPT(self.data.vocab_size).to(DEVICE)
            if self.world_size > 1:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=CONFIG["LR"]))

    def _load_state(self):
        if os.path.exists(PATHS["MEM_PKL"]):
            try:
                with open(PATHS["MEM_PKL"], "rb") as f:
                    self.unwrap(self.population[0]).load_state_dict(pickle.load(f), strict=False)
                self.memory.load()
                if self.rank == 0: logging.info(">>> ANCESTRAL STATE RESTORED")
            except Exception as e: logging.error(f"Load Failed: {e}")

    def unwrap(self, m): return m.module if hasattr(m, "module") else m

    # Save
    def save_extended_state(self, gen, tag=""):
        if self.rank != 0: return
        state = {
            "model": self.unwrap(self.population[0]).state_dict(),
            "agency": self.agency.state_dict(),
            "gen": gen
        }
        atomic_save(state, PATHS["CHECKPOINT"], use_torch=True)
        self.memory.save()
        logging.info(f"STATE SAVED: Gen {gen}")

    # Core Lifecycle
    def run_cycle(self, gen):
        # 1. Observation
        xb, _ = self.data.get_batch()
        with torch.no_grad():
            _, _, state = self.unwrap(self.population[0])(xb)
            state = state.mean(0)
        
        # 2. Decision
        action = self.agency.decide(state)
        if self.rank == 0: logging.info(f"CYCLE {gen} | ACTION: {action}")

        # 3. Execution
        if action == "train": self.train_epoch(gen)
        elif action == "evolve": self.evolve_population(gen)
        elif action == "reflect": self.reflect()
        elif action == "explore": self.train_epoch(gen, noise=0.05) # High noise

        # 4. Update Policy
        # Using simple reward: -Loss. Better implementation requires tracking delta-loss.
        # This is a placeholder for the full RL loop described in AgencyCore.
        pass

    def train_epoch(self, gen, noise=None):
        if noise is None:
            noise = max(0, 0.01 * (1 - gen/CONFIG["GENERATIONS"]))

        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                diff = random.choice(CONFIG["CURRICULUM"])
                x, y = self.data.get_batch(diff)
                
                _, loss, _ = model(x, y, noise_scale=noise)
                
                # Meta-Learning Update
                grad_norm = 0.0 # Placeholder
                lr_scale = self.meta_opt.get_lr_scale(loss.item(), grad_norm)
                
                # Meta-Optimization Logic (Simplified for brevity)
                # target = 1.0 if loss decreased else 0.5
                # meta_loss = (lr_scale - target)**2
                # self.meta_optimizer.zero_grad(); meta_loss.backward(); self.meta_optimizer.step()

                for g in opt.param_groups: g['lr'] = CONFIG["LR"] * lr_scale.item()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                if self.rank == 0 and i == 0 and step % 50 == 0:
                    logging.info(f"[Train] Loss: {loss.item():.4f} | LR: {lr_scale.item():.2f}")

    def evolve_population(self, gen):
        scores = []
        for model in self.population:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(CONFIG["EVAL_BATCHES"]):
                    x, y = self.data.get_batch()
                    _, l, _ = model(x, y)
                    losses.append(l.item())
            scores.append(-np.mean(losses))
        
        best_idx = np.argmax(scores)
        best_model = self.unwrap(self.population[best_idx])
        seed = IdentitySeed.compress(best_model)
        
        # Civilization Roles
        # Leader = best_idx. 
        # Others = Workers (Merge) or Explorers (Mutate)
        
        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best_idx:
                target = self.unwrap(self.population[i])
                # Reconstruct from seed + noise (Mutation)
                IdentitySeed.reconstruct(target, seed)
                with torch.no_grad():
                     for p in target.parameters():
                         p.add_(torch.randn_like(p) * 0.02)
                self.optimizers[i] = optim.AdamW(target.parameters(), lr=CONFIG["LR"])
        
        if self.rank == 0:
            logging.info(f"[Evolve] Best: {best_idx} | Score: {scores[best_idx]:.4f}")

    def reflect(self):
        x, _ = self.data.get_batch()
        with torch.no_grad():
            _, _, meta = self.unwrap(self.population[0])(x)
            state_vec = meta.mean(0)
            
            # Active Consolidation
            mem_bank = self.unwrap(self.population[0]).memory_bank
            mem_bank.data = 0.99 * mem_bank.data + 0.01 * state_vec.unsqueeze(0)

            self.memory.store(state_vec, {"time": time.time()})
            
            if self.rank == 0:
                 ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                 out = self.unwrap(self.population[0]).generate(ctx)
                 print("\n[REFLECT] ", self.data.decode(out[0].tolist()))

# --- Missing Class Defs for context ---
class MetaLearningEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def get_lr_scale(self, loss, grad_norm):
        inp = torch.tensor([loss, grad_norm], device=DEVICE).float()
        return self.net(inp) * 2.0

class TelemetryLogger:
    def __init__(self): self.file = PATHS["TELEMETRY"]
    def log(self, data):
        try:
            with open(self.file, "a") as f: f.write(json.dumps(data)+"\n")
        except: pass

# ============================================================
# 7. EXECUTION
# ============================================================
def run(rank, world):
    if world > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)

    core = ImmortalCoreController(rank, world)

    try:
        for g in range(CONFIG["GENERATIONS"]):
            core.run_cycle(g)
    except KeyboardInterrupt:
        if rank == 0: logging.info("INTERRUPT: Saving State...")
        core.save_extended_state("interrupt")
    finally:
        if world > 1: dist.destroy_process_group()
        if rank == 0: core.save_extended_state("final")

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        run(0, 1)
