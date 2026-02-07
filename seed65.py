# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v70.0 — THE SOVEREIGN
# ==============================================================================
#
# 🛡️ FINAL STABILITY PATCHES (v70.0):
# 1. DDP SAFETY: Conditional device_ids preventing CPU crash.
# 2. MOE GRAPH: Balance loss returned as tensor (not buffer) to preserve grad.
# 3. WORLD MODEL: Dynamic state resizing to prevent batch-size crashes.
# 4. IDENTITY: Skips LayerNorm/Bias params during reconstruction to preserve stability.
# 5. MEMORY: Enforces dictionary payload schema + safe decoding.
# 6. TRAINING: Hard checks for None batches + Optimizer reset on NaN rollback.
# 7. REFLECTION: Strips padding/artifacts from memory injection.
#
# STATUS: AUTONOMOUS / RESILIENT / PERSISTENT.
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
# 1. CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

CONFIG = {
    "EMBED_DIM": 384, "LAYERS": 6, "HEADS": 6, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "TOP_K": 2, "WINDOW_SIZE": 64, "WORLD_SIM": 5,
    "BATCH_SIZE": 16, "LR": 3e-4, "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
    "GRAD_EXPLOSION_THRESHOLD": 5.0,
    "AUX_LOSS_WEIGHT": 0.01,
    "POPULATION_SIZE": 4, "GENERATIONS": 50, "CYCLES_PER_GEN": 200,
    "REGENERATE_STEPS": 50, "EVAL_BATCHES": 4,
    "MEMORY_CAPACITY": 500_000, "IDENTITY_SEED_SIZE": 512,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0], "SYNTH_RATIO_CAP": 0.2, "WIPE_RATIO": 0.1
}

PATHS = {
    "MEM_PKL": "seed_memory.pkl", "MEM_BAK": "seed_memory_backup.pkl",
    "CHECKPOINT": "seed_full_state.pt", "ARCHIVE": "IMMORTAL_ARCHIVE.pt",
    "TELEMETRY": "telemetry.jsonl", "DIR_CKPT": "checkpoints",
    "DIR_ARCHIVE": "archive_history",
    "DATA": "data.txt", "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat", "MEM_META": "memory_meta.pkl"
}

for d in [PATHS["DIR_CKPT"], PATHS["DIR_ARCHIVE"]]:
    if not os.path.exists(d): os.makedirs(d)

# ============================================================
# 2. UTILITIES & SAFETY
# ============================================================
class RewardNormalizer:
    def __init__(self, alpha=0.95):
        self.mean = 0.0; self.var = 1.0; self.alpha = alpha; self.count = 0
    def normalize(self, x):
        self.count += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.var = self.alpha * self.var + (1 - self.alpha) * ((x - self.mean)**2)
        return (x - self.mean) / (math.sqrt(self.var) + 1e-6)
    def state_dict(self): return {"mean": self.mean, "var": self.var, "count": self.count}
    def load_state_dict(self, d): 
        self.mean = d["mean"]; self.var = d["var"]; self.count = d["count"]

def atomic_save(obj, path, use_torch=False):
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
        logging.error(f"Save Failed {path}: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def identity_hash(model):
    vec = torch.cat([p.flatten() for p in model.parameters()])
    if vec.numel() > 1000000: vec = vec[::100]
    return hashlib.sha256(vec.detach().cpu().numpy().tobytes()).hexdigest()[:8]

class TelemetryLogger:
    def __init__(self): self.file = PATHS["TELEMETRY"]
    def log(self, data):
        data["ts"] = time.time()
        try:
            with open(self.file, "a") as f: f.write(json.dumps(data)+"\n")
        except: pass

class SelfAudit:
    @staticmethod
    def run(controller):
        logging.info("🔍 STARTING DEEP SELF-AUDIT...")
        try:
            model = controller.unwrap(controller.population[0])
            x = torch.randint(0, 100, (1, 32)).to(DEVICE)
            y = torch.randint(0, 100, (1, 32)).to(DEVICE)
            logits, loss, _, _ = model(x, y) # Corrected tuple unpacking
            if torch.isnan(loss): raise RuntimeError("Model output is NaN")
            
            controller.optimizers[0].zero_grad()
            loss.backward()
            grad_norm = sum((p.grad.norm() for p in model.parameters() if p.grad is not None), torch.tensor(0.0).to(DEVICE))
            if grad_norm == 0: logging.warning("⚠️ Zero gradients in Self-Audit (Expected if model initialized zero)")
            controller.optimizers[0].step()
            
            if not hasattr(controller.memory, "save"): raise RuntimeError("Memory API mismatch")
            if CONFIG["BATCH_SIZE"] < 1: raise RuntimeError("Invalid Batch Size")

            logging.info("✅ SELF-AUDIT PASSED. SYSTEM NOMINAL.")
        except Exception as e:
            logging.critical(f"❌ AUDIT FAILED: {e}")
            sys.exit(1)

class DataManager:
    def __init__(self, rank):
        self.rank = rank
        self.data_tensor = None
        self.synth_tensor = torch.tensor([], dtype=torch.long)
        self.vocab_size = 0
        self.itos = {}; self.stoi = {}
        self._load_data()

    def _load_data(self):
        if self.rank == 0 and not os.path.exists(PATHS["DATA"]):
            with open(PATHS["DATA"], "w") as f: f.write("SACRSN INITIALIZATION " * 1000)
        if NUM_GPUS > 1: dist.barrier()
        
        with open(PATHS["DATA"], "r", encoding="utf-8") as f: raw = f.read()
        synth = ""
        if os.path.exists(PATHS["SYNTH"]):
            with open(PATHS["SYNTH"], "r", encoding="utf-8") as f: synth = f.read()

        chars = sorted(list(set(raw + synth)))
        self.vocab_size = len(chars) + 1
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i+1: ch for i, ch in enumerate(chars)}
        self.itos[0] = "<PAD>"; self.stoi["<PAD>"] = 0
        
        self.data_tensor = torch.tensor([self.stoi.get(c,0) for c in raw], dtype=torch.long)
        if synth: self.synth_tensor = torch.tensor([self.stoi.get(c,0) for c in synth], dtype=torch.long)

    def get_batch(self, difficulty=1.0):
        if self.data_tensor is None: raise RuntimeError("Data Error")
        use_synth = (len(self.synth_tensor) > CONFIG["BLOCK_SIZE"]) and (random.random() < CONFIG["SYNTH_RATIO_CAP"])
        source = self.synth_tensor if use_synth else self.data_tensor
        
        if len(source) < CONFIG["BLOCK_SIZE"]: source = self.data_tensor
        if len(source) < CONFIG["BLOCK_SIZE"]: return None, None

        seq_len = max(16, int(CONFIG["BLOCK_SIZE"] * difficulty))
        if len(source) < seq_len + 5: seq_len = len(source) - 2
        
        ix = torch.randint(len(source) - seq_len, (CONFIG["BATCH_SIZE"],))
        x = torch.stack([source[i:i+seq_len] for i in ix])
        y = torch.stack([source[i+1:i+seq_len+1] for i in ix])
        
        if seq_len < CONFIG["BLOCK_SIZE"]:
            pad = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"] - seq_len, dtype=torch.long)
            x = torch.cat([x, pad], dim=1); y = torch.cat([y, pad], dim=1)
        return x.to(DEVICE), y.to(DEVICE)

    def decode(self, t): return "".join([self.itos.get(i, "") for i in t if i!=0])
    def encode(self, s): return [self.stoi.get(c, 0) for c in s]

# ============================================================
# 3. COGNITIVE MODULES
# ============================================================
class DiskEpisodicMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.count = 0
        self.payloads = []
        self.file_emb = PATHS["MEM_VECS"]
        self.file_meta = PATHS["MEM_META"]
        if os.path.exists(self.file_emb):
            self.embeddings = np.memmap(self.file_emb, dtype='float32', mode='r+', shape=(self.max, self.dim))
            self.load_meta()
            if self.count > 0 and np.isnan(self.embeddings[0]).any():
                logging.warning("⚠️ MEMORY CORRUPTION DETECTED. RESETTING.")
                self.embeddings[:] = 0.0; self.count = 0; self.payloads = []
        else:
            self.embeddings = np.memmap(self.file_emb, dtype='float32', mode='w+', shape=(self.max, self.dim))

    # FIX 2: Strict Dictionary Payload
    def store(self, embedding, data_dict):
        # FIX 5: Rank safety moved to caller or handled here
        if dist.is_initialized() and dist.get_rank() != 0: return

        emb = embedding.detach().cpu().numpy().flatten()
        idx = self.count % self.max
        self.embeddings[idx] = emb
        
        # Ensure standard schema
        entry = {
            "text": data_dict.get("text", ""),
            "gen": data_dict.get("gen", 0),
            "score": data_dict.get("score", 0.0),
            "time": time.time()
        }
        
        if idx < len(self.payloads): self.payloads[idx] = entry
        else: self.payloads.append(entry)
        
        if len(self.payloads) > self.max: self.payloads = self.payloads[-self.max:]
        self.count += 1
        if self.count % 1000 == 0: self.embeddings.flush()

    def query(self, embedding, top_k=5):
        if self.count == 0: return []
        valid = min(self.count, self.max)
        recent = np.arange(max(0, valid - 500), valid)
        random_idx = np.random.choice(valid, min(valid, 4500), replace=False)
        idx_pool = np.unique(np.concatenate([recent, random_idx]))
        
        mem = self.embeddings[idx_pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top_local = np.argsort(sim)[-top_k:][::-1]
        real_indices = idx_pool[top_local]
        # Safety check
        return [self.payloads[i] for i in real_indices if i < len(self.payloads)]

    def save(self):
        self.embeddings.flush()
        with open(self.file_meta + ".tmp", "wb") as f:
            pickle.dump({"count": self.count, "payloads": self.payloads}, f)
        os.replace(self.file_meta + ".tmp", self.file_meta)

    def load_meta(self):
        if os.path.exists(self.file_meta):
            try:
                with open(self.file_meta, "rb") as f:
                    meta = pickle.load(f)
                    self.count = meta.get("count", 0); self.payloads = meta.get("payloads", [])
            except: pass

class AgencyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONFIG["EMBED_DIM"], 256), nn.ReLU(), nn.Linear(256, 5), nn.Softmax(-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs = []
        self.rewards = []
        self.scaler = RewardNormalizer()
        self.replay = deque(maxlen=2000)

    def decide(self, state):
        if state is None: return "train"
        state = torch.nan_to_num(state, nan=0.0).detach()
        probs = self.net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return ["train", "evolve", "explore", "reflect", "rest"][action.item()]

    def update_policy(self):
        if not self.rewards: return
        policy_loss = []
        # Rewards are now a list of floats, normalized here
        normalized = [self.scaler.normalize(r) for r in self.rewards]
        returns = torch.tensor(normalized).to(DEVICE)
        returns = torch.clamp(returns, -2.0, 2.0)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()
        del self.saved_log_probs[:]; del self.rewards[:]

class NeuralWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.gru = nn.GRUCell(dim, dim)
        self.register_buffer("state", torch.zeros(1, dim))
    
    # FIX 3: Dynamic State Resizing
    def forward(self, x):
        # x: [Batch, Dim] or [Dim]
        if x.dim() == 1: x = x.unsqueeze(0)
        
        # Resize state if batch size changed
        if x.size(0) != self.state.size(0):
            self.state = torch.zeros(x.size(0), x.size(1), device=x.device)
            
        self.state = self.state.detach().clone()
        self.state = self.gru(x, self.state)
        return self.state.squeeze(0)
    
    def reset_state(self): 
        # Reset to size 1, will resize on next forward
        self.state = torch.zeros(1, CONFIG["EMBED_DIM"], device=DEVICE)

class IdentitySeed:
    @staticmethod
    def compress(model):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // CONFIG["IDENTITY_SEED_SIZE"])
        sampled = flat[::step][:CONFIG["IDENTITY_SEED_SIZE"]]
        hash_sig = hashlib.sha256(sampled.detach().cpu().numpy().tobytes()).hexdigest()
        return {"weights": sampled.detach().cpu(), "meta": {"layers": len(model.blocks), "hash": hash_sig}}

    @staticmethod
    def reconstruct(model, seed):
        if isinstance(seed, dict): weights = seed["weights"]
        else: weights = seed
        weights = weights.to(DEVICE)
        
        if weights.std() < 1e-6: weights += torch.randn_like(weights) * 1e-4

        target_layers = seed["meta"].get("layers", CONFIG["LAYERS"])
        while len(model.blocks) < target_layers: model.blocks.append(RecurrentBlock().to(DEVICE))
        while len(model.blocks) > target_layers: del model.blocks[-1]
        
        total = sum(p.numel() for p in model.parameters())
        x_t = torch.linspace(0, 1, total, device=DEVICE)
        x_s = torch.linspace(0, 1, len(weights), device=DEVICE)
        idx = torch.bucketize(x_t, x_s).clamp(0, len(weights)-2)
        val = torch.lerp(weights[idx], weights[idx+1], (x_t-x_s[idx])/(x_s[idx+1]-x_s[idx]+1e-9))
        
        ptr = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                # FIX 4: Skip LayerNorm/Bias to prevent destabilization
                if "ln" in name or "bias" in name or "gate" in name:
                    ptr += p.numel()
                    continue
                    
                n = p.numel()
                p.data.copy_(val[ptr:ptr+n].reshape(p.shape))
                ptr += n

class MetaLearningEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def get_lr_scale(self, loss, grad_norm):
        inp = torch.tensor([loss / 10.0, math.log1p(grad_norm)], device=DEVICE).float()
        return self.net(inp) * 2.0

class HierarchicalMemory(nn.Module):
    def __init__(self, dim=CONFIG["EMBED_DIM"]):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(32, dim))
    def read(self): return self.bank

# ============================================================
# 5. NEURAL ARCHITECTURE
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
        i = torch.arange(b).view(-1,1); j = torch.arange(b).view(1,-1)
        self.register_buffer("local_mask", (torch.abs(i-j) <= self.window).view(1,1,b,b))

    def forward(self, x, memory=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)

        if memory is not None:
            mem_exp = memory.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem_exp, k], 1); v = torch.cat([mem_exp, v], 1)
            T_total = k.size(1)
        else: T_total = T

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_total, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_total, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # FIX 7: Safe Clamping
        start = max(0, k.size(2) - T)
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

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        # FIX 6: Return loss, don't store in buffer to avoid graph detach issues
        balance_loss = (scores.mean(dim=(0,1))**2).sum() * CONFIG["NUM_EXPERTS"]
        
        topk, indices = torch.topk(scores, CONFIG["TOP_K"], dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
        masked = scores * mask
        masked = masked / (masked.sum(-1, keepdim=True) + 1e-9)
        
        out = sum(masked[...,i:i+1] * e(x) for i,e in enumerate(self.experts))
        return out, balance_loss

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.ln1 = nn.LayerNorm(dim); self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(dim); self.moe = MoEBlock()
    def forward(self, x, memory=None):
        x = x + self.attn(self.ln1(x), memory)
        m_out, m_loss = self.moe(self.ln2(x))
        x = x + m_out
        return x, m_loss

class SacrsnSeedGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(CONFIG["BLOCK_SIZE"], dim)
        self.memory_bank = nn.Parameter(torch.randn(32, dim))
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.world_sims = CONFIG["WORLD_SIM"]

    def forward(self, idx, targets=None, noise_scale=0.0):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        mem = self.memory_bank
        
        x_exp = x.repeat_interleave(self.world_sims, 0)
        if noise_scale > 0: x_exp += torch.randn_like(x_exp) * noise_scale
        
        total_aux_loss = 0
        for block in self.blocks:
            x_exp, aux = block(x_exp, memory=mem)
            total_aux_loss += aux
            
        x_final = self.ln_f(x_exp).view(self.world_sims, B, T, -1).mean(0)
        logits = self.head(x_final)
        meta_memory = x_final.mean(dim=1).detach()

        loss = None
        if targets is not None:
            raw = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            loss = raw + CONFIG["AUX_LOSS_WEIGHT"] * total_aux_loss

        return logits, loss, meta_memory, total_aux_loss

    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            logits, _, _, _ = self(idx[:, -CONFIG["BLOCK_SIZE"]:])
            probs = F.softmax(torch.nan_to_num(logits[:, -1, :], nan=-1e9), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

# ============================================================
# 6. IMMORTAL CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data = DataManager(rank)
        self.memory = DiskEpisodicMemory()
        
        self.agency = AgencyCore().to(DEVICE)
        self.world = NeuralWorldModel().to(DEVICE)
        self.meta_opt = MetaLearningEngine().to(DEVICE)
        self.telemetry = TelemetryLogger()
        
        self.population = []
        self.optimizers = []
        self.world_opt = optim.AdamW(self.world.parameters(), lr=1e-4)
        self.meta_optimizer = optim.AdamW(self.meta_opt.parameters(), lr=1e-4)
        self.agency_optimizer = self.agency.optimizer
        
        self._spawn()
        self._load_state()
        if rank == 0: SelfAudit.run(self)

    def _spawn(self):
        for _ in range(CONFIG["POPULATION_SIZE"]):
            model = SacrsnSeedGPT(self.data.vocab_size).to(DEVICE)
            if self.world_size > 1:
                # FIX 1: DDP Safety
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.rank] if DEVICE == "cuda" else None)
            self.population.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=CONFIG["LR"]))

    def _load_state(self):
        if os.path.exists(PATHS["CHECKPOINT"]):
            try:
                ckpt = torch.load(PATHS["CHECKPOINT"], map_location=DEVICE)
                if "population" in ckpt:
                    for i, state in enumerate(ckpt["population"]):
                        if i < len(self.population): self.unwrap(self.population[i]).load_state_dict(state)
                if "optimizers" in ckpt:
                    for opt, st in zip(self.optimizers, ckpt["optimizers"]): opt.load_state_dict(st)
                if "agency" in ckpt: self.agency.load_state_dict(ckpt["agency"])
                if "meta_opt" in ckpt: self.meta_opt.load_state_dict(ckpt["meta_opt"])
                if "world" in ckpt: self.world.load_state_dict(ckpt["world"])
                if "rng" in ckpt: torch.set_rng_state(ckpt["rng"])
                if "cuda_rng" in ckpt and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
                self.memory.load_meta()
                if self.rank == 0: logging.info(f">>> RESTORED GENERATION {ckpt.get('gen', '?')}")
            except Exception as e: logging.error(f"Load Failed: {e}")

    def unwrap(self, m): return m.module if hasattr(m, "module") else m

    def save_system(self, gen, tag=""):
        if self.rank != 0: return
        state = {
            "population": [self.unwrap(p).state_dict() for p in self.population],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "agency": self.agency.state_dict(),
            "agency_opt": self.agency.optimizer.state_dict(),
            "meta_opt": self.meta_opt.state_dict(),
            "meta_opt_opt": self.meta_optimizer.state_dict(),
            "world": self.world.state_dict(),
            "world_opt": self.world_opt.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "gen": gen, "config": CONFIG
        }
        atomic_save(state, PATHS["CHECKPOINT"], use_torch=True)
        atomic_save(state, os.path.join(PATHS["DIR_ARCHIVE"], f"gen_{gen:05d}.pt"), use_torch=True)
        self.memory.save()
        logging.info(f"💾 SYSTEM SAVED | Gen {gen} | Tag: {tag}")

    def run_cycle(self, gen):
        xb, yb = self.data.get_batch()
        if xb is None: return

        self.world.reset_state()

        with torch.no_grad():
            _, _, state, _ = self.unwrap(self.population[0])(xb)
            state_vec = state.mean(0)
        
        action = self.agency.decide(state_vec)
        if self.rank == 0: logging.info(f"🤖 CYCLE {gen} | ACTION: {action}")

        # Train / Evolve returns avg_loss from Agent 0 for reward calc
        avg_loss = 0.0
        if action == "train": 
            avg_loss = self.train(gen)
        elif action == "evolve": 
            self.evolve(gen)
        elif action == "reflect": 
            self.reflect()
        
        # World Model Train (Detached)
        state_vec_detached = state_vec.detach().clone()
        pred_vec = self.world(state_vec_detached)
        wm_loss = F.mse_loss(pred_vec, state_vec_detached)
        
        self.world_opt.zero_grad()
        wm_loss.backward()
        self.world_opt.step()
        
        curiosity = wm_loss.item()
        loss_reward = -avg_loss if action == "train" else 0.0
        
        self.agency.rewards = [loss_reward * 1.0 + curiosity * 0.1]
        self.agency.update_policy()

        self.save_system(gen)

    def train(self, gen):
        noise = max(0, 0.01 * (1 - gen/CONFIG["GENERATIONS"]))
        epoch_loss = 0.0
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            agent_loss_sum = 0.0
            
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                diff = random.choice(CONFIG["CURRICULUM"])
                x, y = self.data.get_batch(diff)
                # FIX 6: Batch None Check
                if x is None: continue
                
                _, loss, _, _ = model(x, y, noise_scale=noise)
                
                # FIX 8: Catastrophic Reset
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.critical(f"⚠️ NAN LOSS (Agent {i}). FULL RESET.")
                    self._load_state()
                    # Reset optimizer to clear bad momentum
                    self.optimizers[i] = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
                    return 0.0

                opt.zero_grad()
                loss.backward()
                
                grad_norm = sum((p.grad.norm() for p in model.parameters() if p.grad is not None), torch.tensor(0.0).to(DEVICE))
                lr_scale = self.meta_opt.get_lr_scale(loss.item(), grad_norm)
                for g in opt.param_groups: g['lr'] = CONFIG["LR"] * lr_scale.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                agent_loss_sum += loss.item()

                if self.rank == 0 and i == 0 and step % 10 == 0:
                    logging.info(f"   Step {step} | Loss: {loss.item():.4f} | LR: {lr_scale.item():.2f}")
            
            if i == 0: epoch_loss = agent_loss_sum / max(1, CONFIG["CYCLES_PER_GEN"])
            
        return epoch_loss

    def evolve(self, gen):
        scores = []
        for i, model in enumerate(self.population):
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(CONFIG["EVAL_BATCHES"]):
                    x, y = self.data.get_batch()
                    if x is None: continue
                    _, l, _, _ = model(x, y)
                    losses.append(l.item())
            score = -np.mean(losses) if losses else -100.0
            scores.append(score)
            if self.rank == 0: logging.info(f"   Agent {i} Score: {score:.4f}")

        best_idx = np.argmax(scores)
        best_model = self.unwrap(self.population[best_idx])
        seed = IdentitySeed.compress(best_model)
        
        if self.rank == 0: 
            drift = identity_hash(best_model)
            logging.info(f"🧬 WINNER: Agent {best_idx} | ID: {drift}")

        best_opt_state = self.optimizers[best_idx].state_dict()

        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best_idx:
                target = self.unwrap(self.population[i])
                IdentitySeed.reconstruct(target, seed)
                with torch.no_grad():
                     for p in target.parameters(): p.add_(torch.randn_like(p) * 0.02)
                self.optimizers[i].load_state_dict(best_opt_state)
            
            if self.world_size > 1: dist.barrier()

    def reflect(self):
        x, _ = self.data.get_batch()
        if x is None: return
        with torch.no_grad():
            _, _, meta, _ = self.unwrap(self.population[0])(x)
            vec = meta.mean(0)
            
            recalled = self.memory.query(vec)
            # FIX 7: Safe Decode
            context_str = ""
            if recalled:
                valid_text = [r["text"] for r in recalled if isinstance(r, dict) and "text" in r]
                context_str = " ".join(valid_text[:2]).strip()
            
            prompt = f"MEMORY: {context_str}\n[REFLECT]:" if context_str else "[REFLECT]:"
            encoded = torch.tensor([self.data.encode(prompt)], device=DEVICE)

            self.memory.store(vec, {"text": self.data.decode(x[0].tolist()), "gen": 0, "type": "reflection"})
            
            if self.rank == 0:
                 out = self.unwrap(self.population[0]).generate(encoded)
                 print(f"\n{self.data.decode(out[0].tolist())}\n")

# ============================================================
# EXECUTION
# ============================================================
def run(rank, world):
    if world > 1:
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)

    core = ImmortalCoreController(rank, world)
    try:
        for g in range(CONFIG["GENERATIONS"]):
            core.run_cycle(g)
    except KeyboardInterrupt:
        if rank == 0: logging.info("INTERRUPT SAVED.")
        core.save_system(999, "interrupt")
    finally:
        if rank == 0: core.memory.embeddings.flush()
        if world > 1: dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        run(0, 1)
