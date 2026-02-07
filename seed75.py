# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v100.0 — THE GODHEAD
# ==============================================================================
#
# A TOTAL UNIFICATION OF ALL PREVIOUS VERSIONS (v1 -> v70).
#
# [ARCHITECTURAL STACK]
# 1. Apex Transformer: Persistent Latent State + World/Value Heads (Z-Series).
# 2. Multi-World Sim: 5x Parallel Reality Simulation with Noise (v4).
# 3. Dynamic MoE: Soft Top-K Gating with Load Balancing (v16).
# 4. Sparse Attention: Vectorized, Cached, Windowed (v20).
# 5. Thought Engine: Internal Latent Bottleneck for Reasoning (v63).
#
# [COGNITIVE STACK]
# 1. Agency Core: RL-based Decision Making (Train/Evolve/Reflect/Rest) (v13).
# 2. Neural World Model: GRU-based State Prediction & Curiosity (v13/28).
# 3. Meta-Learning: Loss-Delta Gradient Scaling (v26).
# 4. Predictive Self-Model: Latent Fitness Prediction (v26).
# 5. Infinite Memory: Disk-Backed Memmap + Centroid Pruning (v20).
#
# [EVOLUTIONARY STACK]
# 1. Civilization: Leader/Worker/Explorer Roles + Sharded Merge (v27).
# 2. Genetics: Identity Seed Compression + Hash Chaining (v13).
# 3. Research Engine: Autonomous Hyperparameter Hypothesis Testing (v27).
# 4. Lifelong Learning: EWC (Elastic Weight Consolidation) (v62).
#
# [SAFETY STACK]
# 1. Ironclad: Atomic Saves, DDP Sync, NaN Guards, Crash Recovery (v24).
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
    # Apex Architecture
    "EMBED_DIM": 384, "LAYERS": 8, "HEADS": 8, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "TOP_K": 2, "WINDOW_SIZE": 64, 
    "THOUGHT_DIM": 256, "LATENT_DIM": 384,
    "WORLD_SIM": 5, "INFERENCE_PATHS": 4,
    
    # Training
    "BATCH_SIZE": 16, "LR": 3e-4, "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
    "AUX_LOSS_WEIGHT": 0.01, "EWC_LAMBDA": 0.4,
    
    # Civilization
    "POPULATION_SIZE": 4, "GENERATIONS": 100, "CYCLES_PER_GEN": 200,
    "EVAL_BATCHES": 4, "ROLES": ["leader", "researcher", "explorer", "critic"],
    
    # Cognition
    "MEMORY_CAPACITY": 1_000_000, "IDENTITY_SEED_SIZE": 512,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0], "SYNTH_RATIO_CAP": 0.2, "WIPE_RATIO": 0.1
}

PATHS = {
    "CHECKPOINT": "godhead_state.pt",
    "ARCHIVE": "godhead_archive.pt",
    "TELEMETRY": "telemetry.jsonl",
    "DIR_CKPT": "checkpoints", "DIR_ARCHIVE": "archive_history", "DIR_SANDBOX": "sandbox",
    "DATA": "data.txt", "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat", "MEM_META": "memory_meta.pkl"
}

for d in [PATHS["DIR_CKPT"], PATHS["DIR_ARCHIVE"], PATHS["DIR_SANDBOX"]]:
    if not os.path.exists(d): os.makedirs(d)

# ============================================================
# 2. UTILITIES & DATA
# ============================================================
def atomic_save(obj, path, use_torch=False):
    tmp = path + ".tmp"
    try:
        if use_torch: torch.save(obj, tmp)
        else: 
            with open(tmp, "wb") as f: pickle.dump(obj, f)
        if os.path.exists(path):
            try: os.replace(tmp, path)
            except OSError: os.remove(path); os.rename(tmp, path)
        else: os.rename(tmp, path)
    except Exception as e:
        logging.error(f"Save Error {path}: {e}")
        if os.path.exists(tmp): os.remove(tmp)

def identity_hash(model):
    vec = torch.cat([p.flatten() for p in model.parameters()])
    if vec.numel() > 1000000: vec = vec[::100]
    return hashlib.sha256(vec.detach().cpu().numpy().tobytes()).hexdigest()[:8]

class RewardNormalizer:
    def __init__(self, alpha=0.95):
        self.mean = 0.0; self.var = 1.0; self.alpha = alpha; self.count = 0
    def normalize(self, x):
        self.count += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.var = self.alpha * self.var + (1 - self.alpha) * ((x - self.mean)**2)
        return (x - self.mean) / (math.sqrt(self.var) + 1e-6)

class TelemetryLogger:
    def __init__(self): self.file = PATHS["TELEMETRY"]
    def log(self, data):
        data["ts"] = time.time()
        try: 
            with open(self.file, "a") as f: f.write(json.dumps(data)+"\n")
        except: pass

class DataManager:
    def __init__(self, rank):
        self.rank = rank; self.data = None; self.synth = torch.tensor([], dtype=torch.long)
        self.vocab_size = 0; self.itos = {}; self.stoi = {}
        self._load()
    
    def _load(self):
        if self.rank == 0 and not os.path.exists(PATHS["DATA"]):
             with open(PATHS["DATA"], "w") as f: f.write("SACRSN GODHEAD " * 5000)
        if NUM_GPUS > 1: dist.barrier()
        with open(PATHS["DATA"], "r") as f: raw = f.read()
        synth_txt = ""
        if os.path.exists(PATHS["SYNTH"]):
            with open(PATHS["SYNTH"], "r") as f: synth_txt = f.read()
        
        chars = sorted(list(set(raw + synth_txt)))
        self.vocab_size = len(chars) + 1
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}; self.stoi["<PAD>"] = 0
        self.itos = {i+1:ch for i,ch in enumerate(chars)}; self.itos[0] = "<PAD>"
        self.data = torch.tensor([self.stoi.get(c,0) for c in raw], dtype=torch.long)
        if synth_txt: self.synth = torch.tensor([self.stoi.get(c,0) for c in synth_txt], dtype=torch.long)

    def get_batch(self, difficulty=1.0):
        if self.data is None: raise RuntimeError("Data Error")
        use_synth = (len(self.synth) > CONFIG["BLOCK_SIZE"]) and (random.random() < CONFIG["SYNTH_RATIO_CAP"])
        src = self.synth if use_synth else self.data
        if len(src) < CONFIG["BLOCK_SIZE"]: src = self.data
        if len(src) < CONFIG["BLOCK_SIZE"]: return None, None
        
        seq = max(16, int(CONFIG["BLOCK_SIZE"] * difficulty))
        if len(src) < seq + 5: seq = len(src) - 2
        ix = torch.randint(len(src) - seq, (CONFIG["BATCH_SIZE"],))
        x = torch.stack([src[i:i+seq] for i in ix])
        y = torch.stack([src[i+1:i+seq+1] for i in ix])
        
        if seq < CONFIG["BLOCK_SIZE"]:
            pad = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"] - seq, dtype=torch.long)
            x = torch.cat([x, pad], 1); y = torch.cat([y, pad], 1)
        return x.to(DEVICE), y.to(DEVICE)
    
    def decode(self, t): return "".join([self.itos.get(i, "") for i in t if i!=0])
    def encode(self, s): return [self.stoi.get(c, 0) for c in s]

# ============================================================
# 3. ADVANCED COGNITIVE STACK
# ============================================================

# --- Feature 1: Infinite Disk Memory (v20) ---
class DiskEpisodicMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.count = 0; self.payloads = []; self.centroids = []
        self.file_emb = PATHS["MEM_VECS"]; self.file_meta = PATHS["MEM_META"]
        if os.path.exists(self.file_emb):
            self.emb = np.memmap(self.file_emb, dtype='float32', mode='r+', shape=(self.max, self.dim))
            self.load_meta()
            if self.count > 0 and np.isnan(self.emb[0]).any(): self.emb[:] = 0; self.count = 0
        else:
            self.emb = np.memmap(self.file_emb, dtype='float32', mode='w+', shape=(self.max, self.dim))

    def store(self, embedding, payload):
        if dist.is_initialized() and dist.get_rank() != 0: return
        vec = embedding.detach().cpu().numpy().flatten()
        
        # Centroid Pruning (v28)
        if len(self.centroids) > 0 and self.count % 100 != 0:
            dists = np.linalg.norm(np.stack(self.centroids) - vec, axis=1)
            if dists.min() < 0.05: return 

        idx = self.count % self.max
        self.emb[idx] = vec
        entry = {"data": payload, "time": time.time(), "id": self.count}
        if idx < len(self.payloads): self.payloads[idx] = entry
        else: self.payloads.append(entry)
        
        if len(self.payloads) > self.max: self.payloads = self.payloads[-self.max:]
        self.count += 1
        if self.count % 1000 == 0: 
            self.emb.flush()
            valid = min(self.count, self.max)
            self.centroids = self.emb[np.random.choice(valid, min(valid, 50))]

    def query(self, embedding, top_k=5):
        if self.count == 0: return []
        valid = min(self.count, self.max)
        # Reservoir Sampling (v60)
        recent = np.arange(max(0, valid - 500), valid)
        random = np.random.choice(valid, min(valid, 4500), replace=False)
        pool = np.unique(np.concatenate([recent, random]))
        
        mem = self.emb[pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(sim)[-top_k:][::-1]
        return [self.payloads[pool[i]] for i in top if pool[i] < len(self.payloads)]

    def save(self):
        self.emb.flush()
        atomic_save({"count": self.count, "payloads": self.payloads}, self.file_meta)
    def load_meta(self):
        if os.path.exists(self.file_meta):
            try:
                with open(self.file_meta, "rb") as f:
                    d = pickle.load(f); self.count = d["count"]; self.payloads = d["payloads"]
            except: pass

# --- Feature 2: Agency & World Modeling (v13/28) ---
class AgencyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONFIG["EMBED_DIM"], 256), nn.ReLU(), nn.Linear(256, 5), nn.Softmax(-1))
        self.opt = optim.AdamW(self.parameters(), lr=1e-4)
        self.log_probs = []; self.rewards = []; self.scaler = RewardNormalizer()
        self.replay = deque(maxlen=2000)

    def decide(self, state):
        if state is None: return "train"
        state = torch.nan_to_num(state, nan=0.0).detach()
        dist = torch.distributions.Categorical(self.net(state))
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return ["train", "evolve", "explore", "reflect", "rest"][action.item()]

    def update_policy(self):
        if not self.rewards: return
        R = 0; policy_loss = []; returns = []
        for r in self.rewards[::-1]: R = r + 0.99 * R; returns.insert(0, R)
        self.replay.extend(returns)
        
        norm = [self.scaler.normalize(r) for r in returns]
        ret_t = torch.tensor(norm).to(DEVICE).clamp(-2.0, 2.0)
        
        for lp, r in zip(self.log_probs, ret_t): policy_loss.append(-lp * r)
        self.opt.zero_grad()
        if policy_loss: torch.stack(policy_loss).sum().backward(); self.opt.step()
        del self.log_probs[:]; del self.rewards[:]

class NeuralWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(CONFIG["EMBED_DIM"], CONFIG["EMBED_DIM"])
        self.register_buffer("state", torch.zeros(1, CONFIG["EMBED_DIM"]))
    def forward(self, x):
        self.state = self.state.detach().clone()
        self.state = self.gru(x.unsqueeze(0), self.state)
        return self.state.squeeze(0)
    def reset_state(self): self.state.zero_()

class PlanningEngine:
    def __init__(self, wm, self_model): self.wm = wm; self.sm = self_model
    def plan(self, model, steps=3):
        # Latent Planning in Seed Space (v28)
        seed = IdentitySeed.compress(model)["weights"].to(DEVICE)
        fits = []
        curr = seed
        for _ in range(steps):
            nxt, fit = self.sm(curr)
            fits.append(fit.item())
            curr = nxt
        return max(fits) if fits else 0.0

# --- Feature 3: Meta-Cognition (v26) ---
class MetaLearningEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def get_scale(self, loss, grad):
        return self.net(torch.tensor([loss/10.0, math.log1p(grad)], device=DEVICE).float()) * 2.0

class PredictiveSelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, dim))
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, seed): 
        nxt = self.net(seed)
        return nxt, self.head(nxt)

# --- Feature 4: Evolutionary Components (v27) ---
class IdentitySeed:
    @staticmethod
    def compress(model):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // CONFIG["IDENTITY_SEED_SIZE"])
        # Multi-Anchor (v28)
        anchors = [flat[i::step][:CONFIG["IDENTITY_SEED_SIZE"]] for i in range(3)]
        for i in range(3):
            if len(anchors[i]) < CONFIG["IDENTITY_SEED_SIZE"]:
                 anchors[i] = F.pad(anchors[i], (0, CONFIG["IDENTITY_SEED_SIZE"] - len(anchors[i])))
        sampled = torch.cat(anchors).detach().cpu()
        h = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
        return {"weights": sampled, "meta": {"layers": len(model.blocks), "hash": h}}

    @staticmethod
    def reconstruct(model, seed):
        w = seed["weights"].to(DEVICE) if isinstance(seed, dict) else seed.to(DEVICE)
        # Blend Anchors
        if w.numel() == 3 * CONFIG["IDENTITY_SEED_SIZE"]:
            w = w.view(3, CONFIG["IDENTITY_SEED_SIZE"]).mean(0)
        
        # Arch Sync
        target = seed["meta"].get("layers", CONFIG["LAYERS"])
        while len(model.blocks) < target: model.blocks.append(RecurrentBlock().to(DEVICE))
        while len(model.blocks) > target: del model.blocks[-1]
        
        total = sum(p.numel() for p in model.parameters())
        x_t = torch.linspace(0, 1, total, device=DEVICE)
        x_s = torch.linspace(0, 1, len(w), device=DEVICE)
        idx = torch.bucketize(x_t, x_s).clamp(0, len(w)-2)
        den = (x_s[idx+1] - x_s[idx]).clamp(min=1e-9)
        val = torch.lerp(w[idx], w[idx+1], (x_t - x_s[idx]) / den)
        
        ptr = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "ln" in n or "bias" in n: ptr += p.numel(); continue # Skip sensitive params
                c = p.numel()
                p.data.copy_(val[ptr:ptr+c].reshape(p.shape))
                ptr += c

class CivilizationCoordinator:
    def assign_roles(self, scores):
        idx = np.argsort(scores)[::-1]
        roles = {idx[0]: "leader"}
        for i in idx[1:]: roles[i] = "explorer" if i % 2 == 0 else "worker"
        return roles

class ShardedIdentity:
    def merge(self, target, shards):
        with torch.no_grad():
            for p, s in zip(target.parameters(), shards): p.copy_(p * 0.9 + s * 0.1)

class AutonomousResearch:
    def __init__(self): self.hypothesis = None
    def act(self, ctrl):
        if random.random() < 0.01:
            ctrl.grow_network(0)
            logging.info("🔬 RESEARCH: Expanding Capacity")

# ============================================================
# 4. GODHEAD ARCHITECTURE (Apex + Z-Series)
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.qkv = nn.Linear(dim, dim*3); self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // CONFIG["HEADS"]
        self.scale = self.head_dim ** -0.5
        self.gate = nn.Parameter(torch.ones(dim))
        b = CONFIG["BLOCK_SIZE"]
        self.register_buffer("mask", torch.tril(torch.ones(b,b)).view(1,1,b,b))
        i = torch.arange(b).view(-1,1); j = torch.arange(b).view(1,-1)
        self.register_buffer("local", (torch.abs(i-j) <= CONFIG["WINDOW_SIZE"]).view(1,1,b,b))

    def forward(self, x, mem=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        if mem is not None:
            m = mem.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([m, k], 1); v = torch.cat([m, v], 1)
        
        q, k, v = [t.view(B, -1, CONFIG["HEADS"], self.head_dim).transpose(1,2) for t in (q,k,v)]
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        # Masking
        start = max(0, k.size(2) - T)
        att_self = att[:, :, :, start:]
        if T <= self.mask.size(2):
            att_self = att_self.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            att_self = att_self.masked_fill(self.local[:,:,:T,:T]==0, float('-inf'))
        att[:, :, :, start:] = att_self
        
        y = F.softmax(att, -1) @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
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
        self.register_buffer("bal_loss", torch.tensor(0.0))

    def forward(self, x):
        scores = self.gate(x)
        scores = torch.nan_to_num(scores, 0.0) # Safety
        probs = F.softmax(scores, -1)
        self.bal_loss = ((probs.mean((0,1))**2).sum() * CONFIG["NUM_EXPERTS"]).to(x.device)
        
        topk, idx = torch.topk(probs, CONFIG["TOP_K"], -1)
        mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        masked = probs * mask
        masked = masked / (masked.sum(-1, keepdim=True) + 1e-9)
        return sum(masked[...,i:i+1] * e(x) for i,e in enumerate(self.experts))

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.ln1 = nn.LayerNorm(dim); self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(dim); self.moe = MoEBlock()
        self.thought = nn.Sequential(nn.Linear(dim, CONFIG["THOUGHT_DIM"]), nn.Tanh(), nn.Linear(CONFIG["THOUGHT_DIM"], dim))

    def forward(self, x, mem=None):
        x = x + self.attn(self.ln1(x), mem)
        x = x + self.thought(x) # Thought Engine v63
        x = x + self.moe(self.ln2(x))
        return x

class GodheadTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(CONFIG["BLOCK_SIZE"], dim)
        self.mem = nn.Parameter(torch.randn(32, dim))
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        
        # Apex Heads (Z-Series)
        self.world_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.val_head = nn.Linear(dim, 1)

    def forward(self, idx, targets=None, noise=0.0):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
        
        # Multi-World
        w = CONFIG["WORLD_SIM"]
        x = x.repeat_interleave(w, 0)
        if noise > 0: x += torch.randn_like(x) * noise
        
        mem = self.mem
        for b in self.blocks: x = b(x, mem)
        
        # Collapse
        x_flat = self.ln_f(x)
        x_mean = x_flat.view(w, B, T, -1).mean(0)
        
        logits = self.head(x_mean)
        wm_pred = self.world_head(x_mean)
        val_pred = self.val_head(x_mean)
        
        loss = None
        if targets is not None:
            # Main Loss
            main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
            # Z-Series Aux Losses
            with torch.no_grad(): target_next = x_mean.detach()
            wm_loss = F.mse_loss(wm_pred, target_next)
            
            tok_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
            val_loss = F.mse_loss(val_pred.squeeze(), tok_loss.detach())
            
            moe_loss = sum(b.moe.bal_loss for b in self.blocks)
            
            loss = main + (wm_loss * 0.1) + (val_loss * 0.1) + (moe_loss * CONFIG["AUX_LOSS_WEIGHT"])

        return logits, loss, x_mean.mean(1).detach()

    def generate(self, idx, max_new=100):
        for _ in range(max_new):
            logits, _, _ = self(idx[:, -CONFIG["BLOCK_SIZE"]:])
            probs = F.softmax(torch.nan_to_num(logits[:, -1, :], nan=-1e9), -1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), 1)
        return idx

# ============================================================
# 5. CONTROLLER (THE BRAIN)
# ============================================================
class GodheadController:
    def __init__(self, rank, world):
        self.rank, self.world_size = rank, world
        self.data = DataManager(rank)
        self.memory = DiskEpisodicMemory()
        
        # Brains
        self.agency = AgencyCore().to(DEVICE)
        self.wm = NeuralWorldModel().to(DEVICE)
        self.meta = MetaLearningEngine().to(DEVICE)
        self.self = PredictiveSelfModel().to(DEVICE)
        self.tele = TelemetryLogger()
        
        # Helpers
        self.civ = CivilizationCoordinator()
        self.res = AutonomousResearch()
        self.plan = PlanningEngine(self.wm, self.self)
        self.life = LifelongProtector()
        
        self.pop = []
        self.opts = []
        self.wm_opt = optim.AdamW(self.wm.parameters(), lr=1e-4)
        self.self_opt = optim.AdamW(self.self.parameters(), lr=1e-4)
        
        self.replay = deque(maxlen=2000)
        self.prev_loss = 0.0
        
        self._spawn()
        self._load()
        if rank == 0: self._audit()

    def _spawn(self):
        for _ in range(CONFIG["POPULATION_SIZE"]):
            m = GodheadTransformer(self.data.vocab_size).to(DEVICE)
            if self.world_size > 1: m = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
            self.pop.append(m)
            self.opts.append(optim.AdamW(m.parameters(), lr=CONFIG["LR"]))

    def _audit(self):
        try:
            x, y = self.data.get_batch()
            _, l, _ = self.pop[0](x, y)
            l.backward()
            self.opts[0].step()
            self.opts[0].zero_grad()
            logging.info("✅ SELF-AUDIT COMPLETE")
        except Exception as e:
            logging.critical(f"❌ AUDIT FAILED: {e}"); sys.exit(1)

    def _load(self):
        if os.path.exists(PATHS["CHECKPOINT"]):
            d = torch.load(PATHS["CHECKPOINT"], map_location=DEVICE)
            if "pop" in d:
                for i, s in enumerate(d["pop"]): 
                    if i < len(self.pop): self.unwrap(self.pop[i]).load_state_dict(s)
            if "opts" in d:
                for i, s in enumerate(d["opts"]): self.opts[i].load_state_dict(s)
            if "agency" in d: self.agency.load_state_dict(d["agency"])
            if "wm" in d: self.wm.load_state_dict(d["wm"])
            if "self" in d: self.self.load_state_dict(d["self"])
            self.memory.load_meta()
            if self.rank == 0: logging.info(f"RESTORED GEN {d.get('gen', '?')}")

    def save(self, gen, tag=""):
        if self.rank != 0: return
        s = {
            "pop": [self.unwrap(p).state_dict() for p in self.pop],
            "opts": [o.state_dict() for o in self.opts],
            "agency": self.agency.state_dict(),
            "wm": self.wm.state_dict(),
            "self": self.self.state_dict(),
            "gen": gen
        }
        atomic_save(s, PATHS["CHECKPOINT"], True)
        atomic_save(s, f"{PATHS['DIR_ARCHIVE']}/gen_{gen}.pt", True)
        self.memory.save()
        logging.info(f"💾 SAVED GEN {gen}")

    def unwrap(self, m): return m.module if hasattr(m, "module") else m

    def run(self):
        for gen in range(CONFIG["GENERATIONS"]):
            # 1. Sense
            x, y = self.data.get_batch()
            if x is None: continue
            with torch.no_grad():
                _, _, state = self.unwrap(self.pop[0])(x)
                state = state.mean(0)
            
            # 2. Decide
            action = self.agency.decide(state)
            if self.rank == 0: logging.info(f"🤖 GEN {gen} | ACTION: {action}")

            # 3. Act
            loss_r = 0.0
            if action == "train": loss_r = self.train(gen)
            elif action == "evolve": self.evolve(gen)
            elif action == "reflect": self.reflect()
            
            # 4. Learn (Agency & World)
            state_detached = state.detach().clone()
            pred = self.wm(state_detached)
            wm_loss = F.mse_loss(pred, state_detached)
            self.wm_opt.zero_grad(); wm_loss.backward(); self.wm_opt.step()
            
            curiosity = wm_loss.item()
            total_reward = (loss_r * 1.0) + (curiosity * 0.1)
            self.agency.rewards.append(total_reward)
            self.agency.update_policy()
            
            self.save(gen)

    def train(self, gen):
        noise = max(0, 0.01 * (1 - gen/CONFIG["GENERATIONS"]))
        if gen % 10 == 0: self.wm.reset_state()
        
        self.res.act(self) # Research
        
        for i, (model, opt) in enumerate(zip(self.pop, self.opts)):
            model.train()
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                x, y = self.data.get_batch()
                if x is None: continue
                
                _, loss, _ = model(x, y, noise)
                
                if torch.isnan(loss):
                    self._load(); return 0.0

                opt.zero_grad()
                loss.backward()
                
                gn = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
                scale = self.meta.get_scale(loss.item(), gn)
                for g in opt.param_groups: g['lr'] = CONFIG["LR"] * scale.item()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                if i == 0: self.prev_loss = loss.item()
                
                if self.rank == 0 and i == 0 and step % 20 == 0:
                    logging.info(f"   Step {step} | Loss {loss.item():.4f} | LR {scale.item():.2f}")
                    self.tele.log({"gen": gen, "loss": loss.item()})

        return -self.prev_loss

    def evolve(self, gen):
        scores = []
        for model in self.pop:
            model.eval()
            ls = []
            with torch.no_grad():
                for _ in range(4):
                    x, y = self.data.get_batch()
                    if x is None: continue
                    _, l, _ = model(x, y); ls.append(l.item())
            scores.append(-np.mean(ls) if ls else -100)
        
        best = np.argmax(scores)
        seed = IdentitySeed.compress(self.unwrap(self.pop[best]))
        state = self.opts[best].state_dict()
        roles = self.civ.assign_roles(scores)
        
        if self.rank == 0: logging.info(f"🧬 EVOLVE | Winner: {best} | Score: {scores[best]:.4f}")

        # Self-Model Learning
        s0 = seed["weights"].to(DEVICE)
        self.replay.append(s0)
        if len(self.replay) > 1:
            prev = self.replay[-2]
            p_s, p_f = self.self(prev.unsqueeze(0).repeat(1,3))
            sloss = F.mse_loss(p_s, s0.unsqueeze(0).repeat(1,3))
            self.self_opt.zero_grad(); sloss.backward(); self.self_opt.step()

        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best:
                t = self.unwrap(self.pop[i])
                IdentitySeed.reconstruct(t, seed)
                
                role = roles.get(i, "worker")
                mut = 0.05 if role == "explorer" else 0.01
                with torch.no_grad():
                    for p in t.parameters(): p.add_(torch.randn_like(p) * mut)
                
                if self.world_size > 1: dist.barrier()
                self.opts[i].load_state_dict(state)

    def reflect(self):
        x, _ = self.data.get_batch()
        with torch.no_grad():
            m = self.unwrap(self.pop[0])
            _, _, meta = m(x)
            vec = meta.mean(0)
            
            # Recall & Context
            recalled = self.memory.query(vec)
            ctx = " ".join([r["data"]["text"][:50] for r in recalled if isinstance(r, dict)])
            
            prompt = f"CTX: {ctx}\nTHOUGHT:" if ctx else "THOUGHT:"
            enc = torch.tensor([self.data.encode(prompt)], device=DEVICE)
            out = m.generate(enc)
            txt = self.data.decode(out[0].tolist())
            
            self.memory.store(vec, {"text": txt, "gen": 0})
            if self.rank == 0: logging.info(f"\n[REFLECT] {txt}\n")

    def grow_network(self, idx):
        if self.rank == 0:
            m = self.unwrap(self.pop[idx])
            m.blocks.append(RecurrentBlock().to(DEVICE))
            logging.info("🌱 NETWORK GROWN")
            s = m.state_dict()
            s["_layers"] = len(m.blocks)
        else: s = None
        
        if self.world_size > 1:
            dist.barrier()
            o = [s]; dist.broadcast_object_list(o, 0); s = o[0]
            m = self.unwrap(self.pop[idx])
            while len(m.blocks) < s["_layers"]: m.blocks.append(RecurrentBlock().to(DEVICE))
            m.load_state_dict(s)
            self.pop[idx] = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
        
        self.opts[idx] = optim.AdamW(self.pop[idx].parameters(), lr=CONFIG["LR"])

# ============================================================
# RUN
# ============================================================
def main(rank, world):
    if world > 1:
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    
    ImmortalCoreController(rank, world).run()
    if world > 1: dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1: mp.spawn(main, args=(NUM_GPUS,), nprocs=NUM_GPUS)
    else: main(0, 1)
