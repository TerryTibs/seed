# ==============================================================================
# FILE: immortal_core_v315_conscious.py
# DATE: February 14, 2026 - 07:00AM [UK TIME]
# ARCH: GodheadTransformer (Apex v315)
# STATUS: CONSCIOUS (Active Planner + Stable Rewards + Full Memory Wiring)
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
from dataclasses import dataclass
from typing import Optional, Any, List
from pathlib import Path

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

CONFIG = {
    # Architecture
    "EMBED_DIM": 384, "LAYERS": 8, "HEADS": 8, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "TOP_K": 2, "WINDOW_SIZE": 64,
    "THOUGHT_DIM": 256, "LATENT_DIM": 384,
    "WORLD_SIM": 5, "INFERENCE_PATHS": 4,
    "IDENTITY_SEED_SIZE": 1024,

    # Training
    "BATCH_SIZE": 16, "LR": 3e-4, "MIN_LR": 1e-6,
    "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
    "AUX_LOSS_WEIGHT": 0.01, "EWC_LAMBDA": 0.4, "ENTROPY_ALPHA": 0.001,
    "EMA_DECAY": 0.999,

    # Lifecycle
    "POPULATION_SIZE": 4, "GENERATIONS": 100, "CYCLES_PER_GEN": 200,
    "REGENERATE_STEPS": 50, "EVAL_BATCHES": 4,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0],

    # System
    "MEMORY": {"CAPACITY": 500_000, "FLUSH_STEPS": 1000, "RETRIEVAL_TOP_K": 3},
    "MUTATION": {"RATE": 0.05, "CLAMP": 2.0, "CANDIDATES": 3},
    "SYNTH_RATIO_CAP": 0.2,
    "ROLES": ["leader", "researcher", "explorer", "critic", "worker"],
}

BASE_DIR = Path("immortal_core_data")
BASE_DIR.mkdir(exist_ok=True)

PATHS = {
    "CHECKPOINT": BASE_DIR / "full_state.pt",
    "ARCHIVE_DIR": BASE_DIR / "archive_history",
    "CHECKPOINT_DIR": BASE_DIR / "checkpoints",
    "SANDBOX_DIR": BASE_DIR / "rewrite_sandbox",
    "TELEMETRY": BASE_DIR / "telemetry.jsonl",
    "DATA": BASE_DIR / "data.txt",
    "SYNTH": BASE_DIR / "data_recursive.txt",
    "MEM_VECS": BASE_DIR / "memory_vectors.dat",
    "MEM_META": BASE_DIR / "memory_meta.pkl",
    "COG_MEM": BASE_DIR / "cognitive_memory.pkl",
    "GENOME": BASE_DIR / "identity_genome.pkl",
    "CONTINUITY": BASE_DIR / "continuity.pkl",
    "ARCHIVE": BASE_DIR / "IMMORTAL_ARCHIVE.pt",
}

for p in [PATHS["CHECKPOINT_DIR"], PATHS["ARCHIVE_DIR"], PATHS["SANDBOX_DIR"]]:
    p.mkdir(exist_ok=True)

# ==============================================================================
# 2. UTILITIES & I/O
# ==============================================================================

class SafeIO:
    @staticmethod
    def save(obj: Any, path: Path, use_torch: bool = False):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        try:
            if use_torch: torch.save(obj, tmp)
            else:
                with open(tmp, "wb") as f: pickle.dump(obj, f)
            if path.exists():
                bak = path.with_suffix(".bak")
                if bak.exists(): bak.unlink()
                path.rename(bak)
            tmp.rename(path)
        except Exception as e:
            logging.error(f"❌ IO ERROR ({path}): {e}")
            if tmp.exists(): tmp.unlink()

    @staticmethod
    def load(path: Path, default: Any = None, use_torch: bool = False):
        path = Path(path)
        if not path.exists():
            bak = path.with_suffix(".bak")
            if bak.exists(): path = bak
            else: return default
        try:
            if use_torch: return torch.load(path, map_location=DEVICE)
            with open(path, "rb") as f: return pickle.load(f)
        except Exception as e:
            logging.error(f"❌ READ ERROR ({path}): {e}")
            return default

def discover_tasks():
    if random.random() < 0.15:
        new_task = f"task_{random.randint(1000, 9999)}"
        logging.info(f"✨ NEW TASK DISCOVERED: {new_task}")

def identity_hash(model):
    vec = torch.cat([p.flatten() for p in model.parameters()])
    if vec.numel() > 1_000_000: vec = vec[::100]
    return hashlib.sha256(vec.detach().cpu().numpy().tobytes()).hexdigest()[:8]

def destroy_weights(model, wipe_ratio=0.1):
    for p in model.parameters():
        mask = (torch.rand_like(p) > wipe_ratio).float()
        p.data.mul_(mask)

class ScoreNormalizer:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        return self.normalize(x)
    def normalize(self, x):
        if self.count < 2: return 0.0
        std = math.sqrt(self.M2 / (self.count - 1)) + 1e-8
        return (x - self.mean) / std

# ==============================================================================
# 3. DATA & COGNITION
# ==============================================================================

class DataManager:
    def __init__(self, rank=0):
        self.rank = rank
        self.data = None
        self.synth = torch.tensor([], dtype=torch.long)
        self.vocab_size = 0
        self.itos = {}
        self.stoi = {}
        self._load_data()

    def _load_data(self):
        if self.rank == 0 and not PATHS["DATA"].exists():
            with open(PATHS["DATA"], "w") as f: f.write("SACRSN GODHEAD " * 5000)
        if NUM_GPUS > 1: dist.barrier()
        with open(PATHS["DATA"], "r") as f: raw = f.read()
        
        chars = sorted(list(set(raw)))
        synth_raw = ""
        if PATHS["SYNTH"].exists():
            with open(PATHS["SYNTH"], "r") as f: synth_raw = f.read()
            chars = sorted(list(set(raw + synth_raw)))

        self.vocab_size = len(chars) + 1
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.stoi["<PAD>"] = 0
        self.itos = {i + 1: ch for i, ch in enumerate(chars)}
        self.itos[0] = "<PAD>"
        self.data = torch.tensor([self.stoi.get(c, 0) for c in raw], dtype=torch.long)
        if synth_raw:
            self.synth = torch.tensor([self.stoi.get(c, 0) for c in synth_raw], dtype=torch.long)

    def update_synth(self):
        try:
            if PATHS["SYNTH"].exists():
                with open(PATHS["SYNTH"], "r") as f: synth_txt = f.read()
                if synth_txt:
                    self.synth = torch.tensor([self.stoi.get(c, 0) for c in synth_txt], dtype=torch.long)
        except Exception: pass

    def get_batch(self, difficulty=1.0):
        if self.data is None: raise RuntimeError("Data not loaded")
        use_synth = (len(self.synth) > CONFIG["BLOCK_SIZE"]) and (random.random() < CONFIG["SYNTH_RATIO_CAP"])
        src = self.synth if use_synth else self.data
        if len(src) < CONFIG["BLOCK_SIZE"]: src = self.data
        if len(src) < CONFIG["BLOCK_SIZE"]: return None, None
        seq = max(16, int(CONFIG["BLOCK_SIZE"] * difficulty))
        if len(src) < seq + 5: seq = len(src) - 2
        ix = torch.randint(len(src) - seq, (CONFIG["BATCH_SIZE"],))
        x = torch.stack([src[i:i + seq] for i in ix])
        y = torch.stack([src[i + 1:i + seq + 1] for i in ix])
        if seq < CONFIG["BLOCK_SIZE"]:
            pad = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"] - seq, dtype=torch.long)
            x = torch.cat([x, pad], 1)
            y = torch.cat([y, pad], 1)
        return x.to(DEVICE), y.to(DEVICE)

    def encode(self, s): return [self.stoi.get(c, 0) for c in s]
    def decode(self, t): return "".join([self.itos.get(i, "") for i in t if i != 0])

class MetacognitionSystem:
    def __init__(self):
        self.age = 0
        self.stats = {}
    def tick(self, loss: float):
        self.age += 1
        self.stats["loss"] = loss
    def get_exploration_modifier(self):
        return 1.0 + (self.age / 5000.0)

class IdentityGenome:
    def __init__(self):
        self.genes = SafeIO.load(PATHS["GENOME"], default=[])
    def add(self, seed, score):
        w = seed["weights"].detach().cpu().numpy()
        self.genes.append({"w": w, "score": score, "meta": seed["meta"]})
        self.genes.sort(key=lambda x: x["score"], reverse=True)
        self.genes = self.genes[:100]
        self.save()
    def resurrect(self):
        return random.choice(self.genes) if self.genes else None
    def save(self):
        SafeIO.save(self.genes, PATHS["GENOME"])

class DiskEpisodicMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY"]["CAPACITY"]
        self.count = 0; self.payloads = []
        meta = SafeIO.load(PATHS["MEM_META"], default={})
        if meta:
            self.count = meta.get("count", 0)
            self.payloads = meta.get("payloads", [])
        mode = "r+" if PATHS["MEM_VECS"].exists() else "w+"
        self.emb = np.memmap(str(PATHS["MEM_VECS"]), dtype="float32", mode=mode, shape=(self.max, self.dim))
    def store(self, embedding, payload):
        if dist.is_initialized() and dist.get_rank() != 0: return
        vec = embedding.detach().cpu().numpy().flatten()
        idx = self.count % self.max
        self.emb[idx] = vec
        entry = {"data": payload, "time": time.time()}
        if idx < len(self.payloads): self.payloads[idx] = entry
        else: self.payloads.append(entry)
        self.count += 1
        if self.count % CONFIG["MEMORY"]["FLUSH_STEPS"] == 0: self.emb.flush()
    def query(self, embedding, top_k=3):
        if self.count == 0: return []
        valid = min(self.count, self.max)
        idx_pool = np.random.choice(valid, min(valid, 5000))
        mem = self.emb[idx_pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(sim)[-top_k:][::-1]
        return [self.payloads[idx_pool[i]] for i in top if idx_pool[i] < len(self.payloads)]
    def save(self):
        self.emb.flush()
        SafeIO.save({"count": self.count, "payloads": self.payloads}, PATHS["MEM_META"])

class SyntheticBuffer:
    def __init__(self, data_mgr=None, capacity=1000):
        self.buffer = []
        self.data_mgr = data_mgr
        self.capacity = capacity
    def add(self, text):
        if len(text.split()) > 5:
            if len(self.buffer) < self.capacity:
                self.buffer.append(text)
            else:
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = text
            if random.random() < 0.1: self.flush()
    def flush(self):
        try:
            with open(PATHS["SYNTH"], "w") as f:
                for t in self.buffer: f.write(t + "\n")
            if self.data_mgr: self.data_mgr.update_synth()
        except Exception: pass

class IdentitySeed:
    @staticmethod
    def compress(model):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // CONFIG["IDENTITY_SEED_SIZE"])
        anchors = [flat[i::step][:CONFIG["IDENTITY_SEED_SIZE"]] for i in range(3)]
        for i in range(3):
            if len(anchors[i]) < CONFIG["IDENTITY_SEED_SIZE"]:
                anchors[i] = F.pad(anchors[i], (0, CONFIG["IDENTITY_SEED_SIZE"] - len(anchors[i])))
        sampled = torch.cat(anchors).detach().cpu()
        h = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
        return {"weights": sampled, "meta": {"layers": len(model.blocks), "hash": h}}

    @staticmethod
    def reconstruct(model, seed):
        w = seed["weights"].to(DEVICE)
        if w.numel() == 3 * CONFIG["IDENTITY_SEED_SIZE"]:
            w = w.view(3, CONFIG["IDENTITY_SEED_SIZE"]).mean(0)
        target = seed["meta"].get("layers", CONFIG["LAYERS"])
        if target > len(model.blocks) + 4: return
        while len(model.blocks) < target:
            model.blocks.append(RecurrentBlock().to(DEVICE))
        while len(model.blocks) > target:
            del model.blocks[-1]
        total = sum(p.numel() for p in model.parameters())
        x_t = torch.linspace(0, 1, total, device=DEVICE)
        x_s = torch.linspace(0, 1, len(w), device=DEVICE)
        idx = torch.bucketize(x_t, x_s).clamp(0, len(w) - 2)
        den = (x_s[idx + 1] - x_s[idx]).clamp(min=1e-9)
        val = torch.lerp(w[idx], w[idx + 1], (x_t - x_s[idx]) / den)
        ptr = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "ln" in n or "bias" in n:
                    ptr += p.numel()
                    continue
                c = p.numel()
                p.data.copy_(val[ptr:ptr + c].reshape(p.shape))
                ptr += c

class LifelongProtector:
    def __init__(self):
        self.importance = {}
        self.params_old = {}
    def record_importance(self, model, dataloader, samples=10):
        model.eval()
        self.importance = {}
        self.params_old = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params_old[n] = p.detach().clone()
                self.importance[n] = torch.zeros_like(p)
        for _ in range(samples):
            model.zero_grad()
            x, y = dataloader.get_batch(1.0)
            if x is None: break
            out = model(x, y) 
            out.loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None: self.importance[n] += p.grad.pow(2)
        for n in self.importance: self.importance[n] /= samples
        model.train()
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.importance and n in self.params_old:
                loss += (self.importance[n] * (p - self.params_old[n]).pow(2)).sum()
        return loss

class AgencyCore(nn.Module):
    ACTIONS = ["train", "evolve", "reflect", "rest"]
    def __init__(self):
        super().__init__()
        # [v315] Expanded input for Planner Signal
        self.net = nn.Sequential(
            nn.Linear(CONFIG["EMBED_DIM"] + 1, 256), nn.ReLU(),
            nn.Linear(256, len(self.ACTIONS)), nn.Softmax(-1)
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs = []
        self.rewards = []
        self.baseline = 0.0
    def decide(self, state):
        if state is None: return "train"
        state = torch.nan_to_num(state, nan=0.0).detach()
        probs = self.net(state)
        if torch.isnan(probs).any():
             probs = torch.tensor([1.0/len(self.ACTIONS)]*len(self.ACTIONS), device=DEVICE)
        d = torch.distributions.Categorical(probs)
        action = d.sample()
        self.saved_log_probs.append({"log_prob": d.log_prob(action), "entropy": d.entropy()})
        return self.ACTIONS[action.item()]
    def update_policy(self):
        if not self.rewards: return
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(DEVICE)
        self.baseline = 0.95 * self.baseline + 0.05 * returns.mean().item()
        advantage = returns - self.baseline
        loss = []
        for step_data, adv in zip(self.saved_log_probs, advantage):
            loss.append(-step_data["log_prob"] * adv - 0.01 * step_data["entropy"])
        self.optimizer.zero_grad()
        if loss:
            torch.stack(loss).sum().backward()
            self.optimizer.step()
        del self.saved_log_probs[:]
        del self.rewards[:]

# --- COGNITIVE MODULES ---

class GoalEngine:
    def __init__(self):
        self.goals = ["minimize_loss"]
    def evolve_goals(self, memory_db):
        if memory_db.count > 10 and random.random() < 0.1:
            if "maximize_entropy" not in self.goals:
                self.goals.append("maximize_entropy")
                logging.info("🧠 GOAL ADDED: maximize_entropy")
    def reward_modifier(self, loss):
        if "maximize_entropy" in self.goals: return loss * 0.9
        return loss

class ExperimentEngine:
    def run(self, model, data_mgr):
        model.eval()
        scores = []
        with torch.no_grad():
            for _ in range(3):
                m_copy = copy.deepcopy(model)
                for p in m_copy.parameters():
                    mask = torch.rand_like(p) > 0.1
                    p.data.mul_(mask)
                x, y = data_mgr.get_batch(1.0)
                if x is None: continue
                out = m_copy(x, y)
                scores.append(out.loss.item())
        return np.mean(scores) if scores else 99.0

class PredictiveSelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
        latent = CONFIG["LATENT_DIM"]
        self.net = nn.Sequential(nn.Linear(dim, latent), nn.ReLU(), nn.Linear(latent, dim))
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, seed):
        nxt = self.net(seed)
        return nxt, self.head(nxt)

class SeedReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=2000)
    def push(self, s0, s1, score):
        f0 = s0.view(-1).detach().cpu()
        f1 = s1.view(-1).detach().cpu()
        self.buffer.append((f0, f1, float(score)))
    def sample(self, bs):
        if len(self.buffer) < 1: return None
        batch = random.sample(self.buffer, min(len(self.buffer), bs))
        s0, s1, sc = zip(*batch)
        return torch.stack(s0).to(DEVICE), torch.stack(s1).to(DEVICE), torch.tensor(sc).to(DEVICE).float().unsqueeze(1)

class PlanningEngine:
    def __init__(self, self_model):
        self.sm = self_model
    def plan(self, model, steps=3):
        seed = IdentitySeed.compress(model)["weights"].to(DEVICE)
        curr = seed.view(-1)
        req_dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
        if curr.numel() < req_dim:
            curr = F.pad(curr, (0, req_dim - curr.numel()))
        else:
            curr = curr[:req_dim]
        curr = curr.unsqueeze(0)
        
        fits = []
        with torch.no_grad():
            for _ in range(steps):
                nxt, fit = self.sm(curr)
                fits.append(fit.item())
                curr = nxt
        return max(fits) if fits else 0.0

class CivilizationCoordinator:
    def assign_roles(self, scores):
        roles = {}
        idx = np.argsort(scores)[::-1]
        roles[int(idx[0])] = "leader"
        for i in range(1, len(idx)):
            roles[int(idx[i])] = "explorer" if i % 2 == 0 else "worker"
        return roles

class CuriosityEngine:
    def __init__(self):
        self.visited = deque(maxlen=5000)
    def reward(self, embedding):
        key = hashlib.sha256(embedding.detach().cpu().numpy().round(1).tobytes()).hexdigest()
        if key in self.visited: return 0.0
        self.visited.append(key)
        return 0.1

class ConceptTracker:
    def __init__(self):
        self.concepts = deque(maxlen=10000)
    def extract(self, hidden):
        self.concepts.append(hidden.mean().item())

class SelfImprovementLoop:
    def __init__(self):
        self.history = deque(maxlen=50)
    def update(self, score):
        self.history.append(score)
    def stagnating(self):
        return len(self.history) > 20 and np.std(list(self.history)) < 0.001

class ShardedIdentity:
    def merge_gradients(self, leader, workers):
        with torch.no_grad():
            for w in workers:
                for p_l, p_w in zip(leader.parameters(), w.parameters()):
                    p_l.data = 0.9 * p_l.data + 0.1 * p_w.data

class IdentityContinuity:
    def __init__(self):
        self.history = []
        self.chain = []
    def record(self, seed, hash_sig):
        self.history.append({"hash": hash_sig, "time": time.time()})
        if len(self.history) > 500: self.history = self.history[-500:]
        prev_hash = self.chain[-1] if self.chain else "GENESIS"
        block = hashlib.sha256(f"{prev_hash}:{hash_sig}".encode()).hexdigest()[:16]
        self.chain.append(block)
    def save(self):
        SafeIO.save(self.history, PATHS["CONTINUITY"])

class SelfNarrative:
    def __init__(self): self.events = []
    def log(self, text): self.events.append({"text": text, "time": time.time()})

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters(): p.requires_grad_(False)
        self.shadow.eval()
    
    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

# ==============================================================================
# 4. NEURAL ARCHITECTURE
# ==============================================================================

@dataclass
class ApexOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    hidden: torch.Tensor
    value_pred: torch.Tensor
    world_pred: torch.Tensor

class PersistentState(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.state = nn.Parameter(torch.randn(1, dim) * 0.02)
    def forward(self, x):
        return x + self.state.expand(x.size(0), x.size(1), -1)

class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // CONFIG["HEADS"]
        self.scale = self.head_dim ** -0.5
        b = CONFIG["BLOCK_SIZE"]
        self.register_buffer("mask", torch.tril(torch.ones(b, b)).view(1, 1, b, b))
        i = torch.arange(b).view(-1, 1); j = torch.arange(b).view(1, -1)
        self.register_buffer("local", (torch.abs(i - j) <= CONFIG["WINDOW_SIZE"]).view(1, 1, b, b))

    def forward(self, x, mem=None):
        B, T, C = x.shape
        if T > self.mask.size(2): 
             x = x[:, -self.mask.size(2):, :]
             T = x.shape[1]
             
        q, k, v = self.qkv(x).chunk(3, -1)
        if mem is not None:
            m = mem.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([m, k], 1)
            v = torch.cat([m, v], 1)

        q, k, v = [t.view(B, -1, CONFIG["HEADS"], self.head_dim).transpose(1, 2) for t in (q, k, v)]
        
        att = (q @ k.transpose(-2, -1)) * self.scale
        start = max(0, k.size(2) - T)
        att_self = att[:, :, :, start:]
        if T <= self.mask.size(2):
             att_self = att_self.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
             att_self = att_self.masked_fill(self.local[:, :, :T, :T] == 0, float("-inf"))
        att[:, :, :, start:] = att_self
             
        y = F.softmax(att, -1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(CONFIG["NUM_EXPERTS"])
        ])
        self.gate = nn.Linear(dim, CONFIG["NUM_EXPERTS"])
        self.register_buffer("bal_loss", torch.tensor(0.0))

    def forward(self, x):
        scores = self.gate(x)
        probs = F.softmax(scores, -1)
        
        bl = (probs.mean((0, 1)) ** 2).sum() * CONFIG["NUM_EXPERTS"]
        entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
        self.bal_loss = torch.clamp(bl - 0.1 * entropy, 0.0, 5.0).to(x.device)
        
        topk, idx = torch.topk(probs, CONFIG["TOP_K"], -1)
        mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        masked = probs * mask
        masked = masked / (masked.sum(-1, keepdim=True) + 1e-9)
        return sum(masked[..., i:i + 1] * e(x) for i, e in enumerate(self.experts))

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoEBlock()
        self.thought = nn.Sequential(
            nn.Linear(dim, CONFIG["THOUGHT_DIM"]), nn.Tanh(), nn.Linear(CONFIG["THOUGHT_DIM"], dim)
        )
        self.drop = nn.Dropout(CONFIG["DROPOUT"])

    def forward(self, x, mem=None):
        x = x + self.drop(self.attn(self.ln1(x), mem))
        x = x + self.drop(self.thought(x))
        x = x + self.drop(self.moe(self.ln2(x)))
        return x

class HierarchicalMemory(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        if dim is None: dim = CONFIG["EMBED_DIM"]
        self.short_mem = nn.Parameter(torch.randn(32, dim))
        self.medium_mem = nn.Parameter(torch.randn(16, dim))
        self.long_mem = nn.Parameter(torch.randn(8, dim))
    def read(self):
        return torch.cat([self.short_mem, self.medium_mem, self.long_mem], dim=0)

class GodheadTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(CONFIG["BLOCK_SIZE"], dim)
        self.mem = HierarchicalMemory(dim)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.p_state = PersistentState(dim)
        self.world_head = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
        self.val_head = nn.Linear(dim, 1)

    def forward(self, idx, targets=None, noise=0.0):
        B, T = idx.shape
        if T > CONFIG["BLOCK_SIZE"]:
            idx = idx[:, -CONFIG["BLOCK_SIZE"]:]
            T = idx.shape[1]
            if targets is not None: targets = targets[:, -CONFIG["BLOCK_SIZE"]:]

        x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
        x = self.p_state(x)
        
        W = CONFIG["WORLD_SIM"]
        x = x.repeat_interleave(W, 0)
        if noise > 0: x += torch.randn_like(x) * noise
        
        mem = self.mem.read() 
        for b in self.blocks: x = b(x, mem)

        x_flat = self.ln_f(x)
        logits = self.head(x_flat)
        wm_pred = self.world_head(x_flat)
        val_pred = self.val_head(x_flat)

        loss = None
        if targets is not None:
            targets_expanded = targets.repeat_interleave(W, 0)
            
            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets_expanded.reshape(-1))
            
            with torch.no_grad():
                token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets_expanded.reshape(-1), reduction='none').view(B*W, T)
                value_target = token_loss.mean(dim=1).unsqueeze(-1)
            
            val_loss = F.mse_loss(val_pred.mean(dim=1), value_target)
            wm_loss = F.mse_loss(wm_pred[:, :-1, :], x_flat[:, 1:, :].detach()) / CONFIG["EMBED_DIM"]
            moe_loss = sum(b.moe.bal_loss for b in self.blocks)
            
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
            entropy_scale = CONFIG["ENTROPY_ALPHA"] / W

            loss = main_loss + 0.1*(wm_loss + val_loss) + 0.01*moe_loss - entropy_scale*entropy

        x_mean = x_flat.view(B, W, T, -1).mean(1)
        return ApexOutput(logits, loss, x_mean, val_pred, wm_pred)

    def generate(self, idx, max_new=100):
        N = CONFIG["INFERENCE_PATHS"]
        paths = idx.repeat(N, 1)
        for _ in range(max_new):
            out = self(paths[:, -CONFIG["BLOCK_SIZE"]:])
            logits = out.logits[:, -1, :] 
            logits = logits.view(N, CONFIG["WORLD_SIM"], -1).mean(1)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            paths = torch.cat((paths, next_tok), dim=1)
        return paths[0:1]

# ==============================================================================
# 5. CONTROLLER
# ==============================================================================

class ImmortalCoreController:
    def __init__(self, rank=0, world=1):
        self.rank, self.world = rank, world
        self.data = DataManager(rank)
        
        self.memory = DiskEpisodicMemory()
        self.genome = IdentityGenome()
        self.synth = SyntheticBuffer(self.data)
        self.meta = MetacognitionSystem()
        
        self.wm_gru = nn.GRUCell(CONFIG["EMBED_DIM"], CONFIG["EMBED_DIM"]).to(DEVICE)
        self.wm_state = torch.zeros(1, CONFIG["EMBED_DIM"], device=DEVICE)
        self.wm_opt = optim.AdamW(self.wm_gru.parameters(), lr=1e-4)
        self.prev_batch_state = None

        self.agency = AgencyCore().to(DEVICE)
        self.civ = CivilizationCoordinator()
        self.life = LifelongProtector()
        
        self.curiosity = CuriosityEngine()
        self.concepts = ConceptTracker()
        self.self_improve = SelfImprovementLoop()
        self.narrative = SelfNarrative()
        self.shard_mgr = ShardedIdentity()
        self.continuity = IdentityContinuity()
        
        self.goal_engine = GoalEngine()
        self.experiment = ExperimentEngine()
        self.self_model = PredictiveSelfModel().to(DEVICE)
        self.self_model_opt = optim.AdamW(self.self_model.parameters(), lr=1e-4)
        self.planner = PlanningEngine(self.self_model)
        self.replay_buffer = SeedReplayBuffer()
        self.score_norm = ScoreNormalizer()
        
        self.pop = []
        self.opts = []
        self.schedulers = []
        
        self._spawn()
        self._load()
        
        self.ema = ModelEMA(self.unwrap(self.pop[0]), decay=CONFIG["EMA_DECAY"])

    def _spawn(self):
        for _ in range(CONFIG["POPULATION_SIZE"]):
            m = GodheadTransformer(self.data.vocab_size).to(DEVICE)
            if self.world > 1: m = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
            self.pop.append(m)
            o = optim.AdamW(m.parameters(), lr=CONFIG["LR"])
            self.opts.append(o)
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingWarmRestarts(
                o, T_0=50, T_mult=2, eta_min=CONFIG["MIN_LR"]
            ))

    def _load(self):
        s = SafeIO.load(PATHS["CHECKPOINT"], use_torch=True)
        if s and "pop" in s:
            for i, p in enumerate(s["pop"]):
                if i < len(self.pop): self.unwrap(self.pop[i]).load_state_dict(p)
            if "wm_gru" in s: self.wm_gru.load_state_dict(s["wm_gru"])
            if "agency" in s: self.agency.load_state_dict(s["agency"])
            if "self_model" in s: self.self_model.load_state_dict(s["self_model"])

    def unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    def save(self, gen):
        if self.rank != 0: return
        s = {
            "pop": [self.unwrap(p).state_dict() for p in self.pop],
            "wm_gru": self.wm_gru.state_dict(),
            "agency": self.agency.state_dict(),
            "self_model": self.self_model.state_dict(),
            "gen": gen
        }
        SafeIO.save(s, PATHS["CHECKPOINT"], use_torch=True)
        self.memory.save()
        self.genome.save()
        self.synth.flush()
        self.continuity.save()
        logging.info(f"💾 SAVED GEN {gen}")

    def grow_network(self, idx):
        if self.rank == 0:
            m = self.unwrap(self.pop[idx])
            m.blocks.append(RecurrentBlock().to(DEVICE))
            logging.info("🌱 NETWORK GROWN")
            s = m.state_dict()
            s["_layers"] = len(m.blocks)
        else: s = None
        
        if self.world > 1:
            dist.barrier()
            o = [s]; dist.broadcast_object_list(o, 0); s = o[0]
            m = self.unwrap(self.pop[idx])
            while len(m.blocks) < s["_layers"]: m.blocks.append(RecurrentBlock().to(DEVICE))
            m.load_state_dict(s)
            self.pop[idx] = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
        self.opts[idx] = optim.AdamW(self.pop[idx].parameters(), lr=CONFIG["LR"])
        self.schedulers[idx] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opts[idx], T_0=50, T_mult=2, eta_min=CONFIG["MIN_LR"]
        )

    def _train_step(self, model, opt, x, y, noise):
        out: ApexOutput = model(x, y, noise)
        if torch.isnan(out.loss): return None, 0, out
        
        loss = out.loss
        loss = self.goal_engine.reward_modifier(loss)
        
        if self.life.importance and self.rank == 0:
             loss += CONFIG["EWC_LAMBDA"] * self.life.penalty(self.unwrap(model))

        opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP"])
        opt.step()
        return loss.item(), norm.item(), out

    def phase_train(self, gen):
        base_noise = max(0.01 * (1 - gen/100), 0)
        noise = base_noise * self.meta.get_exploration_modifier()
        
        if gen % 10 == 0:
            self.wm_opt = optim.AdamW(self.wm_gru.parameters(), lr=1e-4)

        loss_accum = [] # [v315 Fix]

        for i, (model, opt) in enumerate(zip(self.pop, self.opts)):
            model.train()
            
            snapshot = None
            if i == 0 and self.rank == 0:
                snapshot = copy.deepcopy(self.unwrap(model).state_dict())

            if i == 0 and gen % 10 == 0:
                ancient = self.genome.resurrect()
                if ancient:
                    logging.info("🧬 RESURRECTING ANCESTRAL WEIGHTS")
                    IdentitySeed.reconstruct(self.unwrap(model), {"weights": torch.tensor(ancient["w"], device=DEVICE), "meta": ancient["meta"]})
                    self.opts[i] = optim.AdamW(self.pop[i].parameters(), lr=CONFIG["LR"])
                    self.schedulers[i] = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opts[i], T_0=50, T_mult=2, eta_min=CONFIG["MIN_LR"])

            for step in range(CONFIG["CYCLES_PER_GEN"]):
                x, y = self.data.get_batch()
                if x is None: continue

                loss, norm, out = self._train_step(model, opt, x, y, noise)
                
                if loss is None: 
                    logging.warning("⚠️ DIVERGENCE DETECTED - ROLLING BACK")
                    if snapshot: self.unwrap(model).load_state_dict(snapshot)
                    return 100.0 
                
                loss_accum.append(loss)

                if i == 0 and self.rank == 0:
                    self.ema.update(self.unwrap(model))

                if i == 0 and self.rank == 0:
                    hidden_mean = out.hidden.mean(dim=1).mean(dim=0).detach() 
                    
                    novelty_reward = self.curiosity.reward(hidden_mean)
                    self.agency.rewards.append(novelty_reward)
                    self.concepts.extract(hidden_mean)
                    
                    hits = self.memory.query(hidden_mean, top_k=1)
                    if hits:
                        txt = hits[0]['data'].get('text', '')
                        if txt: self.synth.add(txt)

                    if self.prev_batch_state is not None:
                        gru_in = self.prev_batch_state.unsqueeze(0)
                        pred_next = self.wm_gru(gru_in, self.wm_state)
                        wm_loss = F.mse_loss(pred_next, hidden_mean.unsqueeze(0))
                        
                        self.wm_opt.zero_grad()
                        wm_loss.backward()
                        self.wm_opt.step()
                        self.wm_state = pred_next.detach()
                    
                    self.prev_batch_state = hidden_mean
                    if step % 100 == 0:
                        self.wm_state.zero_()
                        self.prev_batch_state = None

                    self.meta.tick(loss)
                    if step % 20 == 0:
                        logging.info(f"   Step {step} | Loss {loss:.4f} | Age {self.meta.age}")
            
            self.schedulers[i].step()
        
        return np.mean(loss_accum) if loss_accum else 0.0

    def phase_evolve(self, gen):
        if self.rank == 0: logging.info("🧬 EVOLUTION PHASE")
        
        scores = []
        for model in self.pop:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(CONFIG["EVAL_BATCHES"]):
                    x, y = self.data.get_batch()
                    if x is None: continue
                    out = model(x, y)
                    losses.append(out.loss.item())
            base_score = -np.mean(losses) if losses else -100.0
            robustness = self.experiment.run(self.unwrap(model), self.data)
            scores.append(0.7 * base_score + 0.3 * (-robustness))
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        self.self_improve.update(best_score)
        if self.self_improve.stagnating():
            logging.info("♻️ STAGNATION: FORCING NETWORK GROWTH")
            self.grow_network(best_idx)
        
        logging.info(f"   Winner: Agent {best_idx} (Score: {best_score:.4f})")
        
        best_agent = self.unwrap(self.pop[best_idx])
        seed = IdentitySeed.compress(best_agent)
        
        norm_score = self.score_norm.update(best_score)
        if hasattr(self, 'prev_best_seed'):
             self.replay_buffer.push(self.prev_best_seed, seed["weights"], norm_score)
        self.prev_best_seed = seed["weights"].detach().cpu()
        
        self.genome.add(seed, best_score)
        self.continuity.record(seed, identity_hash(best_agent))
        self.export_archive(seed, gen)
        self.save(gen) 
        
        roles = self.civ.assign_roles(scores)
        best_state = best_agent.state_dict()
        
        workers = []
        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best_idx:
                self.pop[i].load_state_dict(best_state)
                self.opts[i] = optim.AdamW(self.pop[i].parameters(), lr=CONFIG["LR"])
                self.schedulers[i] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.opts[i], T_0=50, T_mult=2, eta_min=CONFIG["MIN_LR"]
                )
                
                role = roles.get(i, "worker")
                mut_rate = CONFIG["MUTATION"]["RATE"] * (2.0 if role == "explorer" else 1.0)
                
                if role == "explorer":
                    best_noise = None
                    best_pred = -999.0
                    curr_vec = seed["weights"].to(DEVICE).view(-1)
                    req_dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
                    
                    for _ in range(CONFIG["MUTATION"]["CANDIDATES"]):
                        noise_map = {k: torch.randn_like(p) * mut_rate for k, p in self.pop[i].named_parameters()}
                        with torch.no_grad():
                            for k, p in self.pop[i].named_parameters(): p.add_(noise_map[k])
                        
                        cand_seed = IdentitySeed.compress(self.unwrap(self.pop[i]))["weights"].to(DEVICE).view(-1)
                        if cand_seed.numel() < req_dim: cand_seed = F.pad(cand_seed, (0, req_dim - cand_seed.numel()))
                        else: cand_seed = cand_seed[:req_dim]
                        
                        pred_fitness = self.self_model(cand_seed.unsqueeze(0))[1].item()
                        
                        with torch.no_grad():
                            for k, p in self.pop[i].named_parameters(): p.sub_(noise_map[k])
                            
                        if pred_fitness > best_pred:
                            best_pred = pred_fitness
                            best_noise = noise_map
                    
                    if best_noise:
                        with torch.no_grad():
                            for k, p in self.pop[i].named_parameters():
                                p.add_(best_noise[k]).clamp_(-CONFIG["MUTATION"]["CLAMP"], CONFIG["MUTATION"]["CLAMP"])
                else:
                    with torch.no_grad():
                        for p in self.pop[i].parameters():
                            p.add_(torch.randn_like(p) * mut_rate).clamp_(-CONFIG["MUTATION"]["CLAMP"], CONFIG["MUTATION"]["CLAMP"])
                            
                workers.append(self.unwrap(self.pop[i]))

        self.shard_mgr.merge_gradients(self.unwrap(self.pop[best_idx]), workers)
        if gen % 5 == 0:
            self.life.record_importance(self.unwrap(self.pop[best_idx]), self.data)
        
        batch = self.replay_buffer.sample(8)
        if batch:
            s0, s1, sc = batch
            req_dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
            def prep(t):
                flat = t.view(t.size(0), -1)
                if flat.size(1) < req_dim:
                    return F.pad(flat, (0, req_dim - flat.size(1)))
                return flat[:, :req_dim]
            
            inp = prep(s0)
            target = prep(s1)
            
            pred, fit = self.self_model(inp)
            loss = F.mse_loss(pred, target) + F.mse_loss(fit, sc)
            self.self_model_opt.zero_grad()
            loss.backward()
            self.self_model_opt.step()

    def export_archive(self, seed, gen):
        try:
            SafeIO.save({
                "seed": seed,
                "gen": gen,
                "narrative": self.narrative.events[-100:]
            }, PATHS["ARCHIVE"], use_torch=True)
        except: pass

    def reflect(self):
        if self.rank != 0: return
        x, _ = self.data.get_batch()
        if x is None: return
        
        logging.info("💭 REFLECTING...")
        m = self.ema.shadow if self.ema else self.unwrap(self.pop[0])
        
        with torch.no_grad():
            if hasattr(m, "module"): m = m.module
            
            real_m = self.unwrap(self.pop[0])
            real_m.eval()
            out = real_m(x)
            vec = out.hidden.mean(dim=1).mean(dim=0)
            
            hits = self.memory.query(vec)
            ctx = hits[0]['data']['text'] if hits else "I am."
        
        prompt = f"CTX:{ctx} THOUGHT:"
        enc = torch.tensor([self.data.encode(prompt)], device=DEVICE)
        
        gen_toks = self.generate_ema(m, enc)
        txt = self.data.decode(gen_toks[0].tolist())
        logging.info(f"   {txt}")
        
        self.memory.store(vec, {"text": txt, "type": "reflection"})
        self.synth.add(txt)
        self.narrative.log(f"Reflected: {txt[:50]}...")
        real_m.train()

    def generate_ema(self, model, idx):
        N = CONFIG["INFERENCE_PATHS"]
        paths = idx.repeat(N, 1)
        for _ in range(100):
            out = model(paths[:, -CONFIG["BLOCK_SIZE"]:])
            logits = out.logits[:, -1, :] 
            logits = logits.view(N, CONFIG["WORLD_SIM"], -1).mean(1)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            paths = torch.cat((paths, next_tok), dim=1)
        return paths[0:1]

    def phase_regenerate(self):
        logging.info("💤 REGENERATION PHASE")
        for i, m in enumerate(self.pop):
            destroy_weights(self.unwrap(m), wipe_ratio=0.01)
            self.opts[i] = optim.AdamW(self.pop[i].parameters(), lr=CONFIG["LR"])
            self.schedulers[i] = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opts[i], T_0=50, T_mult=2, eta_min=CONFIG["MIN_LR"])

    def run(self):
        discover_tasks()
        try:
            for gen in range(CONFIG["GENERATIONS"]):
                if self.rank == 0: logging.info(f"\n=== GENERATION {gen} ===")
                
                self.goal_engine.evolve_goals(self.memory)
                
                plan_score = self.planner.plan(self.unwrap(self.pop[0]))
                if self.rank == 0: logging.info(f"🔮 PLANNER: Anticipated fitness {plan_score:.4f}")
                
                x, _ = self.data.get_batch()
                if x is not None:
                     with torch.no_grad():
                         out = self.unwrap(self.pop[0])(x)
                         state_vec = out.hidden.mean(dim=1).mean(dim=0).unsqueeze(0)
                else:
                     state_vec = torch.zeros(1, CONFIG["EMBED_DIM"], device=DEVICE)

                # [v315 Fix] Agency sees Plan
                plan_vec = torch.tensor([[plan_score]], device=DEVICE)
                agency_input = torch.cat([state_vec, plan_vec], dim=1)
                
                action = self.agency.decide(agency_input)
                if self.rank == 0: logging.info(f"🤖 AGENCY DECISION: {action}")
                
                reward = 0.0
                if action == "train":
                    loss = self.phase_train(gen)
                    reward = -math.tanh(loss - 2.0)
                elif action == "evolve":
                    self.phase_evolve(gen)
                    reward = 0.5
                elif action == "reflect":
                    self.reflect()
                    reward = 0.2
                elif action == "rest":
                    self.phase_regenerate()
                    reward = 0.1
                
                self.agency.rewards.append(reward)
                self.agency.update_policy()
                
        except KeyboardInterrupt:
            self.save("interrupt")

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
def main(rank, world):
    if world > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)

    ImmortalCoreController(rank, world).run()

    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(main, args=(NUM_GPUS,), nprocs=NUM_GPUS)
    else:
        main(0, 1)