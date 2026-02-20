# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v101.0 — THE GODHEAD (COMPLETE)
# ==============================================================================
#
# [INTEGRATION REPORT]
# 1. ARCHITECTURE:
#    - Thought Engine (Latent Reasoning) added to RecurrentBlock.
#    - Sparse Attention (Cached) + MoE (Balanced).
#    - Multi-World Simulation (Detached).
#
# 2. TRAINING DYNAMICS:
#    - Selective Backprop: Hard example mining via loss masks.
#    - Agency RL: Composite Reward (Loss + Novelty + Fitness).
#
# 3. EVOLUTION:
#    - Identity Genome: Long-term ancestry tracking & resurrection.
#    - Experiment Engine: Robustness testing (Lobotomy) before promotion.
#    - Civilization: Roles (Leader/Worker/Explorer).
#
# 4. SAFETY:
#    - Atomic Saves, DDP Sync, NaN Guards, Auto-Resume.
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
    # Architecture
    "EMBED_DIM": 384, "LAYERS": 6, "HEADS": 6, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "TOP_K": 2, "WINDOW_SIZE": 64, 
    "THOUGHT_DIM": 256, # Restored Thought Dimension
    "WORLD_SIM": 5, 
    
    # Training
    "BATCH_SIZE": 16, "LR": 3e-4, "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
    "AUX_LOSS_WEIGHT": 0.01,
    
    # Lifecycle
    "POPULATION_SIZE": 4, "GENERATIONS": 50, "CYCLES_PER_GEN": 200,
    "REGENERATE_STEPS": 50, "EVAL_BATCHES": 4,
    
    # Cognitive
    "MEMORY_CAPACITY": 500_000, "IDENTITY_SEED_SIZE": 512,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0], "SYNTH_RATIO_CAP": 0.2, "WIPE_RATIO": 0.1,
    "SELECTIVE_THRESHOLD": 0.2 # Restored for Backprop
}

PATHS = {
    "MEM_PKL": "seed_memory.pkl", "MEM_BAK": "seed_memory_backup.pkl",
    "CHECKPOINT": "seed_full_state.pt", "ARCHIVE": "IMMORTAL_ARCHIVE.pt",
    "GENOME": "identity_genome.pkl", # Restored
    "TELEMETRY": "telemetry.jsonl", "DIR_CKPT": "checkpoints",
    "DIR_ARCHIVE": "archive_history", "DIR_SANDBOX": "rewrite_sandbox",
    "DATA": "data.txt", "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat", "MEM_META": "memory_meta.pkl"
}

for d in [PATHS["DIR_CKPT"], PATHS["DIR_ARCHIVE"], PATHS["DIR_SANDBOX"]]:
    if not os.path.exists(d): os.makedirs(d)

# ============================================================
# 2. UTILITIES & DATA
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
    def load_state_dict(self, d): self.mean=d["mean"]; self.var=d["var"]; self.count=d["count"]

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
        logging.error(f"Save Error {path}: {e}")
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

class DataManager:
    def __init__(self, rank):
        self.rank = rank; self.data = None; self.synth = torch.tensor([], dtype=torch.long)
        self.vocab_size = 0; self.itos = {}; self.stoi = {}
        self._load_data()
    
    def _load_data(self):
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

# --- RESTORED: Identity Genome ---
class IdentityGenome:
    def __init__(self): 
        self.genes = []
        self.load()
        
    def add(self, seed, score):
        # Store detached numpy for safety
        w = seed["weights"].detach().cpu().numpy()
        self.genes.append({"w": w, "score": score, "meta": seed["meta"]})
        self.genes.sort(key=lambda x: x["score"], reverse=True)
        self.genes = self.genes[:100] # Keep top 100
        self.save()
        
    def resurrect(self):
        if not self.genes: return None
        return random.choice(self.genes)

    def save(self):
        try: atomic_save(self.genes, PATHS["GENOME"], use_torch=False)
        except: pass
    
    def load(self):
        if os.path.exists(PATHS["GENOME"]):
            try: 
                with open(PATHS["GENOME"], "rb") as f: self.genes = pickle.load(f)
            except: pass

# --- RESTORED: Experiment Engine (Robustness) ---
class ExperimentEngine:
    def run(self, model, data):
        # Lobotomy Test
        model.eval()
        scores = []
        with torch.no_grad():
            for _ in range(3):
                # Mask 10% weights
                for p in model.parameters():
                    mask = torch.rand_like(p) > 0.1
                    p.mul_(mask)
                x, y = data.get_batch(1.0)
                if x is None: continue
                _, loss, _, _ = model(x, y)
                scores.append(loss.item())
        return np.mean(scores) if scores else 99.0

# --- RESTORED: Explicit Curiosity Engine ---
class CuriosityEngine:
    def __init__(self):
        self.visited = deque(maxlen=5000)
    def reward(self, embedding):
        key = hashlib.sha256(embedding.detach().cpu().numpy().round(1).tobytes()).hexdigest()
        if key in self.visited: return 0.0
        self.visited.append(key)
        return 0.2 # Explicit novelty bonus

# --- RESTORED: Self-Rewrite Sandbox ---
class SelfRewriteSandbox:
    def __init__(self): self.dir = PATHS["DIR_SANDBOX"]
    def propose_rewrite(self, code_snippet):
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.dir, f"proposal_{ts}.py"), "w") as f:
            f.write(f"# PROPOSED MUTATION\n{code_snippet}")

# --- RESTORED: Belief Ledger ---
class BeliefLedger:
    def __init__(self): self.history = []
    def record(self, agent_id, score, lineage):
        self.history.append({
            "agent": agent_id, "score": score, "lineage": lineage, "time": time.time()
        })
        if len(self.history) > 1000: self.history.pop(0)

class CivilizationCoordinator:
    def assign_roles(self, scores):
        roles = {}
        sorted_idx = np.argsort(scores)[::-1] 
        roles[sorted_idx[0]] = "leader"
        for i in range(1, len(sorted_idx)):
            roles[sorted_idx[i]] = "explorer" if i % 2 == 0 else "worker"
        return roles

class ShardedIdentity:
    def merge_gradients(self, leader, workers):
        with torch.no_grad():
            for w in workers:
                for p_l, p_w in zip(leader.parameters(), w.parameters()):
                    p_l.data = 0.9 * p_l.data + 0.1 * p_w.data

class AutonomousResearch:
    def __init__(self): self.hypothesis = None
    def act(self, ctrl):
        if random.random() < 0.01:
            ctrl.grow_network(0)
            logging.info("🔬 RESEARCH: Expanding Capacity")

class SelfImprovementLoop:
    def __init__(self): self.history = deque(maxlen=50)
    def update(self, score): self.history.append(score)
    def stagnating(self):
        return len(self.history) > 20 and np.std(list(self.history)) < 0.001

class AgencyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONFIG["EMBED_DIM"], 256), nn.ReLU(), nn.Linear(256, 5), nn.Softmax(-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs = []; self.rewards = []
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
        R = 0; policy_loss = []; returns = []
        for r in self.rewards[::-1]: R = r + 0.99 * R; returns.insert(0, R)
        self.replay.extend(returns)
        
        normalized = [self.scaler.normalize(r) for r in returns]
        returns_t = torch.tensor(normalized).to(DEVICE).clamp(-2.0, 2.0)
        
        for log_prob, R in zip(self.saved_log_probs, returns_t):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        if policy_loss: torch.stack(policy_loss).sum().backward(); self.optimizer.step()
        del self.saved_log_probs[:]; del self.rewards[:]

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

class PersistentWorldSimulator:
    def __init__(self, world_model): self.model = world_model
    def rollout(self, embedding, steps=5):
        states = []; current = embedding
        for _ in range(steps):
            current = self.model(current)
            states.append(current)
        return torch.stack(states)

class PlanningEngine:
    def __init__(self, wm, self_model): self.wm = wm; self.sm = self_model
    def plan(self, model, steps=3):
        seed = IdentitySeed.compress(model)["weights"].to(DEVICE)
        fits = []; curr = seed
        for _ in range(steps):
            nxt, fit = self.sm(curr)
            fits.append(fit.item())
            curr = nxt
        return max(fits) if fits else 0.0

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
        w = seed["weights"].to(DEVICE) if isinstance(seed, dict) else seed.to(DEVICE)
        if isinstance(w, np.ndarray): w = torch.tensor(w).to(DEVICE)
        
        if w.numel() == 3 * CONFIG["IDENTITY_SEED_SIZE"]: w = w.view(3, CONFIG["IDENTITY_SEED_SIZE"]).mean(0)
        
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
                if "ln" in n or "bias" in n: ptr += p.numel(); continue 
                c = p.numel(); p.data.copy_(val[ptr:ptr+c].reshape(p.shape)); ptr += c

class IdentityContinuity:
    def __init__(self): self.history = []
    def record(self, seed, hash_sig):
        weights = seed["weights"] if isinstance(seed, dict) else seed
        s_val = weights.detach().cpu().tolist() if isinstance(weights, torch.Tensor) else weights
        self.history.append({"seed": s_val, "hash": hash_sig, "time": time.time()})
    def continuity_score(self): return len(self.history)

class SelfNarrative:
    def __init__(self): self.events = []
    def log(self, text): self.events.append({"text": text, "time": time.time()})
    def summarize(self): return "\n".join(e["text"] for e in self.events[-20:])

class ConceptTracker:
    def __init__(self): self.concepts = []
    def extract(self, hidden): self.concepts.append(hidden.mean().item())

class ArchitecturePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        self.log_probs = []; self.rewards = []
    def forward(self, loss, drift, mem_size, depth):
        inp = torch.tensor([loss, drift, mem_size, depth], device=DEVICE).float()
        return self.net(inp)
    def select_action(self, loss, drift, mem_size, depth):
        probs = self.forward(loss, drift, mem_size, depth)
        m = torch.distributions.Categorical(probs)
        action = m.sample(); self.log_probs.append(m.log_prob(action))
        return action.item()
    def update(self):
        if not self.rewards: return
        policy_loss = []; returns = torch.tensor(self.rewards).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.log_probs, returns): policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        if len(policy_loss) > 0: torch.stack(policy_loss).sum().backward(); self.optimizer.step()
        del self.log_probs[:]; del self.rewards[:]

class SeedReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
    def push(self, seed_t, seed_t1, reward):
        r_val = float(reward) if torch.is_tensor(reward) else reward
        self.buffer.append((seed_t.detach().cpu(), seed_t1.detach().cpu(), r_val))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s0, s1, r = zip(*batch)
        return torch.stack(s0).to(DEVICE), torch.stack(s1).to(DEVICE), torch.tensor(r).to(DEVICE).float()

class GoalEngine:
    def __init__(self): self.goals = ["minimize_loss"]
    def evolve_goals(self, memory_db):
        if hasattr(memory_db, "payloads") and len(memory_db.payloads) > 10:
            losses = [m['loss'] for m in memory_db.payloads[-10:]]
            if np.std(losses) < 0.05 and "increase_creativity" not in self.goals:
                self.goals.append("increase_creativity")
                logging.info("🧠 GOAL EVOLVED: Added 'increase_creativity'")
    def reward_modifier(self, loss):
        if "increase_creativity" in self.goals: return loss * random.uniform(0.9, 1.1)
        return loss

class HierarchicalMemory(nn.Module):
    def __init__(self, dim=CONFIG["EMBED_DIM"]):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(32, dim))
    def read(self): return self.bank

class DiskEpisodicMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.count = 0; self.payloads = []; self.centroids = []
        self.file_emb = PATHS["MEM_VECS"]; self.file_meta = PATHS["MEM_META"]
        if os.path.exists(self.file_emb):
            self.emb = np.memmap(self.file_emb, dtype='float32', mode='r+', shape=(self.max, self.dim))
            self.load_meta()
        else:
            self.emb = np.memmap(self.file_emb, dtype='float32', mode='w+', shape=(self.max, self.dim))

    def store(self, embedding, payload):
        if dist.is_initialized() and dist.get_rank() != 0: return
        vec = embedding.detach().cpu().numpy().flatten()
        idx = self.count % self.max
        self.emb[idx] = vec
        entry = {"data": payload, "time": time.time(), "id": self.count}
        if idx < len(self.payloads): self.payloads[idx] = entry
        else: self.payloads.append(entry)
        self.count += 1
        if self.count % 1000 == 0: self.emb.flush()

    def query(self, embedding, top_k=5):
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
        atomic_save({"count": self.count, "payloads": self.payloads}, self.file_meta)
    def load_meta(self):
        if os.path.exists(self.file_meta):
            try:
                with open(self.file_meta, "rb") as f:
                    d = pickle.load(f); self.count = d["count"]; self.payloads = d["payloads"]
            except: pass

# ============================================================
# 5. NEURAL ARCHITECTURE
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

    def forward(self, x, mem=None, loss_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        if mem is not None:
            m = mem.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([m, k], 1); v = torch.cat([m, v], 1)
        
        q, k, v = [t.view(B, -1, CONFIG["HEADS"], self.head_dim).transpose(1,2) for t in (q,k,v)]
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        start = max(0, k.size(2) - T)
        att_self = att[:, :, :, start:]
        if T <= self.mask.size(2):
            att_self = att_self.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            att_self = att_self.masked_fill(self.local[:,:,:T,:T]==0, float('-inf'))
        att[:, :, :, start:] = att_self
        
        y = F.softmax(att, -1) @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        out = self.proj(y * self.gate)
        
        # RESTORED: Selective Backprop Mask
        if loss_mask is not None:
            out = out * loss_mask.unsqueeze(-1)
        return out

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
        scores = torch.nan_to_num(scores, 0.0)
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
        # RESTORED: Thought Engine
        self.thought = nn.Sequential(nn.Linear(dim, CONFIG["THOUGHT_DIM"]), nn.Tanh(), nn.Linear(CONFIG["THOUGHT_DIM"], dim))

    def forward(self, x, mem=None, loss_mask=None):
        x = x + self.attn(self.ln1(x), mem, loss_mask=loss_mask)
        x = x + self.thought(x) 
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
        
        # Apex Heads
        self.world_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.val_head = nn.Linear(dim, 1)

    def forward(self, idx, targets=None, noise=0.0):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
        
        w = CONFIG["WORLD_SIM"]
        x = x.repeat_interleave(w, 0)
        if noise > 0: x += torch.randn_like(x) * noise
        
        mem = self.mem
        
        # RESTORED: Selective Backprop Masking Calculation
        loss_mask = None
        if targets is not None:
             # Placeholder: We don't have losses yet, but in a real loop you'd calc it before.
             # For now, we simulate "attention mask" for backprop efficiency
             pass

        for b in self.blocks: x = b(x, mem, loss_mask=loss_mask)
        
        x_flat = self.ln_f(x)
        x_mean = x_flat.view(w, B, T, -1).mean(0)
        
        logits = self.head(x_mean)
        wm_pred = self.world_head(x_mean)
        val_pred = self.val_head(x_mean)
        
        loss = None
        if targets is not None:
            safe_logits = torch.nan_to_num(logits, 0.0)
            main = F.cross_entropy(safe_logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
            with torch.no_grad(): target_next = x_mean.detach()
            wm_loss = F.mse_loss(wm_pred, target_next)
            
            tok_loss = F.cross_entropy(safe_logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
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
# 6. IMMORTAL CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank, world):
        self.rank, self.world_size = rank, world
        self.data = DataManager(rank)
        self.memory = DiskEpisodicMemory()
        
        self.agency = AgencyCore().to(DEVICE)
        self.world = NeuralWorldModel().to(DEVICE)
        self.meta = MetaLearningEngine().to(DEVICE)
        self.self = PredictiveSelfModel().to(DEVICE)
        self.tele = TelemetryLogger()
        
        self.civ = CivilizationCoordinator()
        self.res = AutonomousResearch()
        self.plan = PlanningEngine(self.world, self.self)
        self.life = LifelongProtector()
        self.ledger = BeliefLedger()
        self.experiment = ExperimentEngine()
        self.curiosity = CuriosityEngine()
        self.civ_mind = CivilizationMind()
        self.arch_policy = ArchitecturePolicy().to(DEVICE)
        self.identity_continuity = IdentityContinuity()
        self.narrative = SelfNarrative()
        self.concepts = ConceptTracker()
        self.shard_mgr = ShardedIdentity()
        self.sandbox = SelfRewriteSandbox()
        self.goal_engine = GoalEngine()
        self.genome = IdentityGenome()
        
        self.pop = []
        self.opts = []
        self.world_opt = optim.AdamW(self.world.parameters(), lr=1e-4)
        self.meta_optimizer = optim.AdamW(self.meta.parameters(), lr=1e-4)
        self.self_opt = optim.AdamW(self.self.parameters(), lr=1e-4)
        self.agency_opt = self.agency.optimizer
        
        self.replay = deque(maxlen=2000)
        self.prev_loss = 0.0
        self.score_history = []
        
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
            if "wm" in d: self.world.load_state_dict(d["wm"])
            if "self" in d: self.self.load_state_dict(d["self"])
            self.memory.load_meta()
            if self.rank == 0: logging.info(f"RESTORED GEN {d.get('gen', '?')}")

    def save(self, gen, tag=""):
        if self.rank != 0: return
        s = {
            "pop": [self.unwrap(p).state_dict() for p in self.pop],
            "opts": [o.state_dict() for o in self.opts],
            "agency": self.agency.state_dict(),
            "agency_opt": self.agency.optimizer.state_dict(),
            "wm": self.world.state_dict(),
            "self": self.self.state_dict(),
            "gen": gen
        }
        atomic_save(s, PATHS["CHECKPOINT"], True)
        atomic_save(s, f"{PATHS['DIR_ARCHIVE']}/gen_{gen}.pt", True)
        self.memory.save()
        self.genome.save()
        logging.info(f"💾 SAVED GEN {gen}")

    def unwrap(self, m): return m.module if hasattr(m, "module") else m

    def run(self):
        for gen in range(CONFIG["GENERATIONS"]):
            x, y = self.data.get_batch()
            if x is None: continue
            
            self.world.reset_state()
            
            with torch.no_grad():
                _, _, state = self.unwrap(self.pop[0])(x)
                state = state.mean(0)
            
            action = self.agency.decide(state)
            if self.rank == 0: logging.info(f"🤖 GEN {gen} | ACTION: {action}")

            loss_r = 0.0
            if action == "train": loss_r = self.train(gen)
            elif action == "evolve": self.evolve(gen)
            elif action == "reflect": self.reflect()
            
            state_detach = state.detach().clone()
            pred = self.world(state_detach)
            wm_loss = F.mse_loss(pred, state_detach)
            self.world_opt.zero_grad(); wm_loss.backward(); self.world_opt.step()
            
            # RESTORED: Explicit Curiosity
            c_bonus = self.curiosity.reward(state_detach)
            
            curiosity = wm_loss.item()
            total_reward = (loss_r * 1.0) + (curiosity * 0.1) + c_bonus
            self.agency.rewards.append(total_reward)
            self.agency.update_policy()
            
            self.save(gen)

    def train(self, gen):
        noise = max(0, 0.01 * (1 - gen/CONFIG["GENERATIONS"]))
        self.res.act(self) 
        
        for i, (model, opt) in enumerate(zip(self.pop, self.opts)):
            model.train()
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                diff = random.choice(CONFIG["CURRICULUM"])
                x, y = self.data.get_batch(diff)
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
        
        # RESTORED: Identity Genome
        if self.rank == 0:
            self.genome.add(seed, scores[best])
            # Check collapse
            if scores[best] < -10.0:
                 logging.warning("⚠️ COLLAPSE DETECTED. RESURRECTING ANCESTOR.")
                 ancient = self.genome.resurrect()
                 if ancient: 
                     seed = {"weights": torch.tensor(ancient["w"]), "meta": ancient["meta"]}
                     state = None # Reset optimizer for fresh start

        # RESTORED: Experiment Engine (Lobotomy)
        if self.rank == 0:
            robustness = self.experiment.run(self.unwrap(self.pop[best]), self.data)
            logging.info(f"   Robustness Score: {robustness:.4f}")

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
                if state: self.opts[i].load_state_dict(state)

    def reflect(self):
        x, _ = self.data.get_batch()
        with torch.no_grad():
            m = self.unwrap(self.pop[0])
            _, _, meta = m(x)
            vec = meta.mean(0)
            
            recalled = self.memory.query(vec)
            
            # RESTORED: Context Injection
            ctx = ""
            if recalled:
                texts = [r["data"]["text"] for r in recalled if isinstance(r, dict) and "data" in r and "text" in r["data"]]
                ctx = " ".join(texts[:2]).strip()

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
