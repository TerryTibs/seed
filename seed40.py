# ============================================================
# SACRSN-SEED IMMORTAL CORE v24.0 — THE IRONCLAD UPDATE
# ============================================================
#
# CRITICAL FIXES (v24.0):
# 1. BATCHING: Hard crash on missing data (prevents NoneType errors).
# 2. SEED: Safe attribute access for model.blocks.
# 3. HASH: Safe dictionary get() for metadata.
# 4. BUFFER: Scalar reward storage type fix.
# 5. SYNTH: Correct .numel() check for empty tensors.
# 6. DDP: Guaranteed optimizer rebuild after sync.
# 7. META: NaN guards on meta-gradients.
# 8. WORLD: State clamping [-5, 5] to prevent explosion.
# 9. MEMORY: Payload list capping to prevent RAM leaks.
# 10. GEN: Strict NaN masking in generation softmax.
#
# NEW FEATURES:
# 1. KILL-SWITCH: Auto-reset on divergence (Loss > 100).
# 2. GRAD MONITOR: Warnings for NaN gradients.
# 3. HASH CHAIN: Blockchain-style identity lineage.
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
import hashlib
import json
import shutil
from collections import deque

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

# --- HYPERPARAMETERS ---
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

# Legacy Aliases
EMBED = EMBED_DIM
BLOCK = BLOCK_SIZE
LR = LEARNING_RATE
EXPERT_COUNT = NUM_EXPERTS
WORLD_COUNT = WORLD_SIM
WORLD_BRANCHES = WORLD_SIM
ATTN_WINDOW = WINDOW_SIZE

# Massive Config (seed.py)
LARGE_CONFIG = { "EMBED": 1024, "LAYERS": 24, "HEADS": 16, "BLOCK": 512 }

# Lifecycle
POPULATION_SIZE = 4
GENERATIONS = 20
CYCLES_PER_GEN = 200
REGENERATE_STEPS = 50
EVAL_BATCHES = 4
GRAD_CLIP = 1.0
STEPS_PER_CYCLE = CYCLES_PER_GEN

# Cognitive Config
MEMORY_CAPACITY = 500_000
IDENTITY_SEED_SIZE = 512
CURRICULUM_STEPS = [0.25, 0.5, 0.75, 1.0]
SELECTIVE_THRESHOLD = 0.10
WIPE_RATIO_DEFAULT = 0.1
SYNTHETIC_RATIO_CAP = 0.2

# File Paths
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
PT_FILE = "seed_model.pt"
ARCHIVE_FILE = "IMMORTAL_ARCHIVE.pt"
TELEMETRY_FILE = "telemetry.jsonl"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data.txt"
SYNTH_DATA_PATH = "data_recursive.txt"
MEMMAP_FILE = "memory_vectors.dat"

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
    
    synth_text = ""
    if os.path.exists(SYNTH_DATA_PATH):
        with open(SYNTH_DATA_PATH, "r", encoding="utf-8") as f: synth_text += f.read()

    chars = sorted(list(set(raw_text + synth_text)))
    vocab_size = len(chars) + 1
    stoi = {ch: i+1 for i, ch in enumerate(chars)}
    itos = {i+1: ch for i, ch in enumerate(chars)}
    itos[0] = "<PAD>"; stoi["<PAD>"] = 0
    
    data_tensor = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)
    synth_tensor = torch.tensor([stoi[c] for c in synth_text], dtype=torch.long) if synth_text else torch.tensor([], dtype=torch.long)
    
    if rank == 0: logging.info(f">>> VOCAB: {vocab_size} | REAL: {len(data_tensor)} | SYNTH: {len(synth_tensor)}")
    return data_tensor, synth_tensor, vocab_size, itos, stoi

data_tensor, synth_tensor, vocab_size, itos, stoi = None, None, 0, {}, {}

def encode(s): return [stoi.get(c, 0) for c in s]
def decode(tokens): return "".join([itos.get(t, "") for t in tokens if t != 0])

def get_batch(difficulty=1.0, block=BLOCK):
    global synth_tensor, data_tensor
    
    # Fix 1: Hard crash if no data
    if data_tensor is None or len(data_tensor) == 0:
        raise RuntimeError("Training data not initialized or empty!")
    
    if synth_tensor is None: synth_tensor = torch.tensor([], dtype=torch.long)

    use_synth = (len(synth_tensor) > block) and (random.random() < SYNTHETIC_RATIO_CAP)
    source = synth_tensor if use_synth else data_tensor
    
    if len(source) < block: source = data_tensor 
    if len(source) < block: raise ValueError("Data too small for block size!")

    seq_len = max(16, int(block * difficulty))
    if len(source) < seq_len + 1: seq_len = len(source) - 2
    
    ix = torch.randint(len(source) - seq_len, (BATCH_SIZE,))
    x = torch.stack([source[i:i+seq_len] for i in ix])
    y = torch.stack([source[i+1:i+seq_len+1] for i in ix])
    
    if seq_len < block:
        pad = torch.zeros(BATCH_SIZE, block - seq_len, dtype=torch.long)
        x = torch.cat([x, pad], dim=1); y = torch.cat([y, pad], dim=1)
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 3. UTILITIES & GLOBAL HELPERS
# ============================================================
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
        logging.error(f"Atomic Save Failed for {path}: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def identity_signature(model): return sum(p.mean().item() for p in model.parameters())
def compress_identity(model): return IdentitySeed.compress(model)
def restore_identity(model, seed): IdentitySeed.reconstruct(model, seed)
def destroy_weights(model, wipe_ratio=WIPE_RATIO_DEFAULT):
    with torch.no_grad():
        for p in model.parameters():
            mask = (torch.rand_like(p) > wipe_ratio).float(); p.mul_(mask)
def spawn_agents(base_model, count=3): return [copy.deepcopy(base_model) for _ in range(count)]
def evaluate_agent(agent, batch_size=8):
    x, y = get_batch() 
    with torch.no_grad():
        logits, _, _, _ = agent(x); preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()
def generate(model, prompt, steps=200, temperature=1.0, top_k=None):
    tokens = torch.tensor([encode(prompt)], device=DEVICE); model.eval()
    return decode(model.generate(tokens, steps, temperature, top_k)[0].tolist())

def save_memory(model, path=MEMORY_FILE):
    try:
        state = model.state_dict()
        if os.path.exists(path): shutil.copy2(path, MEMORY_BACKUP)
        atomic_save(state, path, use_torch=False)
        atomic_save(state, PT_FILE, use_torch=True)
    except Exception as e: logging.error(f"Save Error: {e}")

def load_memory(model, path=MEMORY_FILE):
    try:
        with open(path, "rb") as f: model.load_state_dict(pickle.load(f), strict=False)
    except: pass

def save_model(model, optimizer=None, step=None, tag="final"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(CHECKPOINT_DIR, f"model_{tag}_{timestamp}.pt")
    payload = {"model_state_dict": model.state_dict(), "step": step, "timestamp": timestamp}
    if optimizer: payload["optimizer_state_dict"] = optimizer.state_dict()
    atomic_save(payload, save_path, use_torch=True)
    logging.info(f"[SAVED] Model checkpoint -> {save_path}")

# ============================================================
# 4. ADVANCED COGNITIVE MODULES
# ============================================================

# --- Agency Core ---
class AgencyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Linear(256, 5), nn.Softmax(dim=-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs = []
        self.rewards = []
        self.return_buffer = deque(maxlen=50)

    def decide(self, state_embed):
        actions = ["train", "evolve", "explore", "reflect", "rest"]
        if state_embed is None: return "train", torch.zeros(5)
        probs = self.net(state_embed.detach())
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action_idx))
        return actions[action_idx.item()], probs.detach()

    def update_policy(self):
        if not self.rewards: return
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(DEVICE)
        self.return_buffer.extend(returns.tolist())
        baseline = sum(self.return_buffer) / len(self.return_buffer) if self.return_buffer else 0.0
        advantage = returns - baseline
        if advantage.std() > 1e-5: advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
        for log_prob, A in zip(self.saved_log_probs, advantage): policy_loss.append(-log_prob * A)
        self.optimizer.zero_grad()
        if len(policy_loss) > 0: torch.stack(policy_loss).sum().backward(); self.optimizer.step()
        del self.saved_log_probs[:]; del self.rewards[:]

# --- Neural World Model (Fix 8: Clamp) ---
class NeuralWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(EMBED_DIM, EMBED_DIM)
        self.register_buffer('state', torch.zeros(1, EMBED_DIM))
    def update(self, embedding):
        self.state = self.gru(embedding.unsqueeze(0), self.state)
        self.state = self.state * 0.995 
        # Fix 8: Clamp state
        self.state = torch.clamp(self.state, -5.0, 5.0)
    def reset_state(self): self.state.zero_()
    def predict_next(self, current_embedding):
        with torch.no_grad():
            next_state = self.gru(current_embedding.unsqueeze(0), self.state)
        return next_state.squeeze(0)

class PersistentWorldSimulator:
    def __init__(self, world_model): self.model = world_model
    def rollout(self, embedding, steps=5):
        states = []; current = embedding
        for _ in range(steps):
            current = self.model.predict_next(current)
            states.append(current)
        return torch.stack(states)

WorldModel = NeuralWorldModel

# --- Identity Continuity (Hash Chain) ---
class IdentityContinuity:
    def __init__(self): self.history = []
    def record(self, seed, hash_sig):
        weights = seed["weights"] if isinstance(seed, dict) else seed
        s_val = weights.detach().cpu().tolist() if isinstance(weights, torch.Tensor) else weights
        
        # New Feature: Hash Chain
        prev_hash = self.history[-1]["hash"] if self.history else "GENESIS"
        chained_hash = hashlib.sha256((str(prev_hash) + hash_sig).encode()).hexdigest()
        
        self.history.append({"seed": s_val, "hash": chained_hash, "time": time.time()})
    def continuity_score(self): return len(self.history)

# --- Telemetry Logger ---
class TelemetryLogger:
    def __init__(self, filename=TELEMETRY_FILE): self.filename = filename
    def log(self, data_dict):
        data_dict["timestamp"] = time.time()
        try:
            with open(self.filename, "a") as f: f.write(json.dumps(data_dict) + "\n")
        except: pass

# --- Disk-Backed Episodic Memory (Fix 9: Payload Cap) ---
class DiskEpisodicMemory:
    def __init__(self, embed_dim=EMBED_DIM, max_entries=MEMORY_CAPACITY):
        self.embed_dim = embed_dim
        self.max_entries = max_entries
        self.count = 0
        self.payloads = []
        self.centroids = []
        self.file_emb = MEMMAP_FILE
        self.file_meta = "episodic_memory_meta.pkl"
        
        if os.path.exists(self.file_emb):
            self.embeddings = np.memmap(self.file_emb, dtype='float32', mode='r+', shape=(max_entries, embed_dim))
            if os.path.exists(self.file_meta):
                with open(self.file_meta, "rb") as f:
                    meta = pickle.load(f)
                    self.count = meta["count"]
                    self.payloads = meta["payloads"]
        else:
            self.embeddings = np.memmap(self.file_emb, dtype='float32', mode='w+', shape=(max_entries, embed_dim))

    def store(self, embedding, data):
        emb = embedding.detach().cpu().numpy().flatten()
        if self.count > 100 and self.count % 500 == 0:
             valid_count = min(self.count, self.max_entries)
             idx = np.random.choice(valid_count, min(valid_count, 20))
             self.centroids = self.embeddings[idx]

        if len(self.centroids) > 0:
            cents = np.stack(self.centroids)
            dists = np.linalg.norm(cents - emb, axis=1)
            if dists.min() < 0.05: return 

        idx = self.count % self.max_entries
        self.embeddings[idx] = emb
        
        # Fix 9: Cap payload list to avoid RAM leak
        if idx < len(self.payloads): self.payloads[idx] = data
        else: self.payloads.append(data)
        
        if len(self.payloads) > self.max_entries:
            self.payloads = self.payloads[-self.max_entries:]
            
        self.count += 1

    def query(self, embedding, top_k=5):
        if self.count == 0: return []
        valid_count = min(self.count, self.max_entries)
        sample_size = min(10000, valid_count)
        indices = np.random.choice(valid_count, sample_size, replace=False)
        mem_sample = self.embeddings[indices]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem_sample @ q) / (np.linalg.norm(mem_sample, axis=1) * np.linalg.norm(q) + 1e-9)
        top_sample_idx = np.argsort(sim)[-top_k:][::-1]
        real_indices = indices[top_sample_idx]
        # Safety check for index bounds
        return [self.payloads[i] for i in real_indices if i < len(self.payloads)]
    
    def save(self):
        self.embeddings.flush()
        with open(self.file_meta + ".tmp", "wb") as f:
            pickle.dump({"count": self.count, "payloads": self.payloads}, f)
        if os.path.exists(self.file_meta): os.replace(self.file_meta + ".tmp", self.file_meta)
        else: os.rename(self.file_meta + ".tmp", self.file_meta)
    
    def load(self): pass

# --- Synthetic Data Guard (Fix 5: Numel) ---
class SyntheticBuffer:
    def __init__(self, capacity=50):
        self.buffer = []
        self.capacity = capacity
    
    def calculate_entropy(self, text):
        if not text: return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    def is_repetitive(self, text):
        words = text.split()
        if len(words) < 10: return True
        grams = [tuple(words[i:i+3]) for i in range(len(words)-3)]
        if len(grams) == 0: return True
        return (len(set(grams)) / len(grams)) < 0.5 

    def add(self, text, perplexity):
        entropy = self.calculate_entropy(text)
        if perplexity < 50.0 and entropy > 3.5 and not self.is_repetitive(text):
            self.buffer.append(text)
            if len(self.buffer) > self.capacity:
                self.flush()
    
    def flush(self):
        try:
            with open(SYNTH_DATA_PATH, "a") as f:
                for t in self.buffer: f.write(t + "\n")
            self.buffer = []
        except: pass

class SelfNarrative:
    def __init__(self): self.events = []
    def log(self, text): self.events.append({"text": text, "time": time.time()})
    def summarize(self): return "\n".join(e["text"] for e in self.events[-20:])

class ConceptTracker:
    def __init__(self): self.concepts = []
    def extract(self, hidden): self.concepts.append(hidden.mean().item())

# --- Planning Engine ---
class PlanningEngine:
    def __init__(self, world_simulator, self_model):
        self.world_simulator = world_simulator
        self.self_model = self_model
    
    def plan(self, model, steps=3):
        seed_data = IdentitySeed.compress(model)["weights"].to(DEVICE)
        current_seed = seed_data
        fitness_trajectory = []
        for _ in range(steps):
            next_seed, fitness = self.self_model(current_seed)
            fitness_trajectory.append(fitness.item())
            current_seed = next_seed 
        return max(fitness_trajectory) if fitness_trajectory else 0.0

# --- Identity Seed (Fix 2/3: Safety) ---
class IdentitySeed:
    @staticmethod
    def compress(model, optimizer=None, meta=None):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // IDENTITY_SEED_SIZE)
        
        anchors = [
            flat[::step][:IDENTITY_SEED_SIZE],
            flat[1::step][:IDENTITY_SEED_SIZE],
            flat[2::step][:IDENTITY_SEED_SIZE]
        ]
        for i in range(3):
            if len(anchors[i]) < IDENTITY_SEED_SIZE:
                anchors[i] = F.pad(anchors[i], (0, IDENTITY_SEED_SIZE - len(anchors[i])))
                
        sampled = torch.cat(anchors).detach().cpu()
        hash_sig = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
        
        # Fix 2: Check for blocks existence
        layer_count = len(model.blocks) if hasattr(model, "blocks") else 0
        
        meta_blob = {
            "layers": layer_count, 
            "embed": EMBED_DIM, 
            "lr": LEARNING_RATE, 
            "hash": hash_sig,
            "_meta_layers": layer_count
        }
        return {
            "weights": sampled,
            "meta": meta_blob,
            "optim": optimizer.state_dict() if optimizer else None
        }

    @staticmethod
    def reconstruct(model, seed):
        if isinstance(seed, torch.Tensor): weights = seed
        else: weights = seed["weights"]
        
        weights = weights.to(next(model.parameters()).device)
        
        if weights.numel() == 3 * IDENTITY_SEED_SIZE:
            anchors = weights.view(3, IDENTITY_SEED_SIZE)
            weights = anchors.mean(dim=0)
        elif weights.numel() > IDENTITY_SEED_SIZE:
            weights = weights[:IDENTITY_SEED_SIZE]

        # Fix 3: Safe get
        meta = seed.get("meta", {}) if isinstance(seed, dict) else {}
        target_layers = meta.get("layers", len(model.blocks) if hasattr(model, "blocks") else LAYERS)
        
        # Architecture Sync
        if hasattr(model, "blocks"):
            while len(model.blocks) < target_layers:
                model.blocks.append(RecurrentBlock().to(DEVICE))
            while len(model.blocks) > target_layers:
                del model.blocks[-1]
            
        total = sum(p.numel() for p in model.parameters())
        x_t = torch.linspace(0, 1, total, device=weights.device)
        x_s = torch.linspace(0, 1, len(weights), device=weights.device)
        idx = torch.bucketize(x_t, x_s).clamp(0, len(weights)-2)
        w0, w1 = weights[idx], weights[idx+1]
        t = (x_t - x_s[idx]) / (x_s[idx+1] - x_s[idx] + 1e-9)
        rebuilt = torch.lerp(w0, w1, t)
        
        ptr = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel(); p.data.copy_(rebuilt[ptr:ptr+n].reshape(p.shape)); ptr += n

class BeliefLedger:
    def __init__(self): self.beliefs = []
    def record(self, belief, parent=None, score=0):
        self.beliefs.append({"belief": belief["weights"].tolist() if isinstance(belief,dict) else belief.tolist(), "score": score, "time": time.time()})

class ArchitecturePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1))
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        self.log_probs = []
        self.rewards = []
    def forward(self, loss, drift, mem_size, depth):
        inp = torch.tensor([loss, drift, mem_size, depth], device=DEVICE).float()
        return self.net(inp)
    def select_action(self, loss, drift, mem_size, depth):
        probs = self.forward(loss, drift, mem_size, depth)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()
    def update(self):
        if not self.rewards: return
        policy_loss = []
        returns = torch.tensor(self.rewards).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        if len(policy_loss) > 0: torch.stack(policy_loss).sum().backward(); self.optimizer.step()
        del self.log_probs[:]; del self.rewards[:]

# --- Fix 4: Scalar Reward Buffer ---
class SeedReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
    def push(self, seed_t, seed_t1, reward):
        # Fix 4: Ensure float
        r_val = float(reward) if torch.is_tensor(reward) else reward
        self.buffer.append((seed_t.detach().cpu(), seed_t1.detach().cpu(), r_val))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s0, s1, r = zip(*batch)
        return torch.stack(s0).to(DEVICE), torch.stack(s1).to(DEVICE), torch.tensor(r).to(DEVICE).float()

class ExperimentEngine:
    def run(self, model):
        ablated = copy.deepcopy(model)
        destroy_weights(ablated, 0.2)
        return evaluate_agent(ablated)

class PredictiveSelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IDENTITY_SEED_SIZE * 3, 256), nn.ReLU(), nn.Linear(256, IDENTITY_SEED_SIZE * 3))
        self.fitness_head = nn.Sequential(nn.Linear(IDENTITY_SEED_SIZE * 3, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, seed): 
        next_seed = self.net(seed)
        predicted_fitness = self.fitness_head(next_seed)
        return next_seed, predicted_fitness

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

class MetaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, loss_tensor): return self.lr_net(loss_tensor)

class HierarchicalMemory(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, short_len=32, medium_len=16, long_len=8):
        super().__init__()
        self.short_mem = nn.Parameter(torch.randn(short_len, embed_dim))
        self.medium_mem = nn.Parameter(torch.randn(medium_len, embed_dim))
        self.long_mem = nn.Parameter(torch.randn(long_len, embed_dim))
    def read(self): return torch.cat([self.short_mem, self.medium_mem, self.long_mem], dim=0)
    def write(self, updates, short_idx=None, medium_idx=None, long_idx=None):
        if short_idx is not None: self.short_mem.data[short_idx] = updates.data
        elif medium_idx is not None: self.medium_mem.data[medium_idx] = updates.data
        elif long_idx is not None: self.long_mem.data[long_idx] = updates.data

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
        if memory is not None:
            mem_exp = memory.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem_exp, k], dim=1); v = torch.cat([mem_exp, v], dim=1)
            T_total = k.size(1) 
        else: T_total = T

        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T_total,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T_total,self.num_heads,self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        
        self_att_start = T_total - T
        att_self = att[:,:,:,self_att_start:]
        att_self = att_self.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        att_self = att_self.masked_fill(self.local_mask[:,:,:T,:T] == 0, float('-inf'))
        att[:,:,:,self_att_start:] = att_self

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.proj(out * self.gate)
        if loss_mask is not None: out = out * loss_mask.unsqueeze(-1)
        return out

class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM*4), nn.GELU(), nn.Linear(EMBED_DIM*4, EMBED_DIM))
            for _ in range(NUM_EXPERTS)])
        self.gate = nn.Linear(EMBED_DIM, NUM_EXPERTS)
        self.register_buffer('balance_loss', torch.tensor(0.0))

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
        self.global_memory = HierarchicalMemory(EMBED_DIM)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.world_sims = WORLD_SIM
        self.meta_memory = None

    def forward(self, idx, targets=None, noise_scale=0.0, loss_mask=None):
        B,T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        mem_ctx = self.global_memory.read()
        x = x + mem_ctx.mean(dim=0).unsqueeze(0).unsqueeze(0)

        expanded_mask = None
        if loss_mask is not None:
            if loss_mask.dim() == 1: loss_mask = loss_mask.view(B, T)
            expanded_mask = loss_mask.repeat(self.world_sims, 1)

        x_exp = x.repeat(self.world_sims, 1, 1)
        if noise_scale > 0: x_exp += torch.randn_like(x_exp) * noise_scale
        
        for block in self.blocks: 
            x_exp = block(x_exp, memory=mem_ctx, loss_mask=expanded_mask)
        
        x_final = self.ln_f(x_exp).view(self.world_sims, B, T, EMBED_DIM).mean(dim=0)
        logits = self.head(x_final)
        self.meta_memory = logits.mean(dim=1).detach()
        
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
            
            # Fix 10: NaN Guard
            probs = F.softmax(torch.nan_to_num(logits, nan=-1e9), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

SeedGPT = SacrsnSeedGPT; RecurrentWorld = SacrsnSeedGPT; SeedGPTv3 = SacrsnSeedGPT

# ============================================================
# 6. IMMORTAL CORE CONTROLLER
# ============================================================
class ImmortalCoreController:
    def __init__(self, rank=0, world_size=1):
        self.rank, self.world_size = rank, world_size
        self.population, self.optimizers = [], []
        self.score_history = []
        self.prev_loss = 0.0
        self.prev_state = None
        
        self.memory_db = DiskEpisodicMemory(EMBED_DIM)
        self.memory_db.load()
        self.self_model = PredictiveSelfModel().to(DEVICE)
        self.meta_opt = MetaOptimizer().to(DEVICE)
        self.goal_engine = GoalEngine()
        self.telemetry = TelemetryLogger()
        self.replay_buffer = SeedReplayBuffer()
        self.synth_buffer = SyntheticBuffer()
        
        self.agency = AgencyCore().to(DEVICE)
        self.world_env = NeuralWorldModel().to(DEVICE) 
        self.world_simulator = PersistentWorldSimulator(self.world_env) 
        self.identity_continuity = IdentityContinuity()
        self.narrative = SelfNarrative()
        self.planner = PlanningEngine(self.world_simulator, self.self_model)
        self.concepts = ConceptTracker()
        
        self.belief_ledger = BeliefLedger()
        self.arch_policy = ArchitecturePolicy().to(DEVICE)
        self.world_model = WorldModel().to(DEVICE) 
        self.experiment_engine = ExperimentEngine()
        
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

    def probe_weight_stats(self, model):
        means, stds = [], []
        for p in model.parameters():
            means.append(p.data.mean().item()); stds.append(p.data.std().item())
        return np.mean(means), np.mean(stds)

    def save_memory(self, model):
        try:
            state = model.state_dict()
            if os.path.exists(MEMORY_FILE): os.rename(MEMORY_FILE, MEMORY_BACKUP)
            atomic_save(state, MEMORY_FILE, use_torch=False)
            atomic_save(state, PT_FILE, use_torch=True)
            logging.info(">>> MEMORY SAVED (Atomic)")
        except Exception as e: logging.error(f"Save Error: {e}")

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
            atomic_save(checkpoint, path, use_torch=True)
            logging.info(f">>> CHECKPOINT .pt SAVED: {path}")
        except Exception as e: logging.error(f"PT Save Error: {e}")

    def export_identity_archive(self, model, memory_db, narrative):
        try:
            seed = IdentitySeed.compress(model)
            atomic_save({
                "seed": seed,
                "beliefs": memory_db.payloads[:1000],
                "narrative": narrative.events,
                "arch_layers": len(model.blocks),
                "timestamp": time.time()
            }, ARCHIVE_FILE, use_torch=True)
            logging.info(">>> IMMORTAL ARCHIVE EXPORTED")
        except Exception as e: logging.error(f"Archive Error: {e}")

    def grow_network(self, agent_idx):
        if self.rank == 0:
            model = self.unwrap(self.population[agent_idx])
            model.blocks.append(RecurrentBlock().to(DEVICE))
            if len(model.blocks) > model.position_embedding.num_embeddings:
                 model.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM).to(DEVICE)
            logging.info(f"🌱 AGENT {agent_idx} ARCHITECTURE GROWN: {len(model.blocks)} LAYERS")
            state = model.state_dict()
            state["_meta_layers"] = len(model.blocks)
        else:
            state = None

        if self.world_size > 1:
            dist.barrier()
            obj_list = [state]
            dist.broadcast_object_list(obj_list, src=0)
            state = obj_list[0]
            model = self.unwrap(self.population[agent_idx])
            target_layers = state.get("_meta_layers", len(state)//2)
            while len(model.blocks) < target_layers:
                 model.blocks.append(RecurrentBlock().to(DEVICE))
            model.load_state_dict(state, strict=False)

        if self.world_size > 1:
            self.population[agent_idx] = nn.parallel.DistributedDataParallel(
                self.unwrap(self.population[agent_idx]), device_ids=[self.rank], find_unused_parameters=True)
        
        # Fix 6: Guaranteed Rebuild
        self.optimizers[agent_idx] = optim.AdamW(self.population[agent_idx].parameters(), lr=LEARNING_RATE)

    def auto_grow(self):
        loss = self.score_history[-1] if self.score_history else 0
        policy = self.arch_policy.select_action(loss, 0, len(self.memory_db.embeddings), len(self.unwrap(self.population[0]).blocks))
        if policy == 0: 
            self.grow_network(0)
            reward = 1.0 if loss < self.prev_loss else -0.5
            self.arch_policy.rewards.append(reward)
            self.arch_policy.update()

    def simulate_future(self, model):
        with torch.no_grad():
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(ctx, 100)
        return decode(out[0].tolist())

    def destroy_weights(self, model, ratio=0.1):
        with torch.no_grad():
            for p in model.parameters():
                mask = (torch.rand_like(p) > ratio).float(); p.mul_(mask)

    def probe_neurons(self, model, x):
        with torch.no_grad():
            _, _, hidden, _ = model(x)
        scores = hidden.abs().mean(dim=(0,1))
        top = torch.topk(scores, 10)
        return top.indices.tolist(), top.values.tolist()

    def train_agents_distributed(self, steps_per_cycle=STEPS_PER_CYCLE):
        self.phase_train(0)

    def run_agency_step(self, model):
        if model.meta_memory is None: return "train"
        state_embed = model.meta_memory.mean(dim=0)
        action, probs = self.agency.decide(state_embed)
        if self.rank == 0:
            self.telemetry.log({"event": "agency_decision", "action": action, "probs": probs.cpu().tolist()})
        return action

    def run_exploration(self, gen):
        if self.rank == 0: logging.info("🌍 MODE: EXPLORATION (High Noise)")
        self.train_with_noise(gen, 0.05)

    def run_reflection(self, gen):
        if self.rank == 0: logging.info("💭 MODE: REFLECTION (Memory Consolidation)")
        self.generate_demo()
        with torch.no_grad():
            xb, _ = get_batch()
            _, _, hidden, _ = self.unwrap(self.population[0])(xb)
            memories = self.memory_db.query(hidden.mean(dim=1).mean(dim=0))
            if memories and self.rank == 0: logging.info(f"    Recalled {len(memories)} memories")

    def train_with_noise(self, gen, noise_level):
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            for step in range(50): 
                xb, yb = get_batch()
                _, loss, _, _ = model(xb, yb, noise_scale=noise_level)
                opt.zero_grad(); loss.backward(); opt.step()

    def phase_train(self, generation):
        noise_level = max(0.0, 0.01 * (1.0 - generation / GENERATIONS))
        
        if generation % 10 == 0: self.world_env.reset_state()
        
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            log = (i == 0 and self.rank == 0)
            
            for step in range(CYCLES_PER_GEN):
                diff = random.choice(CURRICULUM_STEPS)
                xb, yb = get_batch(difficulty=diff)
                
                _, loss, hidden, _ = model(xb, yb, noise_scale=noise_level)
                loss = self.goal_engine.reward_modifier(loss)
                
                # Check for Kill-Switch Condition
                if torch.isnan(loss) or loss.item() > 100.0:
                    logging.warning(f"⚠️ AGENT {i} DIVERGED (Loss: {loss.item()}). RESETTING...")
                    load_memory(self.unwrap(model))
                    return # Abort this cycle

                with torch.no_grad():
                     current_state = hidden.mean(dim=1).mean(dim=0).detach()
                     if self.prev_state is not None:
                         pred_state = self.world_env.predict_next(self.prev_state)
                         intrinsic_reward = F.mse_loss(pred_state, current_state).item() * 0.1
                         self.agency.rewards.append(intrinsic_reward)
                     self.prev_state = current_state

                if i == 0:
                    state_embed = hidden.mean(dim=1).mean(dim=0)
                    self.world_env.update(state_embed)
                    self.concepts.extract(hidden)
                
                if step % 20 == 0 and self.rank == 0:
                    memories = self.memory_db.query(hidden.mean(dim=1).mean(dim=0))
                    if memories:
                        avg_past_loss = np.mean([m['loss'] for m in memories])
                        if avg_past_loss < loss.item(): noise_level *= 0.9
                        else: noise_level *= 1.1 
                
                meta_in = torch.tensor([[loss.item()]], device=DEVICE)
                lr_scale = self.meta_opt(meta_in)
                
                delta = (self.prev_loss - loss).detach()
                # Fix 7: Meta NaN Guard
                meta_loss = torch.nan_to_num(-(delta * lr_scale).mean(), nan=0.0)
                
                self.meta_opt_optimizer.zero_grad(); meta_loss.backward(); self.meta_opt_optimizer.step()
                for g in opt.param_groups: g['lr'] = LEARNING_RATE * (lr_scale.item() + 0.5)
                
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
                self.prev_loss = loss.item()
                
                if log and step % 50 == 0:
                    # New Feature: Grad Monitor
                    for p in model.parameters():
                        if torch.isnan(p).any(): logging.warning("NaN WEIGHTS DETECTED")

                    logging.info(f"[Agent 0] Step {step} | Loss: {loss.item():.4f} | LR: {lr_scale.item():.2f}")
                    if self.rank == 0: self.telemetry.log({"step": step, "loss": loss.item(), "lr": lr_scale.item()})
                
                if step > 0 and step % 500 == 0 and self.rank == 0:
                    self.save_checkpoint(model, opt, generation, tag=f"step{step}")

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
        sorted_indices = np.argsort(scores)[::-1]
        survivor_count = max(1, len(self.population) // 2)
        survivors = sorted_indices[:survivor_count]
        best_idx = survivors[0]
        
        reward = scores[best_idx] - (self.score_history[-1] if self.score_history else 0)
        self.agency.rewards.append(reward)
        self.agency.update_policy()
        
        self.score_history.append(scores[best_idx])
        if self.rank == 0: 
            logging.info(f"  [EVOLVE] ELITES: {survivors.tolist()} | Best Score: {scores[best_idx]:.4f}")
            w_mean, w_std = self.probe_weight_stats(self.unwrap(self.population[best_idx]))
            logging.info(f"  [TELEMETRY] Weights Mean: {w_mean:.4f} | Std: {w_std:.4f}")

        best_agent = self.unwrap(self.population[best_idx])
        best_state = copy.deepcopy(best_agent.state_dict())
        
        exp_score = self.experiment_engine.run(best_agent)
        if self.rank == 0: logging.info(f"🧪 EXPERIMENT SCORE: {exp_score:.4f}")
        
        for i in range(POPULATION_SIZE):
            if i in survivors: continue
            else:
                parent_state = random.choice([copy.deepcopy(self.unwrap(self.population[s]).state_dict()) for s in survivors])
                model = self.unwrap(self.population[i])
                
                self.sync_architecture(model, parent_state)
                model.load_state_dict(parent_state)
                
                if self.world_size > 1:
                    self.population[i] = nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.rank], find_unused_parameters=True)
                
                with torch.no_grad():
                    for p in self.population[i].parameters(): p.add_(torch.randn_like(p) * 0.01)
                
                # Fix 6: Guaranteed Rebuild
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=LEARNING_RATE)

        if self.rank == 0:
            self.memory_db.save()
            self.save_memory(best_agent)
            self.save_checkpoint(best_agent, self.optimizers[best_idx], gen, tag="best")
            
            seed_export = IdentitySeed.compress(best_agent, self.optimizers[best_idx])
            self.belief_ledger.record(belief=seed_export, score=scores[best_idx])
            self.export_identity_archive(best_agent, self.memory_db, self.narrative)

    def validate_synthetic_data(self, text, model):
        tokens = torch.tensor([encode(text)], device=DEVICE)
        # Fix 5: Correct numel check
        if tokens.numel() == 0: return False
        with torch.no_grad():
            logits, _, _, _ = model(tokens)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tokens.view(-1), ignore_index=0)
        return loss.item() < 3.0

    def run_cycle(self, gen):
        if self.rank == 0: logging.info(f"\n=== CYCLE {gen} ===")
        
        with torch.no_grad():
            xb, _ = get_batch()
            _, _, hidden, _ = self.unwrap(self.population[0])(xb)
            state_embed = hidden.mean(dim=1).mean(dim=0)
        
        action, probs = self.agency.decide(state_embed)
        if self.rank == 0: 
            logging.info(f"🧠 AGENCY DECISION: {action}")
            self.narrative.log(f"Cycle {gen}: Action={action}")

        if action == "evolve": 
            if self.rank == 0: self.grow_network(0)
        elif action == "rest": 
            time.sleep(0.2); return
        elif action == "reflect": 
            self.run_reflection(gen); return
        
        seed_before = IdentitySeed.compress(self.unwrap(self.population[0]), self.optimizers[0])
        
        self.phase_train(gen)
        scores = self.phase_evaluate()
        
        plan_score = self.planner.plan(self.unwrap(self.population[0]))
        if self.rank == 0: logging.info(f"🧭 PLAN SCORE: {plan_score:.4f}")
        
        self.auto_grow() 
        self.phase_evolve(scores, gen)
        self.phase_regenerate()
        
        seed_after = IdentitySeed.compress(self.unwrap(self.population[0]), self.optimizers[0])
        if self.rank == 0:
            # Fix 3 + Hash Chain
            hash_sig = seed_after.get("meta", {}).get("hash", "unknown")
            self.identity_continuity.record(seed_after["weights"], hash_sig)
            if len(self.identity_continuity.history) > 5:
                prev = self.identity_continuity.history[-2]["hash"]
                if prev != hash_sig: logging.info("🧬 IDENTITY DRIFT DETECTED")

        # Fix 4: Scalar Reward
        self.replay_buffer.push(seed_before["weights"], seed_after["weights"], float(scores[0]))
        if len(self.replay_buffer.buffer) > 64:
            s0, s1, r_target = self.replay_buffer.sample(64)
            pred_seed, pred_fit = self.self_model(s0)
            meta_loss = F.mse_loss(pred_seed, s1) + F.mse_loss(pred_fit.squeeze(), r_target)
            self.self_opt.zero_grad(); meta_loss.backward(); self.self_opt.step()
            if self.rank == 0: logging.info(f"🪞 SELF-MODEL LOSS: {meta_loss.item():.6f}")
        
        if self.rank == 0:
            synth_text = self.simulate_future(self.unwrap(self.population[0]))
            if self.validate_synthetic_data(synth_text, self.unwrap(self.population[0])):
                self.synth_buffer.add(synth_text, 1.0)
            
            if gen % 2 == 0:
                global data_tensor
                data_tensor, _, _, _, _ = setup_data(self.rank)

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

    data_tensor, _, vocab_size, itos, stoi = setup_data(rank)
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
