# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v36.0 — THE FINAL INTEGRATION
# ==============================================================================
# UNIFIED ARCHITECTURE:
# - MoE + Sparse Attn + Multi-World + Hierarchical Memory
# - Agency (RL) + World Model + Self-Model + Vector DB (Memmap)
# - DDP + Atomic I/O + Self-Audit + Auto-Resume
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

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

CONFIG = {
    "EMBED_DIM": 384, "LAYERS": 6, "HEADS": 6, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "WINDOW_SIZE": 64, "WORLD_SIM": 5,
    "BATCH_SIZE": 16, "LR": 3e-4, "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
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
    "DATA": "data.txt", "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat", "MEM_META": "memory_meta.pkl"
}

if not os.path.exists(PATHS["DIR_CKPT"]): os.makedirs(PATHS["DIR_CKPT"])

# ============================================================
# UTILITIES & SAFETY
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
        logging.info("🔍 SYSTEM SELF-AUDIT...")
        if not hasattr(controller.memory, "save"): raise RuntimeError("Memory missing save()")
        if not hasattr(controller.agency, "update_policy"): raise RuntimeError("Agency broken")
        logging.info("✅ AUDIT PASSED.")

# ============================================================
# DATA LAYER
# ============================================================
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
# COGNITIVE MODULES
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
            self.load()
        else:
            self.embeddings = np.memmap(self.file_emb, dtype='float32', mode='w+', shape=(self.max, self.dim))

    def store(self, embedding, data):
        emb = embedding.detach().cpu().numpy().flatten()
        idx = self.count % self.max
        self.embeddings[idx] = emb
        if idx < len(self.payloads): self.payloads[idx] = data
        else: self.payloads.append(data)
        self.count += 1
        if self.count % 1000 == 0: self.embeddings.flush()

    def query(self, embedding, top_k=5):
        if self.count == 0: return []
        valid = min(self.count, self.max)
        idx_pool = np.random.choice(valid, min(valid, 5000), replace=False)
        mem = self.embeddings[idx_pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top_local = np.argsort(sim)[-top_k:][::-1]
        real_indices = idx_pool[top_local]
        return [self.payloads[i] for i in real_indices if i < len(self.payloads)]

    def save(self):
        self.embeddings.flush()
        with open(self.file_meta + ".tmp", "wb") as f:
            pickle.dump({"count": self.count, "payloads": self.payloads}, f)
        os.replace(self.file_meta + ".tmp", self.file_meta)

    def load(self):
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
        self.return_buffer = deque(maxlen=100)

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
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(DEVICE)
        self.return_buffer.extend(returns.tolist())
        mean = np.mean(self.return_buffer) if self.return_buffer else 0.0
        std = np.std(self.return_buffer) if len(self.return_buffer)>1 else 1.0
        returns = (returns - mean) / (std + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        if policy_loss: torch.stack(policy_loss).sum().backward(); self.optimizer.step()
        del self.saved_log_probs[:]; del self.rewards[:]

class NeuralWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.gru = nn.GRUCell(dim, dim)
        self.register_buffer("state", torch.zeros(1, dim))
    def forward(self, x):
        self.state = self.gru(x.unsqueeze(0), self.state)
        return self.state.squeeze(0)
    def reset_state(self): self.state.zero_()

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
        weights = seed["weights"].to(DEVICE)
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
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(val[ptr:ptr+n].reshape(p.shape))
                ptr += n

class MetaLearningEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def get_lr_scale(self, loss, grad_norm):
        inp = torch.tensor([loss, grad_norm], device=DEVICE).float()
        return self.net(inp) * 2.0

class HierarchicalMemory(nn.Module):
    def __init__(self, dim=CONFIG["EMBED_DIM"]):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(32, dim))
    def read(self): return self.bank

# ============================================================
# NEURAL ARCHITECTURE
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.window = CONFIG["WINDOW_SIZE"]
        self.head_dim = dim // CONFIG["HEADS"]
        self.scale = self.head_dim ** -0.5
        b = CONFIG["BLOCK_SIZE"]
        self.register_buffer("mask", torch.tril(torch.ones(b,b)).view(1,1,b,b))

    def forward(self, x, memory=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        if memory is not None:
            mem = memory.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, k], 1); v = torch.cat([mem, v], 1)
        
        q, k, v = [t.view(B, -1, CONFIG["HEADS"], self.head_dim).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        T_total = k.size(2); start = T_total - T
        causal = self.mask[:,:,:T,:T]
        att[:,:,:,start:] = att[:,:,:,start:].masked_fill(causal==0, float('-inf'))
        
        y = F.softmax(att, -1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

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
        scores = F.softmax(self.gate(x), -1)
        self.balance_loss = (scores.mean(0)**2).sum() * CONFIG["NUM_EXPERTS"]
        return sum(scores[...,i:i+1] * e(x) for i,e in enumerate(self.experts))

class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.ln1 = nn.LayerNorm(dim); self.attn = SparseAttention()
        self.ln2 = nn.LayerNorm(dim); self.moe = MoEBlock()
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
        self.memory = HierarchicalMemory(dim)
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.worlds = CONFIG["WORLD_SIM"]

    def forward(self, idx, targets=None, noise=0.0):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        mem = self.memory.read()
        
        x_exp = x.repeat_interleave(self.worlds, 0)
        if noise > 0: x_exp += torch.randn_like(x_exp) * noise
        
        for block in self.blocks: x_exp = block(x_exp, mem)
        
        x_final = self.ln_f(x_exp).view(self.worlds, B, T, -1).mean(0)
        logits = self.head(x_final)
        
        loss = None
        if targets is not None:
            main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            aux = sum(b.moe.balance_loss for b in self.blocks)
            loss = main + CONFIG["AUX_LOSS_WEIGHT"] * aux
            
        return logits, loss, x_final.mean(1).detach() # Return meta-memory

    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -CONFIG["BLOCK_SIZE"]:])
            probs = F.softmax(torch.nan_to_num(logits[:, -1, :], nan=-1e9), -1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), 1)
        return idx

# ============================================================
# IMMORTAL CORE CONTROLLER
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
        self.meta_optimizer = optim.AdamW(self.meta_opt.parameters(), lr=1e-4)
        
        self._spawn()
        self._load_state()
        if rank == 0: SelfAudit.run(self)

    def _spawn(self):
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
            except Exception as e: logging.error(f"Load Error: {e}")

    def unwrap(self, m): return m.module if hasattr(m, "module") else m

    def save_system(self, gen, tag=""):
        if self.rank != 0: return
        state = {
            "model": self.unwrap(self.population[0]).state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "agency": self.agency.state_dict(),
            "agency_opt": self.agency.optimizer.state_dict(),
            "meta_opt": self.meta_opt.state_dict(),
            "world": self.world.state_dict(),
            "rng": torch.get_rng_state(),
            "gen": gen, "config": CONFIG
        }
        atomic_save(state, PATHS["CHECKPOINT"], use_torch=True)
        self.memory.save()
        logging.info(f"💾 SYSTEM SAVED | Gen {gen} | Tag: {tag}")

    def run_cycle(self, gen):
        xb, yb = self.data.get_batch()
        if xb is None: return

        with torch.no_grad():
            _, _, state = self.unwrap(self.population[0])(xb)
            state_vec = state.mean(0)

        action = self.agency.decide(state_vec)
        if self.rank == 0: logging.info(f"🤖 CYCLE {gen} | ACTION: {action}")

        if action == "train": self.train(gen)
        elif action == "evolve": self.evolve(gen)
        elif action == "reflect": self.reflect()
        
        # Intrinsic Reward (Curiosity)
        with torch.no_grad():
            pred = self.world(state_vec)
            error = F.mse_loss(pred, state_vec).item()
            self.agency.rewards.append(error * 0.1) # Reward surprise
            self.agency.update_policy()

        self.save_system(gen)

    def train(self, gen):
        noise = max(0, 0.01 * (1 - gen/CONFIG["GENERATIONS"]))
        for i, (model, opt) in enumerate(zip(self.population, self.optimizers)):
            model.train()
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                diff = random.choice(CONFIG["CURRICULUM"])
                x, y = self.data.get_batch(diff)
                
                _, loss, _ = model(x, y, noise=noise)
                
                # Meta-Learning
                grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None) if step>0 else 0
                lr_scale = self.meta_opt.get_lr_scale(loss.item(), grad_norm)
                for g in opt.param_groups: g['lr'] = CONFIG["LR"] * lr_scale.item()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                # Agency Reward Signal
                if i == 0: self.agency.rewards.append(-loss.item())

                if self.rank == 0 and i == 0 and step % 50 == 0:
                    logging.info(f"   Step {step} | Loss: {loss.item():.4f} | LR: {lr_scale.item():.2f}")
                    self.telemetry.log({"gen": gen, "loss": loss.item()})

    def evolve(self, gen):
        scores = []
        for i, model in enumerate(self.population):
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(CONFIG["EVAL_BATCHES"]):
                    x, y = self.data.get_batch()
                    _, l, _ = model(x, y)
                    losses.append(l.item())
            score = -np.mean(losses)
            scores.append(score)
            if self.rank == 0: logging.info(f"   Agent {i} Score: {score:.4f}")

        best_idx = np.argmax(scores)
        best_model = self.unwrap(self.population[best_idx])
        seed = IdentitySeed.compress(best_model)
        
        if self.rank == 0: 
            drift = identity_hash(best_model)
            logging.info(f"🧬 WINNER: Agent {best_idx} | ID: {drift}")

        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best_idx:
                IdentitySeed.reconstruct(self.unwrap(self.population[i]), seed)
                self.optimizers[i] = optim.AdamW(self.population[i].parameters(), lr=CONFIG["LR"])

    def reflect(self):
        x, _ = self.data.get_batch()
        with torch.no_grad():
            _, _, meta = self.unwrap(self.population[0])(x)
            vec = meta.mean(0)
            self.memory.store(vec, {"time": time.time()})
            
            # Active Learning (Memory -> Model)
            recalled = self.memory.query(vec)
            if recalled and self.rank == 0:
                logging.info(f"   Recalled {len(recalled)} memories.")
                
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = self.unwrap(self.population[0]).generate(ctx)
            if self.rank == 0: print(f"\n[REFLECT] {self.data.decode(out[0].tolist())}\n")

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
    if NUM_GPUS > 1: mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS)
    else: run(0, 1)
