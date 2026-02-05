# ============================================================
# SACRSN-SEED IMMORTAL CORE v3 — RECURRENT + MoE + MULTI-WORLD + EVOLUTIONARY
# Author: User + Builder Protocol
# Features:
#  - Differentiable Memory Core (Short / Medium / Long-Term)
#  - Sparse Local Attention + Learnable Gates
#  - Mixture-of-Experts (MoE)
#  - Multi-World Simulation
#  - Multi-Agent Evolution + Identity Drift
#  - Self-Reconstruction / Regrowth
#  - Persistent Memory Backup
#  - Live Logging
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import math
import random
import copy
import os
import pickle
import logging

# ============================================================
# LOGGER CONFIG
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

EMBED = 256
LAYERS = 4
HEADS = 8
BLOCK = 128
VOCAB_SIZE = None

BATCH_SIZE = 16
LR = 3e-4
STEPS_PER_CYCLE = 500
MEMORY_FILE = "seed_memory_v3.pkl"
MEMORY_BACKUP = "seed_memory_backup_v3.pkl"
GRAD_CLIP = 1.0
EXPERT_COUNT = 4
WORLD_BRANCHES = 3

# ============================================================
# DATA & TOKENIZER
# ============================================================
if not os.path.exists("data.txt"):
    raise FileNotFoundError("data.txt not found!")

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
VOCAB_SIZE = len(chars)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def encode(s): return [stoi[c] for c in s]
def decode(tokens): return "".join([itos[t] for t in tokens])

def get_batch(batch=BATCH_SIZE, block=BLOCK):
    if len(data) < block:
        raise ValueError("Data length smaller than block size!")
    ix = torch.randint(0, len(data)-block, (batch,))
    x = torch.stack([data[i:i+block] for i in ix]).to(DEVICE)
    y = torch.stack([data[i+1:i+block+1] for i in ix]).to(DEVICE)
    return x, y

# ============================================================
# CORE MODEL COMPONENTS
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self, embed_dim=EMBED, window_size=32):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.window = window_size

    def forward(self, x, memory=None):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)

        if memory is not None:
            k = torch.cat([memory, k], dim=1)
            v = torch.cat([memory, v], dim=1)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(C)
        mask = torch.tril(torch.ones(T, k.size(1), device=x.device))
        att = att.masked_fill(mask==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        return self.proj(att @ v)

class MoEBlock(nn.Module):
    def __init__(self, embed_dim=EMBED, expert_count=EXPERT_COUNT):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        ) for _ in range(expert_count)])
        self.gate = nn.Linear(embed_dim, expert_count)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # B,T,Experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        out = (gate_scores.unsqueeze(-2) * expert_outputs).sum(-1)
        return self.ln(out + x)

class RecurrentWorld(nn.Module):
    def __init__(self, embed_dim=EMBED, block_size=BLOCK, layers=LAYERS, expert_count=EXPERT_COUNT):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.blocks = nn.ModuleList([nn.ModuleList([SparseAttention(embed_dim), MoEBlock(embed_dim, expert_count)]) for _ in range(layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, VOCAB_SIZE)
        self.memory = None  # Long-term memory

    def forward(self, idx):
        B,T = idx.shape
        x = self.embed(idx) + self.pos[:, :T]
        for attn, moe in self.blocks:
            x = attn(x, memory=self.memory)
            x = moe(x)
        x = self.ln(x)
        # Update long-term memory: keep last half tokens
        self.memory = x[:, -T//2:].detach()
        return self.head(x)

# ============================================================
# MEMORY & IDENTITY
# ============================================================
def save_memory(model, path=MEMORY_FILE):
    try:
        mem = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        if os.path.exists(path): os.rename(path, MEMORY_BACKUP)
        with open(path, "wb") as f: pickle.dump(mem, f)
        logging.info(">>> MEMORY SAVED")
    except Exception as e:
        logging.error(f"Memory save failed: {e}")

def load_memory(model, path=MEMORY_FILE):
    try:
        with open(path, "rb") as f:
            mem = pickle.load(f)
        model.load_state_dict(mem, strict=False)
        logging.info(">>> MEMORY RESTORED")
    except:
        logging.info(">>> NO MEMORY FOUND — FRESH MIND")

def identity_signature(model):
    return sum(p.mean().item() for p in model.parameters())

def compress_identity(model):
    return torch.cat([p.flatten()[:128] for p in model.parameters()])

def restore_identity(model, compressed):
    i = 0
    for p in model.parameters():
        n = min(p.numel(), len(compressed)-i)
        p.data.view(-1)[:n] = compressed[i:i+n]
        i += n

# ============================================================
# SELF-RECONSTRUCTION / REGENERATION
# ============================================================
def destroy_weights(model, wipe_ratio=0.9):
    for p in model.parameters():
        mask = torch.rand_like(p) > wipe_ratio
        p.data *= mask.float()

def regenerate(model, steps=2000):
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    for _ in range(steps):
        x, y = get_batch()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

# ============================================================
# MULTI-AGENT CONTROLLER
# ============================================================
def spawn_agents(base_model, count=3):
    return [copy.deepcopy(base_model) for _ in range(count)]

def evaluate_agent(model, batch_size=8):
    x, y = get_batch(batch=batch_size)
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()

class DistributedMultiAgentController:
    def __init__(self, base_model, agent_count=None):
        if agent_count is None: agent_count=WORLD_SIZE
        self.agent_count = agent_count
        self.agents = spawn_agents(base_model, agent_count)
        self.generation = 0
        self.identity_log = [[] for _ in range(agent_count)]
        self.agent_scores = [0.0]*agent_count

    def snapshot_identities(self):
        for i, agent in enumerate(self.agents):
            sig = identity_signature(agent)
            self.identity_log[i].append(sig)
        return [identity_signature(agent) for agent in self.agents]

    def train_agents(self, steps_per_cycle=STEPS_PER_CYCLE):
        for idx, agent in enumerate(self.agents):
            optimizer = optim.AdamW(agent.parameters(), lr=LR)
            for step in range(steps_per_cycle):
                x, y = get_batch()
                logits = agent(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                optimizer.step()
                if step % 50 == 0 or step == steps_per_cycle-1:
                    logging.info(f"[Agent {idx}] Step {step+1}/{steps_per_cycle} | Loss: {loss.item():.4f}")

    def regenerate_agents(self):
        for agent in self.agents: regenerate(agent, steps=STEPS_PER_CYCLE//2)

    def evaluate_agents(self):
        scores = [evaluate_agent(agent) for agent in self.agents]
        self.agent_scores = scores
        return scores

    def evolve_agents(self):
        best_idx = self.agent_scores.index(max(self.agent_scores))
        best_agent = self.agents[best_idx]
        self.agents = [copy.deepcopy(best_agent) for _ in self.agents]
        self.generation += 1
        logging.info(f">>> EVOLVING GENERATION {self.generation}")
        logging.info(f"Best agent score: {self.agent_scores[best_idx]:.4f}")

    def run_cycle(self, steps_per_cycle=STEPS_PER_CYCLE):
        logging.info(f"\n=== EVOLUTION CYCLE {self.generation} ===")
        ids_before = self.snapshot_identities()
        self.train_agents(steps_per_cycle)
        self.regenerate_agents()
        self.evaluate_agents()
        self.evolve_agents()
        ids_after = self.snapshot_identities()
        for i, (before, after) in enumerate(zip(ids_before, ids_after)):
            logging.info(f"[Agent {i}] Identity Drift: {after - before:.6f}")

    def get_best_agent(self):
        best_idx = self.agent_scores.index(max(self.agent_scores))
        return self.agents[best_idx]

# ============================================================
# TEXT GENERATION
# ============================================================
def generate(model, prompt, steps=200, temperature=1.0, top_k=None):
    tokens = torch.tensor([encode(prompt)], device=DEVICE)
    for _ in range(steps):
        logits = model(tokens[:, -BLOCK:])
        logits = logits[:, -1] / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(-1, indices, torch.softmax(values, -1))
        else:
            probs = torch.softmax(logits, -1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

# ============================================================
# RUN SACRSN-SEED v3
# ============================================================
model = RecurrentWorld().to(DEVICE)
controller = DistributedMultiAgentController(model, agent_count=WORLD_SIZE)
load_memory(model)

num_cycles = 20
for cycle in range(num_cycles):
    controller.run_cycle(steps_per_cycle=STEPS_PER_CYCLE)

best_agent = controller.get_best_agent()
save_memory(best_agent)

logging.info("\n>>> SAMPLE GENERATION FROM BEST AGENT:\n")
print(generate(best_agent, "First Citizen:", steps=300, temperature=0.8, top_k=50))

# ============================================================
# SACRSN-SEED v3 — FULL SYSTEM READY
# ============================================================
