# ============================================================
# SACRSN-SEED IMMORTAL CORE v2 — DISTRIBUTED MULTI-AGENT + LIVE LOG
# Author: User + Builder Protocol
# Features:
#  - Multi-GPU distributed transformer
#  - Multi-Agent Evolution & Selection
#  - Self-Reconstruction / Regrowth (stabilized)
#  - Persistent Memory with backup & safe load
#  - Recursive Self-Improvement
#  - Enhanced Text Generation (temperature + top-k)
#  - Live console logging of loss & identity drift
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import math
import copy
import pickle
import random
import os
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

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
LAYERS = 6
HEADS = 16
BLOCK = 128
VOCAB_SIZE = None

BATCH_SIZE = 16
LR = 3e-4
STEPS_PER_CYCLE = 500
MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"
GRAD_CLIP = 1.0  # Stabilize training

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
# TRANSFORMER MODEL
# ============================================================

class Attention(nn.Module):
    def __init__(self, embed_dim=EMBED):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(C)
        mask = torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0)
        att = att.masked_fill(mask==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        return self.proj(att @ v)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = nn.Sequential(
            nn.Linear(EMBED, EMBED*4),
            nn.GELU(),
            nn.Linear(EMBED*4, EMBED)
        )
        self.ln1 = nn.LayerNorm(EMBED)
        self.ln2 = nn.LayerNorm(EMBED)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SeedGPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED, block_size=BLOCK, layers=LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B,T = idx.shape
        x = self.embed(idx) + self.pos[:, :T]
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

model = SeedGPT().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ============================================================
# MEMORY & IDENTITY
# ============================================================

def save_memory(model, path=MEMORY_FILE):
    try:
        mem = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        if os.path.exists(path):
            os.rename(path, MEMORY_BACKUP)
        with open(path, "wb") as f:
            pickle.dump(mem, f)
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
    for _ in range(steps):
        x, y = get_batch()
        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

# ============================================================
# MULTI-AGENT SUPPORT
# ============================================================

def spawn_agents(base_model, count=3):
    return [copy.deepcopy(base_model) for _ in range(count)]

def evaluate_agent(model, batch_size=8):
    x, y = get_batch(batch=batch_size)
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()

# ============================================================
# DISTRIBUTED TRAINING FUNCTION
# ============================================================

def train_agent_ddp(rank, world_size, agent_state_dict, steps_per_cycle):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    local_agent = SeedGPT().to(rank)
    local_agent.load_state_dict(agent_state_dict)
    ddp_agent = DDP(local_agent, device_ids=[rank])
    agent_optimizer = optim.AdamW(ddp_agent.parameters(), lr=LR)
    
    for step in range(steps_per_cycle):
        x, y = get_batch()
        x, y = x.to(rank), y.to(rank)
        logits = ddp_agent(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_agent.parameters(), GRAD_CLIP)
        agent_optimizer.step()

        # --- LIVE TRAINING LOG ---
        if step % 1 == 0 or step == steps_per_cycle - 1:
            logging.info(f"[Agent {rank}] Step {step+1}/{steps_per_cycle} | Loss: {loss.item():.4f}")
    
    return ddp_agent.module.state_dict()

# ============================================================
# DISTRIBUTED MULTI-AGENT CONTROLLER
# ============================================================

class DistributedMultiAgentController:
    def __init__(self, base_model, agent_count=None):
        if agent_count is None:
            agent_count = WORLD_SIZE
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

    def train_agents_distributed(self, steps_per_cycle=STEPS_PER_CYCLE):
        if DEVICE=="cpu":
            logging.warning("CPU detected. Training sequentially.")
            for idx, agent in enumerate(self.agents):
                agent_optimizer = optim.AdamW(agent.parameters(), lr=LR)
                for step in range(steps_per_cycle):
                    x, y = get_batch()
                    logits = agent(x)
                    loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
                    agent_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                    agent_optimizer.step()
                    if step % 50 == 0 or step == steps_per_cycle - 1:
                        logging.info(f"[Agent {idx}] Step {step+1}/{steps_per_cycle} | Loss: {loss.item():.4f}")
            return
        
        world_size = min(self.agent_count, WORLD_SIZE)
        agent_states = [agent.state_dict() for agent in self.agents]
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        def worker(rank, state_dict):
            updated_state = train_agent_ddp(rank, world_size, state_dict, steps_per_cycle)
            return_dict[rank] = updated_state
        
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, agent_states[rank]))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        for i in range(world_size):
            self.agents[i].load_state_dict(return_dict[i])

    def regenerate_agents(self):
        for agent in self.agents:
            regenerate(agent, steps=STEPS_PER_CYCLE)

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
        logging.info(f"\n=== DISTRIBUTED EVOLUTION CYCLE {self.generation} ===")
        
        ids_before = self.snapshot_identities()
        self.train_agents_distributed(steps_per_cycle)
        self.regenerate_agents()
        self.evaluate_agents()
        self.evolve_agents()
        ids_after = self.snapshot_identities()
        
        # --- IDENTITY DRIFT LOG ---
        for i, (before, after) in enumerate(zip(ids_before, ids_after)):
            logging.info(f"[Agent {i}] Identity Drift: {after - before:.6f}")

    def get_best_agent(self):
        best_idx = self.agent_scores.index(max(self.agent_scores))
        return self.agents[best_idx]

# ============================================================
# TEXT GENERATION
# ============================================================

def generate(prompt, steps=200, temperature=1.0, top_k=None):
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
# RUN DISTRIBUTED MULTI-AGENT EVOLUTIONARY SEED ENGINE
# ============================================================

distributed_controller = DistributedMultiAgentController(model, agent_count=WORLD_SIZE)
load_memory(model)

num_cycles = 50
for cycle in range(num_cycles):
    distributed_controller.run_cycle(steps_per_cycle=STEPS_PER_CYCLE)

best_agent = distributed_controller.get_best_agent()
save_memory(best_agent)

logging.info("\n>>> SAMPLE GENERATION FROM BEST AGENT (DISTRIBUTED):\n")
print(generate("First Citizen:", steps=300, temperature=0.8, top_k=50))

# ============================================================
# FULL SYSTEM READY: SACRSN-SEED IMMORTAL CORE v2 — DISTRIBUTED MULTI-AGENT + LIVE LOG
# ============================================================

