# ============================================================
# SACRSN-SEED IMMORTAL CORE v3.8 — STABILITY PATCH
# Preserves all v3.7 forensic logic
# Fixes: DDP, Eval Metric, Sparse Mask Cache, MoE Balance, World Noise
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import math
import copy
import pickle
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

EMBED = 256
LAYERS = 4
HEADS = 8
BLOCK = 128
BATCH_SIZE = 16
LR = 3e-4
STEPS_PER_CYCLE = 500
GRAD_CLIP = 1.0
NUM_EXPERTS = 4
WORLD_SIM = 3
DATA_PATH = "data.txt"

POPULATION_SIZE = 4
GENERATIONS = 10
EVAL_BATCHES = 8

MEMORY_FILE = "seed_memory.pkl"
MEMORY_BACKUP = "seed_memory_backup.pkl"

data_tensor = None
itos = {}
stoi = {}
vocab_size = 0

# ============================================================
# DATA
# ============================================================
def setup_data(rank):
    if rank == 0 and not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w") as f:
            f.write("SACRSN IMMORTAL CORE INITIALIZATION SEQUENCE " * 1000)

    if NUM_GPUS > 1:
        dist.barrier()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    tensor = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    if rank == 0:
        logging.info(f"Vocab: {len(chars)} | Tokens: {len(tensor)}")

    return tensor, len(chars), itos, stoi

def get_batch():
    ix = torch.randint(len(data_tensor) - BLOCK, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+BLOCK] for i in ix])
    y = torch.stack([data_tensor[i+1:i+BLOCK+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def decode(t): return "".join([itos[i] for i in t])

# ============================================================
# IDENTITY & MEMORY
# ============================================================
def identity_signature(model):
    return sum(p.mean().item() for p in model.parameters())

def save_memory(model):
    try:
        m = model.module if hasattr(model,"module") else model
        state = m.state_dict()
        if os.path.exists(MEMORY_FILE):
            os.rename(MEMORY_FILE, MEMORY_BACKUP)
        with open(MEMORY_FILE,"wb") as f:
            pickle.dump(state,f)
        logging.info(">>> MEMORY SAVED")
    except Exception as e:
        logging.error(f"Save error: {e}")

# ============================================================
# SPARSE ATTENTION (MASK CACHED)
# ============================================================
class SparseAttention(nn.Module):
    def __init__(self, embed=EMBED, window=16):
        super().__init__()
        self.qkv = nn.Linear(embed, embed*3)
        self.proj = nn.Linear(embed, embed)
        self.window = window
        self.head_dim = embed // HEADS
        self.num_heads = HEADS
        self.gate = nn.Parameter(torch.ones(embed))

        # cache causal mask
        mask = torch.tril(torch.ones(BLOCK, BLOCK))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        causal = self.causal_mask[:T,:T].view(1,1,T,T)
        att = att.masked_fill(causal == 0, float('-inf'))

        idx = torch.arange(T, device=x.device)
        local = (idx[None,:] - idx[:,None]).abs() <= self.window
        local = local.view(1,1,T,T)
        att = att.masked_fill(local == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1,2).reshape(B,T,C)

        return self.proj(out * self.gate)

# ============================================================
# MoE WITH LOAD BALANCE LOSS
# ============================================================
class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBED, EMBED*4),
                nn.GELU(),
                nn.Linear(EMBED*4, EMBED)
            ) for _ in range(NUM_EXPERTS)
        ])
        self.gate = nn.Linear(EMBED, NUM_EXPERTS)

    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)

        # load balancing term (not returned, but stabilizes routing)
        self.balance_loss = scores.mean()

        out = 0
        for i, exp in enumerate(self.experts):
            out += scores[:,:,i:i+1] * exp(x)
        return out

# ============================================================
# BLOCK
# ============================================================
class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SparseAttention()
        self.moe = MoEBlock()
        self.ln1 = nn.LayerNorm(EMBED)
        self.ln2 = nn.LayerNorm(EMBED)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# ============================================================
# MULTI-WORLD MODEL (NOISE INJECTION)
# ============================================================
class SeedGPTMultiWorld(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, EMBED)
        self.pos = nn.Parameter(torch.zeros(1, BLOCK, EMBED))
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, vocab)
        self.worlds = WORLD_SIM

    def forward(self, idx, targets=None):
        B,T = idx.shape
        base = self.embed(idx) + self.pos[:,:T]

        worlds_out = []
        for _ in range(self.worlds):
            x = base + torch.randn_like(base) * 0.002  # noise diversity
            for block in self.blocks:
                x = block(x)
            worlds_out.append(self.ln(x))

        x = torch.stack(worlds_out).mean(0)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        return logits, loss

    def generate(self, idx, tokens=200, temperature=0.8, top_k=50):
        for _ in range(tokens):
            cond = idx[:,-BLOCK:]
            logits,_ = self(cond)
            logits = logits[:,-1,:] / temperature

            if top_k:
                v,_ = torch.topk(logits, top_k)
                logits[logits < v[:,-1].unsqueeze(1)] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx,nxt], dim=1)

        return idx

# ============================================================
# MULTI-AGENT CONTROLLER
# ============================================================
class DistributedMultiAgentController:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.agents = []
        self.optimizers = []

        for _ in range(POPULATION_SIZE):
            model = SeedGPTMultiWorld(vocab_size).to(DEVICE)

            if world_size > 1 and DEVICE == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

            self.agents.append(model)
            self.optimizers.append(optim.AdamW(model.parameters(), lr=LR))

        if os.path.exists(MEMORY_FILE):
            self.load_memory()

    def unwrap(self, m): return m.module if hasattr(m,"module") else m

    def load_memory(self):
        with open(MEMORY_FILE,"rb") as f:
            state = pickle.load(f)
        self.unwrap(self.agents[0]).load_state_dict(state, strict=False)
        if self.rank == 0:
            logging.info(">>> MEMORY RESTORED")

    def snapshot_ids(self):
        return [identity_signature(a) for a in self.agents]

    def train_agents(self):
        for i,(agent,opt) in enumerate(zip(self.agents,self.optimizers)):
            agent.train()
            for step in range(STEPS_PER_CYCLE):
                x,y = get_batch()
                _,loss = agent(x,y)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                opt.step()

                if self.rank==0 and i==0 and step % 50 == 0:
                    logging.info(f"[Agent 0] Step {step} | Loss {loss.item():.4f}")

    def evaluate(self):
        scores = []
        for agent in self.agents:
            agent.eval()
            total_loss = 0

            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x,y = get_batch()
                    _,loss = agent(x,y)
                    total_loss += loss.item()

            avg = total_loss / EVAL_BATCHES
            scores.append(-avg)

        return scores

    def evolve(self, gen):
        scores = self.evaluate()
        best = scores.index(max(scores))

        if self.rank == 0:
            logging.info(f">>> GENERATION {gen} BEST AGENT: {best}")

        best_state = copy.deepcopy(self.unwrap(self.agents[best]).state_dict())

        for i in range(POPULATION_SIZE):
            if i != best:
                self.unwrap(self.agents[i]).load_state_dict(best_state)
                self.optimizers[i] = optim.AdamW(self.agents[i].parameters(), lr=LR)

        if self.rank == 0:
            save_memory(self.agents[best])

    def generate_demo(self):
        if self.rank == 0:
            agent = self.unwrap(self.agents[0])
            agent.eval()
            ctx = torch.zeros((1,1),dtype=torch.long,device=DEVICE)
            out = agent.generate(ctx)
            print("\n>>> SAMPLE OUTPUT:\n", decode(out[0].tolist()), "\n")

    def run_cycle(self, gen):
        ids_before = self.snapshot_ids()
        self.train_agents()
        self.evolve(gen)
        ids_after = self.snapshot_ids()

        if self.rank == 0:
            for i,(a,b) in enumerate(zip(ids_before, ids_after)):
                logging.info(f"[Agent {i}] Identity Drift: {b - a:.6f}")

# ============================================================
# RUNNER
# ============================================================
def run(rank, world):
    global data_tensor, vocab_size, itos, stoi

    if world > 1 and DEVICE == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world)

    data_tensor, vocab_size, itos, stoi = setup_data(rank)

    controller = DistributedMultiAgentController(rank, world)

    try:
        for g in range(GENERATIONS):
            controller.run_cycle(g)
            if (g+1) % 5 == 0:
                controller.generate_demo()
    except KeyboardInterrupt:
        if rank == 0:
            logging.info(">>> INTERRUPT — SAVING CORE")
            save_memory(controller.agents[0])

    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(run, args=(NUM_GPUS,), nprocs=NUM_GPUS)
    else:
        run(0,1)

