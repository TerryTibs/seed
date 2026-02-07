# ==============================================================================
# SACRSN-SEED IMMORTAL CORE v64.1 — APEX FIX
# ==============================================================================
#
# 🟢 BUG FIX:
# 1. ApexController.__init__ signature updated to accept (rank, world_size).
#
# Z-SERIES FEATURES (RETAINED):
# - Z-1: Persistent State
# - Z-2: World Model
# - Z-3: Planning Head
# - Z-4: External Memory (Rolling Cache)
# - Z-5: Auto-Tuner
# - Z-6: Multi-Path Inference
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
    format="%(asctime)s | APEX | %(message)s",
    datefmt="%H:%M:%S"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

CONFIG = {
    "EMBED_DIM": 384, "LAYERS": 8, "HEADS": 8, "BLOCK_SIZE": 256,
    "NUM_EXPERTS": 4, "TOP_K": 2, "WINDOW_SIZE": 64, 
    "LATENT_STATE_DIM": 384, 
    "WORLD_PRED_STEPS": 1,   
    "INFERENCE_PATHS": 4,    
    
    "BATCH_SIZE": 16, "LR": 3e-4, "DROPOUT": 0.1, "GRAD_CLIP": 1.0,
    "POPULATION_SIZE": 1,    
    "GENERATIONS": 50, "CYCLES_PER_GEN": 500,
    
    "MEMORY_CAPACITY": 100_000,
    "DATA_PATH": "data.txt"
}

PATHS = {
    "CHECKPOINT": "apex_checkpoint.pt",
    "BEST_MODEL": "apex_best.pt",
    "MEMORY": "apex_memory.dat",
    "TELEMETRY": "apex_telemetry.csv",
    "DIR_CKPT": "checkpoints"
}

if not os.path.exists(PATHS["DIR_CKPT"]): os.makedirs(PATHS["DIR_CKPT"])

# ============================================================
# 2. DATA INFRASTRUCTURE
# ============================================================
class DataManager:
    def __init__(self, rank):
        self.rank = rank
        self.data = None
        self.vocab_size = 0
        self.stoi = {}; self.itos = {}
        self._load()

    def _load(self):
        if self.rank == 0 and not os.path.exists(CONFIG["DATA_PATH"]):
            with open(CONFIG["DATA_PATH"], "w") as f: f.write("APEX INITIALIZATION " * 5000)
        if NUM_GPUS > 1: dist.barrier()
        
        with open(CONFIG["DATA_PATH"], "r", encoding="utf-8") as f: text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 1 
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}; self.stoi["<PAD>"] = 0
        self.itos = {i+1:ch for i,ch in enumerate(chars)}; self.itos[0] = "<PAD>"
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
    def get_batch(self):
        if len(self.data) < CONFIG["BLOCK_SIZE"]: return None, None
        ix = torch.randint(len(self.data) - CONFIG["BLOCK_SIZE"], (CONFIG["BATCH_SIZE"],))
        x = torch.stack([self.data[i:i+CONFIG["BLOCK_SIZE"]] for i in ix])
        y = torch.stack([self.data[i+1:i+CONFIG["BLOCK_SIZE"]+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)
    
    def decode(self, t): return "".join([self.itos.get(i, "") for i in t if i != 0])
    def encode(self, s): return [self.stoi.get(c, 0) for c in s]

# ============================================================
# 3. Z-4: EXTERNAL MEMORY (Rolling Cache)
# ============================================================
class ApexMemory:
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.ptr = 0
        self.full = False
        
        self.keys = torch.zeros(self.max, self.dim, device="cpu")
        self.vals = torch.zeros(self.max, self.dim, device="cpu")

    def write(self, k, v):
        b = k.size(0)
        end = self.ptr + b
        
        if end > self.max:
            overflow = end - self.max
            self.keys[self.ptr:] = k[:-overflow].cpu()
            self.vals[self.ptr:] = v[:-overflow].cpu()
            self.keys[:overflow] = k[-overflow:].cpu()
            self.vals[:overflow] = v[-overflow:].cpu()
            self.ptr = overflow
            self.full = True
        else:
            self.keys[self.ptr:end] = k.cpu()
            self.vals[self.ptr:end] = v.cpu()
            self.ptr = end

    def read(self, query, k=5):
        if self.ptr == 0 and not self.full: return None
        
        valid_sz = self.max if self.full else self.ptr
        keys_view = self.keys[:valid_sz].to(query.device) 
        vals_view = self.vals[:valid_sz].to(query.device)
        
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(keys_view, dim=-1)
        scores = q_norm @ k_norm.T
        
        top_scores, top_idx = torch.topk(scores, k, dim=-1)
        retrieved = vals_view[top_idx] 
        return retrieved

# ============================================================
# 4. APEX MODEL COMPONENTS
# ============================================================

class PersistentState(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.state = nn.Parameter(torch.randn(1, dim) * 0.02)
    
    def forward(self, x):
        s = self.state.expand(x.size(0), x.size(1), -1)
        return x + s

class WorldModelHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        return self.net(x)

class ValueHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1)
    
    def forward(self, x):
        return self.net(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config["EMBED_DIM"], 3 * config["EMBED_DIM"])
        self.c_proj = nn.Linear(config["EMBED_DIM"], config["EMBED_DIM"])
        self.n_head = config["HEADS"]
        self.n_embd = config["EMBED_DIM"]
        self.register_buffer("bias", torch.tril(torch.ones(config["BLOCK_SIZE"], config["BLOCK_SIZE"]))
                                     .view(1, 1, config["BLOCK_SIZE"], config["BLOCK_SIZE"]))

    def forward(self, x, ext_mem=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if ext_mem is not None:
            mem_k = ext_mem.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            mem_v = ext_mem.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            k = torch.cat([mem_k, k], dim=2)
            v = torch.cat([mem_v, v], dim=2)
            
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        mem_len = k.size(2) - T
        mask = self.bias[:, :, :T, :T]
        full_mask = torch.ones(1, 1, T, mem_len + T, device=x.device)
        full_mask[:, :, :, mem_len:] = mask
        
        att = att.masked_fill(full_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config["EMBED_DIM"], 4 * config["EMBED_DIM"])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config["EMBED_DIM"], config["EMBED_DIM"])
        self.dropout = nn.Dropout(config["DROPOUT"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["EMBED_DIM"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config["EMBED_DIM"])
        self.mlp = MLP(config)

    def forward(self, x, ext_mem=None):
        x = x + self.attn(self.ln1(x), ext_mem)
        x = x + self.mlp(self.ln2(x))
        return x

# --- The APEX Transformer ---
class ApexTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = CONFIG
        self.token_emb = nn.Embedding(CONFIG["vocab_size"], CONFIG["EMBED_DIM"])
        self.pos_emb = nn.Embedding(CONFIG["BLOCK_SIZE"], CONFIG["EMBED_DIM"])
        
        self.core_state = PersistentState(CONFIG["EMBED_DIM"])
        self.blocks = nn.ModuleList([Block(CONFIG) for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(CONFIG["EMBED_DIM"])
        
        self.lm_head = nn.Linear(CONFIG["EMBED_DIM"], CONFIG["vocab_size"])
        self.world_head = WorldModelHead(CONFIG["EMBED_DIM"])
        self.value_head = ValueHead(CONFIG["EMBED_DIM"])

    def forward(self, idx, targets=None, memory=None, num_paths=1):
        B, T = idx.size()
        
        if num_paths > 1 and self.training:
            idx = idx.repeat(num_paths, 1)
            if targets is not None: targets = targets.repeat(num_paths, 1)
            if memory is not None: memory = memory.repeat(num_paths, 1, 1)
            B = B * num_paths

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=DEVICE))
        x = tok + pos
        
        x = self.core_state(x)
        
        for block in self.blocks:
            x = block(x, ext_mem=memory)
            
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        next_state_pred = self.world_head(x)
        state_value = self.value_head(x)
        
        loss = None
        if targets is not None:
            # 1. Main Language Loss
            loss_lm = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
            # 2. World Model Loss
            pred = next_state_pred[:, :-1, :]
            target = x[:, 1:, :].detach() 
            loss_wm = F.mse_loss(pred, target)
            
            # 3. Value Head Loss
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none', ignore_index=0)
            token_loss = token_loss.view(B, T)
            loss_val = F.mse_loss(state_value.squeeze(), token_loss.detach())
            
            loss = loss_lm + (loss_wm * 0.1) + (loss_val * 0.1)
            
            if num_paths > 1:
                loss = loss / num_paths

        return logits, loss, x.detach()

# ============================================================
# 5. AUTO-TUNER
# ============================================================
class AutoTuner:
    def __init__(self, optimizer):
        self.opt = optimizer
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience = 0
    
    def step(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > 100: self.loss_history.pop(0)
        
        if loss < self.best_loss * 0.99:
            self.best_loss = loss
            self.patience = 0
            return "improve"
        else:
            self.patience += 1
        
        if self.patience > 20:
            self.patience = 0
            self._adjust_lr(0.8)
            return "stagnant"
        
        return "stable"
    
    def _adjust_lr(self, factor):
        for g in self.opt.param_groups:
            g['lr'] *= factor
            g['lr'] = max(g['lr'], 1e-6)
        logging.info(f"📉 AutoTuner: LR adjusted to {self.opt.param_groups[0]['lr']:.2e}")

# ============================================================
# 6. APEX CONTROLLER
# ============================================================
class ApexController:
    # 🟢 FIX: ACCEPT WORLD_SIZE IN INIT
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data = DataManager(rank)
        CONFIG["vocab_size"] = self.data.vocab_size
        
        self.model = ApexTransformer().to(DEVICE)
        
        if self.world_size > 1:
            self.ddp_model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        else:
            self.ddp_model = self.model
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG["LR"])
        self.memory = ApexMemory()
        self.tuner = AutoTuner(self.optimizer)
        
        self.load()

    def save(self, tag="latest"):
        if self.rank != 0: return
        state = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "config": CONFIG
        }
        torch.save(state, os.path.join(PATHS["DIR_CKPT"], f"{tag}.pt"))
        
    def load(self):
        path = os.path.join(PATHS["DIR_CKPT"], "latest.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["opt"])
            logging.info(">>> RESTORED APEX STATE")

    def run(self):
        self.model.train()
        
        for gen in range(CONFIG["GENERATIONS"]):
            total_loss = 0
            
            for step in range(CONFIG["CYCLES_PER_GEN"]):
                x, y = self.data.get_batch()
                if x is None: continue
                
                with torch.no_grad():
                    query_emb = self.model.token_emb(x).mean(dim=1)
                    mems = self.memory.read(query_emb)
                
                paths = CONFIG["INFERENCE_PATHS"]
                logits, loss, latents = self.ddp_model(x, y, memory=mems, num_paths=paths)
                
                if torch.isnan(loss):
                    logging.error("NaN Loss! Skipping step.")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["GRAD_CLIP"])
                self.optimizer.step()
                
                self.memory.write(latents.mean(dim=1), latents.mean(dim=1)) 
                
                total_loss += loss.item()
                
                if step % 50 == 0 and self.rank == 0:
                    logging.info(f"Gen {gen} | Step {step} | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / CONFIG["CYCLES_PER_GEN"]
            status = self.tuner.step(avg_loss)
            
            if status == "improve" and self.rank == 0:
                self.save(tag="best")
            
            self.save(tag="latest")

# ============================================================
# EXECUTION
# ============================================================
def main(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    # 🟢 FIX: PASS BOTH ARGUMENTS
    controller = ApexController(rank, world_size)
    controller.run()
    
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(main, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
    else:
        main(0, 1)
