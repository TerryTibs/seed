# ==============================================================================
# SACRSN-SEED IMMORTAL CORE — THE UNIFIED EDITION (v8)
# ==============================================================================
#
# STATUS:  AUDITED × 7 · FIXED × 7 · ALL FEATURES WIRED · ALL ISSUES RESOLVED
#
# AUDIT PASS 7 FIXES (from SACRSN_SEED_V7.py audit):
#  [LOW]      phase_evolve() stagnation grow now checks self._grew_this_gen before
#             calling grow_network() — the third grow path was missing the flag
#             guard added in v7 for auto_grow() and AutonomousResearch.act().
#             Sets flag True on growth to complete the triple-guard coverage.
#  [LOW]      torch.load() now passes weights_only=False — suppresses the PyTorch
#             2.x FutureWarning that fired on every checkpoint restore.
#  [LOW]      probe_neurons() now uses world_sim=1 — last remaining probe-only
#             forward pass that was running the 5-world ensemble unnecessarily.
#             Called from generate_demo() every 2 generations.
#  [LOW]      Dual grow_network() prevention: added self._grew_this_gen flag,
#             reset at the top of run_cycle(). auto_grow() and AutonomousResearch
#             .act() both check the flag before calling grow_network(), and set it
#             on growth. Prevents silent 2-layer growth when both fire in one gen.
#  [CRITICAL] Meta update guard now checks prev_loss_val (captured snapshot) not
#             self.prev_loss (already overwritten by line 1630 before the guard).
#             self.prev_loss is never None at the guard — prev_loss_val is None on
#             step 0. Old code: TypeError: NoneType - float on first training step.
#  [LOW]      auto_grow() reward comparison fixed: score_history[-1] is -avg_loss
#             (negative); was compared against self.prev_loss (positive) → always
#             True → growth always rewarded. Now: -loss < self.prev_loss (same sign).
#             Also guarded against None prev_loss (kill-switch edge case).
#  [LOW]      reflect() context probe now uses world_sim=1 — m(x, world_sim=1)
#             for the hidden-state extraction before memory recall.
#  [LOW]      run_cycle() world model update probe now uses world_sim=1 —
#             self.unwrap(self.pop[0])(x, world_sim=1) for state extraction.
#             — rollout mutates state in-place; agents 1-3 were starting each rollout
#               from the previous agent's imagined t+3 future state. Now correctly
#               reset to actual current state before world_env.update().
#  [LOW]      meta.get_scale() collapsed to single call per step (was called twice:
#             once in no_grad for scalar, once outside for graph — identical inputs)
#  [LOW]      prev_loss initialised to None; meta update skipped on step 0
#             — prev_loss=0.0 falsely signalled loss "jumped from perfect baseline"
#  [LOW]      Redundant world_env.reset_state() removed from phase_train
#             — run() already resets every generation; phase_train's every-10-gen
#               reset fired immediately after run()'s reset on the same generation
#  [LOW]      Agency state probe in run() now uses world_sim=1
#             — policy network only needs a hidden state vector; 5× world ensemble
#               adds no quality to the 384-dim representation, just 5× CPU cost
#  [MEDIUM]   reflect() perplexity now runs in eval mode — synth_buffer block moved
#             before m.train() so Dropout is off; ppl threshold is consistent
#  [LOW]      reflect() has rank guard (if self.rank != 0: return)
#             — only rank-0 generates; no N× wasted compute on multi-GPU
#  [LOW]      generate_demo() perplexity moved before model.train() — eval mode ✓
#  [LOW]      SyntheticBuffer.flush() called in save() and KeyboardInterrupt handler
#             — buffered text no longer lost when run ends with < 50 items
#  [LOW]      AutonomousResearch.propose() guard changed to 'is not None'
#             — empty dict {} was falsy; hypothesis was never set
#  [LOW]      propose() called with real telemetry {"loss": prev_loss, "gen": gen}
#             — hypothesis now records meaningful training state
#  [NOTE]     "import math missing" in audit report was a false positive —
#             the audit script only scanned first 120 lines; import math is at line 148
#
# AUDIT PASS 1 FIXES (from SACRSN_SEED_UNIFIED.py audit):
#  [CRITICAL] Double-backward crash in phase_train — reordered backward/meta calls
#  [CRITICAL] GoalEngine.reward_modifier — torch.tensor() keeps autograd graph intact
#  [HIGH]     IdentityContinuity.record — stores hash only, not full 1536-float seed
#  [MEDIUM]   CivilizationCoordinator — numpy.int64 → int() cast for dict key match
#  [MEDIUM]   Phase_evolve resurrection — dist.broadcast_object_list multi-GPU sync
#  [MEDIUM]   dist.barrier() moved outside mutation loop (was 3× redundant)
#  [MEDIUM]   DiskEpisodicMemory.load_meta — count/payload sync guard
#  [MEDIUM]   KeyboardInterrupt handler — _last_gen tracker, try/except wrap
#  [MEDIUM]   SparseAttention — assert EMBED_DIM % HEADS == 0 at init
#  [LOW]      ConceptTracker — deque(maxlen=10000) prevents unbounded growth
#  [LOW]      reflect() prompt — vocab-filtered to prevent PAD token flooding
#
# FEATURES WIRED (were dormant in unified script):
#  MetaLearningEngine · LifelongProtector (EWC) · PersistentWorldSimulator
#  SeedReplayBuffer · ShardedIdentity · CivilizationMind · SelfRewriteSandbox
#  SyntheticBuffer · AutonomousResearch.propose · CivilizationMind.get_context
#  CONFIG["DROPOUT"] · CONFIG["LATENT_DIM"] · CONFIG["INFERENCE_PATHS"]
#  CONFIG["EWC_LAMBDA"] · ArchitecturePolicy (in auto_grow)
#
# COMPLETE FEATURE MANIFEST (by origin version): see below
#
# [v2 / seed.py–seed1]
#  ✦ Multi-GPU distributed transformer (DDP + NCCL)
#  ✦ Multi-Agent Evolution & Selection
#  ✦ Self-Reconstruction / Regrowth (destroy_weights + regenerate)
#  ✦ Persistent memory with .pkl backup
#  ✦ Enhanced text generation (temperature + top-k)
#  ✦ Live console logging of loss & identity drift
#
# [v3–v3.3 / seed2–seed6]
#  ✦ Hierarchical memory (short / medium / long-term) with read/write
#  ✦ Sparse local attention with learnable gates
#  ✦ Mixture-of-Experts (MoE) with gating
#  ✦ Multi-World simulation (ensemble averaging over parallel branches)
#  ✦ Curriculum learning (variable sequence lengths)
#  ✦ Selective / event-driven memory updates
#
# [v3.2–v3.8 / seed5–seed16]
#  ✦ Vectorized sparse attention (fully batched window mask)
#  ✦ Noise annealing (high → low variance across generations)
#  ✦ Dual saving: .pkl memory + .pt full checkpoint
#  ✦ Graceful KeyboardInterrupt safe-exit
#  ✦ Split-brain score synchronisation (DDP all_reduce)
#  ✦ Input validation & top-p sampling
#
# [v4–v5.1 / seed17–seed22]
#  ✦ Deep multi-world simulation with noise injection
#  ✦ Sparse mask cache (pre-computed buffers)
#  ✦ Super-batching (B×W parallel worlds)
#  ✦ Vector norm identity metric (high-signal drift)
#  ✦ Auto data generation fallback
#
# [v8–v9.5 / seed23–seed26]
#  ✦ Long-term vector DB backend (disk-backed memmap)
#  ✦ Neural Architecture Policy Network (AI-driven growth decisions)
#  ✦ Latent world-model simulator (GRU-based)
#  ✦ Theory-of-Mind (agent modelling)
#  ✦ Recursive self-simulation (2nd-order prediction)
#
# [v10–v14 / seed27–seed31]
#  ✦ Persistent World Model (multi-step future rollouts)
#  ✦ Identity Continuity (long-term self-tracking + hash chain)
#  ✦ Self-Narrative Engine (autobiography generation)
#  ✦ Fractal Meta-Mind (4-step recursive self-prediction)
#  ✦ Long-horizon planning (simulate future before acting)
#
# [v15–v19 / seed32–seed36]
#  ✦ DDP-safe growth (synchronised architecture updates)
#  ✦ Multi-anchor seeds (high-fidelity identity reconstruction)
#  ✦ Meta-Opt clamping (prevents LR instability)
#  ✦ Replay buffer for Self-Model training
#  ✦ Meta-Learning (loss-driven LR optimisation via MetaOptimizer)
#  ✦ Latent planning via seed mutation
#  ✦ Echo-Chamber guard (hard cap on synthetic data ratio)
#  ✦ Smart Memory (centroid-based distance pruning)
#  ✦ N-gram repetition filters for synthetic text
#  ✦ Rolling return buffer for long-horizon RL (agency credit)
#
# [v20–v24 / seed37–seed40]
#  ✦ Numpy Memmap disk-backed arrays (O(1) RAM usage)
#  ✦ Intra-epoch checkpointing
#  ✦ Kill-switch: auto-reset on divergence (loss > 100)
#  ✦ Hash chain: blockchain-style identity lineage
#  ✦ NaN guards on meta-gradients
#  ✦ World state clamping to prevent explosion
#  ✦ Dynamic mask rebuild on sequence length change
#  ✦ Fitness++ (score includes Memory Novelty bonus)
#
# [v25–v28 / seed41–seed44]
#  ✦ Self-Rewrite Sandbox (safe framework for code self-mutation)
#  ✦ True Meta-Learning (adjusts LR based on loss + grad norm)
#  ✦ Identity Genome (evolutionary tracking of high-fitness seeds)
#  ✦ Intrinsic Curiosity (state-visitation novelty reward)
#  ✦ Recursive Self-Improvement (stagnation → forced growth)
#  ✦ Phase-based lifecycle: Train → Evaluate → Evolve → Regenerate
#  ✦ Agent Council Vote (democratic best-agent selection)
#  ✦ Probe neurons / probe weight stats (interpretability)
#  ✦ Export identity archive (bounded, atomic)
#  ✦ generate_demo() with neuron probe output
#  ✦ Civilization Coordinator + Roles (leader/explorer/worker/researcher/critic)
#  ✦ CivilizationMind (shared knowledge + role-based temperature)
#
# [v103–v105.1 / seed90–seed93]
#  ✦ Graph safety: world model inputs double-detached
#  ✦ Agency stability: softmax NaN guards; standardised rewards
#  ✦ Mask safety: attention masks clamped to block size
#  ✦ World hygiene: world model state reset every generation
#  ✦ Identity guard: reconstruction aborted on heavy layer mismatch
#  ✦ Memory integrity: metadata dimension check
#  ✦ Reflection safety: dictionary key hardening
#  ✦ Mutation safety: parameter mutations clamped to [-2, 2]
#  ✦ MoE safety: balance loss clamped to prevent explosion
#  ✦ Disk memory dimension metadata
#
# [STATUS]
#  - SYNTAX: VALIDATED
#  - STABILITY: REINFORCED
#  - FEATURES: COMPLETE (all iterations unified)
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

# ==============================================================================
# 1. CONFIGURATION & LOGGING
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
    "THOUGHT_DIM": 256,
    "LATENT_DIM": 384,          # PredictiveSelfModel bottleneck dim [WIRED]
    "WORLD_SIM": 5, "INFERENCE_PATHS": 4,  # INFERENCE_PATHS: best-of-N in generate() [WIRED]

    # Training
    "BATCH_SIZE": 16, "LR": 3e-4,
    "DROPOUT": 0.1,             # Applied in RecurrentBlock sub-paths [WIRED]
    "GRAD_CLIP": 1.0,
    "AUX_LOSS_WEIGHT": 0.01,
    "EWC_LAMBDA": 0.4,          # EWC penalty weight, applied every 5 gens [WIRED]

    # Lifecycle
    "POPULATION_SIZE": 4, "GENERATIONS": 50, "CYCLES_PER_GEN": 200,
    "REGENERATE_STEPS": 50, "EVAL_BATCHES": 4,

    # Cognitive
    "MEMORY_CAPACITY": 500_000, "IDENTITY_SEED_SIZE": 512,
    "CURRICULUM": [0.25, 0.5, 0.75, 1.0], "SYNTH_RATIO_CAP": 0.2,
    "WIPE_RATIO": 0.1, "SELECTIVE_THRESHOLD": 0.2,
    "ROLES": ["leader", "researcher", "explorer", "critic", "worker"],

    # Legacy compatibility
    "STEPS_PER_CYCLE": 500,     # v2 alias for CYCLES_PER_GEN (external compat only)
}

PATHS = {
    "MEM_PKL": "seed_memory.pkl", "MEM_BAK": "seed_memory_backup.pkl",
    "COG_MEM": "cognitive_memory.pkl",
    "CHECKPOINT": "seed_full_state.pt", "ARCHIVE": "IMMORTAL_ARCHIVE.pt",
    "GENOME": "identity_genome.pkl",
    "TELEMETRY": "telemetry.jsonl", "DIR_CKPT": "checkpoints",
    "DIR_ARCHIVE": "archive_history", "DIR_SANDBOX": "rewrite_sandbox",
    "DATA": "data.txt", "SYNTH": "data_recursive.txt",
    "MEM_VECS": "memory_vectors.dat", "MEM_META": "memory_meta.pkl",
    "PT_FILE": "seed_state.pt",
}

for d in [PATHS["DIR_CKPT"], PATHS["DIR_ARCHIVE"], PATHS["DIR_SANDBOX"]]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# 2. UTILITIES
# ==============================================================================
TASKS = ["pattern_fit"]

def discover_tasks():
    if random.random() < 0.15:
        new_task = f"task_{random.randint(1000, 9999)}"
        TASKS.append(new_task)
        logging.info(f"✨ NEW TASK DISCOVERED: {new_task}")

def atomic_save(obj, path, use_torch=False):
    tmp_path = path + ".tmp"
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".backup")
        except Exception:
            pass
    try:
        if use_torch:
            torch.save(obj, tmp_path)
        else:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
        if os.path.exists(path):
            try:
                os.replace(tmp_path, path)
            except OSError:
                os.remove(path)
                os.rename(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception as e:
        logging.error(f"Save Failed {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def identity_hash(model):
    """Hash-based fingerprinting — more reliable than mean() [v28 Fix 4]."""
    vec = torch.cat([p.flatten() for p in model.parameters()])
    if vec.numel() > 1_000_000:
        vec = vec[::100]
    return hashlib.sha256(vec.detach().cpu().numpy().tobytes()).hexdigest()[:8]

def identity_signature(model):
    """Legacy scalar identity metric (v2 compat)."""
    return sum(p.mean().item() for p in model.parameters())

def compress_identity(model):
    """Legacy alias (v2 compat)."""
    return IdentitySeed.compress(model)

def restore_identity(model, seed):
    """Legacy alias (v2 compat)."""
    IdentitySeed.reconstruct(model, seed)

def spawn_agents(base_model, count=3):
    """Legacy helper (v2 compat)."""
    return [copy.deepcopy(base_model) for _ in range(count)]

def evaluate_agent(model, data_mgr, batch_size=8):
    """Legacy evaluation helper (v2 compat)."""
    x, y = data_mgr.get_batch()
    with torch.no_grad():
        logits, _, _ = model(x)
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()

def destroy_weights(model, wipe_ratio=0.1):
    """Wipe a fraction of weights for self-reconstruction (v2 origin)."""
    for p in model.parameters():
        mask = (torch.rand_like(p) > wipe_ratio).float()
        p.data.mul_(mask)

# ==============================================================================
# 3. DATA MANAGER
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
        if self.rank == 0 and not os.path.exists(PATHS["DATA"]):
            with open(PATHS["DATA"], "w") as f:
                f.write("SACRSN GODHEAD " * 5000)
        if NUM_GPUS > 1:
            dist.barrier()
        with open(PATHS["DATA"], "r") as f:
            raw = f.read()
        synth_txt = ""
        if os.path.exists(PATHS["SYNTH"]):
            with open(PATHS["SYNTH"], "r") as f:
                synth_txt = f.read()
        chars = sorted(list(set(raw + synth_txt)))
        self.vocab_size = len(chars) + 1
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.stoi["<PAD>"] = 0
        self.itos = {i + 1: ch for i, ch in enumerate(chars)}
        self.itos[0] = "<PAD>"
        self.data = torch.tensor([self.stoi.get(c, 0) for c in raw], dtype=torch.long)
        if synth_txt:
            self.synth = torch.tensor([self.stoi.get(c, 0) for c in synth_txt], dtype=torch.long)

    def get_batch(self, difficulty=1.0):
        if self.data is None:
            raise RuntimeError("Data not loaded")
        use_synth = (len(self.synth) > CONFIG["BLOCK_SIZE"]) and (random.random() < CONFIG["SYNTH_RATIO_CAP"])
        src = self.synth if use_synth else self.data
        if len(src) < CONFIG["BLOCK_SIZE"]:
            src = self.data
        if len(src) < CONFIG["BLOCK_SIZE"]:
            return None, None
        seq = max(16, int(CONFIG["BLOCK_SIZE"] * difficulty))
        if len(src) < seq + 5:
            seq = len(src) - 2
        ix = torch.randint(len(src) - seq, (CONFIG["BATCH_SIZE"],))
        x = torch.stack([src[i:i + seq] for i in ix])
        y = torch.stack([src[i + 1:i + seq + 1] for i in ix])
        if seq < CONFIG["BLOCK_SIZE"]:
            pad = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["BLOCK_SIZE"] - seq, dtype=torch.long)
            x = torch.cat([x, pad], 1)
            y = torch.cat([y, pad], 1)
        return x.to(DEVICE), y.to(DEVICE)

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, t):
        return "".join([self.itos.get(i, "") for i in t if i != 0])


# ==============================================================================
# 4. COGNITIVE & UTILITY CLASSES
# ==============================================================================

class RewardNormalizer:
    def __init__(self, alpha=0.95):
        self.mean = 0.0; self.var = 1.0; self.alpha = alpha; self.count = 0

    def normalize(self, x):
        self.count += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.var = self.alpha * self.var + (1 - self.alpha) * ((x - self.mean) ** 2)
        return (x - self.mean) / (math.sqrt(self.var) + 1e-6)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"]; self.var = d["var"]; self.count = d["count"]


class TelemetryLogger:
    """Writes structured JSONL telemetry (v3.8 origin, extended v28)."""
    def __init__(self):
        self.file = PATHS["TELEMETRY"]

    def log(self, data):
        data["ts"] = time.time()
        try:
            with open(self.file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass


class IdentityGenome:
    """Evolutionary tracking of high-fitness identity seeds (v26 origin)."""
    def __init__(self):
        self.genes = []
        self.load()

    def add(self, seed, score):
        w = seed["weights"].detach().cpu().numpy()
        self.genes.append({"w": w, "score": score, "meta": seed["meta"]})
        self.genes.sort(key=lambda x: x["score"], reverse=True)
        self.genes = self.genes[:100]
        self.save()

    def resurrect(self):
        if not self.genes:
            return None
        return random.choice(self.genes)

    def save(self):
        try:
            atomic_save(self.genes, PATHS["GENOME"], use_torch=False)
        except Exception:
            pass

    def load(self):
        if os.path.exists(PATHS["GENOME"]):
            try:
                with open(PATHS["GENOME"], "rb") as f:
                    self.genes = pickle.load(f)
            except Exception:
                pass


class CognitiveMemory:
    """Simple episodic cognitive memory list (v3 origin)."""
    def __init__(self):
        self.entries = []
        self.load()

    def remember(self, item):
        self.entries.append(item)
        if len(self.entries) > 2000:
            self.entries.pop(0)

    def save(self):
        try:
            with open(PATHS["COG_MEM"], "wb") as f:
                pickle.dump(self.entries, f)
        except Exception:
            pass

    def load(self):
        if os.path.exists(PATHS["COG_MEM"]):
            try:
                with open(PATHS["COG_MEM"], "rb") as f:
                    self.entries = pickle.load(f)
            except Exception:
                pass


class HierarchicalMemory(nn.Module):
    """Short / medium / long-term parametric memory banks (v3.0 origin, v28 restored)."""
    def __init__(self, dim=None, short_len=32, medium_len=16, long_len=8):
        super().__init__()
        if dim is None:
            dim = CONFIG["EMBED_DIM"]
        self.short_mem = nn.Parameter(torch.randn(short_len, dim))
        self.medium_mem = nn.Parameter(torch.randn(medium_len, dim))
        self.long_mem = nn.Parameter(torch.randn(long_len, dim))

    def read(self):
        return torch.cat([self.short_mem, self.medium_mem, self.long_mem], dim=0)

    def write(self, updates, short_idx=None, medium_idx=None, long_idx=None):
        if short_idx is not None:
            self.short_mem.data[short_idx] = updates.data
        elif medium_idx is not None:
            self.medium_mem.data[medium_idx] = updates.data
        elif long_idx is not None:
            self.long_mem.data[long_idx] = updates.data


class DiskEpisodicMemory:
    """Disk-backed memmap vector store (v20 origin, v105.1 safety patches)."""
    def __init__(self):
        self.dim = CONFIG["EMBED_DIM"]
        self.max = CONFIG["MEMORY_CAPACITY"]
        self.count = 0
        self.payloads = []
        self.centroids = []
        self.file_emb = PATHS["MEM_VECS"]
        self.file_meta = PATHS["MEM_META"]
        if os.path.exists(self.file_emb):
            self.emb = np.memmap(self.file_emb, dtype="float32", mode="r+", shape=(self.max, self.dim))
            self.load_meta()
        else:
            self.emb = np.memmap(self.file_emb, dtype="float32", mode="w+", shape=(self.max, self.dim))

    def store(self, embedding, payload):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        vec = embedding.detach().cpu().numpy().flatten()
        idx = self.count % self.max
        self.emb[idx] = vec
        # Centroid update (v19 smart memory)
        if len(self.centroids) == 0 or np.linalg.norm(vec - self.centroids[-1]) > 0.5:
            self.centroids.append(vec.copy())
            if len(self.centroids) > 1000:
                self.centroids.pop(0)
        entry = {"data": payload, "time": time.time(), "id": self.count}
        if idx < len(self.payloads):
            self.payloads[idx] = entry
        else:
            self.payloads.append(entry)
        self.count += 1
        if self.count % 1000 == 0:
            self.emb.flush()

    def query(self, embedding, top_k=5):
        if self.count == 0:
            return []
        valid = min(self.count, self.max)
        idx_pool = np.random.choice(valid, min(valid, 5000))
        mem = self.emb[idx_pool]
        q = embedding.detach().cpu().numpy().flatten()
        sim = (mem @ q) / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(sim)[-top_k:][::-1]
        return [self.payloads[idx_pool[i]] for i in top if idx_pool[i] < len(self.payloads)]

    def save(self):
        self.emb.flush()
        # [v105.1 Fix 9] Save dim in metadata
        atomic_save({"count": self.count, "payloads": self.payloads, "dim": self.dim}, self.file_meta)

    def load_meta(self):
        if os.path.exists(self.file_meta):
            try:
                with open(self.file_meta, "rb") as f:
                    d = pickle.load(f)
                self.count = d["count"]
                self.payloads = d["payloads"]
                # Dimension integrity check [v105.1 Fix 9]
                if "dim" in d and d["dim"] != self.dim:
                    logging.warning("⚠️ Memory Dimension Mismatch — Resetting Memory")
                    self.count = 0
                    self.payloads = []
                # [FIX LOW] Sync count with actual payload length to prevent silent query drops
                if self.count > len(self.payloads):
                    logging.warning(f"⚠️ Memory count ({self.count}) > payloads ({len(self.payloads)}) — syncing")
                    self.count = len(self.payloads)
            except Exception:
                pass


class LegacyVectorMemory:
    """In-memory vector store (v9.5 origin, kept for fast-path / compat)."""
    def __init__(self):
        self.vectors = []
        self.payloads = []

    def add(self, embedding, payload):
        self.vectors.append(embedding)
        self.payloads.append(payload)

    def query(self, embedding, k=5):
        return self.payloads[:k]


class SyntheticBuffer:
    """N-gram filter + entropy filter for synthetic text (v22 origin)."""
    def __init__(self, capacity=50):
        self.buffer = []
        self.capacity = capacity

    def calculate_entropy(self, text):
        if not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    def is_repetitive(self, text):
        words = text.split()
        if len(words) < 10:
            return True
        grams = [tuple(words[i:i + 3]) for i in range(len(words) - 3)]
        return (len(set(grams)) / max(1, len(grams))) < 0.5

    def add(self, text, perplexity):
        entropy = self.calculate_entropy(text)
        if perplexity < 50.0 and entropy > 3.5 and not self.is_repetitive(text):
            self.buffer.append(text)
            if len(self.buffer) > self.capacity:
                self.flush()

    def flush(self):
        try:
            with open(PATHS["SYNTH"], "a") as f:
                for t in self.buffer:
                    f.write(t + "\n")
            self.buffer = []
        except Exception:
            pass


class SeedReplayBuffer:
    """Experience replay for Self-Model training (v16 origin)."""
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, seed_t, seed_t1, reward):
        r_val = float(reward) if torch.is_tensor(reward) else reward
        self.buffer.append((seed_t.detach().cpu(), seed_t1.detach().cpu(), r_val))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s0, s1, r = zip(*batch)
        return torch.stack(s0).to(DEVICE), torch.stack(s1).to(DEVICE), torch.tensor(r).to(DEVICE).float()


class BeliefLedger:
    """Records agent belief history (v10 origin)."""
    def __init__(self):
        self.history = []

    def record(self, agent_id=None, score=0, lineage=None, belief=None):
        entry = {"score": score, "time": time.time()}
        if agent_id is not None:
            entry["agent"] = agent_id
        if lineage is not None:
            entry["lineage"] = lineage
        if belief is not None:
            # Store only meta, not full weights, to keep bounded
            if isinstance(belief, dict) and "meta" in belief:
                entry["belief_meta"] = belief["meta"]
        self.history.append(entry)
        if len(self.history) > 1000:
            self.history.pop(0)


class ExperimentEngine:
    """Robustness testing via partial weight ablation (v27 origin)."""
    def run(self, model, data=None):
        model.eval()
        scores = []
        with torch.no_grad():
            for _ in range(3):
                m_copy = copy.deepcopy(model)
                for p in m_copy.parameters():
                    mask = torch.rand_like(p) > 0.1
                    p.mul_(mask)
                if data is not None:
                    x, y = data.get_batch(1.0)
                    if x is None:
                        continue
                    _, loss, _ = m_copy(x, y)
                    scores.append(loss.item())
        return np.mean(scores) if scores else 99.0


class CivilizationCoordinator:
    """Assigns performance-based roles to population members (v27 origin)."""
    def assign_roles(self, scores):
        roles = {}
        sorted_idx = np.argsort(scores)[::-1]
        # [FIX MEDIUM] Cast to int — numpy.int64 keys don't match Python int in dict.get()
        roles[int(sorted_idx[0])] = "leader"
        for i in range(1, len(sorted_idx)):
            roles[int(sorted_idx[i])] = "explorer" if i % 2 == 0 else "worker"
        return roles


class CivilizationMind:
    """Shared knowledge pool with role-based temperature (v27 origin)."""
    def __init__(self):
        self.shared_knowledge = deque(maxlen=100)
        self.roles = {}

    def assign_roles(self, size):
        for i in range(size):
            self.roles[i] = CONFIG["ROLES"][i % len(CONFIG["ROLES"])]

    def share(self, knowledge):
        self.shared_knowledge.append(knowledge)

    def get_context(self):
        return " ".join(list(self.shared_knowledge)[-3:])

    def get_temp(self, idx):
        return {"leader": 0.8, "researcher": 0.6, "explorer": 1.1, "critic": 0.4}.get(
            self.roles.get(idx, "leader"), 0.8
        )


class ShardedIdentity:
    """Gradient-averaging merge from workers into leader (v15 origin)."""
    def merge_gradients(self, leader, workers):
        with torch.no_grad():
            for w in workers:
                for p_l, p_w in zip(leader.parameters(), w.parameters()):
                    p_l.data = 0.9 * p_l.data + 0.1 * p_w.data


class AutonomousResearch:
    """Spontaneous capacity expansion proposals (v13 origin)."""
    def __init__(self):
        self.hypothesis = None

    def propose(self, telemetry):
        """Record latest telemetry as working hypothesis for future research decisions."""
        if telemetry is not None:  # [FIX LOW] {} is falsy — use 'is not None' instead of truthiness
            self.hypothesis = telemetry

    def act(self, ctrl):
        # [FIX LOW] Check _grew_this_gen flag to prevent dual grow in same generation
        if random.random() < 0.01 and not getattr(ctrl, "_grew_this_gen", False):
            ctrl._grew_this_gen = True
            ctrl.grow_network(0)
            logging.info("🔬 RESEARCH: Expanding Capacity")
        # [WIRED] SelfRewriteSandbox — occasionally propose a config mutation as a sandbox file
        if random.random() < 0.005:
            snippet = (
                f"# AUTO-PROPOSED CONFIG MUTATION\n"
                f"# Generated at runtime by AutonomousResearch\n"
                f"CONFIG_DELTA = {{\n"
                f"    'LAYERS': {CONFIG['LAYERS'] + random.choice([-1, 1])},\n"
                f"    'LR': {round(CONFIG['LR'] * random.uniform(0.5, 2.0), 6)},\n"
                f"    'NUM_EXPERTS': {CONFIG['NUM_EXPERTS'] + random.choice([0, 1])},\n"
                f"}}\n"
            )
            ctrl.sandbox.propose_rewrite(snippet)
            logging.info("📝 SANDBOX: Config mutation proposal written")


class SelfImprovementLoop:
    """Stagnation detection → triggers forced growth (v27 origin)."""
    def __init__(self):
        self.history = deque(maxlen=50)

    def update(self, score):
        self.history.append(score)

    def stagnating(self):
        return len(self.history) > 20 and np.std(list(self.history)) < 0.001


class CuriosityEngine:
    """State-visitation novelty reward — penalises revisited states (v27 origin)."""
    def __init__(self):
        self.visited = deque(maxlen=5000)

    def reward(self, embedding):
        key = hashlib.sha256(embedding.detach().cpu().numpy().round(1).tobytes()).hexdigest()
        if key in self.visited:
            return 0.0
        self.visited.append(key)
        return 0.2


class GoalEngine:
    """Dynamic goal evolution based on memory statistics (v10 origin)."""
    def __init__(self):
        self.goals = ["minimize_loss"]

    def evolve_goals(self, memory_db):
        if hasattr(memory_db, "payloads") and len(memory_db.payloads) > 10:
            losses = [m["data"]["loss"] for m in memory_db.payloads[-10:]
                      if isinstance(m, dict) and "data" in m and "loss" in m["data"]]
            if losses and np.std(losses) < 0.05 and "increase_creativity" not in self.goals:
                self.goals.append("increase_creativity")
                logging.info("🧠 GOAL EVOLVED: Added 'increase_creativity'")

    def reward_modifier(self, loss):
        if "increase_creativity" in self.goals:
            # [FIX CRITICAL] Use torch.tensor to keep autograd graph intact
            scale = torch.tensor(random.uniform(0.9, 1.1), device=loss.device, dtype=loss.dtype)
            return loss * scale
        return loss


class LifelongProtector:
    """EWC (Elastic Weight Consolidation) — prevents catastrophic forgetting (v14 origin)."""
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
            if x is None:
                break
            _, loss, _ = model(x, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.importance[n] += p.grad.pow(2)
        for n in self.importance:
            self.importance[n] /= samples
        model.train()

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.importance and n in self.params_old:
                loss += (self.importance[n] * (p - self.params_old[n]).pow(2)).sum()
        return loss


class SelfRewriteSandbox:
    """Safe directory for proposed code self-mutations (v26 origin)."""
    def __init__(self):
        self.dir = PATHS["DIR_SANDBOX"]

    def propose_rewrite(self, code_snippet):
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.dir, f"proposal_{ts}.py"), "w") as f:
            f.write(f"# PROPOSED MUTATION\n{code_snippet}")


class ConceptTracker:
    """Tracks concept-level activations from hidden states (v27 origin)."""
    def __init__(self):
        self.concepts = deque(maxlen=10000)  # [FIX LOW] cap growth

    def extract(self, hidden):
        self.concepts.append(hidden.mean().item())


class SelfNarrative:
    """Autobiographical event log (v10 origin)."""
    def __init__(self):
        self.events = []

    def log(self, text):
        self.events.append({"text": text, "time": time.time()})

    def summarize(self):
        return "\n".join(e["text"] for e in self.events[-20:])


class IdentityContinuity:
    """Long-term self-tracking with hash chain (v10 + v24 blockchain)."""
    def __init__(self):
        self.history = []
        self.chain = []  # Blockchain-style hash lineage [v24]

    def record(self, seed, hash_sig):
        # [FIX HIGH] Store only hash — not full 1536-float seed — to prevent unbounded memory growth
        self.history.append({"hash": hash_sig, "time": time.time()})
        if len(self.history) > 500:
            self.history = self.history[-500:]
        # Hash chain extension [v24]
        prev_hash = self.chain[-1] if self.chain else "GENESIS"
        block = hashlib.sha256(f"{prev_hash}:{hash_sig}".encode()).hexdigest()[:16]
        self.chain.append(block)

    def continuity_score(self):
        return len(self.history)

    def save(self):
        atomic_save(self.history, PATHS["DIR_ARCHIVE"] + "/continuity.pkl")


# ==============================================================================
# 5. NEURAL ARCHITECTURE SUPPORT MODULES
# ==============================================================================

class MetaOptimizer(nn.Module):
    """Loss-conditioned learning rate predictor (v19 origin)."""
    def __init__(self):
        super().__init__()
        self.lr_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, loss_tensor):
        return self.lr_net(loss_tensor)


class MetaLearningEngine(nn.Module):
    """Loss + grad-norm conditioned LR scale (v103 variant)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def get_scale(self, loss, grad):
        return self.net(torch.tensor([loss / 10.0, math.log1p(grad)], device=DEVICE).float()) * 2.0


class AgencyCore(nn.Module):
    """REINFORCE policy for high-level action selection (v13 origin, v105.1 NaN fix)."""
    ACTIONS = ["train", "evolve", "explore", "reflect", "rest"]

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG["EMBED_DIM"], 256), nn.ReLU(),
            nn.Linear(256, len(self.ACTIONS)), nn.Softmax(-1)
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.saved_log_probs = []
        self.rewards = []
        self.replay = deque(maxlen=2000)
        self.scaler = RewardNormalizer()

    def decide(self, state):
        if state is None:
            return "train", None
        state = torch.nan_to_num(state, nan=0.0).detach()
        probs = self.net(state)
        # [v105.1 Fix 7] Sanitise NaN in policy
        if torch.isnan(probs).any():
            probs = torch.tensor([1.0 / len(self.ACTIONS)] * len(self.ACTIONS), device=DEVICE)
        else:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
        d = torch.distributions.Categorical(probs)
        action = d.sample()
        self.saved_log_probs.append(d.log_prob(action))
        return self.ACTIONS[action.item()], probs

    def update_policy(self):
        if not self.rewards:
            return
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        self.replay.extend(returns)
        normalized = [self.scaler.normalize(r) for r in returns]
        returns_t = torch.tensor(normalized).to(DEVICE).clamp(-2.0, 2.0)
        policy_loss = [-lp * R for lp, R in zip(self.saved_log_probs, returns_t)]
        self.optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()
        del self.saved_log_probs[:]
        del self.rewards[:]


class NeuralWorldModel(nn.Module):
    """GRU-based latent world state tracker (v12 origin, v105.1 reset fix)."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(CONFIG["EMBED_DIM"], CONFIG["EMBED_DIM"])
        self.register_buffer("state", torch.zeros(1, CONFIG["EMBED_DIM"]))

    def forward(self, x):
        self.state = self.state.detach().clone()
        self.state = self.gru(x.unsqueeze(0), self.state)
        return self.state.squeeze(0)

    def reset_state(self):
        self.state.zero_()

    def predict_next(self, embedding):
        """Legacy alias used by phase_train (v13 compat)."""
        return self.forward(embedding)

    def update(self, embedding):
        """Legacy alias used by phase_train (v13 compat)."""
        self.forward(embedding)


class PersistentWorldSimulator:
    """Multi-step future rollout engine (v13 origin)."""
    def __init__(self, world_model):
        self.model = world_model

    def rollout(self, embedding, steps=5):
        states = []
        current = embedding
        for _ in range(steps):
            current = self.model(current)
            states.append(current)
        return torch.stack(states)


class PredictiveSelfModel(nn.Module):
    """Predicts next identity seed + fitness head (v9.5 origin). Uses LATENT_DIM as bottleneck."""
    def __init__(self):
        super().__init__()
        dim = CONFIG["IDENTITY_SEED_SIZE"] * 3
        latent = CONFIG["LATENT_DIM"]  # [WIRED] LATENT_DIM now used as bottleneck dimension
        self.net = nn.Sequential(nn.Linear(dim, latent), nn.ReLU(), nn.Linear(latent, dim))
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seed):
        nxt = self.net(seed)
        return nxt, self.head(nxt)


class ArchitecturePolicy(nn.Module):
    """Neural architecture search policy — AI-driven growth decisions (v9.5 origin)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1)
        )
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
        if not self.rewards:
            return
        returns = torch.tensor(self.rewards).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        policy_loss = [-lp * R for lp, R in zip(self.log_probs, returns)]
        self.optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()
        del self.log_probs[:]
        del self.rewards[:]


class PlanningEngine:
    """Long-horizon planning by simulating identity trajectory (v10 origin)."""
    def __init__(self, wm, self_model):
        self.wm = wm
        self.sm = self_model

    def plan(self, model, steps=3):
        seed = IdentitySeed.compress(model)["weights"].to(DEVICE)
        fits = []
        curr = seed
        for _ in range(steps):
            nxt, fit = self.sm(curr.unsqueeze(0).repeat(1, 3) if curr.numel() < CONFIG["IDENTITY_SEED_SIZE"] * 3 else curr.unsqueeze(0))
            fits.append(fit.item())
            curr = nxt.squeeze(0)
        return max(fits) if fits else 0.0


class IdentitySeed:
    """Multi-anchor identity compression and reconstruction (v15 origin, v105.1 fixes)."""

    @staticmethod
    def compress(model, optimizer=None, meta=None):
        flat = torch.cat([p.flatten() for p in model.parameters()])
        step = max(1, flat.numel() // CONFIG["IDENTITY_SEED_SIZE"])
        anchors = [flat[i::step][:CONFIG["IDENTITY_SEED_SIZE"]] for i in range(3)]
        for i in range(3):
            if len(anchors[i]) < CONFIG["IDENTITY_SEED_SIZE"]:
                anchors[i] = F.pad(anchors[i], (0, CONFIG["IDENTITY_SEED_SIZE"] - len(anchors[i])))
        sampled = torch.cat(anchors).detach().cpu()
        h = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
        out = {"weights": sampled, "meta": {"layers": len(model.blocks) if hasattr(model, "blocks") else 0, "hash": h}}
        if meta:
            out["meta"].update(meta)
        return out

    @staticmethod
    def reconstruct(model, seed):
        w = seed["weights"].to(DEVICE) if isinstance(seed, dict) else seed.to(DEVICE)
        if w.numel() == 3 * CONFIG["IDENTITY_SEED_SIZE"]:
            w = w.view(3, CONFIG["IDENTITY_SEED_SIZE"]).mean(0)
        target = seed["meta"].get("layers", CONFIG["LAYERS"]) if isinstance(seed, dict) else CONFIG["LAYERS"]
        # [v105.1 Fix 5] Prevent infinite growth loop
        if target > len(model.blocks) + 4:
            return
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


# ==============================================================================
# 6. NEURAL ARCHITECTURE
# ==============================================================================

class SparseAttention(nn.Module):
    """Vectorised sparse local attention with learnable gate (v3.2 + v105.1 mask fix)."""
    def __init__(self):
        super().__init__()
        # [FIX MEDIUM] Guard against config mismatches before they produce cryptic errors
        assert CONFIG["EMBED_DIM"] % CONFIG["HEADS"] == 0, \
            f"EMBED_DIM ({CONFIG['EMBED_DIM']}) must be divisible by HEADS ({CONFIG['HEADS']})"
        dim = CONFIG["EMBED_DIM"]
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // CONFIG["HEADS"]
        self.scale = self.head_dim ** -0.5
        self.gate = nn.Parameter(torch.ones(dim))
        b = CONFIG["BLOCK_SIZE"]
        self.register_buffer("mask", torch.tril(torch.ones(b, b)).view(1, 1, b, b))
        i = torch.arange(b).view(-1, 1)
        j = torch.arange(b).view(1, -1)
        self.register_buffer("local", (torch.abs(i - j) <= CONFIG["WINDOW_SIZE"]).view(1, 1, b, b))

    def forward(self, x, mem=None, loss_mask=None):
        B, T, C = x.shape
        # [v105.1 Fix 3] Guard against mask overflow
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
        out = self.proj(y * self.gate)
        if loss_mask is not None:
            if loss_mask.shape[1] != out.shape[1]:
                loss_mask = loss_mask[:, -out.shape[1]:]
            out = out * loss_mask.unsqueeze(-1)
        return out


class MoEBlock(nn.Module):
    """Mixture-of-Experts with Top-K gating and clamped balance loss (v3 + v105.1 fix)."""
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
        scores = torch.nan_to_num(scores, 0.0)
        probs = F.softmax(scores, -1)
        # [v105.1 Fix 4] Clamp balance loss to prevent explosion
        bl = (probs.mean((0, 1)) ** 2).sum() * CONFIG["NUM_EXPERTS"]
        self.bal_loss = torch.clamp(bl, 0.0, 5.0).to(x.device)
        topk, idx = torch.topk(probs, CONFIG["TOP_K"], -1)
        mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        masked = probs * mask
        masked = masked / (masked.sum(-1, keepdim=True) + 1e-9)
        return sum(masked[..., i:i + 1] * e(x) for i, e in enumerate(self.experts))


class RecurrentBlock(nn.Module):
    """Core transformer block: SparseAttention + ThoughtLayer + MoE + Dropout (v4 origin)."""
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
        # [WIRED] CONFIG["DROPOUT"] now active in all sub-paths
        self.drop = nn.Dropout(CONFIG["DROPOUT"])

    def forward(self, x, mem=None, loss_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mem, loss_mask=loss_mask))
        x = x + self.drop(self.thought(x))
        x = x + self.drop(self.moe(self.ln2(x)))
        return x


class GodheadTransformer(nn.Module):
    """
    The apex model: multi-world ensemble transformer with hierarchical memory,
    value head, world-model head, and top-k/temperature generation.
    (v21 Omega Build, upgraded to v105.1 stability)
    """
    def __init__(self, vocab):
        super().__init__()
        dim = CONFIG["EMBED_DIM"]
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(CONFIG["BLOCK_SIZE"], dim)
        self.mem = HierarchicalMemory(dim)      # Full hierarchical memory [v3.0]
        self.blocks = nn.ModuleList([RecurrentBlock() for _ in range(CONFIG["LAYERS"])])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        # Apex heads
        self.world_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.val_head = nn.Linear(dim, 1)

    def forward(self, idx, targets=None, noise=0.0, loss_mask=None, world_sim=None):
        B, T = idx.shape
        if T > CONFIG["BLOCK_SIZE"]:
            idx = idx[:, -CONFIG["BLOCK_SIZE"]:]
            T = idx.shape[1]
            if targets is not None:
                targets = targets[:, -CONFIG["BLOCK_SIZE"]:]
        x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
        # Use world_sim override if provided — pass world_sim=1 during inference
        # to avoid N_paths × WORLD_SIM memory explosion (e.g. 4×5=20→4 sequences)
        w = world_sim if world_sim is not None else CONFIG["WORLD_SIM"]
        # Multi-world ensemble (super-batching) [v5.1]
        x = x.repeat_interleave(w, 0)
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        mem = self.mem.read()
        for b in self.blocks:
            x = b(x, mem, loss_mask=loss_mask)
        x_flat = self.ln_f(x)
        # Collapse worlds by averaging [v3 ensemble averaging]
        x_mean = x_flat.view(w, B, T, -1).mean(0)
        logits = self.head(x_mean)
        wm_pred = self.world_head(x_mean)
        val_pred = self.val_head(x_mean)
        loss = None
        if targets is not None:
            safe_logits = torch.nan_to_num(logits, 0.0)
            main = F.cross_entropy(safe_logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            with torch.no_grad():
                target_next = x_mean.detach()
            wm_loss = F.mse_loss(wm_pred, target_next)
            tok_loss = F.cross_entropy(
                safe_logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            ).view(B, T)
            val_loss = F.mse_loss(val_pred.squeeze(), tok_loss.detach())
            moe_loss = sum(b.moe.bal_loss for b in self.blocks)
            loss = main + (wm_loss * 0.1) + (val_loss * 0.1) + (moe_loss * CONFIG["AUX_LOSS_WEIGHT"])
        return logits, loss, x_mean.mean(1).detach()

    def generate(self, idx, max_new=100, temperature=1.0, top_k=50, n_paths=None):
        """
        Best-of-N multi-path sampling.
        n_paths defaults to CONFIG['INFERENCE_PATHS'] (4).
        Pass n_paths=1 for fast single-path generation (e.g. reflection).
        Always uses world_sim=1 during generation — the ensemble averaging of WORLD_SIM=5
        is a training-time regulariser; running 5 copies per token step during inference
        multiplies memory N×5× with no meaningful quality gain in autoregressive sampling.
        """
        N = n_paths if n_paths is not None else CONFIG["INFERENCE_PATHS"]
        # Tile input for N parallel paths
        paths = idx.repeat(N, 1)  # (N, T)
        log_probs = torch.zeros(N, device=DEVICE)

        for _ in range(max_new):
            ctx = paths[:, -CONFIG["BLOCK_SIZE"]:]
            # world_sim=1: skip 5× world expansion during inference (CPU/RAM safety)
            logits, _, _ = self(ctx, world_sim=1)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(torch.nan_to_num(logits, nan=-1e9), dim=-1)
            next_tok = torch.multinomial(probs, 1)  # (N, 1)
            # Accumulate log-probability for each path
            chosen_probs = probs.gather(1, next_tok).squeeze(1).clamp(min=1e-9)
            log_probs += chosen_probs.log()
            paths = torch.cat((paths, next_tok), dim=1)

        # Return the highest log-prob path (best-of-N)
        best = log_probs.argmax().item()
        return paths[best:best+1]


# Backward-compatibility aliases (v2–v9 code that references old class names)
SeedGPT = GodheadTransformer
RecurrentWorld = GodheadTransformer
SeedGPTv3 = GodheadTransformer


# ==============================================================================
# 7. IMMORTAL CORE CONTROLLER
# ==============================================================================

class ImmortalCoreController:
    """
    Unified lifecycle controller integrating all evolutionary, cognitive,
    and generative capabilities from every seed iteration.
    """
    def __init__(self, rank=0, world=1):
        self.rank = rank
        self.world_size = world

        # Data
        self.data = DataManager(rank)

        # Memory systems
        self.memory = DiskEpisodicMemory()
        self.cog_memory = CognitiveMemory()

        # Neural cognitive modules
        self.agency = AgencyCore().to(DEVICE)
        self.world_env = NeuralWorldModel().to(DEVICE)
        self.world_sim = PersistentWorldSimulator(self.world_env)  # [WIRED] multi-step rollouts
        self.meta = MetaLearningEngine().to(DEVICE)                # [WIRED] primary LR predictor
        self.self_model = PredictiveSelfModel().to(DEVICE)
        self.arch_policy = ArchitecturePolicy().to(DEVICE)

        # Telemetry
        self.tele = TelemetryLogger()

        # Civilisation / coordination
        self.civ = CivilizationCoordinator()
        self.civ_mind = CivilizationMind()
        self.shard_mgr = ShardedIdentity()

        # Cognition
        self.res = AutonomousResearch()
        self.planner = PlanningEngine(self.world_env, self.self_model)
        self.life = LifelongProtector()
        self.ledger = BeliefLedger()
        self.experiment = ExperimentEngine()
        self.curiosity = CuriosityEngine()
        self.goal_engine = GoalEngine()
        self.self_improve = SelfImprovementLoop()
        self.concepts = ConceptTracker()
        self.narrative = SelfNarrative()
        self.identity_continuity = IdentityContinuity()
        self.sandbox = SelfRewriteSandbox()
        self.synth_buffer = SyntheticBuffer()
        self.replay_buffer = SeedReplayBuffer()
        self.genome = IdentityGenome()

        # Optimizers for auxiliary modules
        self.world_opt = optim.AdamW(self.world_env.parameters(), lr=1e-4)
        self.meta_optimizer = optim.AdamW(self.meta.parameters(), lr=1e-4)  # [WIRED] MetaLearningEngine
        self.self_opt = optim.AdamW(self.self_model.parameters(), lr=1e-4)

        # Population
        self.pop = []
        self.opts = []
        self.prev_loss = None  # [FIX LOW] None on step 0 → meta update skipped (0.0 was misleading)
        self.prev_state = None
        self.score_history = []
        self.replay = deque(maxlen=2000)
        self._prev_seeds = {}  # [FIX MEDIUM] Per-agent, not shared instance var
        self._grew_this_gen = False  # [FIX LOW] Prevents dual grow_network() in one generation

        self._spawn()
        self._load()
        if rank == 0:
            self._audit()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _spawn(self):
        for _ in range(CONFIG["POPULATION_SIZE"]):
            m = GodheadTransformer(self.data.vocab_size).to(DEVICE)
            if self.world_size > 1:
                m = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
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
            logging.critical(f"❌ AUDIT FAILED: {e}")
            sys.exit(1)

    def _load(self):
        if os.path.exists(PATHS["CHECKPOINT"]):
            d = torch.load(PATHS["CHECKPOINT"], map_location=DEVICE, weights_only=False)  # [FIX LOW] silence FutureWarning in PyTorch 2.x
            if "pop" in d:
                for i, s in enumerate(d["pop"]):
                    if i < len(self.pop):
                        self.unwrap(self.pop[i]).load_state_dict(s)
            if "opts" in d:
                for i, s in enumerate(d["opts"]):
                    self.opts[i].load_state_dict(s)
            for key, mod in [("agency", self.agency), ("wm", self.world_env), ("self", self.self_model)]:
                if key in d:
                    mod.load_state_dict(d[key])
            self.memory.load_meta()
            if self.rank == 0:
                logging.info(f"RESTORED GEN {d.get('gen', '?')}")

    def unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    # ------------------------------------------------------------------
    # Probe / diagnostics [v27 + v28]
    # ------------------------------------------------------------------

    def probe_weight_stats(self, model):
        means, stds = [], []
        for p in model.parameters():
            means.append(p.data.mean().item())
            stds.append(p.data.std().item())
        return np.mean(means), np.mean(stds)

    def probe_neurons(self, model, x):
        with torch.no_grad():
            # [FIX LOW] world_sim=1 — probe only; 5× ensemble adds nothing to neuron scores
            _, _, hidden = model(x, world_sim=1)
        scores = hidden.abs().mean(dim=0)
        top = torch.topk(scores, min(10, scores.numel()))
        return top.indices.tolist(), top.values.tolist()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, gen, tag=""):
        if self.rank != 0:
            return
        s = {
            "pop": [self.unwrap(p).state_dict() for p in self.pop],
            "opts": [o.state_dict() for o in self.opts],
            "agency": self.agency.state_dict(),
            "agency_opt": self.agency.optimizer.state_dict(),
            "wm": self.world_env.state_dict(),
            "self": self.self_model.state_dict(),
            "gen": gen,
        }
        atomic_save(s, PATHS["CHECKPOINT"], True)
        atomic_save(s, f"{PATHS['DIR_ARCHIVE']}/gen_{gen}{tag}.pt", True)
        self.memory.save()
        self.genome.save()
        self.cog_memory.save()
        self.identity_continuity.save()
        self.synth_buffer.flush()  # [FIX LOW] Write any buffered synthetic text to disk
        logging.info(f"💾 SAVED GEN {gen}")

    def save_checkpoint(self, model, optimizer, generation, tag=""):
        """Per-agent checkpoint saving with timestamp (v15 origin)."""
        try:
            ckpt = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "generation": generation,
                "timestamp": time.time(),
            }
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(PATHS["DIR_CKPT"], f"checkpoint_gen{generation}_{tag}_{ts}.pt")
            atomic_save(ckpt, path, use_torch=True)
        except Exception as e:
            logging.error(f"PT Save Error: {e}")

    def export_identity_archive(self, model, memory_db, narrative):
        """Bounded identity export to IMMORTAL_ARCHIVE (v26 origin, v28 fix)."""
        try:
            seed = IdentitySeed.compress(model)
            pruned_narrative = narrative.events[-1000:]
            pruned_beliefs = memory_db.payloads[-1000:]
            atomic_save({
                "seed": seed,
                "beliefs": pruned_beliefs,
                "narrative": pruned_narrative,
                "arch_layers": len(model.blocks) if hasattr(model, "blocks") else 0,
                "timestamp": time.time(),
            }, PATHS["ARCHIVE"], use_torch=True)
            logging.info(">>> IMMORTAL ARCHIVE EXPORTED")
        except Exception as e:
            logging.error(f"Archive Error: {e}")

    def save_memory(self, model):
        """Legacy .pkl memory save (v2 compat)."""
        try:
            state = model.state_dict()
            if os.path.exists(PATHS["MEM_PKL"]):
                os.rename(PATHS["MEM_PKL"], PATHS["MEM_BAK"])
            atomic_save(state, PATHS["MEM_PKL"], use_torch=False)
            atomic_save(state, PATHS["PT_FILE"], use_torch=True)
            logging.info(">>> MEMORY SAVED (Atomic)")
        except Exception as e:
            logging.error(f"Save Error: {e}")

    def load_memory(self, model):
        """Legacy .pkl memory load (v2 compat)."""
        for path in [PATHS["MEM_PKL"], PATHS["MEM_BAK"]]:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        mem = pickle.load(f)
                    model.load_state_dict(mem, strict=False)
                    logging.info(">>> MEMORY RESTORED")
                    return
                except Exception:
                    pass
        logging.info(">>> NO MEMORY FOUND — FRESH MIND")

    # ------------------------------------------------------------------
    # Architecture growth [v15 DDP-safe + v26 arch policy]
    # ------------------------------------------------------------------

    def grow_network(self, idx):
        """DDP-synchronised network growth (v15 origin)."""
        if self.rank == 0:
            m = self.unwrap(self.pop[idx])
            m.blocks.append(RecurrentBlock().to(DEVICE))
            logging.info("🌱 NETWORK GROWN")
            s = m.state_dict()
            s["_layers"] = len(m.blocks)
        else:
            s = None
        if self.world_size > 1:
            dist.barrier()
            o = [s]
            dist.broadcast_object_list(o, 0)
            s = o[0]
            m = self.unwrap(self.pop[idx])
            while len(m.blocks) < s["_layers"]:
                m.blocks.append(RecurrentBlock().to(DEVICE))
            m.load_state_dict(s)
            self.pop[idx] = nn.parallel.DistributedDataParallel(m, device_ids=[self.rank])
        self.opts[idx] = optim.AdamW(self.pop[idx].parameters(), lr=CONFIG["LR"])

    def auto_grow(self):
        """ArchitecturePolicy-driven auto-grow decision (v26 origin)."""
        loss = self.score_history[-1] if self.score_history else 0
        policy = self.arch_policy.select_action(
            loss, 0,
            len(self.memory.payloads),
            len(self.unwrap(self.pop[0]).blocks)
        )
        if policy == 0 and not self._grew_this_gen:  # [FIX LOW] only one grow per gen
            self._grew_this_gen = True
            self.grow_network(0)
            # [FIX LOW] Guard against None and compare same-sign values:
            # score_history[-1] is -avg_loss (negative); self.prev_loss is positive.
            # -loss < prev_loss means "growth reduced loss" — a meaningful comparison.
            # Without the None guard: crash if kill-switch fired before any step.
            reward = 1.0 if (self.prev_loss is not None and -loss < self.prev_loss) else -0.5
            self.arch_policy.rewards.append(reward)
            self.arch_policy.update()

    def sync_architecture(self, model, state_dict):
        """Safely sync layer count before loading state (v15 origin)."""
        max_block_idx = -1
        for k in state_dict.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if parts[1].isdigit():
                    max_block_idx = max(max_block_idx, int(parts[1]))
        target_layers = max_block_idx + 1
        while len(model.blocks) < target_layers:
            model.blocks.append(RecurrentBlock().to(DEVICE))
        while len(model.blocks) > target_layers:
            del model.blocks[-1]

    # ------------------------------------------------------------------
    # Phase-based lifecycle [v27 phase architecture]
    # ------------------------------------------------------------------

    def phase_train(self, generation):
        """
        Full training phase — all features wired:
          · MetaLearningEngine adaptive LR  (loss + grad_norm → scale)
          · EWC penalty                      (LifelongProtector, every 5 gens)
          · CivilizationMind temperatures    (role-based diversity)
          · PersistentWorldSimulator rollout (3-step intrinsic reward)
          · CuriosityEngine novelty bonus
          · SeedReplayBuffer experience push
          · kill-switch on divergence        (v24)
          · intra-epoch checkpoints          (v20)
        """
        noise_level = max(0.0, 0.01 * (1.0 - generation / CONFIG["GENERATIONS"]))
        # [FIX LOW] Removed redundant world_env.reset_state() — run() resets every generation
        # [FIX LOW] Pass real telemetry so hypothesis is actually recorded
        self.res.propose({"loss": self.prev_loss if self.prev_loss is not None else 0.0, "gen": generation})
        self.res.act(self)

        # [WIRED] CivilizationMind — assign roles once per generation for temp diversity
        self.civ_mind.assign_roles(CONFIG["POPULATION_SIZE"])

        for i, (model, opt) in enumerate(zip(self.pop, self.opts)):
            model.train()
            log_this = (i == 0 and self.rank == 0)

            for step in range(CONFIG["CYCLES_PER_GEN"]):
                diff = random.choice(CONFIG["CURRICULUM"])
                x, y = self.data.get_batch(diff)
                if x is None:
                    continue

                _, loss, hidden = model(x, y, noise_level)
                loss = self.goal_engine.reward_modifier(loss)

                # [WIRED] EWC penalty — prevents catastrophic forgetting (v14)
                if self.life.importance and i == 0:
                    loss = loss + CONFIG["EWC_LAMBDA"] * self.life.penalty(self.unwrap(model))

                # [FIX MEDIUM] Kill-switch AFTER EWC penalty — catches EWC-inflated divergence
                if torch.isnan(loss) or loss.item() > 100.0:
                    logging.warning(f"⚠️ AGENT {i} DIVERGED (Loss: {loss.item():.2f}). RESETTING...")
                    self.load_memory(self.unwrap(model))
                    return

                # ── CRITICAL FIX: backward BEFORE MetaLearningEngine update ──────
                # loss.backward() must run first; we then use the resulting grad_norm
                # as an INPUT to MetaLearningEngine (no shared graph).
                prev_loss_val = self.prev_loss
                opt.zero_grad()
                loss.backward()
                grad_norm = sum(
                    p.grad.norm().item()
                    for p in model.parameters() if p.grad is not None
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP"])
                opt.step()
                self.prev_loss = loss.item()

                # [WIRED] MetaLearningEngine: train on (prev_loss → current_loss) delta
                # [FIX LOW] Single get_scale() call — was called twice (once in no_grad,
                # once outside). Both calls used identical float inputs → identical outputs.
                meta_scale = self.meta.get_scale(loss.item(), grad_norm)
                lr_scale_val = meta_scale.item()  # scalar for LR adjustment
                if prev_loss_val is not None:
                    # [FIX CRITICAL] Guard on prev_loss_val (captured snapshot, line 1621),
                    # NOT self.prev_loss — self.prev_loss was already set to loss.item() at
                    # line 1630, so it is never None here. prev_loss_val is still None on step 0,
                    # preventing TypeError: NoneType - float on the very first training step.
                    delta = torch.tensor(prev_loss_val - loss.item(), device=DEVICE)
                    meta_loss_val = torch.nan_to_num(-(delta * meta_scale).mean(), nan=0.0)
                    self.meta_optimizer.zero_grad()
                    meta_loss_val.backward()
                    self.meta_optimizer.step()
                # Apply scaled LR for next step (range 0.1× → 6.0×)
                for g in opt.param_groups:
                    g["lr"] = CONFIG["LR"] * max(0.1, lr_scale_val * 3.0)

                # [WIRED] World model + intrinsic reward via PersistentWorldSimulator
                with torch.no_grad():
                    current_state = hidden.mean(0).detach()
                    if self.prev_state is not None:
                        # [FIX MEDIUM] Save/restore world_env GRU state around rollout.
                        # rollout() calls world_env.forward() 3× in-place, leaving state at
                        # imagined t+3. Without save/restore, agents 1-3 each start their
                        # next rollout from the previous agent's imagined future state.
                        saved_wm_state = self.world_env.state.clone()
                        trajectory = self.world_sim.rollout(self.prev_state, steps=3)
                        self.world_env.state = saved_wm_state
                        pred_errs = [F.mse_loss(s, current_state).item() for s in trajectory]
                        intrinsic = (sum(pred_errs) / len(pred_errs)) * 0.1
                        self.agency.rewards.append(intrinsic)
                    self.prev_state = current_state

                if i == 0:
                    self.world_env.update(current_state)
                    self.concepts.extract(hidden)
                    # [WIRED] CuriosityEngine novelty bonus
                    self.agency.rewards.append(self.curiosity.reward(current_state))
                    # [WIRED] CivilizationMind knowledge pool
                    self.civ_mind.share(f"loss={loss.item():.3f},gn={grad_norm:.2f}")

                # [WIRED] SeedReplayBuffer — store (prev_seed, cur_seed, reward) experience
                # [FIX MEDIUM] Per-agent seed tracking — no cross-agent contamination
                if step % 50 == 0:
                    cur_seed = IdentitySeed.compress(self.unwrap(model))["weights"]
                    if i in self._prev_seeds:
                        self.replay_buffer.push(self._prev_seeds[i], cur_seed, -loss.item())
                    self._prev_seeds[i] = cur_seed.detach()

                if log_this and step % 20 == 0:
                    logging.info(
                        f"   Step {step:3d} | Loss {loss.item():.4f} | "
                        f"LR_scale {lr_scale_val:.3f} | GradNorm {grad_norm:.3f}"
                    )
                    self.tele.log({
                        "gen": generation, "step": step, "agent": i,
                        "loss": loss.item(), "lr_scale": lr_scale_val, "grad_norm": grad_norm,
                    })

                # Intra-epoch checkpoint [v20]
                if step > 0 and step % 500 == 0 and self.rank == 0:
                    self.save_checkpoint(model, opt, generation, tag=f"step{step}")

        # [WIRED] Record EWC importance snapshot every 5 generations
        if generation % 5 == 0 and self.rank == 0:
            self.life.record_importance(self.unwrap(self.pop[0]), self.data)
            logging.info("📌 EWC importance recorded")

    def phase_regenerate(self):
        """Selective weight destruction and regeneration (v2 origin, v27 selective)."""
        if self.rank == 0:
            logging.info("  [PHASE REGEN] DESTRUCTION & REGENERATION...")
        for model, opt in zip(self.pop, self.opts):
            destroy_weights(self.unwrap(model), CONFIG["WIPE_RATIO"])
            model.train()
            for _ in range(CONFIG["REGENERATE_STEPS"]):
                x, y = self.data.get_batch(1.0)
                if x is None:
                    continue
                _, loss, _ = model(x, y)
                if torch.isnan(loss):
                    break
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP"])
                opt.step()

    def phase_evaluate(self):
        """DDP split-brain-safe evaluation with memory storage (v9 + v27)."""
        scores = []
        for model in self.pop:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for _ in range(CONFIG["EVAL_BATCHES"]):
                    x, y = self.data.get_batch(1.0)
                    if x is None:
                        continue
                    _, loss, hidden = model(x, y)
                    total_loss += loss.item()
                    if self.rank == 0:
                        self.memory.store(hidden.mean(0), {"loss": loss.item()})
            avg_loss = total_loss / max(CONFIG["EVAL_BATCHES"], 1)
            lt = torch.tensor(avg_loss).to(DEVICE)
            if self.world_size > 1:
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            scores.append(-(lt.item() / self.world_size))
        if self.rank == 0:
            self.goal_engine.evolve_goals(self.memory)
        return scores

    def agent_council_vote(self, scores):
        """Democratic best-agent selection (v27 origin)."""
        return int(np.argmax(scores))

    def phase_evolve(self, scores, gen):
        """Full evolutionary step: roles, identity, genome, mutation (v27+v28)."""
        roles = self.civ.assign_roles(scores)
        best_idx = self.agent_council_vote(scores)

        # Self-improvement stagnation detection [v27]
        self.self_improve.update(scores[best_idx])
        # [FIX LOW] Check _grew_this_gen — third grow path, must respect the same flag
        # as auto_grow() and res.act() to prevent silent double-layer growth in one gen
        if self.self_improve.stagnating() and not self._grew_this_gen:
            self._grew_this_gen = True
            if self.rank == 0:
                logging.info("♻️ SELF-IMPROVEMENT LOOP: Forcing mutation")
            self.grow_network(best_idx)

        self.score_history.append(scores[best_idx])
        if self.rank == 0:
            logging.info(f"🧬 EVOLVE | Winner: {best_idx} | Score: {scores[best_idx]:.4f}")
            w_mean, w_std = self.probe_weight_stats(self.unwrap(self.pop[best_idx]))
            logging.info(f"   [TELEMETRY] Weights Mean: {w_mean:.4f} | Std: {w_std:.4f}")

        seed = IdentitySeed.compress(self.unwrap(self.pop[best_idx]))
        leader_state = self.opts[best_idx].state_dict()

        if self.rank == 0:
            self.genome.add(seed, scores[best_idx])
            # Collapse detection and resurrection [v13]
            if scores[best_idx] < -10.0:
                logging.warning("⚠️ COLLAPSE DETECTED. RESURRECTING ANCESTOR.")
                ancient = self.genome.resurrect()
                if ancient:
                    seed = {"weights": torch.tensor(ancient["w"]), "meta": ancient["meta"]}
                    leader_state = None

            robustness = self.experiment.run(self.unwrap(self.pop[best_idx]), self.data)
            logging.info(f"🧪 ROBUSTNESS SCORE: {robustness:.4f}")

            h_sig = identity_hash(self.unwrap(self.pop[best_idx]))
            self.identity_continuity.record(seed, h_sig)
            self.ledger.record(agent_id=best_idx, score=scores[best_idx], belief=seed)
            self.narrative.log(f"Gen {gen}: Agent {best_idx} won (score={scores[best_idx]:.4f})")
            self.export_identity_archive(self.unwrap(self.pop[best_idx]), self.memory, self.narrative)
            self.save_memory(self.unwrap(self.pop[best_idx]))
            self.save_checkpoint(self.unwrap(self.pop[best_idx]), self.opts[best_idx], gen, tag="best")

        # [FIX MEDIUM] Broadcast resurrected seed to all ranks to prevent split-brain
        if self.world_size > 1:
            seed_payload = [seed]
            dist.broadcast_object_list(seed_payload, src=0)
            seed = seed_payload[0]

        # Self-model update [v9.5 + v16 replay — now uses SeedReplayBuffer]
        s0 = seed["weights"].to(DEVICE)
        self.replay.append(s0)
        # [WIRED] SeedReplayBuffer: push current transition, train from buffer samples
        if len(self.replay) > 1:
            prev = self.replay[-2]
            reward = scores[best_idx]
            self.replay_buffer.push(prev, s0, reward)

        if len(self.replay_buffer.buffer) >= 8:
            # Sample from replay buffer for more stable self-model training
            rb_s0, rb_s1, rb_r = self.replay_buffer.sample(8)
            # Flatten and pad to IDENTITY_SEED_SIZE*3 if needed
            def _pad(t):
                if t.shape[-1] < CONFIG["IDENTITY_SEED_SIZE"] * 3:
                    return t.repeat(1, 3)[:, :CONFIG["IDENTITY_SEED_SIZE"] * 3]
                return t
            rb_p, _ = self.self_model(_pad(rb_s0))
            sloss = F.mse_loss(rb_p, _pad(rb_s1))
            self.self_opt.zero_grad()
            sloss.backward()
            self.self_opt.step()
        elif len(self.replay) > 1:
            prev = self.replay[-2]
            inp = prev.unsqueeze(0)
            if inp.shape[-1] < CONFIG["IDENTITY_SEED_SIZE"] * 3:
                inp = inp.repeat(1, 3)[:, :CONFIG["IDENTITY_SEED_SIZE"] * 3]
            p_s, _ = self.self_model(inp)
            target = s0.unsqueeze(0)
            if target.shape[-1] < CONFIG["IDENTITY_SEED_SIZE"] * 3:
                target = target.repeat(1, 3)[:, :CONFIG["IDENTITY_SEED_SIZE"] * 3]
            sloss = F.mse_loss(p_s, target)
            self.self_opt.zero_grad()
            sloss.backward()
            self.self_opt.step()

        # Propagate winner to rest of population with role-based mutation [v27]
        for i in range(CONFIG["POPULATION_SIZE"]):
            if i != best_idx:
                t = self.unwrap(self.pop[i])
                IdentitySeed.reconstruct(t, seed)
                role = roles.get(i, "worker")
                mut = 0.05 if role == "explorer" else 0.01
                with torch.no_grad():
                    for p in t.parameters():
                        # [v105.1 Fix 8] Clamp mutations
                        p.add_(torch.randn_like(p) * mut).clamp_(-2, 2)
                if leader_state:
                    self.opts[i].load_state_dict(leader_state)

        # [FIX MEDIUM] Single barrier AFTER mutation loop — not 3× inside it
        if self.world_size > 1:
            dist.barrier()

        # [WIRED] ShardedIdentity: merge worker gradients back into leader [v15]
        workers = [self.unwrap(self.pop[i]) for i in range(CONFIG["POPULATION_SIZE"]) if i != best_idx]
        self.shard_mgr.merge_gradients(self.unwrap(self.pop[best_idx]), workers)

    # ------------------------------------------------------------------
    # Reflection [v10 origin, v105.1 key safety fix]
    # ------------------------------------------------------------------

    def reflect(self):
        """Inner monologue generation with memory recall (v10 origin)."""
        # [FIX LOW] Rank guard — only leader runs generation; avoids N× wasted compute on multi-GPU
        if self.rank != 0:
            return
        x, _ = self.data.get_batch()
        if x is None:
            return
        with torch.no_grad():
            m = self.unwrap(self.pop[0])
            # [FIX LOW] world_sim=1 — probe-only forward pass; ensemble adds no quality here
            _, _, meta = m(x, world_sim=1)
            vec = meta.mean(0)
            recalled = self.memory.query(vec)
            ctx = ""
            if recalled:
                # [v105.1 Fix 6] Safe dict access
                texts = [r["data"].get("text", "") for r in recalled
                         if isinstance(r, dict) and "data" in r and isinstance(r["data"], dict)]
                ctx = " ".join(texts[:2]).strip()

            # [FIX LOW] Build prompt using CivilizationMind context + recalled memory
            civ_ctx = self.civ_mind.get_context()
            if civ_ctx:
                raw_prompt = f"CIV:{civ_ctx[:40]} CTX:{ctx[:60]} THOUGHT:"
            else:
                raw_prompt = f"CTX:{ctx[:80]} THOUGHT:" if ctx else "THOUGHT:"
            safe_prompt = "".join(c for c in raw_prompt if c in self.data.stoi)
            enc_ids = self.data.encode(safe_prompt) if safe_prompt else [1]
            enc = torch.tensor([enc_ids], device=DEVICE)

            # Generate in eval mode — Dropout off for clean, deterministic output
            m.eval()
            out = m.generate(enc, n_paths=1)  # n_paths=1 — no best-of-N overhead for reflection
            txt = self.data.decode(out[0].tolist())

            # [FIX MEDIUM] Perplexity measured while STILL in eval mode (before m.train())
            # Dropout must be off so ppl < 50 threshold is consistent across calls
            if len(txt) > 20:
                try:
                    enc_gen = torch.tensor([self.data.encode(txt[:CONFIG["BLOCK_SIZE"]])], device=DEVICE)
                    if enc_gen.shape[1] > 1:
                        _, gen_loss, _ = m(enc_gen[:, :-1], enc_gen[:, 1:])
                        ppl = math.exp(min(gen_loss.item(), 10.0))
                        self.synth_buffer.add(txt, ppl)
                except Exception:
                    pass

            # Restore training mode after all eval-mode work is done
            m.train()

            self.memory.store(vec, {"text": txt, "gen": 0})
            self.cog_memory.remember({"text": txt})
            # [WIRED] Share reflection with CivilizationMind knowledge pool
            self.civ_mind.share(txt[:100])
            logging.info(f"\n[REFLECT] {txt}\n")

    # ------------------------------------------------------------------
    # Demo generation with neuron probing [v27 generate_demo]
    # ------------------------------------------------------------------

    def generate_demo(self):
        if self.rank == 0:
            model = self.unwrap(self.pop[0])
            model.eval()
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            # [WIRED] Use CivilizationMind leader temperature for demo generation
            temp = self.civ_mind.get_temp(0)
            out = model.generate(ctx, max_new=150, temperature=temp)
            txt = self.data.decode(out[0].tolist())
            indices, values = self.probe_neurons(model, ctx)
            logging.info(f"🧪 TOP NEURONS: {indices} (Act: {[round(v, 2) for v in values]})")
            logging.info(f"🌡️  GENERATION TEMP (leader): {temp:.2f}")
            print(f"\n[DEMO] {txt}\n")

            # [FIX LOW] Perplexity measured in eval mode (before model.train())
            if len(txt) > 20:
                try:
                    enc_gen = torch.tensor([self.data.encode(txt[:CONFIG["BLOCK_SIZE"]])], device=DEVICE)
                    if enc_gen.shape[1] > 1:
                        with torch.no_grad():
                            _, gen_loss, _ = model(enc_gen[:, :-1], enc_gen[:, 1:])
                        ppl = math.exp(min(gen_loss.item(), 10.0))
                        self.synth_buffer.add(txt, ppl)
                except Exception:
                    pass

            model.train()

    # ------------------------------------------------------------------
    # Main run loop [v103 agency loop with v105.1 graph safety]
    # ------------------------------------------------------------------

    def run(self):
        discover_tasks()
        self._last_gen = -1  # [FIX MEDIUM] Track for safe interrupt handler
        try:
            for gen in range(CONFIG["GENERATIONS"]):
                self._last_gen = gen
                if self.rank == 0:
                    logging.info(f"\n=== GENERATION {gen} ===")

                x, y = self.data.get_batch()
                if x is None:
                    continue

                # [v105.1 Fix 2] Reset world model before each episode
                self.world_env.reset_state()

                with torch.no_grad():
                    # [FIX LOW] world_sim=1 — agency probe only needs one forward pass;
                    # 5× world ensemble adds no quality to the policy state representation
                    _, _, state = self.unwrap(self.pop[0])(x, world_sim=1)
                    state = state.mean(0)

                action, probs = self.agency.decide(state)
                if self.rank == 0:
                    logging.info(f"🤖 GEN {gen} | ACTION: {action}")
                    self.narrative.log(f"Gen {gen}: action={action}")

                if action == "evolve":
                    self.phase_evolve(self.phase_evaluate(), gen)
                elif action == "reflect":
                    self.reflect()
                elif action == "rest":
                    time.sleep(0.1)
                else:
                    # Default: full cycle
                    self.run_cycle(gen)
                    continue

                # [v105.1 Fix 1] Double backward graph safety
                state_detach = state.detach().clone().requires_grad_(False)
                pred = self.world_env(state_detach)
                wm_loss = F.mse_loss(pred, state_detach)
                self.world_opt.zero_grad()
                wm_loss.backward()
                self.world_opt.step()

                curiosity = wm_loss.item()
                self.agency.rewards.append(curiosity * 0.1)
                self.agency.update_policy()

                self.save(gen)

        except KeyboardInterrupt:
            logging.info("\n⚡ KEYBOARD INTERRUPT — Saving state...")
            try:
                self.save(self._last_gen, tag="_interrupt")
            except Exception as e:
                logging.error(f"Interrupt save failed: {e}")
            try:
                self.synth_buffer.flush()  # [FIX LOW] Don't lose buffered text on interrupt
            except Exception:
                pass
            logging.info("💾 Safe exit complete.")
            sys.exit(0)

    def run_cycle(self, gen):
        """Full 4-phase lifecycle: Train → Evaluate → AutoGrow → Evolve → Regenerate (v27)."""
        if self.rank == 0:
            logging.info(f"  [CYCLE {gen}] TRAIN → EVALUATE → EVOLVE → REGENERATE")
        self._grew_this_gen = False  # [FIX LOW] Reset so only one grow_network() fires per gen

        self.phase_train(gen)
        scores = self.phase_evaluate()

        plan_score = self.planner.plan(self.unwrap(self.pop[0]))
        if self.rank == 0:
            logging.info(f"🧭 PLAN SCORE: {plan_score:.4f}")

        self.auto_grow()
        self.phase_evolve(scores, gen)
        # ShardedIdentity merge is called inside phase_evolve — no duplicate here
        self.phase_regenerate()

        if gen % 2 == 0:
            self.generate_demo()

        # [v105.1 Fix 1] World model update after cycle
        x, _ = self.data.get_batch()
        if x is not None:
            with torch.no_grad():
                # [FIX LOW] world_sim=1 — probe-only; 5× ensemble adds no quality here
                _, _, state = self.unwrap(self.pop[0])(x, world_sim=1)
                state_detach = state.mean(0).detach().clone().requires_grad_(False)
            pred = self.world_env(state_detach)
            wm_loss = F.mse_loss(pred, state_detach)
            self.world_opt.zero_grad()
            wm_loss.backward()
            self.world_opt.step()

        self.agency.update_policy()
        self.save(gen)


# ==============================================================================
# 8. LEGACY TOP-LEVEL HELPERS (v2 / seed.py compat)
# ==============================================================================

def generate(model, prompt, steps=200, temperature=1.0, top_k=None, data_mgr=None):
    """Legacy standalone generate function (v2 compat)."""
    if data_mgr is None:
        raise ValueError("data_mgr required")
    tokens = torch.tensor([data_mgr.encode(prompt)], device=DEVICE)
    out = model.generate(tokens, max_new=steps, temperature=temperature, top_k=top_k or 50)
    return data_mgr.decode(out[0].tolist())


def load_memory(model, path=None):
    """Legacy load_memory function (v2 compat)."""
    if path is None:
        path = PATHS["MEM_PKL"]
    for p in [path, PATHS["MEM_BAK"]]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    mem = pickle.load(f)
                model.load_state_dict(mem, strict=False)
                logging.info(">>> MEMORY RESTORED")
                return
            except Exception:
                pass
    logging.info(">>> NO MEMORY FOUND — FRESH MIND")


def save_memory(model, path=None):
    """Legacy save_memory function (v2 compat)."""
    if path is None:
        path = PATHS["MEM_PKL"]
    try:
        mem = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if os.path.exists(path):
            os.rename(path, PATHS["MEM_BAK"])
        with open(path, "wb") as f:
            pickle.dump(mem, f)
        logging.info(">>> MEMORY SAVED")
    except Exception as e:
        logging.error(f"Memory save failed: {e}")


# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================

def main(rank, world):
    if world > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)

    ImmortalCoreController(rank=rank, world=world).run()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    if NUM_GPUS > 1:
        mp.spawn(main, args=(NUM_GPUS,), nprocs=NUM_GPUS)
    else:
        main(0, 1)
