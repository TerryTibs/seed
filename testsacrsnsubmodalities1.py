# ============================================================
# SACRSN v36 — IMMORTAL CORE EDITION
# Stable • Persistent • Self-Reflective • Cognitive
# ============================================================

import os, time, random, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ==========================================
# 0. Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# ==========================================
# 1. CONFIG
# ==========================================
CONFIG = {
    "seq_len": 128,
    "embedding_dim": 512,
    "epochs": 1000,
    "learning_rate": 5e-4,
    "grad_clip": 0.5,

    "max_recursion_depth": 8,
    "act_threshold": 0.999,
    "ponder_penalty": 0.0001,

    "feature_variance_weight": 0.1,
    "feature_independence_weight": 0.1,
    "feature_buffer_size": 32,

    "commitment_cost": 0.01,
    "graph_bias_scale": 0.8,

    "ethical_weight": 0.005,
    "diversity_weight": 0.5,
    "focus_weight": 0.001,

    "stack_size": 32,
    "use_stack": True,

    "eps": 1e-6
}

# ==========================================
# 2. DATA
# ==========================================
TEXT_DATA = """The neural architecture of the mind is a mirror of the cosmos itself..."""

def tokenize(text):
    return re.findall(r"[\w']+|[^\s\w]", text)

tokens = tokenize(TEXT_DATA)
vocab = sorted(set(tokens))
word_to_ix = {w:i for i,w in enumerate(vocab)}
ix_to_word = {i:w for w,i in word_to_ix.items()}

data_tensor = torch.tensor([word_to_ix[t] for t in tokens], device=DEVICE)
vocab_size = len(vocab)
CONFIG["n_symbols"] = max(int(vocab_size * 1.2), 32)

# ==========================================
# 3. COMPLEX PRIMITIVES
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Linear(dim, dim, bias=False)
        self.i = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.r.weight)
        nn.init.orthogonal_(self.i.weight)

    def forward(self, z):
        return torch.complex(
            self.r(z.real) - self.i(z.imag),
            self.r(z.imag) + self.i(z.real)
        )

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        mean = mag.mean(-1, keepdim=True)
        var = mag.var(-1, keepdim=True)
        norm = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        phase = torch.angle(z)
        norm = norm * self.scale + self.shift
        return torch.complex(norm * torch.cos(phase), norm * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(mag + self.bias) / mag
        return z * scale

# ==========================================
# 4. GRAPH MEMORY VQ
# ==========================================
class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        # [CRITICAL] Uniform Init spreads symbols out
        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim*2))
        nn.init.uniform_(self.codebook, -0.5, 0.5)
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    
    def forward(self, z, prev_symbol_idx=None, sensory_offset=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        if sensory_offset is not None:
            z_flat = z_flat + sensory_offset
        
        # Euclidean Distance
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            graph_prior = self.adjacency[prev_symbol_idx]
            # [FIX E] Clamp bias to prevent overpower
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior).clamp(max=0.3)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices

# ==========================================
# 5. CORE CELL
# ==========================================
class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt = nn.Linear(dim*2, 1)
        nn.init.constant_(self.halt.bias, -1.2)

    def forward(self, z):
        z = self.norm(self.lin(z))
        z = self.act(z)
        flat = torch.cat([z.real, z.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt(flat))
        return z, halt_prob

# ==========================================
# 6. Master Model (UberCRSN v36: Feature Discovery)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        # Sensory Embeddings
        self.perspective_emb = nn.Embedding(CONFIG["n_perspectives"], dim)
        self.audio_dir_emb = nn.Embedding(CONFIG["n_audio_locs"], dim)
        self.olfactory_loc_emb = nn.Embedding(CONFIG["n_chem_locs"], dim*2)
        self.gustatory_loc_emb = nn.Embedding(CONFIG["n_chem_locs"], dim*2)
        
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        
        # --- FEATURE DISCOVERY HEADS (The "ICA" Outputs) ---
        # 7 Independent heads that the AI must learn to use
        self.head_1 = nn.Linear(dim*2, 1)
        self.head_2 = nn.Linear(dim*2, 1)
        self.head_3 = nn.Linear(dim*2, 1)
        self.head_4 = nn.Linear(dim*2, 1)
        self.head_5 = nn.Linear(dim*2, 1)
        self.head_6 = nn.Linear(dim*2, 1)
        self.head_7 = nn.Linear(dim*2, 1)
        # -----------------------------------------------
        
        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
            
        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        
        # Sensory Injection
        perspective_idx = torch.randint(0, CONFIG["n_perspectives"], (batch_size,), device=z.device)
        p_emb = self.perspective_emb(perspective_idx)
        z = z + torch.complex(p_emb, torch.zeros_like(p_emb))

        audio_idx = torch.randint(0, CONFIG["n_audio_locs"], (batch_size,), device=z.device)
        a_emb = self.audio_dir_emb(audio_idx)
        z = z + torch.complex(a_emb, torch.zeros_like(a_emb))

        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        # ACT Loop Variables
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        focus_reg_accum = 0 # Track focus usage
        stack_history = [] 
        
        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        ethical_loss_total = 0
        
        # --- RECURSION LOOP ---
        for t in range(CONFIG["max_recursion_depth"]):
            # 1. Cell Step
            z_proc, p_halt, stack_ctrl, focus_mask = self.cell(z)
            
            # Accumulate Focus Regularization (L2 norm of mask)
            focus_reg_accum += torch.sum(focus_mask**2)
            
            # Kinesthetic Temp
            temp_val = torch.tensor(0.1 * (t+1), device=z.device).view(1,1)
            temp_emb = temp_val.repeat(batch_size, self.dim)
            z_proc = z_proc + torch.complex(temp_emb, torch.zeros_like(temp_emb))
            
            # 2. Stack Step
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1).to(z.device))

            # 3. VQ & Sensory Context
            olf_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=z.device)
            gust_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=z.device)
            sensory_offset = self.olfactory_loc_emb(olf_idx) + self.gustatory_loc_emb(gust_idx)
            
            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym, sensory_offset=sensory_offset)
            
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            
            # 4. Ponder & Accumulate
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss

        # --- FINAL OUTPUT GENERATION ---
        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        # Feature Heads (Using Sigmoid for 0-1 range for statistical stability)
        feats = [
            torch.sigmoid(self.head_1(features)),
            torch.sigmoid(self.head_2(features)),
            torch.sigmoid(self.head_3(features)),
            torch.sigmoid(self.head_4(features)),
            torch.sigmoid(self.head_5(features)),
            torch.sigmoid(self.head_6(features)),
            torch.sigmoid(self.head_7(features))
        ]
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        
        if len(stack_history) > 0: avg_stack = torch.stack(stack_history).mean()
        else: avg_stack = torch.tensor(0.0)
            
        # Return 9 Values (This matches your new train loop)
        return logits, feats, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack, focus_reg_accum

# ==========================================
# 7. Training Engine (v36: Robust & Optimized)
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    print(f"--- Training SACRSN v36 (Robust ICA Edition) ---")
    print("Updates: NaN Guards, ReLU Loss, Orthogonality, Gradient Accumulation")
    
    feature_buffer = [] 
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            total_feat_loss = 0
            
            # [FIX C] Smoother entropy decay
            entropy_weight = 0.01 * (0.5 + 0.5 * (1 - epoch / CONFIG["epochs"]))
            
            for i in range(len(data_tensor) - 1):
                x = data_tensor[i].view(1, 1)
                y = data_tensor[i+1].view(1)
                
                # Forward Pass
                logits, feats, hidden, sym_idx, ponder, vq_loss, eth_loss, _, focus_sum = model(x, hidden, prev_sym)
                
                # --- ROBUST FEATURE ACCUMULATION ---
                current_features = torch.cat(feats, dim=1) # [1, 7]
                
                # Add DETACHED copy to buffer
                feature_buffer.append(current_features.detach())
                if len(feature_buffer) > CONFIG["feature_buffer_size"]:
                    feature_buffer.pop(0) 
                
                # --- OPTIMIZED STATISTICAL LOSS ---
                loss_feature_stat = torch.tensor(0.0, device=DEVICE)
                
                # Performance Optimization: Only compute O(N^2) stats every 8 steps
                if i % 8 == 0 and len(feature_buffer) > 5:
                    # 1. Prepare History (Detached)
                    history_tensor = torch.cat(feature_buffer[:-1], dim=0)
                    
                    # 2. Combine with LIVE current features
                    combined_batch = torch.cat([history_tensor, current_features], dim=0)
                    
                    # [FIX A] Variance Loss with NaN Guard
                    # clamp_min prevents division by zero in gradients
                    std_devs = torch.std(combined_batch, dim=0).clamp_min(1e-6)
                    
                    # [FIX B] Prevent negative loss with ReLU
                    std_loss = F.relu(1.0 - std_devs.mean())
                    
                    # [FIX A/Conceptual] Independence Loss with Identity Matrix Target
                    decorr_loss = torch.tensor(0.0, device=DEVICE)
                    
                    if combined_batch.shape[0] > 1:
                        # Normalize columns to prevent scale issues in correlation
                        centered = combined_batch - combined_batch.mean(dim=0)
                        # Orthogonality Constraint (Soft ICA)
                        # We want X.T @ X to resemble Identity (Uncorrelated features)
                        cov = (centered.T @ centered) / (combined_batch.shape[0] - 1)
                        target = torch.eye(7, device=DEVICE)
                        decorr_loss = F.mse_loss(cov, target)
                    
                    loss_feature_stat = (std_loss * CONFIG["feature_variance_weight"]) + \
                                        (decorr_loss * CONFIG["feature_independence_weight"])
                
                # Standard Losses
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum())
                
                # [FIX D] Temperature smoothed symbol buffer update
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                if curr_onehot.dim() > 1: curr_onehot = curr_onehot.view(-1)
                with torch.no_grad():
                    model.prev_sym_soft.copy_(0.95 * model.prev_sym_soft + 0.05 * curr_onehot)
                
                loss_diversity = CONFIG["diversity_weight"] * (model.prev_sym_soft * torch.log(model.prev_sym_soft + 1e-9)).sum()
                loss_ethics = CONFIG["ethical_weight"] * eth_loss
                loss_focus_reg = CONFIG["focus_weight"] * focus_sum
                
                # TOTAL LOSS
                loss = loss_pred + \
                       loss_feature_stat + \
                       loss_ponder + 0.1*vq_loss + loss_entropy + loss_diversity + loss_ethics + loss_focus_reg
                
                # Backprop
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                # [FIX F] Robust Hidden State Detachment
                h_z, h_mem, h_ptr = hidden
                h_mem_det = h_mem.detach() if h_mem is not None else None
                h_ptr_det = h_ptr.detach() if h_ptr is not None else None
                hidden = (h_z.detach(), h_mem_det, h_ptr_det)
                
                prev_sym = sym_idx.detach()
                
                total_loss += loss.item()
                total_feat_loss += loss_feature_stat.item()
                total_ponder += ponder.item()
                
                usage_dist = model.prev_sym_soft.detach() + 1e-10
                entropy_val = -(usage_dist * torch.log(usage_dist)).sum()
                total_ppx += torch.exp(entropy_val).item()
                
            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = total_loss / len(data_tensor)
                # Adjust feat loss display for the fact we compute it every 8 steps
                avg_feat = (total_feat_loss * 8) / len(data_tensor) 
                avg_ponder = total_ponder / len(data_tensor)
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | ICA_Reg: {avg_feat:.4f} | Steps: {avg_ponder:.2f} | LR: {lr:.6f}")
                
                if avg_loss < 0.001:
                    print("--- CONVERGED ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "sacrsn_v36.pth")
    print("Training complete.")
