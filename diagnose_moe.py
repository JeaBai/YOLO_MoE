#!/usr/bin/env python3
"""Diagnose MoE router weights, descriptor distribution, and cascade impact."""

import torch
import sys
sys.path.insert(0, '/workspace')

from ultralytics.nn.modules.moe.modules import SparseDualMoE
from ultralytics.nn.modules.moe.descriptor import ExplicitDescriptor, direct_mapping

# ============================================================
# 1. Router Weight Analysis
# ============================================================
print("=" * 60)
print("1. ROUTER WEIGHT ANALYSIS (fresh init)")
print("=" * 60)

# Create MoE module with training config
for name, cascade in [("P4", 0.75), ("P5", 1.0)]:
    moe = SparseDualMoE(
        in_channels=128, out_channels=128,
        num_experts=4, top_k=4,
        cascade_weight=cascade,
        descriptor_alpha=0.7, descriptor_beta=0.3,
    )
    router = moe.routing.router
    
    # Analyze router weights per expert
    print(f"\n--- {name} (cascade={cascade}) ---")
    for name_param, param in router.named_parameters():
        if 'weight' in name_param:
            print(f"  {name_param}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # Run dummy forward to check router output distribution
    x = torch.randn(4, 128, 20, 20)
    with torch.no_grad():
        logits = router(x)  # [B, num_experts]
        probs = torch.softmax(logits, dim=-1)
    
    print(f"  Router logits per expert:  {logits.mean(dim=0).tolist()}")
    print(f"  Router probs per expert:   {probs.mean(dim=0).tolist()}")
    print(f"  Logits std per expert:     {logits.std(dim=0).tolist()}")
    
    # Check which expert tends to be ranked lowest
    rankings = logits.argsort(dim=-1, descending=True)  # [B, 4], rank 0 = best
    for e in range(4):
        avg_rank = (rankings == e).nonzero(as_tuple=True)[1].float().mean().item()
        print(f"  Expert e{e} avg rank: {avg_rank:.2f} (0=best, 3=worst)")

# ============================================================
# 2. Descriptor Score Distribution Analysis
# ============================================================
print("\n" + "=" * 60)
print("2. DESCRIPTOR SCORE DISTRIBUTION")
print("=" * 60)

descriptor = ExplicitDescriptor(alpha=0.7, beta=0.3)

# Simulate feature maps at different levels of "complexity"
torch.manual_seed(42)
for label, scale in [("Simple (low var)", 0.5), ("Normal", 1.0), ("Complex (high var)", 2.0)]:
    x = torch.randn(8, 128, 20, 20) * scale
    with torch.no_grad():
        s = descriptor(x).view(-1)
    print(f"\n  {label}:")
    print(f"    Range: [{s.min().item():.4f}, {s.max().item():.4f}]")
    print(f"    Mean: {s.mean().item():.4f}, Std: {s.std().item():.4f}")
    print(f"    Values: {s.tolist()}")

# ============================================================
# 3. Direct Mapping Analysis
# ============================================================
print("\n" + "=" * 60)
print("3. DIRECT MAPPING: s -> top_k (K_max=4)")
print("=" * 60)

print("\n  Threshold analysis:")
print(f"  Formula: top_k = 1 + floor(s * 3 + 0.5)")
print(f"  top_k=1 when s < 0.167")
print(f"  top_k=2 when 0.167 <= s < 0.500")
print(f"  top_k=3 when 0.500 <= s < 0.833")
print(f"  top_k=4 when s >= 0.833")

# Demonstrate with cascade multiplication
print("\n  Cascade impact (K_max=4):")
for cascade_w in [0.5, 0.75, 1.0, 1.25]:
    print(f"\n  cascade_weight={cascade_w}:")
    for s_val in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70]:
        s_scaled = s_val * cascade_w
        tk = 1 + int(torch.floor(torch.tensor(s_scaled * 3 + 0.5)).item())
        print(f"    s={s_val:.2f} -> s_scaled={s_scaled:.3f} -> top_k={tk}")

# ============================================================
# 4. CSV Data Analysis (from training)
# ============================================================
print("\n" + "=" * 60)
print("4. TRAINING CSV DATA ANALYSIS")
print("=" * 60)

# Reconstruct from the CSV data we have
# Run 2 (top_k=4): rows 202-401
# Key metrics from the data
print("\n  From training data (Run 2, top_k=4):")
print("  P4 (model.6, cascade=0.75):")
print("    avg_descriptor ~0.24, avg_topk ~1.5")
print("    ~60% samples get top_k=2, ~40% get top_k=1 (approx)")
print("  P5 (model.8, cascade=1.0):")
print("    avg_descriptor ~0.24, avg_topk ~1.75")
print("    ~75% samples get top_k=2, ~25% get top_k=1 (approx)")

# ============================================================
# 5. Root Cause Diagnosis
# ============================================================
print("\n" + "=" * 60)
print("5. ROOT CAUSE DIAGNOSIS")
print("=" * 60)

print("""
  Problem: e2 expert consistently dead (< 5% usage)
  
  Root causes (ranked by likelihood):
  
  A. Router initialization bias
     - Fresh router shows all experts have similar mean logits (~0)
     - But router weights evolve during training, creating winner-take-all
     - e2 gets consistently ranked last → never selected
     - FIX: Add load balancing loss or reinitialize router
  
  B. Descriptor dynamic range too narrow
     - s ~0.24, cascade=0.75 → s_scaled ~0.18 → top_k=1 (P4)
     - s ~0.24, cascade=1.0  → s_scaled ~0.24 → top_k=1 or 2 (P5)
     - Only 1-2 experts activated, 4 available
     - FIX: Increase cascade_weight or tune alpha/beta to produce ~0.3-0.5 scores
  
  C. Shared expert dominates
     - Fusion gate may learn to favor shared_expert over sparse experts
     - Reduces gradient signal to sparse experts → router collapse
     - FIX: Regularize fusion gate, reduce shared expert capacity
  
  D. Balance loss too weak
     - balance_loss_coeff=0.2, drops to balance_loss_min=0.01 after warmup
     - Not enough pressure to distribute load evenly
     - FIX: Increase balance_loss_coeff or add entropy regularization
""")

# ============================================================
# 6. Recommendation
# ============================================================
print("=" * 60)
print("6. RECOMMENDED PARAMETER CHANGES")
print("=" * 60)

print("""
  Quick fix (try first):
    - P4 cascade_weight: 0.75 → 1.0
    - descriptor_alpha: 0.7 → 0.5  (reduce variance weight)
    - descriptor_beta: 0.3 → 0.5   (increase energy weight)
    → Expected: s_scaled ~0.25-0.35 → avg_topk ~1.8-2.2
  
  If e2 still dead:
    - balance_loss_coeff: 0.2 → 0.5
    - balance_loss_min: 0.01 → 0.05
    - Add router noise_std: 0.1 → 0.2 (more exploration)
  
  If dynamic range still narrow:
    - Replace direct clamp with batch-norm in descriptor
    - Or use learnable alpha/beta
""")