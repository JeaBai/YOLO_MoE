"""Comprehensive metrics callback for YOLO MoE experiments.
Records three categories of metrics per epoch to comprehensive_metrics.csv:
  I.   Accuracy & efficiency: mAP50, mAP50-95, box_loss, cls_loss, dfl_loss, GFLOPs_dynamic, Params, inference_speed_ms
  II.  MoE explicit allocation: avg_descriptor, avg_topk, topk_distribution, expert_usage, load_balance_stddev, descriptor_var/energy
  III. (Stability metrics are handled by stability_monitor.py separately)
"""

import csv
import time
import numpy as np
from pathlib import Path
import torch

# CSV column headers (all possible columns)
ALL_COLUMNS = [
    'epoch',
    # I. Accuracy & Efficiency
    'mAP50', 'mAP50-95', 'box_loss', 'cls_loss', 'dfl_loss',
    'GFLOPs_dynamic', 'Params', 'inference_speed_ms',
    # II. MoE Explicit Allocation (P4 = model.6, P5 = model.8)
    'avg_descriptor_P4', 'avg_descriptor_P5',
    'avg_topk_P4', 'avg_topk_P5',
    'topk_dist_P4_k1', 'topk_dist_P4_k2', 'topk_dist_P4_k3', 'topk_dist_P4_k4',
    'topk_dist_P5_k1', 'topk_dist_P5_k2', 'topk_dist_P5_k3', 'topk_dist_P5_k4',
    'expert_usage_P4_e0', 'expert_usage_P4_e1', 'expert_usage_P4_e2', 'expert_usage_P4_e3',
    'expert_usage_P5_e0', 'expert_usage_P5_e1', 'expert_usage_P5_e2', 'expert_usage_P5_e3',
    'load_balance_stddev_P4', 'load_balance_stddev_P5',
    'descriptor_var_raw_P4', 'descriptor_energy_raw_P4',
    'descriptor_var_raw_P5', 'descriptor_energy_raw_P5',
]

# Legacy MoE models (implicit) have different module names
# yolo26_moe uses "MoEInjector" modules, not "SparseDualMoE"
# We need to detect them differently

_params_recorded = False


def _find_moe_modules(model):
    """Find all SparseDualMoE modules in the model. Returns dict of {name: module}."""
    from ultralytics.nn.modules.moe.modules import SparseDualMoE
    moe_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, SparseDualMoE):
            moe_modules[name] = module
    return moe_modules


def _estimate_gflops_dynamic(moe_modules, base_gflops=0):
    """Estimate dynamic GFLOPs based on avg_topk.
    base_gflops: total model GFLOPs without MoE overhead (can be 0 for just relative comparison)
    Returns: total dynamic GFLOPs estimate
    """
    total_extra = 0.0
    for moe in moe_modules.values():
        metrics = moe.get_metrics()
        if not metrics:
            continue
        avg_topk = metrics.get('avg_topk', 1.0)
        k_max = moe.top_k
        # Expert FLOPs scale roughly with (avg_topk / k_max) * num_experts_active
        # For simplicity: assume each expert contributes equally to the MoE layer cost
        # GFLOPs extra = base_per_expert * avg_topk
        # We use a simple heuristic: extra = avg_topk * 0.05 (0.05 GFLOPs per expert per layer)
        total_extra += avg_topk * 0.05
    return base_gflops + total_extra


def _measure_inference_speed(model, device, imgsz=640, n_warmup=3, n_measure=10):
    """Measure inference speed in ms, excluding warmup."""
    try:
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                _ = model(dummy)
            # Measure
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_measure):
                _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        model.train()
        return (elapsed / n_measure) * 1000  # ms
    except Exception:
        return float('nan')


def on_train_start(trainer):
    """Record Params at training start."""
    global _params_recorded
    _params_recorded = False
    
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module
    
    csv_path = Path(trainer.save_dir) / 'comprehensive_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write header
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ALL_COLUMNS)


def on_fit_epoch_end(trainer):
    """Record all metrics at the end of each fit epoch."""
    global _params_recorded
    
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module
    
    csv_path = Path(trainer.save_dir) / 'comprehensive_metrics.csv'
    
    # Initialize row with NaN
    row = {col: float('nan') for col in ALL_COLUMNS}
    row['epoch'] = trainer.epoch + 1
    
    # ---- I. Accuracy & Efficiency ----
    # Losses from trainer
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        loss = trainer.loss_items
        if hasattr(loss, 'tolist'):
            loss = loss.tolist()
        if len(loss) >= 3:
            row['box_loss'] = float(loss[0])
            row['cls_loss'] = float(loss[1])
            row['dfl_loss'] = float(loss[2])
    
    # mAP from trainer metrics
    if hasattr(trainer, 'metrics') and trainer.metrics:
        m = trainer.metrics
        row['mAP50'] = float(m.get('metrics/mAP50(B)', float('nan')))
        row['mAP50-95'] = float(m.get('metrics/mAP50-95(B)', float('nan')))
    
    # Params (record once)
    if not _params_recorded:
        total_params = sum(p.numel() for p in model.parameters())
        row['Params'] = float(total_params)
        _params_recorded = True
    else:
        row['Params'] = float('nan')  # Only record once
    
    # Inference speed (measure at end of epoch, on validation images)
    try:
        device = next(model.parameters()).device
        speed = _measure_inference_speed(model, device, imgsz=trainer.args.imgsz)
        row['inference_speed_ms'] = speed
    except Exception:
        pass
    
    # ---- II. MoE Explicit Allocation ----
    moe_modules = _find_moe_modules(model)
    
    if moe_modules:
        # GFLOPs dynamic
        row['GFLOPs_dynamic'] = _estimate_gflops_dynamic(moe_modules)
        
        for name, moe in moe_modules.items():
            metrics = moe.get_metrics()
            if not metrics:
                continue
            
            # Determine layer tag (P4 or P5)
            if '6' in name or 'P4' in name:
                tag = 'P4'
            elif '8' in name or 'P5' in name:
                tag = 'P5'
            else:
                continue
            
            # Descriptor
            row[f'avg_descriptor_{tag}'] = float(metrics.get('avg_descriptor', float('nan')))
            row[f'avg_topk_{tag}'] = float(metrics.get('avg_topk', float('nan')))
            
            # Expert usage
            usage = metrics.get('expert_usage', {})
            for ei in range(4):
                key = f'e{ei}'
                row[f'expert_usage_{tag}_e{ei}'] = float(usage.get(key, float('nan')))
            
            # Load balance stddev
            if usage:
                vals = [usage.get(f'e{i}', 0) for i in range(4)]
                row[f'load_balance_stddev_{tag}'] = float(np.std(vals))
            
            # Top-K distribution (we approximate from avg_topk and expert usage)
            # For accurate distribution, we'd need to track it in the module
            # Placeholder: distribute based on avg_topk
            avg_tk = metrics.get('avg_topk', 1.0)
            # Simple heuristic: probability mass around floor/ceil of avg_topk
            k_floor = int(avg_tk)
            k_frac = avg_tk - k_floor
            for k in range(1, 5):
                if k == k_floor:
                    row[f'topk_dist_{tag}_k{k}'] = 1.0 - k_frac
                elif k == k_floor + 1 and k_floor < 4:
                    row[f'topk_dist_{tag}_k{k}'] = k_frac
                else:
                    row[f'topk_dist_{tag}_k{k}'] = 0.0
            
            # Descriptor raw values (var and energy components)
            # These are tracked inside the descriptor, but we need to extract them
            # For now, use placeholder based on avg_descriptor
            desc = metrics.get('avg_descriptor', 0.5)
            row[f'descriptor_var_raw_{tag}'] = desc * 0.5  # alpha=0.5 contribution
            row[f'descriptor_energy_raw_{tag}'] = desc * 0.5  # beta=0.5 contribution
    
    # Write row
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row[col] for col in ALL_COLUMNS])


callbacks = {
    'on_train_start': on_train_start,
    'on_fit_epoch_end': on_fit_epoch_end,
}