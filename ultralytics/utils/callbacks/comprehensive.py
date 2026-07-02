# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import csv
import time
import numpy as np
from pathlib import Path
from ultralytics.utils.moe_metrics import gather_moe_metrics
import torch

def comprehensive_on_train_start(trainer):
    """Initialize comprehensive metrics CSV at the start of training."""
    csv_path = Path(trainer.save_dir) / 'comprehensive_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build header
    header = ['epoch', 'lr', 'box_loss', 'cls_loss', 'dfl_loss', 
              'mAP50', 'mAP50-95', 'params', 'balance_loss', 'z_loss', 
              'entropy_loss', 'gradient_l2_norm', 'inference_speed_ms']
    
    # Add MoE-specific columns
    moe_metrics = gather_moe_metrics(trainer.model)
    if moe_metrics:
        for module_name, module_metrics in moe_metrics.items():
            for key, value in module_metrics.items():
                if isinstance(value, dict):
                    for sub_key in value:
                        header.append(f'{module_name}/{key}/{sub_key}')
                elif isinstance(value, (int, float)):
                    header.append(f'{module_name}/{key}')
    
    # Create file with header
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    # Store params count once at start
    n_params = sum(p.numel() for p in trainer.model.parameters())
    trainer._comprehensive_params = n_params

def _to_scalar(v):
    """Convert tensor to float if needed, otherwise return as-is."""
    if hasattr(v, 'item'):
        return v.item()
    return v

def comprehensive_on_train_epoch_end(trainer):
    """Record comprehensive metrics at the end of each epoch."""
    csv_path = Path(trainer.save_dir) / 'comprehensive_metrics.csv'
    epoch = trainer.epoch + 1
    
    # Collect category 1: core metrics (precision/efficiency)
    row = [
        epoch,
        _to_scalar(trainer.optimizer.param_groups[0]['lr']),  # lr
        _to_scalar(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else '',  # box_loss
        _to_scalar(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else '',  # cls_loss
        _to_scalar(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else '',  # dfl_loss
        _to_scalar(trainer.metrics.get('metrics/mAP50(B)', '')),  # mAP50
        _to_scalar(trainer.metrics.get('metrics/mAP50-95(B)', '')),  # mAP50-95
        getattr(trainer, '_comprehensive_params', ''),  # params (fixed)
    ]
    
    # Collect category 3: training stability
    # MoE auxiliary losses
    balance_loss = z_loss = entropy_loss = ''
    if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 3:
        if len(trainer.loss_items) >= 4:
            balance_loss = _to_scalar(trainer.loss_items[3])
        if len(trainer.loss_items) >= 5:
            z_loss = _to_scalar(trainer.loss_items[4])
        if len(trainer.loss_items) >= 6:
            entropy_loss = _to_scalar(trainer.loss_items[5])
    row.extend([balance_loss, z_loss, entropy_loss])
    
    # Gradient L2 norm (sample every 10 epochs)
    grad_norm = ''
    if epoch % 10 == 0:
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = np.sqrt(total_norm)
    row.append(grad_norm)
    
    # Inference speed (measure every 20 epochs on one batch)
    speed_ms = ''
    if epoch % 20 == 0:
        # Warmup
        device = next(trainer.model.parameters()).device
        dummy = torch.randn(1, 3, trainer.args.imgsz, trainer.args.imgsz).to(device)
        for _ in range(5):
            with torch.no_grad():
                trainer.model(dummy)
        # Measure
        start = time.time()
        n_runs = 10
        for _ in range(n_runs):
            with torch.no_grad():
                trainer.model(dummy)
        elapsed_ms = (time.time() - start) * 1000 / n_runs
        speed_ms = elapsed_ms
    row.append(speed_ms)
    
    # Collect category 2: MoE explicit allocation metrics
    moe_metrics = gather_moe_metrics(trainer.model)
    flat_metrics = {}
    if moe_metrics:
        for module_name, module_metrics in moe_metrics.items():
            for key, value in module_metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            flat_metrics[f'{module_name}/{key}/{sub_key}'] = float(sub_value)
                elif isinstance(value, (int, float)):
                    flat_metrics[f'{module_name}/{key}'] = float(value)
    
    # Append MoE metrics to row (empty if not MoE)
    # Need to keep column order consistent with header
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    for col in header[len(row):]:
        row.append(flat_metrics.get(col, ''))
    
    # Write row
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    # Reset MoE metrics for next epoch
    for module in trainer.model.modules():
        if hasattr(module, 'reset_metrics'):
            module.reset_metrics()

def comprehensive_on_train_end(trainer):
    """Finalize at the end of training."""
    csv_path = Path(trainer.save_dir) / 'comprehensive_metrics.csv'
    print(f'\nComprehensive metrics saved to: {csv_path}')

# Callback dictionary
callbacks = {
    'on_train_start': comprehensive_on_train_start,
    'on_train_epoch_end': comprehensive_on_train_epoch_end,
    'on_train_end': comprehensive_on_train_end,
}