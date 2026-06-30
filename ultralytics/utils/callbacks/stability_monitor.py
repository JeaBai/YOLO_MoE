"""Stability monitoring callback for YOLO MoE experiments.
Records gradient health, MoE auxiliary losses, optimizer state, and GPU memory
every N steps (default 50) to stability_metrics.csv.
"""

import csv
import torch
from pathlib import Path

# Log every N steps
LOG_INTERVAL = 50

CSV_COLUMNS = [
    'epoch', 'step', 'global_step',
    'gradient_l2_norm',
    'balance_loss', 'z_loss', 'entropy_loss',
    'learning_rate', 'weight_decay',
    'gpu_memory_mib',
]

_step_counter = 0


def _compute_gradient_l2_norm(model):
    """Compute mean L2 norm of all parameter gradients."""
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
            count += 1
    if count == 0:
        return 0.0
    return (total_norm / count) ** 0.5


def _get_moe_aux_losses(model):
    """Extract MoE auxiliary losses from the model's loss registry."""
    from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
    
    balance_loss = 0.0
    z_loss = 0.0
    entropy_loss = 0.0
    
    for module in model.modules():
        if module in MOE_LOSS_REGISTRY:
            loss_dict = MOE_LOSS_REGISTRY[module]
            if isinstance(loss_dict, dict):
                balance_loss += float(loss_dict.get('balance_loss', 0))
                z_loss += float(loss_dict.get('z_loss', 0))
                entropy_loss += float(loss_dict.get('entropy_loss', 0))
            elif isinstance(loss_dict, (int, float)):
                balance_loss += float(loss_dict)
    
    return balance_loss, z_loss, entropy_loss


def _get_gpu_memory():
    """Get GPU memory usage in MiB, or 0 if CPU."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def on_train_start(trainer):
    """Initialize CSV at training start."""
    csv_path = Path(trainer.save_dir) / 'stability_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def on_train_batch_end(trainer):
    """Record stability metrics every LOG_INTERVAL steps."""
    global _step_counter
    _step_counter += 1
    
    if _step_counter % LOG_INTERVAL != 0:
        return
    
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module
    
    csv_path = Path(trainer.save_dir) / 'stability_metrics.csv'
    
    # Gradient L2 norm
    grad_norm = _compute_gradient_l2_norm(model)
    
    # MoE auxiliary losses
    bal_loss, z_loss, entropy_loss = _get_moe_aux_losses(model)
    
    # Optimizer state
    lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer, 'scheduler') and trainer.scheduler else float('nan')
    wd = trainer.optimizer.param_groups[0].get('weight_decay', 0) if hasattr(trainer, 'optimizer') else float('nan')
    
    # GPU memory
    gpu_mem = _get_gpu_memory()
    
    # Global step from trainer
    global_step = trainer.global_step if hasattr(trainer, 'global_step') else _step_counter
    
    row = [
        trainer.epoch + 1,
        _step_counter,
        global_step,
        grad_norm,
        bal_loss,
        z_loss,
        entropy_loss,
        lr,
        wd,
        gpu_mem,
    ]
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


callbacks = {
    'on_train_start': on_train_start,
    'on_train_batch_end': on_train_batch_end,
}
