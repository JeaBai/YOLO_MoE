# Stability monitor callback: records training stability metrics (LR, weight decay, etc.).
import csv
from pathlib import Path


def on_train_start(trainer):
    """Initialize CSV path and write headers."""
    trainer._stab_metrics_csv = Path(trainer.save_dir) / 'metrics_stability.csv'
    trainer._stab_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = ['epoch', 'learning_rate', 'weight_decay']
    if not trainer._stab_metrics_csv.exists():
        with open(trainer._stab_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def on_train_batch_end(trainer):
    """Record stability metrics periodically (every epoch via batch-level check)."""
    # We use batch end to capture the last batch of each epoch
    csv_path = getattr(trainer, '_stab_metrics_csv', None)
    if csv_path is None:
        return

    # Only record at the last batch of each epoch
    if not hasattr(trainer, '_stab_last_recorded_epoch'):
        trainer._stab_last_recorded_epoch = -1

    current_epoch = trainer.epoch
    if current_epoch == trainer._stab_last_recorded_epoch:
        return
    trainer._stab_last_recorded_epoch = current_epoch

    epoch = current_epoch + 1

    # Learning rate and weight decay from optimizer
    lr = 0.0
    wd = 0.0
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        for pg in trainer.optimizer.param_groups:
            if 'lr' in pg:
                lr = float(pg['lr'])
            if 'weight_decay' in pg:
                wd = float(pg['weight_decay'])
            break  # first param group

    row = [epoch, lr, wd]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


callbacks = {
    'on_train_start': on_train_start,
    'on_train_batch_end': on_train_batch_end,
}