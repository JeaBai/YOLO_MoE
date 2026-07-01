# Comprehensive metrics callback: records core accuracy/efficiency metrics per epoch.
import csv
import time
from pathlib import Path


def on_train_start(trainer):
    """Initialize CSV paths and write headers."""
    trainer._comp_metrics_csv = Path(trainer.save_dir) / 'metrics_core.csv'
    trainer._comp_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = ['epoch', 'mAP50', 'mAP50_95', 'box_loss', 'cls_loss', 'dfl_loss', 'inference_speed_ms']
    if not trainer._comp_metrics_csv.exists():
        with open(trainer._comp_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def on_fit_epoch_end(trainer):
    """After each fit epoch (train + val), record core metrics."""
    csv_path = getattr(trainer, '_comp_metrics_csv', None)
    if csv_path is None:
        return

    epoch = trainer.epoch + 1

    # Validation metrics (mAP)
    metrics = getattr(trainer, 'metrics', {}) or {}
    mAP50 = float(metrics.get('metrics/mAP50(B)', 0.0))
    mAP50_95 = float(metrics.get('metrics/mAP50-95(B)', 0.0))

    # Training losses
    loss_items = getattr(trainer, 'loss_items', None)
    if loss_items is not None and hasattr(loss_items, 'tolist'):
        losses = loss_items.tolist()
        box_loss = float(losses[0]) if len(losses) > 0 else 0.0
        cls_loss = float(losses[1]) if len(losses) > 1 else 0.0
        dfl_loss = float(losses[2]) if len(losses) > 2 else 0.0
    else:
        box_loss = cls_loss = dfl_loss = 0.0

    # Inference speed (ms/img, excluding warmup)
    speed = 0.0
    if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'speed'):
        val_speed = trainer.validator.speed
        if val_speed and len(val_speed) >= 3:
            # val_speed = [preprocess, inference, loss, postprocess] in ms
            speed = float(val_speed[1]) if val_speed[1] is not None else 0.0

    row = [epoch, mAP50, mAP50_95, box_loss, cls_loss, dfl_loss, speed]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


callbacks = {
    'on_train_start': on_train_start,
    'on_fit_epoch_end': on_fit_epoch_end,
}