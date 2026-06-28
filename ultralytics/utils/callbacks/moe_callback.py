# ultralytics-main/ultralytics/utils/callbacks/moe_callback.py
import csv
from pathlib import Path
from ultralytics.utils.moe_metrics import gather_moe_metrics

def on_train_epoch_end(trainer):
    """每个 epoch 结束后记录 MoE 指标"""
    model = trainer.model
    if hasattr(model, 'module'):  # DDP
        model = model.module

    # 收集指标
    metrics = gather_moe_metrics(model)
    if not metrics:
        return

    # 展平为标量 key-value
    flat_metrics = {}
    for module_name, module_metrics in metrics.items():
        for key, value in module_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat_metrics[f'moe/{module_name}/{key}/{sub_key}'] = float(sub_value)
            elif isinstance(value, (int, float)):
                flat_metrics[f'moe/{module_name}/{key}'] = float(value)

    if not flat_metrics:
        return

    # 写入独立的 moe_metrics.csv
    csv_path = Path(trainer.save_dir) / 'moe_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(flat_metrics.keys())

    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + keys)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([trainer.epoch + 1] + [flat_metrics[k] for k in keys])

    # 重置模型内部记录，以便下一个 epoch 重新收集
    for module in model.modules():
        if hasattr(module, 'reset_metrics'):
            module.reset_metrics()

# 注册回调（通常在训练脚本中）
callbacks = {
    'on_train_epoch_end': on_train_epoch_end
}