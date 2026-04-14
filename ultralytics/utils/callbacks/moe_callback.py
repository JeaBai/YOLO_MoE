# ultralytics-main/ultralytics/utils/callbacks/moe_callback.py
from ultralytics.utils.moe_metrics import gather_moe_metrics

def on_train_epoch_end(trainer):
    """每个 epoch 结束后记录 MoE 指标"""
    model = trainer.model
    if hasattr(model, 'module'):  # DDP
        model = model.module

    # 收集指标
    metrics = gather_moe_metrics(model)

    # 将指标写入 trainer 的日志中，以便后续读取
    for module_name, module_metrics in metrics.items():
        for key, value in module_metrics.items():
            if isinstance(value, (int, float)):
                trainer.metrics[f'moe/{module_name}/{key}'] = value

    # 可选：重置模型内部记录，以便下一个 epoch 重新收集
    for module in model.modules():
        if hasattr(module, 'reset_metrics'):
            module.reset_metrics()

# 注册回调（通常在训练脚本中）
callbacks = {
    'on_train_epoch_end': on_train_epoch_end
}