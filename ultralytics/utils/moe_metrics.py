# ultralytics-main/ultralytics/utils/moe_metrics.py
import torch
import torch.distributed as dist

def gather_moe_metrics(model):
    """收集模型中所有 SparseDualMoE 层的负载指标"""
    metrics = {}
    for name, module in model.named_modules():
        if hasattr(module, 'get_metrics') and callable(module.get_metrics):
            metrics[name] = module.get_metrics()
    return metrics

def sync_metrics_across_ranks(metrics):
    """分布式训练时同步指标（简化示例）"""
    if dist.is_available() and dist.is_initialized():
        # 这里只做简单的同步，实际可以收集所有 rank 的指标并平均
        # 根据实际需求实现
        pass
    return metrics