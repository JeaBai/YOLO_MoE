# Checklist

- [x] `ExplicitDescriptor.forward()` 包含 within-batch min-max 归一化逻辑
- [x] `ExplicitDescriptor.__init__()` 默认 `alpha=0.5, beta=0.5`
- [x] `ExplicitDescriptor` 无任何可学习参数（`list(parameters()) == []`）
- [x] `ExplicitDescriptor` 无 running statistics（`state_dict()` 返回 `{}`）
- [x] B=1 边界情况不崩溃（除零保护）
- [x] 归一化后分数在 [0, alpha+beta] 区间
- [x] 单元测试全部通过（11/11 tests passed）
- [x] YAML 中 P4 `cascade_weight=1.0`
- [x] 训练 200 epoch 正常完成无报错
- [x] `moe_metrics.csv` 中 avg_descriptor 不再集中在 ~0.24 (P4=0.388, P5=0.488)
- [x] `moe_metrics.csv` 中 avg_topk >= 2.0 (P4=2.15, P5=2.46)
- [x] `moe_metrics.csv` 中 e2 专家使用率 > 5%（不再 dead）(P4=13.35%, P5=20.24%)
- [x] `moe_metrics.csv` 中 dead_experts < 2（P4=0.0, P5=0.0）