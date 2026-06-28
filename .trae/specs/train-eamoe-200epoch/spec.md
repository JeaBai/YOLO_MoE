# 训练 EAMoE 200 Epoch + 专家诊断 Spec

## Why
验证显式分配 MoE (EAMoE) 在训练中的实际表现：专家负载分布是否均衡、top_k 分配是否符合"按需分配"预期、训练稳定性。当前环境无 GPU，使用 coco8 数据集进行快速验证。

## What Changes
- **新增** 注册 `moe_callback` 到默认回调链，实现每个 epoch 结束后自动收集 MoE 专家指标
- **新增** 训练运行：coco8 数据集，200 epoch，记录专家诊断日志
- **新增** 训练后诊断：运行 `ExpertUsageTracker` 生成专家使用热图

## Impact
- Affected code: `ultralytics/utils/callbacks/base.py`, 训练输出目录

---

## ADDED Requirements

### Requirement: MoE Callback 自动注册
系统 SHALL 在每个训练 epoch 结束时自动收集 MoE 专家负载指标，无需用户手动配置。

#### Scenario: 训练时自动收集
- **WHEN** 训练进入 `on_train_epoch_end` 阶段
- **THEN** `gather_moe_metrics(model)` 被调用，指标写入 `trainer.metrics` 前缀 `moe/`

### Requirement: 200 Epoch 训练运行
系统 SHALL 支持使用 coco8 数据集训练 EAMoE 模型 200 epoch，并输出专家诊断日志。

#### Scenario: 训练完成
- **WHEN** 训练 200 epoch 完成
- **THEN** 输出 `runs/detect/train*/` 包含 `results.csv`（含 MoE 指标列）、模型权重、专家诊断日志