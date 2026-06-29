# 修复 Descriptor 归一化策略 Spec

## Why
当前 ExplicitDescriptor 在 forward 中直接对 raw var_sample 和 energy_sample 做 `alpha * var + beta * energy` 再 `clamp(0,1)`。由于 raw 值极小（~0.001 量级），clamp 截断后分数集中在 0.19-0.26 窄区间，导致 top_k 仅 1-2，4 个专家利用率仅 25-50%，e2 专家永久死亡。

根因：缺少归一化步骤，raw 值的量级差异被 clamp 抹平，丧失动态范围。

## What Changes
- **修复** `ExplicitDescriptor.forward()`：对 var_sample 和 energy_sample 分别做 within-batch min-max 归一化到 [0,1]，再加权组合
- **微调** 默认参数：`alpha=0.5, beta=0.5`（等权，归一化后各贡献 0-0.5）
- **微调** YAML 配置：P4 `cascade_weight=1.0`（与 P5 对齐）
- **明确**：归一化逻辑为纯数学操作（min-max），零可学习参数，零 running statistics

## Impact
- Affected specs: `add-explicit-allocation-moe`（descriptor 实现变更）
- Affected code: `ultralytics/nn/modules/moe/descriptor.py`, `ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml`
- Not affected: `modules.py`（接口不变）, router, experts, fusion gate

---

## MODIFIED Requirements

### Requirement: ExplicitDescriptor 归一化
`ExplicitDescriptor.forward()` SHALL 对 raw var_sample 和 energy_sample 分别应用 within-batch min-max 归一化到 [0, 1]，再进行加权组合，确保输出分数始终覆盖 [0, 1] 全区间。

#### Scenario: 正常 batch (B >= 2)
- **WHEN** 输入 batch 中有多个样本，var_sample 和 energy_sample 的值域跨度 > 1e-8
- **THEN** 归一化后 var_norm 和 energy_norm 均在 [0, 1] 区间，且 min=0, max=1
- **AND** 输出 `s = alpha * var_norm + beta * energy_norm` 在 [0, alpha+beta] 区间

#### Scenario: 单样本 batch (B = 1)
- **WHEN** batch_size = 1，var_sample 或 energy_sample 的 min == max
- **THEN** 对应归一化值设为 0（避免除零），不崩溃

#### Scenario: 零参数保证
- **WHEN** 检查 `ExplicitDescriptor` 的 `parameters()` 和 `state_dict()`
- **THEN** 返回空（无任何可学习参数或 running statistics）

### Requirement: 默认参数微调
系统 SHALL 使用以下默认参数：
- `descriptor_alpha=0.5, descriptor_beta=0.5`（等权归一化）
- YAML 中 P4 层 `cascade_weight=1.0`（与 P5 对齐）
- 其他参数不变