# Tasks

## Task 1: 修复 ExplicitDescriptor 归一化逻辑
**Files:**
- Modify: `ultralytics/nn/modules/moe/descriptor.py`

- [x] **SubTask 1.1: 实现 within-batch min-max 归一化**
  在 `ExplicitDescriptor.forward()` 中：
  1. 计算 `var_sample` 和 `energy_sample`（保持现有逻辑）
  2. 对每个做 within-batch min-max 归一化：`v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)`
  3. 处理 B=1 边界情况：max==min 时 `v_norm` 设为 0
  4. 组合：`s = alpha * var_norm + beta * energy_norm`
  5. 移除 `torch.clamp(s, 0.0, 1.0)`（归一化已保证 [0, 1] 区间）

- [x] **SubTask 1.2: 更新默认参数**
  修改 `__init__` 默认值：`alpha=0.5, beta=0.5`

- [x] **SubTask 1.3: 验证零参数承诺**
  确认 `list(descriptor.parameters())` 返回 `[]`，`descriptor.state_dict()` 返回 `{}`

## Task 2: 更新单元测试
**Files:**
- Modify: `tests/nn/modules/moe/test_explicit_descriptor.py`
- Modify: `tests/nn/modules/moe/test_direct_mapping.py`

- [x] **SubTask 2.1: 添加归一化行为测试**
  - 测试 batch 内 min-max 归一化后分数在 [0, alpha+beta] 区间
  - 测试 B=1 时不崩溃
  - 测试零参数（`list(descriptor.parameters()) == []`）
  - 测试不同 batch 的分数分布覆盖 [0, 1] 全区间

- [x] **SubTask 2.2: 更新 direct_mapping 集成测试**
  验证归一化后 descriptor 输出 + direct_mapping 能产生 top_k >= 2 的结果

- [x] **SubTask 2.3: 运行全部测试**
  11/11 tests passed

## Task 3: 更新 YAML 配置
**Files:**
- Modify: `ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml`

- [x] **SubTask 3.1: P4 cascade_weight 对齐**
  将 P4 层的 `cascade_weight` 从 `0.75` 改为 `1.0`，与 P5 保持一致
  Also added `alpha=0.5, beta=0.5` to both P4 and P5

- [x] **SubTask 3.2: 更新 modules.py 默认值**
  将 `descriptor_alpha` 从 `0.7` 改为 `0.5`，`descriptor_beta` 从 `0.3` 改为 `0.5`

## Task 4: 运行 200 Epoch 验证训练
**Files:**
- Run: CLI 训练命令

- [x] **SubTask 4.1: 确认环境可用**
  PyTorch 2.12.1, coco8 数据集已下载

- [x] **SubTask 4.2: 启动训练**
  200 epoch，top_k=4，完成。moe_metrics.csv 已生成

- [x] **SubTask 4.3: 诊断验证**
  - avg_descriptor: P4=0.388, P5=0.488 (不再集中在 ~0.24)
  - avg_topk: P4=2.15, P5=2.46 (均 >= 2.0)
  - e2 使用率: P4=13.35%, P5=20.24% (均 > 5%，不再 dead)
  - dead_experts: P4=0.0, P5=0.0 (均 < 2)

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 is independent of Task 1, 2
- Task 4 depends on Task 1, 2, 3