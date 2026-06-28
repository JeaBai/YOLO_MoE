# 显式分配 MoE (Explicit Allocation MoE) Spec

## Why
当前 YOLO_MoE 的 `SparseDualMoE` 使用隐式路由机制：`complexity_estimator`（黑盒 Conv2d+Sigmoid）学习动态 top_k，`UltraEfficientRouter` 通过 softmax+top-k 隐式决定专家激活。这导致专家负载不均、小分辨率特征图失效、分配过程不可解释。本 spec 将分配决策从"黑盒神经网络梯度隐式涌现"改为"显式确定性数学公式驱动"，实现真正的按需分配。

## What Changes
- **新增** `ExplicitDescriptor`：0 参数确定性特征描述子，基于逐样本方差+能量计算复杂度分数
- **新增** 层级预算级联（Cascade Budget）：通过 YAML 配置的预设权重前验注入，替代后验截断
- **新增** 直接映射（Direct Mapping）：闭式解 `top_k = 1 + round(clamp(s', 0, 1) × (K_max - 1))`，逐样本独立
- **移除** `complexity_estimator`（黑盒 Conv2d+Sigmoid）— **BREAKING**
- **移除** 强制激活专家机制（`forced_experts`、`hunger_counters`、`random_force_prob`）— **BREAKING**
- **移除** 知识蒸馏损失（`distill_coeff`）— **BREAKING**
- **移除** 动态容量因子（`capacity_factor`、复杂度估计器熵正则）— **BREAKING**
- **保留** 共享专家、`UltraEfficientRouter`（仅决定"哪个专家"）、融合门控、MoE 辅助损失
- **修改** `SparseDualMoE` 构造函数和 forward 流程

## Impact
- Affected specs: MoE routing, expert allocation, YAML model config
- Affected code: `ultralytics/nn/modules/moe/modules.py`, `ultralytics/nn/modules/moe/routers.py`, `ultralytics/cfg/models/moe26/*.yaml`, `ultralytics/nn/modules/moe/__init__.py`

---

## ADDED Requirements

### Requirement: ExplicitDescriptor — 确定性特征描述子
系统 SHALL 提供 `ExplicitDescriptor` 模块，替代黑盒 `complexity_estimator`，使用纯确定性数学公式计算每个样本的复杂度分数。

**公式：**
```
For each sample in batch:
  var_sample = mean_{h,w}(Var_{c}(x))          # 逐通道方差，在空间上取均值
  energy_sample = ||mean_{c,h,w}(x)||²          # 通道-空间均值向量的 L2 能量
  s = α × normalize(var_sample) + β × normalize(energy_sample)
```
其中 `normalize` 为 batch 内 min-max 归一化，`α` 和 `β` 为超参数（默认 `α=0.7, β=0.3`）。

#### Scenario: 正常前向传播
- **WHEN** 输入特征图 `x [B, C, H, W]` 进入 `ExplicitDescriptor`
- **THEN** 输出 `s [B, 1, 1, 1]`，每个样本的标量分数，值域 `[0, 1]`

#### Scenario: 小分辨率输入
- **WHEN** 输入特征图尺寸为 `2×2` 或更小
- **THEN** 方差和能量计算仍然有效（不会像 Sigmoid 输出接近 0），输出有效的复杂度分数

#### Scenario: 0 参数
- **WHEN** 检查 `ExplicitDescriptor` 的参数量
- **THEN** `sum(p.numel() for p in module.parameters()) == 0`

### Requirement: 层级预算级联（Cascade Budget）
系统 SHALL 支持通过 YAML 配置为不同特征层级指定预算权重，在显式分数上做前验缩放。

**公式：**
```
s' = s × w_layer
```
其中 `w_layer` 是预设超参数，默认值：`w_P3=0.5, w_P4=0.75, w_P5=1.0`（浅层压缩，深层放宽）。

#### Scenario: 层级权重配置
- **WHEN** 在 YAML 配置的 `SparseDualMoE` 参数中指定 `cascade_weight: 0.75`
- **THEN** 该层的复杂度分数按 `0.75` 缩放，影响后续 top_k 计算

#### Scenario: 默认权重
- **WHEN** 未指定 `cascade_weight`
- **THEN** 默认使用 `1.0`（不缩放）

### Requirement: 直接映射（Direct Mapping）
系统 SHALL 使用确定性闭式解将复杂度分数映射为每样本的 top_k。

**公式：**
```
top_k = 1 + round(clamp(s', 0, 1) × (K_max - 1))
```
其中 `K_max` 为最大专家激活数（即 `top_k` 超参数）。输出为 `[B]` 形状的整数张量，每个样本一个标量。

#### Scenario: 高复杂度样本
- **WHEN** 样本的 `s' = 0.9`，`K_max = 4`
- **THEN** `top_k = 1 + round(0.9 × 3) = 1 + 3 = 4`，激活全部专家

#### Scenario: 低复杂度样本
- **WHEN** 样本的 `s' = 0.1`，`K_max = 4`
- **THEN** `top_k = 1 + round(0.1 × 3) = 1 + 0 = 1`，仅激活 1 个专家

#### Scenario: 逐样本一致性
- **WHEN** 一个样本被分配 `top_k = 3`
- **THEN** 该样本的所有 H×W 个空间位置共享同一个 `top_k`，无空间位置间差异

### Requirement: 路由器角色简化
系统 SHALL 保留 `UltraEfficientRouter`，但其角色从"决定激活几个 + 激活谁"简化为仅"决定激活哪个专家"。

#### Scenario: 路由器接收动态 top_k
- **WHEN** 调用 `router(x, top_k=sample_top_k)`
- **THEN** 路由器按传入的 `top_k` 值选择专家，而非使用自身 `self.top_k`

#### Scenario: 推断模式
- **WHEN** 模型处于 `eval()` 模式
- **THEN** 不注入噪声、不计算 Z-loss、不计算 usage frequency

---

## MODIFIED Requirements

### Requirement: SparseDualMoE 构造函数
`SparseDualMoE` 构造函数 SHALL 移除以下参数：`capacity_factor`、`estimator_entropy_coeff`、`hysteresis_low`、`hysteresis_high`、`distill_coeff`、`forced_experts`、`hunger_threshold`、`forced_expert_weight`、`random_force_prob`。

新增以下参数：`cascade_weight: float = 1.0`、`descriptor_alpha: float = 0.7`、`descriptor_beta: float = 0.3`。

#### Scenario: 新构造函数签名
- **WHEN** 创建 `SparseDualMoE(in_channels=512, out_channels=512, num_experts=4, top_k=4, cascade_weight=0.75)`
- **THEN** 模块成功初始化，包含 `ExplicitDescriptor`、`shared_expert`、`experts`、`routing`、`fusion_gate`、`moe_loss_fn`

### Requirement: SparseDualMoE forward 流程
`SparseDualMoE.forward()` SHALL 按以下顺序执行：

1. `ExplicitDescriptor(x)` → 每样本复杂度分数 `s [B, 1, 1, 1]`
2. 层级预算注入：`s' = s × cascade_weight`
3. 直接映射：`top_k_per_sample = 1 + round(clamp(s', 0, 1) × (K_max - 1))`
4. 路由器：`routing_weights, routing_indices = router(x, top_k=top_k_per_sample)`
5. 有效性掩码：按 `top_k_per_sample` 截断路由权重
6. 共享专家 + 稀疏专家计算
7. 融合门控（使用 `top_k_per_sample` 信息）
8. 辅助损失计算

#### Scenario: 不再有动态容量和强制激活
- **WHEN** `SparseDualMoE.forward()` 执行
- **THEN** 不调用 `complexity_estimator`、不计算 `dynamic_top_k_float`、不执行 `forced_experts` 逻辑、不计算蒸馏损失

---

## REMOVED Requirements

### Requirement: complexity_estimator（黑盒复杂度估计器）
**Reason**: 黑盒 Conv2d+Sigmoid，不可解释，小分辨率时输出接近 0 导致 top_k=1，加剧专家负载不均。
**Migration**: 由 `ExplicitDescriptor` 替代。

### Requirement: forced_experts（强制激活专家机制）
**Reason**: 饥饿计数器、随机强制激活是为弥补隐式路由负载不均的补丁。显式分配从根源解决负载均衡，不再需要此机制。
**Migration**: 直接移除，无替代。

### Requirement: distill_coeff（知识蒸馏损失）
**Reason**: 蒸馏损失用于让未激活专家模仿共享专家输出，是隐式路由的补偿机制。显式分配下不适用。
**Migration**: 直接移除，无替代。

### Requirement: capacity_factor、hysteresis、estimator_entropy_coeff
**Reason**: 这些参数均为 `complexity_estimator` 的配套机制，随其一并移除。
**Migration**: 直接移除，无替代。