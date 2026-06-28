# Tasks

## Task 1: 实现 ExplicitDescriptor 模块
**Files:**
- Create: `tests/nn/modules/moe/test_explicit_descriptor.py`
- Create: `ultralytics/nn/modules/moe/descriptor.py`

- [ ] **SubTask 1.1: 编写 ExplicitDescriptor 的单元测试（TDD-RED）**
  在 `tests/nn/modules/moe/test_explicit_descriptor.py` 中编写测试：
  - `test_output_shape`: 输入 `[B, C, H, W]`，输出 `[B, 1, 1, 1]`
  - `test_output_range`: 输出值域在 `[0, 1]` 内
  - `test_zero_parameters`: `sum(p.numel() for p in module.parameters()) == 0`
  - `test_deterministic`: 相同输入产生相同输出（无随机性）
  - `test_small_resolution`: 输入 `[2, 64, 2, 2]` 不崩溃，输出有效
  - `test_high_variance_gives_high_score`: 高方差输入 → 高分数
  - `test_low_variance_gives_low_score`: 低方差输入 → 低分数

- [ ] **SubTask 1.2: 运行测试验证失败**
  Run: `pytest tests/nn/modules/moe/test_explicit_descriptor.py -v`
  Expected: 全部 FAIL（模块未实现）

- [ ] **SubTask 1.3: 实现 ExplicitDescriptor**
  在 `ultralytics/nn/modules/moe/descriptor.py` 中实现：
  ```python
  import torch
  import torch.nn as nn

  class ExplicitDescriptor(nn.Module):
      """0-parameter deterministic complexity descriptor using variance + energy."""
      def __init__(self, alpha: float = 0.7, beta: float = 0.3):
          super().__init__()
          self.alpha = alpha
          self.beta = beta

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          B, C, H, W = x.shape
          # per-sample variance: Var_c(x) then mean_{h,w}
          var_c = x.var(dim=1, unbiased=False)  # [B, H, W]
          var_sample = var_c.mean(dim=[1, 2])    # [B]
          # per-sample energy: ||mean_{c,h,w}(x)||²
          mean_spatial = x.mean(dim=[2, 3])      # [B, C]
          energy_sample = mean_spatial.pow(2).sum(dim=1)  # [B]

          # batch内 min-max normalize
          def batch_norm(v):
              vmin = v.min()
              vmax = v.max()
              if vmax - vmin < 1e-8:
                  return torch.zeros_like(v)
              return (v - vmin) / (vmax - vmin)

          var_norm = batch_norm(var_sample)
          energy_norm = batch_norm(energy_sample)

          s = self.alpha * var_norm + self.beta * energy_norm
          return s.view(B, 1, 1, 1)
  ```

- [ ] **SubTask 1.4: 运行测试验证通过**
  Run: `pytest tests/nn/modules/moe/test_explicit_descriptor.py -v`
  Expected: 全部 PASS

- [ ] **SubTask 1.5: 提交**
  ```bash
  git add tests/nn/modules/moe/test_explicit_descriptor.py ultralytics/nn/modules/moe/descriptor.py
  git commit -m "feat: add ExplicitDescriptor with deterministic variance+energy scoring"
  ```

## Task 2: 实现直接映射（Direct Mapping）函数
**Files:**
- Create: `tests/nn/modules/moe/test_direct_mapping.py`
- Modify: `ultralytics/nn/modules/moe/descriptor.py`

- [ ] **SubTask 2.1: 编写 direct_mapping 的单元测试（TDD-RED）**
  在 `tests/nn/modules/moe/test_direct_mapping.py` 中编写测试：
  - `test_max_complexity`: `s=0.95, K_max=4` → `top_k=4`
  - `test_min_complexity`: `s=0.05, K_max=4` → `top_k=1`
  - `test_mid_complexity`: `s=0.5, K_max=4` → `top_k=2` (round(0.5×3)=2)
  - `test_clamp_upper`: `s=1.5, K_max=4` → `top_k=4`（clamp 到 1）
  - `test_clamp_lower`: `s=-0.5, K_max=4` → `top_k=1`（clamp 到 0）
  - `test_output_shape`: 输入 `s [B, 1, 1, 1]`，输出 `top_k [B]` 整数张量
  - `test_output_dtype`: 输出为整数类型

- [ ] **SubTask 2.2: 运行测试验证失败**
  Run: `pytest tests/nn/modules/moe/test_direct_mapping.py -v`
  Expected: 全部 FAIL（函数未实现）

- [ ] **SubTask 2.3: 实现 direct_mapping 函数**
  在 `ultralytics/nn/modules/moe/descriptor.py` 中追加：
  ```python
  def direct_mapping(s: torch.Tensor, k_max: int) -> torch.Tensor:
      """Closed-form per-sample top_k mapping.
      top_k = 1 + round(clamp(s, 0, 1) * (k_max - 1))
      """
      s_clamped = s.clamp(0.0, 1.0)
      top_k = 1 + torch.round(s_clamped * (k_max - 1)).int()
      return top_k.view(-1)  # [B]
  ```

- [ ] **SubTask 2.4: 运行测试验证通过**
  Run: `pytest tests/nn/modules/moe/test_direct_mapping.py -v`
  Expected: 全部 PASS

- [ ] **SubTask 2.5: 提交**
  ```bash
  git add tests/nn/modules/moe/test_direct_mapping.py ultralytics/nn/modules/moe/descriptor.py
  git commit -m "feat: add direct_mapping closed-form top_k computation"
  ```

## Task 3: 重构 SparseDualMoE — 移除隐式组件，集成显式分配
**Files:**
- Modify: `ultralytics/nn/modules/moe/modules.py`
- Modify: `ultralytics/nn/modules/moe/__init__.py`

- [ ] **SubTask 3.1: 编写 SparseDualMoE 新 forward 的集成测试（TDD-RED）**
  在 `tests/nn/modules/moe/test_explicit_descriptor.py` 中追加：
  - `test_sparse_dual_moe_explicit_forward`: 创建 `SparseDualMoE`，前向传播不崩溃，输出形状正确
  - `test_sparse_dual_moe_no_complexity_estimator`: 确认 `complexity_estimator` 属性不存在
  - `test_sparse_dual_moe_no_forced_experts`: 确认 `forced_experts` 属性不存在
  - `test_sparse_dual_moe_has_descriptor`: 确认 `descriptor` 属性存在且为 `ExplicitDescriptor`
  - `test_sparse_dual_moe_has_cascade_weight`: 确认 `cascade_weight` 属性存在

- [ ] **SubTask 3.2: 运行测试验证失败**
  Run: `pytest tests/nn/modules/moe/test_explicit_descriptor.py -v -k "sparse_dual"`
  Expected: 全部 FAIL（新参数/属性未实现）

- [ ] **SubTask 3.3: 重构 SparseDualMoE**
  修改 `ultralytics/nn/modules/moe/modules.py`：
  - 移除构造函数参数：`capacity_factor`、`estimator_entropy_coeff`、`hysteresis_low`、`hysteresis_high`、`distill_coeff`、`forced_experts`、`hunger_threshold`、`forced_expert_weight`、`random_force_prob`
  - 新增构造函数参数：`cascade_weight: float = 1.0`、`descriptor_alpha: float = 0.7`、`descriptor_beta: float = 0.3`
  - 移除 `self.complexity_estimator`、`self.hunger_counters`、`self._prev_infer_topk` 的初始化
  - 新增 `self.descriptor = ExplicitDescriptor(alpha=descriptor_alpha, beta=descriptor_beta)`
  - 新增 `self.cascade_weight = cascade_weight`
  - 移除蒸馏损失、强制激活、动态容量相关逻辑
  - 修改 `forward()` 为新的显式分配流程

  具体代码变更：

  移除构造函数中以下参数定义：
  ```
  capacity_factor, estimator_entropy_coeff, hysteresis_low, hysteresis_high,
  distill_coeff, forced_experts, hunger_threshold, forced_expert_weight, random_force_prob
  ```

  新增：
  ```python
  cascade_weight: float = 1.0,
  descriptor_alpha: float = 0.7,
  descriptor_beta: float = 0.3,
  ```

  移除属性存储：
  ```python
  # 删除这些
  self.capacity_factor = capacity_factor
  self.estimator_entropy_coeff = estimator_entropy_coeff
  self.hysteresis_low = hysteresis_low
  self.hysteresis_high = hysteresis_high
  self.distill_coeff = distill_coeff
  self.forced_experts = forced_experts
  self.hunger_threshold = hunger_threshold
  self.forced_expert_weight = forced_expert_weight
  self.random_force_prob = random_force_prob
  self.hunger_counters = ...
  self._prev_infer_topk = ...
  ```

  新增：
  ```python
  self.cascade_weight = cascade_weight
  self.descriptor = ExplicitDescriptor(alpha=descriptor_alpha, beta=descriptor_beta)
  ```

  移除 `self.complexity_estimator = nn.Sequential(...)` 初始化

  forward() 替换为：
  ```python
  def forward(self, x):
      B, C, H, W = x.shape

      # 1. ExplicitDescriptor: per-sample complexity score
      s = self.descriptor(x)  # [B, 1, 1, 1]

      # 2. Cascade budget injection
      s_scaled = s * self.cascade_weight

      # 3. Direct mapping: per-sample top_k
      top_k_per_sample = direct_mapping(s_scaled, self.top_k)  # [B]

      # 4. Router: determines which experts
      routing_result = self.routing(x)
      routing_weights = routing_result[0]  # [B, top_k, 1, 1]
      routing_indices = routing_result[1]  # [B, top_k, 1, 1]

      # 5. Validity mask: filter by per-sample top_k
      k_range = torch.arange(self.top_k, device=x.device).view(1, -1, 1, 1)
      top_k_expanded = top_k_per_sample.view(B, 1, 1, 1)
      mask = (k_range < top_k_expanded).to(routing_weights.dtype)
      masked_weights = routing_weights * mask
      sum_masked = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
      normalized_weights = masked_weights / sum_masked

      # 6. Shared expert + sparse experts
      shared_out = self.shared_expert(x)
      from .utils import BatchedExpertComputation
      expert_out = BatchedExpertComputation.compute_sparse_experts_batched(
          x, self.experts, normalized_weights, routing_indices,
          self.top_k, self.num_experts
      )

      # 7. Fusion gate
      gate_pool = F.adaptive_avg_pool2d(shared_out + expert_out, 1)
      topk_info = top_k_per_sample.view(B, 1, 1, 1).float() / self.top_k
      gate_input = torch.cat([gate_pool, topk_info], dim=1)
      gate_weights = self.fusion_gate(gate_input)
      output = gate_weights[:, 0:1, :, :] * shared_out + gate_weights[:, 1:2, :, :] * expert_out

      # 8. Auxiliary loss
      if self.training:
          step = self.global_step.item()
          if step < self.balance_warmup_steps:
              progress = step / max(1, self.balance_warmup_steps - 1)
              coeff = self.balance_loss_coeff * (1 - progress) + self.balance_loss_min * progress
          else:
              coeff = self.balance_loss_min
          self.moe_loss_fn.balance_loss_coeff = coeff

          probs = routing_result[5]
          aux_loss = self.moe_loss_fn(probs, self.routing.router(x).clamp(-30, 30), routing_indices)
          MOE_LOSS_REGISTRY[self] = aux_loss
          self.global_step.add_(1)

      return output
  ```

- [ ] **SubTask 3.4: 运行测试验证通过**
  Run: `pytest tests/nn/modules/moe/test_explicit_descriptor.py -v -k "sparse_dual"`
  Expected: 全部 PASS

- [ ] **SubTask 3.5: 更新 __init__.py 导出**
  在 `ultralytics/nn/modules/moe/__init__.py` 中添加：
  ```python
  from .descriptor import ExplicitDescriptor, direct_mapping
  ```
  在 `__all__` 中添加 `"ExplicitDescriptor"`, `"direct_mapping"`

- [ ] **SubTask 3.6: 提交**
  ```bash
  git add ultralytics/nn/modules/moe/modules.py ultralytics/nn/modules/moe/__init__.py
  git commit -m "refactor: replace implicit complexity estimator with explicit descriptor allocation"
  ```

## Task 4: 更新 YAML 模型配置文件
**Files:**
- Modify: `ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml`

- [ ] **SubTask 4.1: 更新 YAML 配置**
  将 `YOLO_SparseDualMoE.yaml` 中 `SparseDualMoE` 的参数从旧参数迁移到新参数：
  - 第 6 层（P4/16）：`SparseDualMoE, [512, 4, 2, 0.2, 0.01, 5000, 1e-3, 0.001, 1.5, 8, 0.01, 0.3, 0.7, 0.0, True, 3, 0.5, 0.2]`
    → `SparseDualMoE, [512, 4, 2, 0.2, 0.01, 5000, 1e-3, 0.001, 8, 0.75]`
    （保留：out_channels, num_experts, top_k, balance_loss_coeff, balance_loss_min, balance_warmup_steps, router_z_loss_coeff, entropy_loss_coeff, num_groups；新增：cascade_weight=0.75）
  - 第 8 层（P5/32）：`SparseDualMoE, [1024, 4, 2, 0.2, 0.01, 5000, 1e-3, 0.001, 1.5, 8, 0.01, 0.3, 0.7, 0.0, True, 3, 0.5, 0.2]`
    → `SparseDualMoE, [1024, 4, 2, 0.2, 0.01, 5000, 1e-3, 0.001, 8, 1.0]`
    （cascade_weight=1.0，深层不压缩）

- [ ] **SubTask 4.2: 提交**
  ```bash
  git add ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml
  git commit -m "config: update MoE YAML to use explicit allocation parameters"
  ```

## Task 5: 端到端验证
**Files:**
- Run: 集成测试

- [ ] **SubTask 5.1: 模型加载验证**
  确认 YAML 配置可以正确解析并构建模型：
  ```bash
  python -c "
  from ultralytics import YOLO
  model = YOLO('ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml')
  print('Model built successfully')
  # Check descriptor exists
  for name, m in model.model.named_modules():
      if hasattr(m, 'descriptor'):
          print(f'{name}: descriptor found, params={sum(p.numel() for p in m.descriptor.parameters())}')
  "
  ```
  Expected: 模型构建成功，descriptor 参数为 0

- [ ] **SubTask 5.2: 前向传播验证**
  确认模型可以正常前向传播：
  ```bash
  python -c "
  import torch
  from ultralytics import YOLO
  model = YOLO('ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml')
  x = torch.randn(2, 3, 640, 640)
  model.model.eval()
  with torch.no_grad():
      out = model.model(x)
  print('Forward pass OK, output shape:', out.shape if isinstance(out, torch.Tensor) else 'tuple')
  "
  ```
  Expected: 前向传播成功，无 NaN/Inf

- [ ] **SubTask 5.3: 完整单元测试套件**
  ```bash
  pytest tests/nn/modules/moe/ -v
  ```
  Expected: 全部 PASS

# Task Dependencies
- Task 2 depends on Task 1 (direct_mapping 在 descriptor.py 中)
- Task 3 depends on Task 1, Task 2 (SparseDualMoE 依赖 ExplicitDescriptor 和 direct_mapping)
- Task 4 depends on Task 3 (YAML 参数需匹配新构造函数)
- Task 5 depends on Task 1, Task 2, Task 3, Task 4