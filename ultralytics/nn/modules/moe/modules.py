import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import weakref
from typing import Tuple, Dict, Optional, Union
from .utils import FlopsUtils, get_safe_groups, BatchedExpertComputation
from .experts import (
    OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert,
    InvertedResidualExpert, EfficientExpertGroup
)
from .routers import (
    UltraEfficientRouter
)
from torch.amp import autocast
from .loss import MoELoss

# Global registry to store auxiliary losses for MoE modules
# This prevents storing non-leaf tensors in the module instance, avoiding deepcopy errors
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()

# ---------------------------------------------------------------------------
#   稀疏双路MoE
# ---------------------------------------------------------------------------

class SparseDualMoE(nn.Module):
    indtance_counter = 0    # 全局实例

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_coeff: float = 0.2,
        balance_loss_min: float = 0.01,
        balance_warmup_steps: int = 5000,
        router_z_loss_coeff: float = 1e-3,
        entropy_loss_coeff: float = 0.001,
        capacity_factor: float = 1.5,
        num_groups: int = 8,

        estimator_entropy_coeff: float = 0.01,  # 估计器熵正则系数
        hysteresis_low: float = 0.3,            # 推理滞后低阈值
        hysteresis_high: float = 0.7,           # 推理滞后高阈值

        distill_coeff: float = 0.0,             # 知识共享系数，0表示关闭

        forced_experts: bool = True,            # 强制激活专家，False关闭
        hunger_threshold: int = 3,              # 连续未被选中的次数阈值
        forced_expert_weight: float = 0.5,      # 强制激活专家的输出权重
        random_force_prob: float = 0.2,         # 无饥饿专家时随机激活的概率
    ):

        super().__init__()
        # 参数存储...
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.balance_loss_min = balance_loss_min
        self.balance_warmup_steps = balance_warmup_steps
        self.capacity_factor = capacity_factor

        self.estimator_entropy_coeff = estimator_entropy_coeff
        self.hysteresis_low = hysteresis_low
        self.hysteresis_high = hysteresis_high

        self.distill_coeff = distill_coeff
        self.forced_experts = forced_experts
        self.hunger_threshold = hunger_threshold
        self.forced_expert_weight = forced_expert_weight
        self.random_force_prob = random_force_prob
        # 饥饿计数器 buffer（不参与梯度）
        self.register_buffer('hunger_counters', torch.zeros(self.num_experts, dtype=torch.long))

        # 分配唯一序号
        self.modelID = SparseDualMoE.indtance_counter
        SparseDualMoE.indtance_counter += 1

        # 推理时滞后状态缓存（不参与梯度，不保存到 state_dict）
        self.register_buffer('_prev_infer_topk', torch.tensor(self.top_k, dtype=torch.long), persistent=False)

        # 注册 buffer
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_balance_coeff', torch.tensor(balance_loss_coeff))

        # 1. 复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        # 2. 使用 UltraEfficientRouter（轻量且保留空间信息）
        self.routing = UltraEfficientRouter(
            in_channels, num_experts,
            reduction=16,           # 更激进的压缩
            top_k=top_k,
            noise_std=0.1,
            temperature=1.0,
            pool_scale=8
        )

        # 3. 使用分组卷积
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            # 可选异构：不同kernel_size
            ks = 3 if i % 2 == 0 else 5
            self.experts.append(
                InvertedResidualExpert(
                    in_channels, out_channels,
                    expand_ratio=2.0,
                    kernel_size=ks,
                    groups=num_groups
                )
            )

        # 4. 共享专家
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False, groups=num_groups),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            nn.SiLU(inplace=True)
        )

        # 5. 可学习融合门控
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 2, 1),
            nn.Sigmoid()
        )

        # 6. 辅助损失（增加软路由支持）
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=router_z_loss_coeff,
            entropy_loss_coeff=entropy_loss_coeff,
            num_experts=num_experts,
            top_k=top_k,
            use_soft_balancing=True   # 使用概率代替硬计数，更平滑
        )

        # 修改融合门控：额外接收复杂度信息（通道数+1）
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels + 1, 2, 1),  # 多一个通道用于拼接 dynamic_top_k 信息
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ---------- 1. 复杂度估计与动态 Top-K（逐样本）----------
        complexity = self.complexity_estimator(x)  # [B, 1, 1, 1]
        complexity_flat = complexity.view(B)  # [B]

        if self.training:

            # 计算动态 top_k（逐样本）
            dynamic_top_k_float = self.top_k * complexity_flat * self.capacity_factor
            # # 加入微小噪声，增加探索性
            # if self.training:
            #     if self.global_step % 50 == 0:
            #         print(f'无噪dynamic_top_k_float = {dynamic_top_k_float}')
            #     noise = torch.rand_like(dynamic_top_k_float) - 0.5
            #     dynamic_top_k_float = dynamic_top_k_float + noise * 0.1
            #     if self.global_step % 50 == 0:
            #         print(f'加噪dynamic_top_k_float = {dynamic_top_k_float}')

            dynamic_top_k = torch.round(dynamic_top_k_float).int().clamp(min=1, max=self.top_k)  # [B]

            # 每 500步 打印一次诊断信息
            if self.global_step % 372 == 0:
                label_name = f"model.{6 if self.modelID % 2 == 0 else 8}"
                print(f"\n{label_name}  [Step {self.global_step.item()}]"
                      f"复杂度(complexity): min={complexity_flat.min().item():.3f}, "
                      f"max={complexity_flat.max().item():.3f}, "
                      f"mean={complexity_flat.mean().item():.3f}"
                      f"计算动态 top_k（逐样本）dynamic_top_k: {dynamic_top_k}\n"
                )

            # 熵正则化损失：鼓励估计器输出在 0.5 附近
            if self.estimator_entropy_coeff > 0:
                # 二元熵： - (p*log(p) + (1-p)*log(1-p))
                p = complexity_flat.clamp(1e-6, 1 - 1e-6)
                entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
                est_entropy_loss = entropy.mean()
                # 注意：此损失需要加到总辅助损失中，我们稍后处理
            else:
                est_entropy_loss = torch.tensor(0.0, device=x.device)
        else:
            # 推理时：滞后阈值平滑
            # 计算原始浮点 top_k
            raw_top_k_float = self.top_k * complexity_flat * self.capacity_factor
            # 四舍五入
            new_top_k = torch.round(raw_top_k_float).int().clamp(min=1, max=self.top_k)
            # 仅当复杂度超过阈值区间才更新
            high_mask = complexity_flat > self.hysteresis_high
            low_mask = complexity_flat < self.hysteresis_low
            # 其他样本保持上一次的值（使用 buffer）
            prev = self._prev_infer_topk.expand_as(new_top_k)
            dynamic_top_k = torch.where(high_mask, new_top_k, prev)
            dynamic_top_k = torch.where(low_mask, new_top_k, dynamic_top_k)
            # 更新 buffer（对于整个批次，取众数或均值？为简化，逐样本存储需改为状态向量，这里简单处理：记录最后一个样本的值用于下次）
            self._prev_infer_topk.fill_(dynamic_top_k[-1].item())  # 简单起见，后续可改为完整向量

        # ---------- 2. 路由（获取固定数量的 Top-K 专家）----------
        # 注意：路由器的 forward 返回 topk_vals, topk_indices, ... 且形状固定为 [B, self.top_k, 1, 1]
        routing_result = self.routing(x)
        routing_weights = routing_result[0]  # [B, top_k, 1, 1]
        routing_indices = routing_result[1]  # [B, top_k, 1, 1]
        # ... 其他返回值（用于损失）...

        # ---------- 3. 构建动态有效性掩码 ----------
        # 创建掩码：对于每个样本 i，前 dynamic_top_k[i] 个专家有效，其余无效
        # 方法：生成一个与 routing_weights 相同空间形状的布尔掩码
        k_range = torch.arange(self.top_k, device=x.device).view(1, -1, 1, 1)  # [1, top_k, 1, 1]
        # dynamic_top_k 形状 [B, 1, 1, 1]
        dynamic_top_k_expanded = dynamic_top_k.view(B, 1, 1, 1)
        mask = (k_range < dynamic_top_k_expanded).to(routing_weights.dtype)  # [B, top_k, 1, 1]

        # 应用掩码并重新归一化权重
        masked_weights = routing_weights * mask
        # 归一化：确保有效权重之和为 1（避免因 mask 导致总和小于 1）
        sum_masked = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = masked_weights / sum_masked

        # ---------- 4. 共享专家与稀疏专家计算 ----------
        shared_out = self.shared_expert(x)

        # 使用掩码后的权重进行稀疏专家计算
        # BatchedExpertComputation 需要传入 normalized_weights 和 routing_indices
        expert_out = BatchedExpertComputation.compute_sparse_experts_batched(
            x, self.experts, normalized_weights, routing_indices,
            self.top_k, self.num_experts
        )

        # 共享专家知识蒸馏
        if self.training and self.distill_coeff > 0:
            # 获取本批次中至少被一个样本选中的专家ID (用于跳过活跃专家)
            active_experts = routing_indices.flatten().unique()  # shape: [num_active]
            distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            for eid in range(self.num_experts):
                if eid in active_experts:
                    continue
                # 未激活专家直接对当前输入x前向一次
                expert_e_out = self.experts[eid](x)
                # 与共享专家输出计算MSE损失，detach共享输出以阻断其梯度
                mse = F.mse_loss(expert_e_out, shared_out.detach())
                distill_loss = distill_loss + mse

            # 将蒸馏损失加到后续要存储的辅助损失中
            # 注意：我们将在最后统一处理辅助损失，这里先记录
            # 需要把distill_loss附加到aux_loss上
            self._last_distill_loss = distill_loss  # 临时存一下

            if self.global_step % 124 == 0:

                label_name = f"model.{6 if self.modelID % 2 == 0 else 8}"
                print(f"\n{label_name}  [Step {self.global_step}] 蒸馏损失：{self._last_distill_loss.item ():.6f}"
                      f"    选中专家ID：{active_experts}")

        else:
            self._last_distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training and self.forced_experts == True:
            # 获取本批次专家ID
            # 去掉多余的维度，压缩为 [B, top_k]
            indices_flat = routing_indices.view(B, self.top_k)
            # print(indices_flat)
            active_mask = mask.squeeze(-1).squeeze(-1).bool()  # [B, top_k]
            # print(active_mask)
            active_experts_flat = indices_flat[active_mask]
            active_experts = active_experts_flat.unique()
            # active_experts = indices_flat[active_mask].unique()
            # print(active_experts)

            # 更新饥饿计数器
            for expert_id in range(self.num_experts):
                if expert_id in active_experts:
                    self.hunger_counters[expert_id] = 0
                else:
                    self.hunger_counters[expert_id] += 1

            # 筛选满足饥饿条件的专家
            f_experts = []
            for expert_id in range(self.num_experts):
                if expert_id not in active_experts and self.hunger_counters[expert_id] >= self.hunger_threshold:
                    f_experts.append(expert_id)

            # 如果没有饥饿专家，则随机探索一个未激活专家
            if len(f_experts) == 0 and torch.rand(1).item() < self.random_force_prob:
                inactive = [eid for eid in range(self.num_experts) if eid not in active_experts]
                if inactive:
                    forced_idx = inactive[torch.randint(0, len(inactive), (1,)).item()]
                    f_experts.append(forced_idx)

            # 计算强制激活专家的输出并添加到稀疏输出上
            if f_experts:
                # 计算共享输出
                forced_out = torch.zeros_like(expert_out)
                for eid in f_experts:
                    out_e = self.experts[eid](x)
                    forced_out = forced_out + out_e * self.forced_expert_weight
                    self.hunger_counters[eid] = 0   # 饥饿计数清0
                expert_out = expert_out + forced_out

            if self.global_step % 124 == 0:
                unique, counts = dynamic_top_k.unique(return_counts=True)
                topk_dist = dict(zip(unique.tolist(), counts.tolist()))

                # 本批次的专家使用比例
                usage = torch.bincount(active_experts_flat.long(),
                                       minlength=self.num_experts).float() / active_experts_flat.numel()

                label_name = f"model.{6 if self.modelID % 2 == 0 else 8}"
                print(f"\n{label_name}  [Step {self.global_step}]  "
                      f"饥饿计数器: {self.hunger_counters}  "
                      f"硬激活专家: {f_experts if f_experts else 'NULL'}  "
                      f"动态 Top‑K 分布: {topk_dist}  "
                      f"\n         实际专家使用占比：{[round(x.item(), 2) for x in usage]}"
                      f"\n         实际被选中专家ID：{active_experts}"
                      f"\n         本批次未去 0专家：{routing_indices.flatten().unique()}")


        # ---------- 5. 条件感知门控融合 ----------
        # 将 dynamic_top_k 信息拼接到门控输入（全局池化后的特征）
        gate_pool = F.adaptive_avg_pool2d(shared_out + expert_out, 1)  # [B, out_channels, 1, 1]
        # 扩展 dynamic_top_k 到 [B, 1, 1, 1]
        topk_info = dynamic_top_k.view(B, 1, 1, 1).float() / self.top_k  # 归一化到 [0,1]
        gate_input = torch.cat([gate_pool, topk_info], dim=1)  # [B, out_channels+1, 1, 1]
        gate_weights = self.fusion_gate(gate_input)  # [B, 2, 1, 1]

        output = gate_weights[:, 0:1, :, :] * shared_out + gate_weights[:, 1:2, :, :] * expert_out

        # ---------- 6. 辅助损失汇总 ----------
        if self.training:
            # 更新主 MoE 损失系数（原有逻辑）
            step = self.global_step.item()
            if step < self.balance_warmup_steps:
                progress = step / max(1, self.balance_warmup_steps - 1)
                coeff = self.balance_loss_coeff * (1 - progress) + self.balance_loss_min * progress
            else:
                coeff = self.balance_loss_min
            self.current_balance_coeff.fill_(coeff)
            self.moe_loss_fn.balance_loss_coeff = coeff

            # 计算原有 MoE 辅助损失
            probs = routing_result[5]  # 假设第6个返回值是概率分布 pooled_weights
            aux_loss = self.moe_loss_fn(probs, self.routing.router(x).clamp(-30, 30), routing_indices)

            # 添加估计器熵正则化
            if self.estimator_entropy_coeff > 0:
                aux_loss = aux_loss + self.estimator_entropy_coeff * est_entropy_loss

            # 3. 知识共享蒸馏损失
            if self.distill_coeff > 0:
                aux_loss = aux_loss + self.distill_coeff * self._last_distill_loss

            MOE_LOSS_REGISTRY[self] = aux_loss
            self.global_step.add_(1)

        return output


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
