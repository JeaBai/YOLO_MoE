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
    ):

        super().__init__()
        # 参数存储...
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.balance_loss_min = balance_loss_min
        self.balance_warmup_steps = balance_warmup_steps
        self.capacity_factor = capacity_factor

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

    def forward(self, x):
        B, C, H, W = x.shape

        # 动态 top_k
        complexity = self.complexity_estimator(x).mean()
        dynamic_top_k = max(1, min(self.top_k, int(self.top_k * complexity * self.capacity_factor)))

        # 路由（支持动态 top_k）
        routing_weights, routing_indices, usage_freq, importance, z_loss_val, probs = self.routing(x, top_k=dynamic_top_k)
        # 共享专家
        shared_out = self.shared_expert(x)

        # 稀疏专家计算
        expert_out = BatchedExpertComputation.compute_sparse_experts_batched(
            x, self.experts, routing_weights, routing_indices,
            dynamic_top_k, self.num_experts
        )

        # 可学习融合
        gate_weights = self.fusion_gate(shared_out + expert_out)  # [B,2,1,1]
        output = gate_weights[:, 0:1, :, :] * shared_out + gate_weights[:, 1:2, :, :] * expert_out

        # 辅助损失（训练时）
        if self.training:
            # 更新平衡系数（指数衰减）
            step = self.global_step.item()
            if step < self.balance_warmup_steps:
                progress = step / max(1, self.balance_warmup_steps - 1)
                coeff = self.balance_loss_coeff * (1 - progress) + self.balance_loss_min * progress
            else:
                coeff = self.balance_loss_min
            self.current_balance_coeff.fill_(coeff)
            self.moe_loss_fn.balance_loss_coeff = coeff

            # 计算辅助损失（使用软概率）
            aux_loss = self.moe_loss_fn(probs, self.routing.router(x).clamp(-30,30), routing_indices)
            MOE_LOSS_REGISTRY[self] = aux_loss
            self.global_step.add_(1)

        return output


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
