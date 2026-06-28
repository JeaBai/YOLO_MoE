import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from .utils import get_safe_groups, BatchedExpertComputation
from .experts import InvertedResidualExpert
from .routers import UltraEfficientRouter
from .loss import MoELoss

MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()


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
        estimator_entropy_coeff: float = 0.01,
        hysteresis_low: float = 0.3,
        hysteresis_high: float = 0.7,
        distill_coeff: float = 0.0,             # 知识共享系数，0表示关闭
        forced_experts: bool = True,            # 强制激活专家，False关闭
        hunger_threshold: int = 3,              # 连续未被选中的次数阈值
        forced_expert_weight: float = 0.5,      # 强制激活专家的输出权重
        random_force_prob: float = 0.2,         # 无饥饿专家时随机激活概率
    ):
        print("{" +
              f"num_experts: {num_experts}, "
              f"top_k: {top_k}, "
              f"balance_loss_coeff: {balance_loss_coeff}, "
              f"balance_loss_min: {balance_loss_min}, "
              f"balance_warmup_steps: {balance_warmup_steps}, "
              f"router_z_loss_coeff: {router_z_loss_coeff}, "
              f"entropy_loss_coeff: {entropy_loss_coeff}, "
              f"capacity_factor: {capacity_factor}, "
              f"num_groups: {num_groups}, "
              f"estimator_entropy_coeff: {estimator_entropy_coeff}, "
              f"hysteresis_low: {hysteresis_low}, "
              f"hysteresis_high: {hysteresis_high}, "
              f"distill_coeff: {distill_coeff}, "
              f"forced_experts: {forced_experts}, "
              f"hunger_threshold: {hunger_threshold}, "
              f"forced_expert_weight: {forced_expert_weight}, "
              f"random_force_prob: {random_force_prob}, "
              + "}"
              )

        super().__init__()
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

        self.register_buffer('hunger_counters', torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer('_prev_infer_topk', torch.tensor(self.top_k, dtype=torch.long), persistent=False)
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

        # 复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        # 路由
        self.routing = UltraEfficientRouter(
            in_channels, num_experts,
            reduction=16,
            top_k=top_k,
            noise_std=0.1,
            temperature=1.0,
            pool_scale=8
        )

        # 专家（支持可选异构）
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            ks = 3 if i % 2 == 0 else 5
            self.experts.append(
                InvertedResidualExpert(
                    in_channels, out_channels,
                    expand_ratio=2.0,
                    kernel_size=ks,
                    groups=num_groups
                )
            )

        # 共享专家
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False, groups=num_groups),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            nn.SiLU(inplace=True)
        )

        # 融合门控（额外拼接动态 top-k 信息）
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels + 1, 2, 1),
            nn.Sigmoid()
        )

        # 辅助损失
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=router_z_loss_coeff,
            entropy_loss_coeff=entropy_loss_coeff,
            num_experts=num_experts,
            top_k=top_k,
            use_soft_balancing=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 复杂度估计与动态 Top-K
        complexity = self.complexity_estimator(x)                    # [B, 1, 1, 1]
        complexity_flat = complexity.view(B)                        # [B]

        if self.training:
            dynamic_top_k_float = self.top_k * complexity_flat * self.capacity_factor
            dynamic_top_k = torch.round(dynamic_top_k_float).int().clamp(min=1, max=self.top_k)

            if self.estimator_entropy_coeff > 0:
                p = complexity_flat.clamp(1e-6, 1 - 1e-6)
                entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
                est_entropy_loss = entropy.mean()
            else:
                est_entropy_loss = torch.tensor(0.0, device=x.device)
        else:
            # 推理：滞后阈值平滑
            raw_top_k_float = self.top_k * complexity_flat * self.capacity_factor
            new_top_k = torch.round(raw_top_k_float).int().clamp(min=1, max=self.top_k)
            high_mask = complexity_flat > self.hysteresis_high
            low_mask = complexity_flat < self.hysteresis_low
            prev = self._prev_infer_topk.expand_as(new_top_k)
            dynamic_top_k = torch.where(high_mask, new_top_k, prev)
            dynamic_top_k = torch.where(low_mask, new_top_k, dynamic_top_k)
            self._prev_infer_topk.fill_(dynamic_top_k[-1].item())

        # 2. 路由
        routing_result = self.routing(x)
        routing_weights = routing_result[0]                         # [B, top_k, 1, 1]
        routing_indices = routing_result[1]                         # [B, top_k, 1, 1]

        # 3. 动态有效性掩码
        k_range = torch.arange(self.top_k, device=x.device).view(1, -1, 1, 1)
        dynamic_top_k_expanded = dynamic_top_k.view(B, 1, 1, 1)
        mask = (k_range < dynamic_top_k_expanded).to(routing_weights.dtype)
        masked_weights = routing_weights * mask
        sum_masked = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = masked_weights / sum_masked

        # 4. 共享专家与稀疏专家计算
        shared_out = self.shared_expert(x)
        expert_out = BatchedExpertComputation.compute_sparse_experts_batched(
            x, self.experts, normalized_weights, routing_indices,
            self.top_k, self.num_experts
        )

        # 蒸馏损失
        if self.training and self.distill_coeff > 0:
            active_experts = routing_indices.flatten().unique()
            distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            for eid in range(self.num_experts):
                if eid in active_experts:
                    continue
                expert_e_out = self.experts[eid](x)
                mse = F.mse_loss(expert_e_out, shared_out.detach())
                distill_loss = distill_loss + mse
            self._last_distill_loss = distill_loss
        else:
            self._last_distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 强制激活专家
        if self.training and self.forced_experts:
            indices_flat = routing_indices.view(B, self.top_k)
            active_mask = mask.squeeze(-1).squeeze(-1).bool()
            active_experts_flat = indices_flat[active_mask]
            active_experts = active_experts_flat.unique()

            for expert_id in range(self.num_experts):
                if expert_id in active_experts:
                    self.hunger_counters[expert_id] = 0
                else:
                    self.hunger_counters[expert_id] += 1

            f_experts = []
            for expert_id in range(self.num_experts):
                if expert_id not in active_experts and self.hunger_counters[expert_id] >= self.hunger_threshold:
                    f_experts.append(expert_id)

            if len(f_experts) == 0 and torch.rand(1).item() < self.random_force_prob:
                inactive = [eid for eid in range(self.num_experts) if eid not in active_experts]
                if inactive:
                    forced_idx = inactive[torch.randint(0, len(inactive), (1,)).item()]
                    f_experts.append(forced_idx)

            if f_experts:
                forced_out = torch.zeros_like(expert_out)
                for eid in f_experts:
                    out_e = self.experts[eid](x)
                    forced_out = forced_out + out_e * self.forced_expert_weight
                    self.hunger_counters[eid] = 0
                expert_out = expert_out + forced_out

        # 5. 条件感知门控融合
        gate_pool = F.adaptive_avg_pool2d(shared_out + expert_out, 1)
        topk_info = dynamic_top_k.view(B, 1, 1, 1).float() / self.top_k
        gate_input = torch.cat([gate_pool, topk_info], dim=1)
        gate_weights = self.fusion_gate(gate_input)

        output = gate_weights[:, 0:1, :, :] * shared_out + gate_weights[:, 1:2, :, :] * expert_out

        # 6. 辅助损失汇总
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

            if self.estimator_entropy_coeff > 0:
                aux_loss = aux_loss + self.estimator_entropy_coeff * est_entropy_loss
            if self.distill_coeff > 0:
                aux_loss = aux_loss + self.distill_coeff * self._last_distill_loss

            MOE_LOSS_REGISTRY[self] = aux_loss
            self.global_step.add_(1)

        return output