import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from .utils import get_safe_groups, BatchedExpertComputation
from .experts import InvertedResidualExpert
from .routers import UltraEfficientRouter
from .loss import MoELoss
from .descriptor import ExplicitDescriptor, direct_mapping

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
        num_groups: int = 8,
        cascade_weight: float = 1.0,
        descriptor_alpha: float = 0.7,
        descriptor_beta: float = 0.3,
    ):
        print("{" +
              f"num_experts: {num_experts}, "
              f"top_k: {top_k}, "
              f"balance_loss_coeff: {balance_loss_coeff}, "
              f"balance_loss_min: {balance_loss_min}, "
              f"balance_warmup_steps: {balance_warmup_steps}, "
              f"router_z_loss_coeff: {router_z_loss_coeff}, "
              f"entropy_loss_coeff: {entropy_loss_coeff}, "
              f"num_groups: {num_groups}, "
              f"cascade_weight: {cascade_weight}, "
              f"descriptor_alpha: {descriptor_alpha}, "
              f"descriptor_beta: {descriptor_beta}, "
              + "}"
              )

        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.balance_loss_min = balance_loss_min
        self.balance_warmup_steps = balance_warmup_steps

        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

        self.cascade_weight = cascade_weight
        self.descriptor = ExplicitDescriptor(alpha=descriptor_alpha, beta=descriptor_beta)

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

        # Metrics tracking buffers
        self.register_buffer('_expert_hits', torch.zeros(num_experts))
        self.register_buffer('_token_count', torch.zeros(1))
        self.register_buffer('_descriptor_sum', torch.zeros(1))
        self.register_buffer('_descriptor_count', torch.zeros(1))
        self.register_buffer('_topk_sum', torch.zeros(1))
        self.register_buffer('_topk_count', torch.zeros(1))

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

        # Metrics accumulation (training and eval)
        with torch.no_grad():
            # Expert hits: count which experts were selected
            valid_mask = k_range < top_k_expanded
            for e in range(self.num_experts):
                self._expert_hits[e] += ((routing_indices == e) & valid_mask).sum().float()
            self._token_count += routing_indices.numel() * valid_mask.float().mean().item()
            # Descriptor stats
            self._descriptor_sum += s.sum()
            self._descriptor_count += B
            # Top-k stats
            self._topk_sum += top_k_per_sample.float().sum()
            self._topk_count += B

        return output

    def get_metrics(self):
        """Return MoE diagnostic metrics for the current epoch."""
        total = self._token_count.item()
        if total < 1:
            return {}
        hits = self._expert_hits / total  # normalized
        ideal = 1.0 / self.num_experts
        return {
            'expert_usage': {f'e{i}': hits[i].item() for i in range(self.num_experts)},
            'max_usage': hits.max().item(),
            'min_usage': hits.min().item(),
            'usage_std': hits.std().item(),
            'dead_experts': int((hits < 0.1 * ideal).sum().item()),
            'avg_descriptor': (self._descriptor_sum / self._descriptor_count).item(),
            'avg_topk': (self._topk_sum / self._topk_count).item(),
        }

    def reset_metrics(self):
        """Reset accumulated metrics for the next epoch."""
        self._expert_hits.zero_()
        self._token_count.zero_()
        self._descriptor_sum.zero_()
        self._descriptor_count.zero_()
        self._topk_sum.zero_()
        self._topk_count.zero_()