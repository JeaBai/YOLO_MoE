from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FlopsUtils, get_safe_groups


# ==========================================
# Ultra-lightweight Router (core optimization)
# ==========================================
class UltraEfficientRouter(nn.Module):
    """Ultra-efficient router: 1) Depthwise-separable convolution instead of standard conv 2) Aggressive downsampling
    (8x) 3) Early channel compression 4) Improved numerical stability.

    Expected FLOPs reduction: ~95% vs a local router baseline.
    """

    def __init__(
        self, in_channels, num_experts, reduction=16, top_k=2, noise_std=1.0, temperature: float = 1.0, pool_scale=8
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale

        # More aggressive channel compression
        reduced_channels = max(in_channels // reduction, 4)

        # Depthwise-separable conv: compute ~ 1/(kernel_size^2) of standard conv
        self.router = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=True),
            # Pointwise compression
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced_channels, 4), reduced_channels),
            nn.SiLU(inplace=True),
            # Expert projection
            nn.Conv2d(reduced_channels, num_experts, 1, bias=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x, top_k=None
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        B, _C, H, W = x.shape

        # 1) Aggressive downsampling (core optimization)
        if H > self.pool_scale and W > self.pool_scale:
            x_down = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
            # print(f"[Router] Downsampled shape: {x_down.shape}")
        else:
            x_down = x

        # 2) Lightweight convolutional routing
        logits = self.router(x_down)

        # 3) Z-loss computation (numerical stability)
        z_loss_metric = None
        if self.training:
            # Use clamp instead of tanh for better performance
            logits_safe = logits.clamp(-10.0, 10.0)
            z_loss_metric = torch.logsumexp(logits_safe, dim=1).pow(2).mean()
            # print(f"[Router] Z-loss: {z_loss_metric.item():.6f}")

        # 4) Noise injection
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits).mul_(self.noise_std)

        # 5+)  使用传入的 top_k 代替固定的 self.top_k
        if top_k is None:
            top_k = self.top_k

        # 5) Softmax + TopK (fused operation)
        # Clamp logits again before division to be safe
        logits_clamped = logits.clamp(-30.0, 30.0)
        weights = F.softmax((logits_clamped / self.temperature).float(), dim=1).type_as(x)
        pooled_weights = weights.mean(dim=[2, 3], keepdim=True)

        # 关键：使用 top_k 变量
        topk_vals, topk_indices = torch.topk(pooled_weights, top_k, dim=1)

        # In-place normalization
        topk_vals.div_(topk_vals.sum(dim=1, keepdim=True).add_(1e-6))

        if self.training:
            importance = pooled_weights.sum(dim=0).view(self.num_experts)

            # Optimization: use one_hot instead of scatter
            topk_indices_flat = topk_indices.view(B, top_k, 1, 1)[:, :, 0, 0]
            mask = F.one_hot(topk_indices_flat, num_classes=self.num_experts).float()
            usage_frequency = mask.sum(dim=[0, 1]) / (B * top_k)

            return topk_vals, topk_indices, usage_frequency, importance, z_loss_metric, pooled_weights
        else:
            return topk_vals, topk_indices, None, None, None, None

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down = max(H // self.pool_scale, 1)
        w_down = max(W // self.pool_scale, 1)

        flops = B * C * H * W  # AvgPool

        input_down_shape = (B, C, h_down, w_down)

        # Depthwise conv
        flops += FlopsUtils.count_conv2d(self.router[0], input_down_shape)
        # Pointwise conv
        flops += FlopsUtils.count_conv2d(self.router[3], (B, self.router[0].out_channels, h_down, w_down))
        # Expert projection
        flops += FlopsUtils.count_conv2d(self.router[6], (B, self.router[3].out_channels, h_down, w_down))

        return flops
