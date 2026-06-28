import torch
import torch.nn as nn


class ExplicitDescriptor(nn.Module):
    """0-parameter deterministic complexity descriptor using variance + energy.
    
    Computes per-sample complexity score based on:
    - Channel-wise variance averaged over spatial dimensions
    - Channel-spatial mean energy (L2 norm of mean vector)
    
    Both are normalized via batch min-max, then combined with learnable weights.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # per-sample variance: Var_c(x) then mean_{h,w}
        var_c = x.var(dim=1, unbiased=False)  # [B, H, W]
        var_sample = var_c.mean(dim=[1, 2])    # [B]
        
        # per-sample energy: ||mean_{c,h,w}(x)||² / C
        mean_spatial = x.mean(dim=[2, 3])      # [B, C]
        energy_sample = mean_spatial.pow(2).sum(dim=1) / C  # [B]

        s = self.alpha * var_sample + self.beta * energy_sample
        s = torch.clamp(s, 0.0, 1.0)
        return s.view(B, 1, 1, 1)


def direct_mapping(s: torch.Tensor, k_max: int) -> torch.Tensor:
    """Closed-form per-sample top_k mapping.
    
    Formula: top_k = 1 + round(clamp(s, 0, 1) * (k_max - 1))
    
    Args:
        s: Complexity scores [B, 1, 1, 1] with values nominally in [0, 1]
        k_max: Maximum number of experts to activate
        
    Returns:
        top_k: Integer tensor [B] with values in [1, k_max]
    """
    s_clamped = s.clamp(0.0, 1.0)
    top_k = 1 + torch.floor(s_clamped * (k_max - 1) + 0.5).int()
    return top_k.view(-1)  # [B]