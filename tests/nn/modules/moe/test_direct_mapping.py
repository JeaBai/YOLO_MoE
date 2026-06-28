import torch
import pytest
import sys
sys.path.insert(0, '.')

from ultralytics.nn.modules.moe.descriptor import direct_mapping


class TestDirectMapping:
    def test_max_complexity(self):
        s = torch.tensor([0.95, 0.95, 0.95, 0.95]).view(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert top_k.shape == (4,)
        assert (top_k == 4).all(), f"Expected all 4, got {top_k}"

    def test_min_complexity(self):
        s = torch.tensor([0.05, 0.05, 0.05, 0.05]).view(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert (top_k == 1).all(), f"Expected all 1, got {top_k}"

    def test_mid_complexity(self):
        # s=0.5, K_max=4 -> round(0.5*3) = round(1.5) = 2 -> top_k = 1+2 = 3
        s = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        expected = 1 + round(0.5 * 3)
        assert (top_k == expected).all(), f"Expected all {expected}, got {top_k}"

    def test_clamp_upper(self):
        s = torch.tensor([1.5, 2.0, 3.0]).view(3, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert (top_k == 4).all(), f"Expected all 4 (clamped), got {top_k}"

    def test_clamp_lower(self):
        s = torch.tensor([-0.5, -1.0, -2.0]).view(3, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert (top_k == 1).all(), f"Expected all 1 (clamped), got {top_k}"

    def test_output_shape(self):
        s = torch.randn(8, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert top_k.shape == (8,), f"Expected (8,), got {top_k.shape}"

    def test_output_dtype(self):
        s = torch.randn(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert top_k.dtype in (torch.int32, torch.int64, torch.long), f"Expected integer dtype, got {top_k.dtype}"

    def test_varying_k_max(self):
        s = torch.tensor([0.9, 0.9]).view(2, 1, 1, 1)
        top_k = direct_mapping(s, k_max=6)
        # 1 + round(0.9 * 5) = 1 + round(4.5) = 1 + 5 = 6
        assert (top_k == 6).all(), f"Expected 6, got {top_k}"

    def test_edge_zero(self):
        s = torch.zeros(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert (top_k == 1).all(), f"Expected all 1, got {top_k}"

    def test_edge_one(self):
        s = torch.ones(4, 1, 1, 1)
        top_k = direct_mapping(s, k_max=4)
        assert (top_k == 4).all(), f"Expected all 4, got {top_k}"