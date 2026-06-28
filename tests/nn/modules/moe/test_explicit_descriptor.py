import torch
import pytest
import sys
sys.path.insert(0, '.')

# Import the module that will be created
from ultralytics.nn.modules.moe.descriptor import ExplicitDescriptor


class TestExplicitDescriptor:
    @pytest.fixture
    def descriptor(self):
        return ExplicitDescriptor(alpha=0.7, beta=0.3)

    def test_output_shape(self, descriptor):
        x = torch.randn(4, 64, 32, 32)
        s = descriptor(x)
        assert s.shape == (4, 1, 1, 1), f"Expected (4,1,1,1), got {s.shape}"

    def test_output_range(self, descriptor):
        x = torch.randn(4, 64, 32, 32)
        s = descriptor(x)
        assert (s >= 0.0).all() and (s <= 1.0).all(), f"Output out of [0,1]: min={s.min().item():.4f}, max={s.max().item():.4f}"

    def test_zero_parameters(self, descriptor):
        n_params = sum(p.numel() for p in descriptor.parameters())
        assert n_params == 0, f"Expected 0 parameters, got {n_params}"

    def test_deterministic(self, descriptor):
        x = torch.randn(4, 64, 32, 32)
        s1 = descriptor(x)
        s2 = descriptor(x)
        assert torch.equal(s1, s2), "Output should be deterministic"

    def test_small_resolution(self, descriptor):
        x = torch.randn(2, 64, 2, 2)
        s = descriptor(x)
        assert s.shape == (2, 1, 1, 1), f"Expected (2,1,1,1), got {s.shape}"
        assert not torch.isnan(s).any(), "Output contains NaN"
        assert not torch.isinf(s).any(), "Output contains Inf"

    def test_high_variance_gives_high_score(self, descriptor):
        # High variance input: alternating 1 and -1
        x_high = torch.ones(4, 64, 32, 32)
        x_high[:, ::2] = -1.0  # alternating channels
        s_high = descriptor(x_high)

        # Low variance input: all zeros
        x_low = torch.zeros(4, 64, 32, 32)
        s_low = descriptor(x_low)

        # High variance should give higher score
        assert (s_high > s_low).all(), f"high={s_high.view(-1)}, low={s_low.view(-1)}"

    def test_low_variance_gives_low_score(self, descriptor):
        # Constant input should give very low score
        x_constant = torch.ones(4, 64, 32, 32) * 0.5
        s = descriptor(x_constant)

        # Normal random input
        x_random = torch.randn(4, 64, 32, 32)
        s_random = descriptor(x_random)

        # Constant input should have lower score than random
        assert (s_random > s).all(), f"constant={s.view(-1)}, random={s_random.view(-1)}"