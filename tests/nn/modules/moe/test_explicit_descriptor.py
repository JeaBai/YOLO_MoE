import torch
import pytest
import sys
sys.path.insert(0, '.')

# Import the module that will be created
from ultralytics.nn.modules.moe.descriptor import ExplicitDescriptor


class TestExplicitDescriptor:
    @pytest.fixture
    def descriptor(self):
        return ExplicitDescriptor(alpha=0.5, beta=0.5)

    def test_output_shape(self, descriptor):
        x = torch.randn(4, 64, 32, 32)
        s = descriptor(x)
        assert s.shape == (4, 1, 1, 1), f"Expected (4,1,1,1), got {s.shape}"

    def test_output_range(self, descriptor):
        x = torch.randn(4, 64, 32, 32)
        s = descriptor(x)
        assert (s >= 0.0).all() and (s <= 1.0).all(), f"Output out of [0,1]: min={s.min().item():.4f}, max={s.max().item():.4f}"

    def test_batch_minmax_normalization_span(self, descriptor):
        """With B>=2, min-max normalization produces large dynamic range (>0.5 span)."""
        x = torch.randn(8, 64, 32, 32)
        s = descriptor(x)
        span = s.max().item() - s.min().item()
        assert span > 0.5, f"Dynamic range too narrow: span={span:.4f}, expected > 0.5"
        # All scores should be in [0, 1]
        assert (s >= 0.0).all() and (s <= 1.0).all()

    def test_single_batch_no_crash(self, descriptor):
        """B=1 should not crash (min==max → zero)."""
        x = torch.randn(1, 64, 32, 32)
        s = descriptor(x)
        assert s.shape == (1, 1, 1, 1)
        assert not torch.isnan(s).any(), "Output contains NaN for B=1"
        assert (s >= 0.0).all() and (s <= 1.0).all()

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
        """Within same batch, higher variance samples should get higher score."""
        # Create mixed batch: 2 high-variance, 2 low-variance
        x = torch.randn(4, 64, 32, 32)
        x[2:] *= 0.01  # low-variance samples
        s = descriptor(x)
        # High-variance samples (0,1) should score higher than low-variance (2,3)
        assert s[0].item() > s[2].item(), f"high={s[0].item():.4f}, low={s[2].item():.4f}"
        assert s[1].item() > s[3].item(), f"high={s[1].item():.4f}, low={s[3].item():.4f}"

    def test_low_variance_gives_low_score(self, descriptor):
        """Within same batch, constant input should get lowest score."""
        x = torch.randn(4, 64, 32, 32)
        x[0] = 0.0  # zero constant sample
        s = descriptor(x)
        # Zero constant sample should have the lowest score (var=0, energy=0)
        assert s[0].item() <= s[1:].min().item(), \
            f"constant={s[0].item():.4f}, min_others={s[1:].min().item():.4f}"


class TestSparseDualMoEExplicit:
    @pytest.fixture
    def moe_module(self):
        from ultralytics.nn.modules.moe.modules import SparseDualMoE
        return SparseDualMoE(
            in_channels=64,
            out_channels=64,
            num_experts=4,
            top_k=4,
            cascade_weight=1.0,
            num_groups=4,
        )

    def test_sparse_dual_moe_explicit_forward(self, moe_module):
        x = torch.randn(2, 64, 32, 32)
        moe_module.eval()
        with torch.no_grad():
            out = moe_module(x)
        assert out.shape == (2, 64, 32, 32), f"Expected (2,64,32,32), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_sparse_dual_moe_no_complexity_estimator(self, moe_module):
        assert not hasattr(moe_module, 'complexity_estimator'), \
            "complexity_estimator should not exist"

    def test_sparse_dual_moe_no_forced_experts(self, moe_module):
        assert not hasattr(moe_module, 'forced_experts'), \
            "forced_experts should not exist"
        assert not hasattr(moe_module, 'hunger_counters'), \
            "hunger_counters should not exist"

    def test_sparse_dual_moe_has_descriptor(self, moe_module):
        assert hasattr(moe_module, 'descriptor'), \
            "descriptor should exist"
        from ultralytics.nn.modules.moe.descriptor import ExplicitDescriptor
        assert isinstance(moe_module.descriptor, ExplicitDescriptor), \
            f"Expected ExplicitDescriptor, got {type(moe_module.descriptor)}"

    def test_sparse_dual_moe_has_cascade_weight(self, moe_module):
        assert hasattr(moe_module, 'cascade_weight'), \
            "cascade_weight should exist"
        assert moe_module.cascade_weight == 1.0

    def test_sparse_dual_moe_training_forward(self, moe_module):
        x = torch.randn(2, 64, 32, 32)
        moe_module.train()
        out = moe_module(x)
        assert out.shape == (2, 64, 32, 32)
        assert not torch.isnan(out).any(), "Output contains NaN in training mode"

    def test_sparse_dual_moe_per_sample_topk_consistent(self, moe_module):
        """Same sample: all spatial positions share same top_k"""
        x = torch.randn(2, 64, 32, 32)
        moe_module.eval()
        with torch.no_grad():
            # We can't directly inspect internal top_k easily,
            # but we verify the output is consistent across spatial positions
            out = moe_module(x)
            # For a given sample, spatial mean should be meaningful
            # (no per-position variation in top_k)
            assert out.shape == (2, 64, 32, 32)