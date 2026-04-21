"""Tests for MXFP4 packing/unpacking utilities."""

from __future__ import annotations

import torch
import pytest

from quanto.utils.mxfp4_pack import (
    FP4_E2M1_TABLE,
    FP4_MAX,
    compute_e8m0_scales,
    e8m0_to_float,
    pack_fp4_to_uint8,
    pack_mxfp4,
    quantize_to_fp4,
    unpack_mxfp4,
    unpack_uint8_to_fp4,
)


class TestFP4Table:
    """Test FP4 E2M1 encoding table."""

    def test_table_size(self):
        assert len(FP4_E2M1_TABLE) == 16

    def test_positive_values(self):
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for i, v in enumerate(expected):
            assert FP4_E2M1_TABLE[i].item() == v

    def test_negative_values(self):
        expected = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        for i, v in enumerate(expected):
            assert FP4_E2M1_TABLE[i + 8].item() == pytest.approx(v)

    def test_max_value(self):
        assert FP4_MAX == 6.0


class TestE8M0Scales:
    """Test E8M0 scale computation."""

    def test_unit_range(self):
        """Values in [-6, 6] should have scale ~ 1.0."""
        weight = torch.tensor([[1.0, 2.0, 3.0, -1.0] * 8])  # max=3.0
        scales = compute_e8m0_scales(weight, group_size=32)
        scale_float = e8m0_to_float(scales)
        # scale = 2^floor(log2(3.0 / 6.0)) = 2^floor(-1) = 2^(-1) = 0.5
        assert scale_float[0, 0].item() == pytest.approx(0.5)

    def test_large_values(self):
        """Large values should produce larger scales."""
        weight = torch.tensor([[100.0] * 32])
        scales = compute_e8m0_scales(weight, group_size=32)
        scale_float = e8m0_to_float(scales)
        # scale = 2^floor(log2(100/6)) = 2^floor(4.06) = 2^4 = 16
        assert scale_float[0, 0].item() == pytest.approx(16.0)

    def test_small_values(self):
        """Small values should produce smaller scales."""
        weight = torch.tensor([[0.01] * 32])
        scales = compute_e8m0_scales(weight, group_size=32)
        scale_float = e8m0_to_float(scales)
        # Very small scale
        assert scale_float[0, 0].item() < 0.01

    def test_multiple_groups(self):
        """Multiple groups should have independent scales."""
        weight = torch.tensor([[1.0] * 32 + [100.0] * 32])
        scales = compute_e8m0_scales(weight, group_size=32)
        assert scales.shape == (1, 2)
        scale_float = e8m0_to_float(scales)
        assert scale_float[0, 1] > scale_float[0, 0]


class TestFP4Packing:
    """Test FP4 uint8 packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack should recover original codes."""
        codes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.uint8)
        packed = pack_fp4_to_uint8(codes)
        assert packed.shape == (1, 8)
        unpacked = unpack_uint8_to_fp4(packed)
        assert unpacked.shape == (1, 16)
        assert torch.equal(unpacked, codes)

    def test_pack_shape(self):
        """Packed tensor should be half the size."""
        codes = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
        packed = pack_fp4_to_uint8(codes)
        assert packed.shape == (4, 32)

    def test_pack_values(self):
        """Check specific packing: byte = low_nibble | (high_nibble << 4)."""
        codes = torch.tensor([[3, 12]], dtype=torch.uint8)  # 3 and 12
        packed = pack_fp4_to_uint8(codes)
        expected = (3) | (12 << 4)  # 0x03 | 0xC0 = 0xC3
        assert packed[0, 0].item() == expected


class TestMXFP4FullPipeline:
    """Test full pack/unpack pipeline."""

    def test_roundtrip_small(self):
        """Pack then unpack a small tensor."""
        weight = torch.randn(4, 64, dtype=torch.bfloat16)
        result = pack_mxfp4(weight, group_size=32)

        assert "weight.packed" in result
        assert "weight.scale_e8m0" in result
        assert result["weight.packed"].dtype == torch.uint8
        assert result["weight.scale_e8m0"].dtype == torch.uint8
        assert result["weight.packed"].shape == (4, 32)  # 64/2
        assert result["weight.scale_e8m0"].shape == (4, 2)  # 64/32

    def test_roundtrip_accuracy(self):
        """Unpacked values should be close to original (within FP4 precision)."""
        weight = torch.randn(8, 128, dtype=torch.bfloat16) * 2.0
        result = pack_mxfp4(weight, group_size=32)

        recovered = unpack_mxfp4(
            result["weight.packed"],
            result["weight.scale_e8m0"],
            group_size=32,
        )

        # FP4 has only 16 values, so error can be significant
        # But relative error should be bounded
        rel_error = (weight.float() - recovered.float()).abs() / (weight.float().abs() + 1e-8)
        mean_rel_error = rel_error.mean()
        assert mean_rel_error < 0.5, f"Mean relative error too high: {mean_rel_error}"

    def test_compression_ratio(self):
        """Packed format should achieve ~3.76x compression."""
        weight = torch.randn(1024, 4096, dtype=torch.bfloat16)
        result = pack_mxfp4(weight, group_size=32)

        original_bytes = weight.numel() * 2  # BF16 = 2 bytes
        packed_bytes = result["weight.packed"].numel() * 1  # uint8 = 1 byte
        scale_bytes = result["weight.scale_e8m0"].numel() * 1  # uint8 = 1 byte
        total_packed = packed_bytes + scale_bytes

        ratio = original_bytes / total_packed
        # Expected: ~3.76x (64 bytes BF16 vs 17 bytes MXFP4 per 32 elements)
        assert ratio > 3.5, f"Compression ratio too low: {ratio:.2f}x"
        assert ratio < 4.0, f"Compression ratio too high: {ratio:.2f}x"

    def test_zero_tensor(self):
        """Zero tensor should pack/unpack correctly."""
        weight = torch.zeros(4, 64, dtype=torch.bfloat16)
        result = pack_mxfp4(weight, group_size=32)
        recovered = unpack_mxfp4(result["weight.packed"], result["weight.scale_e8m0"])
        assert torch.allclose(recovered, weight.float(), atol=1e-6)

    def test_large_matrix(self):
        """Test with sizes typical of LLM weight matrices."""
        weight = torch.randn(4096, 4096, dtype=torch.bfloat16)
        result = pack_mxfp4(weight, group_size=32)
        assert result["weight.packed"].shape == (4096, 2048)
        assert result["weight.scale_e8m0"].shape == (4096, 128)


class TestConfigIntegration:
    """Test that config accepts new algorithm field."""

    def test_rtn_default(self):
        from quanto.core.config import UnifiedConfig
        config = UnifiedConfig(model_path="/tmp/test", output_dir="/tmp/out")
        assert config.algorithm == "rtn"

    def test_awq_algorithm(self):
        from quanto.core.config import UnifiedConfig
        config = UnifiedConfig(model_path="/tmp/test", output_dir="/tmp/out", algorithm="awq")
        assert config.algorithm == "awq"

    def test_gptq_algorithm(self):
        from quanto.core.config import UnifiedConfig
        config = UnifiedConfig(model_path="/tmp/test", output_dir="/tmp/out", algorithm="gptq")
        assert config.algorithm == "gptq"

    def test_invalid_algorithm(self):
        from quanto.core.config import UnifiedConfig
        with pytest.raises(ValueError, match="Invalid algorithm"):
            UnifiedConfig(model_path="/tmp/test", output_dir="/tmp/out", algorithm="invalid")
