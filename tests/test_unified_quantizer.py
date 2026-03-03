"""
End-to-End Tests for UnifiedQuantizer

Tests all three memory strategies:
- full: Load entire model to GPU
- layerwise_cpu: Load to CPU, quantize layers on GPU
- lazy: Load weights on-demand from disk

Run with: pytest tests/test_unified_quantizer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quanto import UnifiedConfig, UnifiedQuantizer


class TestUnifiedConfig:
    """Tests for UnifiedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
        )

        assert config.precision == "int4"
        assert config.memory_strategy == "auto"
        assert config.pack_int4 is True
        assert config.calibration_data == "pileval"
        assert config.num_calib_samples == 128
        assert config.device == "cuda"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
            precision="int8",
            memory_strategy="lazy",
            pack_int4=False,
            num_calib_samples=256,
            sensitivity_threshold=0.1,
        )

        assert config.precision == "int8"
        assert config.memory_strategy == "lazy"
        assert config.pack_int4 is False
        assert config.num_calib_samples == 256
        assert config.sensitivity_threshold == 0.1

    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="Invalid precision"):
            UnifiedConfig(
                model_path="/tmp/model",
                output_dir="/tmp/output",
                precision="invalid",
            )

    def test_invalid_memory_strategy(self):
        """Test that invalid memory strategy raises error."""
        with pytest.raises(ValueError, match="Invalid memory_strategy"):
            UnifiedConfig(
                model_path="/tmp/model",
                output_dir="/tmp/output",
                memory_strategy="invalid",
            )

    def test_pack_int4_disabled_for_non_int4(self):
        """Test that pack_int4 is disabled for non-int4 precision."""
        config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
            precision="int8",
            pack_int4=True,  # Should be silently disabled
        )

        assert config.pack_int4 is False

    def test_to_dict(self):
        """Test config serialization."""
        config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
        )
        d = config.to_dict()

        assert d["model_path"] == "/tmp/model"
        assert d["output_dir"] == "/tmp/output"
        assert d["precision"] == "int4"
        assert d["memory_strategy"] == "auto"

    def test_from_dict(self):
        """Test config deserialization."""
        d = {
            "model_path": "/tmp/model",
            "output_dir": "/tmp/output",
            "precision": "fp8",
        }
        config = UnifiedConfig.from_dict(d)

        assert config.model_path == "/tmp/model"
        assert config.precision == "fp8"


class TestAutoDetectStrategy:
    """Tests for auto-detect strategy logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
            memory_strategy="auto",
        )
        self.quantizer = UnifiedQuantizer(self.config)

    def _create_mock_config(self, num_layers, hidden_size, intermediate_size):
        """Create a mock HuggingFace config."""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = num_layers
        mock_config.hidden_size = hidden_size
        mock_config.intermediate_size = intermediate_size
        mock_config.num_parameters = None  # Explicitly set to avoid MagicMock formatting issues
        return mock_config

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_small_model_full_strategy(self, mock_gpu_props, mock_cuda_available):
        """Test that small models use 'full' strategy."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24GB GPU
        mock_gpu_props.return_value = mock_props

        # Small model (~3B params, ~6GB in FP16)
        # This should fit under the 70% GPU threshold (16.8GB for 24GB GPU)
        self.quantizer.hf_config = self._create_mock_config(
            num_layers=16,
            hidden_size=2560,
            intermediate_size=6912,
        )
        self.quantizer.model_type = "llama"

        with patch("psutil.virtual_memory") as mock_ram:
            mock_mem = MagicMock()
            mock_mem.total = 128 * 1024**3
            mock_mem.available = 112 * 1024**3
            mock_ram.return_value = mock_mem

            strategy = self.quantizer._auto_detect_strategy()
            assert strategy == "full"

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_medium_model_layerwise_cpu(self, mock_gpu_props, mock_cuda_available):
        """Test that medium models use 'layerwise_cpu' strategy."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24GB GPU
        mock_gpu_props.return_value = mock_props

        # Medium model (~32GB, fits in CPU RAM but not GPU)
        self.quantizer.hf_config = self._create_mock_config(
            num_layers=60,
            hidden_size=6656,
            intermediate_size=17920,
        )
        self.quantizer.model_type = "llama"

        with patch("psutil.virtual_memory") as mock_ram:
            mock_mem = MagicMock()
            mock_mem.total = 128 * 1024**3
            mock_mem.available = 112 * 1024**3
            mock_ram.return_value = mock_mem

            strategy = self.quantizer._auto_detect_strategy()
            assert strategy == "layerwise_cpu"

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_large_model_lazy(self, mock_gpu_props, mock_cuda_available):
        """Test that large models use 'lazy' strategy."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24GB GPU
        mock_gpu_props.return_value = mock_props

        # Large model (~140GB, doesn't fit in CPU RAM)
        self.quantizer.hf_config = self._create_mock_config(
            num_layers=80,
            hidden_size=8192,
            intermediate_size=28672,
        )
        self.quantizer.model_type = "llama"

        with patch("psutil.virtual_memory") as mock_ram:
            mock_mem = MagicMock()
            mock_mem.total = 128 * 1024**3
            mock_mem.available = 112 * 1024**3
            mock_ram.return_value = mock_mem

            strategy = self.quantizer._auto_detect_strategy()
            assert strategy == "lazy"

    @patch("torch.cuda.is_available")
    def test_no_gpu_falls_back_to_cpu(self, mock_cuda_available):
        """Test behavior when no GPU is available."""
        mock_cuda_available.return_value = False

        self.quantizer.hf_config = self._create_mock_config(
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
        )
        self.quantizer.model_type = "llama"

        with patch("psutil.virtual_memory") as mock_ram:
            mock_mem = MagicMock()
            mock_mem.total = 32 * 1024**3
            mock_mem.available = 24 * 1024**3
            mock_ram.return_value = mock_mem

            strategy = self.quantizer._auto_detect_strategy()
            # Without GPU, should fall back to layerwise_cpu or lazy
            assert strategy in ["layerwise_cpu", "lazy"]

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_low_memory_system_uses_lazy(self, mock_gpu_props, mock_cuda_available):
        """Test that systems with low available RAM use lazy strategy."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB GPU
        mock_gpu_props.return_value = mock_props

        # Model that would normally use layerwise_cpu
        self.quantizer.hf_config = self._create_mock_config(
            num_layers=40,
            hidden_size=5120,
            intermediate_size=13824,
        )
        self.quantizer.model_type = "llama"

        # But system has low available memory (e.g., 16GB available)
        with patch("psutil.virtual_memory") as mock_ram:
            mock_mem = MagicMock()
            mock_mem.total = 32 * 1024**3
            mock_mem.available = 16 * 1024**3  # Only 16GB available
            mock_ram.return_value = mock_mem

            strategy = self.quantizer._auto_detect_strategy()
            # Should use lazy because model doesn't fit in available RAM
            assert strategy == "lazy"


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_auto_quantizer_alias(self):
        """Test that AutoQuantizer is an alias for UnifiedQuantizer."""
        from quanto import AutoQuantizer

        assert AutoQuantizer is UnifiedQuantizer

    def test_quantization_config_alias(self):
        """Test that QuantizationConfig is an alias for UnifiedConfig."""
        from quanto import QuantizationConfig

        assert QuantizationConfig is UnifiedConfig

    def test_old_import_style(self):
        """Test that old import style still works."""
        from quanto.core.auto_quantize import AutoQuantizer, QuantizationConfig

        config = QuantizationConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
        )
        quantizer = AutoQuantizer(config)

        assert isinstance(quantizer, UnifiedQuantizer)
        assert isinstance(config, UnifiedConfig)


class TestLayerExclusion:
    """Tests for layer exclusion logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
        )
        self.quantizer = UnifiedQuantizer(self.config)

    def test_default_exclusions(self):
        """Test that default exclusions are applied."""
        # Mock the weight index
        self.quantizer.weight_index = {
            "model.layers.0.self_attn.q_proj.weight": "/tmp/file1.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "/tmp/file1.safetensors",
            "model.embed_tokens.weight": "/tmp/file1.safetensors",
            "lm_head.weight": "/tmp/file1.safetensors",
            "model.norm.weight": "/tmp/file1.safetensors",
        }

        exclusions = self.quantizer._determine_exclude_layers()

        # Should include default patterns
        assert "lm_head" in exclusions
        assert any("*embed*" in e for e in exclusions)
        assert any("*norm*" in e for e in exclusions)

    def test_aggressive_exclusions(self):
        """Test aggressive exclusion mode."""
        self.config.aggressive_exclusion = True
        self.quantizer.weight_index = {
            "model.layers.0.mlp.gate_proj.weight": "/tmp/file1.safetensors",
        }

        exclusions = self.quantizer._determine_exclude_layers()

        # Should include gate patterns in aggressive mode
        assert any("*gate*" in e for e in exclusions)

    def test_custom_exclusions(self):
        """Test custom exclusion patterns."""
        self.config.exclude_layers = ["custom_layer.*", "special_weight"]
        self.quantizer.weight_index = {}

        exclusions = self.quantizer._determine_exclude_layers()

        assert "custom_layer.*" in exclusions
        assert "special_weight" in exclusions


@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that require actual model files.

    Run with: pytest tests/test_unified_quantizer.py -v -m integration
    """

    @pytest.fixture
    def tiny_model(self, tmp_path):
        """Create a tiny model for testing."""
        pytest.importorskip("transformers")

        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

        # Create a tiny Llama model
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=512,
        )

        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Save model
        model_path = tmp_path / "tiny_model"
        model_path.mkdir()
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        return str(model_path)

    def test_full_strategy_integration(self, tiny_model, tmp_path):
        """Test full strategy with a tiny model."""
        output_dir = tmp_path / "output_full"

        config = UnifiedConfig(
            model_path=tiny_model,
            output_dir=str(output_dir),
            memory_strategy="full",
            precision="int4",
            num_calib_samples=4,
            skip_evaluation=True,
        )

        quantizer = UnifiedQuantizer(config)
        result = quantizer.run()

        assert result.success
        assert result.output_dir == str(output_dir)
        # Full strategy uses Quark's export_safetensors which creates model files
        # Check that output directory has some content
        assert output_dir.exists()
        # Check for tokenizer files
        assert (output_dir / "tokenizer_config.json").exists() or (output_dir / "tokenizer.json").exists()

    def test_lazy_strategy_integration(self, tiny_model, tmp_path):
        """Test lazy strategy with a tiny model."""
        output_dir = tmp_path / "output_lazy"

        config = UnifiedConfig(
            model_path=tiny_model,
            output_dir=str(output_dir),
            memory_strategy="lazy",
            precision="int4",
            pack_int4=True,
            num_calib_samples=4,
            skip_evaluation=True,
        )

        quantizer = UnifiedQuantizer(config)
        result = quantizer.run()

        assert result.success
        assert result.output_dir == str(output_dir)
        # Check that quantized layers were saved
        assert (output_dir / "quantized_layers").exists()

    def test_auto_strategy_integration(self, tiny_model, tmp_path):
        """Test auto strategy with a tiny model."""
        output_dir = tmp_path / "output_auto"

        config = UnifiedConfig(
            model_path=tiny_model,
            output_dir=str(output_dir),
            memory_strategy="auto",
            precision="int4",
            num_calib_samples=4,
            skip_evaluation=True,
        )

        quantizer = UnifiedQuantizer(config)
        result = quantizer.run()

        assert result.success
        assert result.model_type is not None
        # Check that output directory has some content
        assert output_dir.exists()

    def test_awq_export_integration(self, tiny_model, tmp_path):
        """Test AWQ export format with a tiny model."""
        output_dir = tmp_path / "output_awq"

        config = UnifiedConfig(
            model_path=tiny_model,
            output_dir=str(output_dir),
            memory_strategy="full",
            precision="int4",
            export_format="awq",
            num_calib_samples=4,
            skip_evaluation=True,
        )

        quantizer = UnifiedQuantizer(config)
        result = quantizer.run()

        assert result.success
        assert output_dir.exists()
        # Check for AWQ format config (quant_method should be "awq")
        import json
        with open(output_dir / "config.json") as f:
            hf_config = json.load(f)
        assert hf_config.get("quantization_config", {}).get("quant_method") == "awq"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
