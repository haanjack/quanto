"""Tests for bugfixes in _add_quantization_config and CLI argument parsing."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quanto import UnifiedConfig, UnifiedQuantizer
from quanto.core.auto_quantize import main as cli_main


class TestAddQuantizationConfig:
    """Tests for the extracted _add_quantization_config method."""

    def setup_method(self):
        self.config = UnifiedConfig(
            model_path="/tmp/model",
            output_dir="/tmp/output",
            precision="int4",
        )
        self.quantizer = UnifiedQuantizer(self.config)
        self.quantizer.hf_config = MagicMock()

    def test_exclude_layers_passed_through(self):
        """Bug fix: exclude_layers should appear in the quantization config."""
        exclude = ["lm_head", "*embed*"]
        self.quantizer._add_quantization_config(exclude)

        qconfig = self.quantizer.hf_config.quantization_config
        assert qconfig["exclude"] == exclude

    def test_exclude_layers_default_empty(self):
        """When no exclude_layers passed, should default to []."""
        self.quantizer._add_quantization_config()
        qconfig = self.quantizer.hf_config.quantization_config
        assert qconfig["exclude"] == []

    def test_int4_dtype(self):
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["dtype"] == "int4"

    def test_int8_dtype(self):
        self.config.precision = "int8"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["dtype"] == "int8"

    def test_fp8_dtype(self):
        self.config.precision = "fp8"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["dtype"] == "fp8"

    def test_mxfp4_dtype(self):
        self.config.precision = "mxfp4"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["dtype"] == "mxfp4"

    def test_mxfp6_dtype(self):
        self.config.precision = "mxfp6"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["dtype"] == "mxfp6"

    def test_group_size_64(self):
        """Group size should be 64 when quant_scheme contains '64'."""
        self.quantizer._get_quant_scheme = lambda: "W4A16_G64"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["group_size"] == 64

    def test_group_size_128_default(self):
        """Group size should default to 128."""
        self.quantizer._get_quant_scheme = lambda: "W4A16"
        self.quantizer._add_quantization_config()
        assert self.quantizer.hf_config.quantization_config["global_quant_config"]["weight"]["group_size"] == 128


class TestCLIArgParsing:
    """Tests for CLI argument parsing fixes."""

    def test_pack_int4_default_true(self):
        """--pack_int4 should default to True."""
        with patch("sys.argv", ["prog", "--model_path", "/m", "--output_dir", "/o"]):
            with patch.object(sys, "exit"):
                with patch("quanto.core.auto_quantize.UnifiedQuantizer") as MockQ:
                    MockQ.return_value.run.return_value = MagicMock(success=True, output_dir="/o", quantized_ppl=None)
                    cli_main()
                    call_config = MockQ.call_args[0][0]
                    assert call_config.pack_int4 is True

    def test_no_pack_int4_flag(self):
        """--no-pack_int4 should set pack_int4 to False."""
        with patch("sys.argv", ["prog", "--model_path", "/m", "--output_dir", "/o", "--no-pack_int4"]):
            with patch.object(sys, "exit"):
                with patch("quanto.core.auto_quantize.UnifiedQuantizer") as MockQ:
                    MockQ.return_value.run.return_value = MagicMock(success=True, output_dir="/o", quantized_ppl=None)
                    cli_main()
                    call_config = MockQ.call_args[0][0]
                    assert call_config.pack_int4 is False

    def test_trust_remote_code_default_true(self):
        """--trust_remote_code should default to True."""
        with patch("sys.argv", ["prog", "--model_path", "/m", "--output_dir", "/o"]):
            with patch.object(sys, "exit"):
                with patch("quanto.core.auto_quantize.UnifiedQuantizer") as MockQ:
                    MockQ.return_value.run.return_value = MagicMock(success=True, output_dir="/o", quantized_ppl=None)
                    cli_main()
                    call_config = MockQ.call_args[0][0]
                    assert call_config.trust_remote_code is True

    def test_no_trust_remote_code_flag(self):
        """--no-trust_remote_code should set trust_remote_code to False."""
        with patch("sys.argv", ["prog", "--model_path", "/m", "--output_dir", "/o", "--no-trust_remote_code"]):
            with patch.object(sys, "exit"):
                with patch("quanto.core.auto_quantize.UnifiedQuantizer") as MockQ:
                    MockQ.return_value.run.return_value = MagicMock(success=True, output_dir="/o", quantized_ppl=None)
                    cli_main()
                    call_config = MockQ.call_args[0][0]
                    assert call_config.trust_remote_code is False
