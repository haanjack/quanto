# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Quanto

Quanto is an LLM quantization toolkit built on AMD Quark. It quantizes HuggingFace models to INT4/INT8/FP8/MXFP4/MXFP6 precisions with multiple memory strategies for different GPU constraints. Source code lives in `src/quanto/`.

## Commands

```bash
# Install
pip install -e ".[dev]"              # dev (pytest, ruff)
pip install -e ".[nvidia]"           # with NVIDIA extras
pip install -e ".[rocm]"             # with ROCm extras

# Tests
pytest tests/ -v                     # all tests
pytest tests/test_unified_quantizer.py -v   # single file
pytest tests/test_unified_quantizer.py::TestUnifiedConfig::test_default_config -v  # single test

# Lint & format
ruff check src/                      # lint
ruff check src/ --fix                # lint with autofix
ruff format src/                     # format

# Quantize a model (CLI)
python -m quanto --model_path /path/to/model --output_dir ./output --precision int4
python -m quanto --model_path /path --sensitivity_analysis --sensitivity_threshold 0.12

# Dequantize
python -m quanto --dequantize --model_path ./quantized --output_dir ./dequantized

# Docker-based integration tests
./scripts/run_tests.sh --gpu nvidia --test all
```

## Architecture

### Pipeline flow
CLI (`__main__.py`) -> `UnifiedConfig` (dataclass validation) -> `UnifiedQuantizer.run()` -> `QuantizationResult`

### Core modules (`src/quanto/core/`)
- **`config.py`** — `UnifiedConfig` dataclass with ~23 fields and `__post_init__` validation. `QuantizationConfig` is a backward-compat alias.
- **`unified_quantizer.py`** — Main quantizer implementing 4 memory strategies: `full` (entire model on GPU), `layerwise_cpu` (model on CPU, layers quantized one-by-one on GPU), `lazy` (weights loaded on-demand from safetensors), `auto` (selects based on model size vs GPU memory).
- **`base_quantizer.py`** — Abstract base class, `QuantizationResult` dataclass.
- **`dequantize.py`** — INT4 -> BF16/FP16 conversion.
- **`sensitivity/`** — Sequential sensitivity analysis: `SequentialSensitivityAnalyzer` scores per-layer quantization impact, `ActivationCache` manages GPU/CPU caching, `SensitivityScorer` computes perplexity-based metrics.

### Supporting modules
- **`constants.py`** — `PRECISION_TO_SCHEME` mapping (e.g., `"int4"` -> `"int4_wo_128"`), `MODEL_TYPE_MAPPINGS`, `DEFAULT_EXCLUDE_PATTERNS`.
- **`analysis/layer_analyzer.py`** — Automatic detection of layers to exclude (lm_head, MoE gates, embeddings/norms with aggressive mode).
- **`utils/calibration.py`** — `CalibrationDataManager` loads from HuggingFace datasets or local files.
- **`utils/int4_pack.py`** — INT4 <-> INT32 packing/unpacking.
- **`utils/memory.py`** — GPU memory tracking and cleanup.
- **`utils/model_utils.py`** — Model type detection and Quark template lookup.

### External dependency
AMD Quark is vendored as a git submodule in `contribs/quark/`. It provides the quantization scheme templates for each model architecture.

## Code style

- Ruff configured: 100-char line length, Python 3.10 target
- Lint rules: E, W, F, I (isort), B (bugbear), C4, UP, ARG, SIM
- Double quotes, space indentation
- `contribs/` directory is excluded from linting

## Key patterns

- **Backward compatibility aliases**: `QuantizationConfig = UnifiedConfig`, `AutoQuantizer` wraps `UnifiedQuantizer`
- **Valid precisions**: `int4`, `int4_64`, `int4_32`, `int8`, `fp8`, `mxfp4`, `mxfp6`, `uint4`
- **Memory strategies**: `full`, `layerwise_cpu`, `lazy`, `auto`
- **Export formats**: `quark` (native, default), `awq`, `gptq` (vLLM compat, INT4 only)
