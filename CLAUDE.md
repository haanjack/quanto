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

# Tests (requires Quark — run on remote server with amd-quark installed)
pytest tests/ -v                     # all tests
pytest tests/test_unified_quantizer.py -v   # single file
pytest tests/test_unified_quantizer.py::TestUnifiedConfig::test_default_config -v  # single test

# Lint & format
ruff check src/                      # lint
ruff check src/ --fix                # lint with autofix
ruff format src/                     # format

# Quantize a model (CLI)
python -m quanto \
    --model_path model/path \
    --output_dir ./output \
    --precision mxfp4 \
    --sensitivity_analysis \
    --sensitivity_threshold 0.12

# Quantize with explicit exclude list (e.g., attn-excl strategy)
python -m quanto \
    --model_path model/path \
    --output_dir ./output \
    --precision mxfp4 \
    --exclude_layers_file exclude.json

# Quantize (Python API)
from quanto import UnifiedQuantizer, UnifiedConfig
config = UnifiedConfig(
    model_path='model/path', output_dir='./output',
    precision='mxfp4', sensitivity_analysis=True,
    sensitivity_threshold=0.12,
)
UnifiedQuantizer(config).run()

# Dequantize
python -m quanto --dequantize --model_path ./quantized --output_dir ./dequantized

# Docker-based integration tests
./scripts/run_e2e_tests.sh rocm       # all ROCm tests
./scripts/run_e2e_tests.sh cuda 1,2   # specific CUDA tests
```

## Architecture

### Pipeline flow
`UnifiedConfig` (dataclass validation) -> `UnifiedQuantizer.run()` -> strategy dispatch -> `QuantizationResult`

### Quantization paths

**MXFP4/MXFP6** — Uses Quark's `quantize_model_per_safetensor` (file2file). Processes each safetensors shard independently without loading the full model. Produces packed uint8 weights + E8M0 scales compatible with vLLM's Quark loader.

**INT4/INT8/FP8** — Uses in-memory quantization via `ModelQuantizer` + `export_safetensors`. Three memory strategies:
- `full` — entire model on GPU
- `layerwise_cpu` — model on CPU, layers quantized one-by-one on GPU
- `lazy` — weights loaded on-demand from safetensors

### Core modules (`src/quanto/core/`)
- **`config.py`** — `UnifiedConfig` dataclass. Key fields: `precision`, `memory_strategy`, `algorithm` (rtn/awq/gptq), `sensitivity_analysis`, `sensitivity_threshold`, `exclude_layers`.
- **`unified_quantizer.py`** — Main quantizer. `run()` dispatches to `_run_file2file_quantization()` for MXFP or `_run_full_gpu_quantization()` / `_run_lazy_quantization()` for INT4/INT8. Contains `_determine_exclude_layers()` with sensitivity analysis and `_align_exclude_groups()` for vLLM fused layer compatibility.
- **`sensitivity/sequential_analyzer.py`** — Iterative sensitivity analysis. Scores each layer using the actual target precision (MXFP4 uses `OCP_MXFP4Spec`, not INT4 proxy). `_build_quant_config_for_scoring()` maps precision to the correct Quark spec class.

### Supporting modules
- **`constants.py`** — `PRECISION_TO_SCHEME` mapping, `MODEL_TYPE_MAPPINGS` (includes `solar_open` -> `qwen3_moe`, `kimi_k2` -> `kimi_k25`), `SUPPORTED_ALGORITHMS`.
- **`auto_quantize.py`** — CLI `main()` entry point. Parses args and creates `UnifiedConfig`. Supports `--exclude_layers_file` for JSON exclude lists.
- **`utils/model_utils.py`** — `detect_model_type()` and `get_template()` for Quark `LLMTemplate` lookup.
- **`utils/calibration.py`** — `CalibrationDataManager` loads from HuggingFace datasets or local files.
- **`utils/int4_pack.py`** — INT4 <-> INT32 packing/unpacking.

### External dependency
AMD Quark is vendored as a git submodule in `contribs/quark/`. Key Quark APIs used:
- `LLMTemplate.get_config(scheme, algorithm, exclude_layers)` — generates per-architecture quantization configs
- `quantize_model_per_safetensor()` — file-to-file quantization (MXFP4 path)
- `ModelQuantizer` / `export_safetensors()` — in-memory quantization (INT4/INT8 path)
- `OCP_MXFP4Spec`, `Int4PerGroupSpec` — precision-specific quantization specs

## Code style

- Ruff configured: 100-char line length, Python 3.10 target
- Lint rules: E, W, F, I (isort), B (bugbear), C4, UP, ARG, SIM
- Double quotes, space indentation
- `contribs/` directory is excluded from linting

## Key patterns

- **vLLM fused layer alignment**: `_align_exclude_groups()` ensures q/k/v projections and gate/up projections are excluded together (vLLM fuses these into `qkv_proj` and `gate_up_proj`)
- **AWQ/GPTQ**: Set `algorithm="awq"` or `"gptq"` in config — passed to `LLMTemplate.get_config(algorithm=...)`. Quark handles execution internally via `AwqProcessor`/`GptqProcessor`.
- **Backward compat aliases**: `QuantizationConfig = UnifiedConfig`, `AutoQuantizer = UnifiedQuantizer`
- **HF hub resolution**: File2file path auto-resolves HF hub IDs to local cache via `snapshot_download`

## Testing environment

Remote server mi355-gpu-16 (aac14 cluster) with MI355 GPUs. Use podman containers with `rocm/vllm-dev:nightly` image which includes PyTorch, Quark, and all dependencies. See `memory/reference_mi355_server.md` for access details.
