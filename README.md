# Quanto: LLM Quantization Tool

An automated tool for quantizing Large Language Models using AMD Quark.

## Features

- **Automatic Model Detection**: Automatically detects model architecture (Llama, Qwen, etc.)
- **Intelligent Layer Exclusion**: Automatically identifies sensitive layers to exclude from quantization
- **Multiple Quantization Schemes**: Supports INT4, INT8, FP8, MXFP4, and more
- **Quality Evaluation**: Evaluates model quality before and after quantization using perplexity
- **Flexible Calibration Data**: Supports HuggingFace datasets or local data
- **Memory Efficient Mode**: Supports file-to-file quantization for limited GPU memory
- **Layer-wise Quantization**: GPU-only quantization with CPU weight offloading for large models
- **HuggingFace Export**: Export quantized models to HuggingFace format
- **Dequantization**: Convert quantized models back to BF16/FP16 for re-quantization or deployment
- **NVIDIA and AMD ROCm Support**: Works with both CUDA and ROCm backends

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/your-org/quanto.git
cd quanto

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install AMD Quark first
pip install contribs/quark

# Install quanto
pip install -e .

# Install quark
pip install contribs/quark

# For NVIDIA CUDA
pip install -e ".[nvidia]"

# For AMD ROCm (first install PyTorch with ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -e ".[rocm]"
```

### Option 2: Using Docker

Pre-built images include Quark and all dependencies, eliminating compilation time.

#### NVIDIA CUDA
```bash
# Build pre-built image (quanto baked in)
docker build -f docker/Dockerfile.cuda -t quanto:cuda .

# Run quantization
docker run --gpus all \
    -v ~/models:/models \
    -v ./outputs:/output \
    quanto:cuda \
    python -m quanto --model_path /models/meta-llama/Meta-Llama-3-8B --output_dir /output/quantized --precision int4

# For development (volume mount source)
docker build -f docker/Dockerfile.cuda.dev -t quanto:cuda-dev .
docker run --gpus all -v $(pwd):/workspace -w /workspace quanto:cuda-dev bash
```

#### AMD ROCm
```bash
# Build pre-built image (quanto baked in)
docker build -f docker/Dockerfile.rocm -t quanto:rocm .

# Run quantization
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -v ~/models:/models \
    -v ./outputs:/output \
    quanto:rocm \
    python -m quanto --model_path /models/meta-llama/Meta-Llama-3-8B --output_dir /output/quantized --precision int4

# For development (volume mount source)
docker build -f docker/Dockerfile.rocm.dev -t quanto:rocm-dev .
docker run --device=/dev/kfd --device=/dev/dri --group-add video -v $(pwd):/workspace -w /workspace quanto:rocm-dev bash
```

## Project Structure

```
quanto/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── requirements.txt        # Base requirements
├── requirements-nvidia.txt # NVIDIA-specific deps
├── requirements-rocm.txt   # ROCm-specific deps
├── contribs/
│   └── quark/              # AMD Quark (submodule)
├── docker/
│   ├── Dockerfile.cuda       # Pre-built for CUDA
│   ├── Dockerfile.cuda.dev   # Development for CUDA
│   ├── Dockerfile.rocm       # Pre-built for ROCm
│   └── Dockerfile.rocm.dev   # Development for ROCm
├── docs/
│   └── examples.md         # Experiment results
├── examples/               # Example scripts
├── scripts/
│   └── repack.py           # Weight packing utilities
├── src/quanto/             # Main package
│   ├── __init__.py
│   ├── __main__.py         # CLI entry point
│   ├── constants.py        # Shared constants
│   ├── core/               # Quantization engines
│   │   ├── base_quantizer.py
│   │   ├── auto_quantize.py
│   │   ├── layerwise_quant.py
│   │   ├── lazy_layerwise_quant.py
│   │   ├── iterative_quantizer.py
│   │   └── dequantize.py
│   ├── analysis/           # Layer analysis
│   │   ├── layer_analyzer.py
│   │   └── sensitivity_analyzer.py
│   ├── export/             # Export utilities
│   │   ├── hf_export.py
│   │   └── model_assembler.py
│   └── utils/              # Shared utilities
│       ├── calibration.py
│       ├── int4_pack.py
│       ├── logging.py
│       ├── memory.py
│       └── model_utils.py
└── tests/                  # Test suite
```

## Usage

### Basic Usage

```bash
python -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4
```

### Layer-wise Quantization (For Large Models)

For models larger than GPU memory, use `--layerwise` to quantize one layer at a time:

```bash
python -m quanto \
    --model_path /models/qwen/qwen3-32b \
    --output_dir ./quantized/qwen3-32b-int4 \
    --precision int4 \
    --layerwise \
    --skip_evaluation
```

### With Custom Calibration Data

```bash
python -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4 \
    --calibration_data /datasets/calibration-data \
    --num_calib_samples 256
```

### Dequantization (INT4 → BF16/FP16)

```bash
python -m quanto --dequantize \
    --model_path ./quantized/qwen3-32b-int4 \
    --output_dir ./dequantized/qwen3-32b-bf16
```

### Weight Packing

Pack BF16 weights to INT4 format for 4x compression:

```bash
python scripts/repack.py pack \
    --input_dir ./quantized_layers \
    --output_dir ./packed_layers
```

## Python API

```python
from quanto import AutoQuantizer, QuantizationConfig

config = QuantizationConfig(
    model_path="/models/meta-llama/Meta-Llama-3-8B",
    output_dir="./quantized",
    precision="int4",
    calibration_data="pileval",
    num_calib_samples=128,
)

quantizer = AutoQuantizer(config)
result = quantizer.run()

print(f"Original PPL: {result.original_ppl:.4f}")
print(f"Quantized PPL: {result.quantized_ppl:.4f}")
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to the model (local or HuggingFace) | Required |
| `--output_dir` | Output directory for quantized model | Required |
| `--precision` | Quantization precision (int8, int4, int4_64, int4_32, fp8, mxfp4, mxfp6, uint4) | int4 |
| `--calibration_data` | Calibration dataset (HuggingFace name or local path) | pileval |
| `--num_calib_samples` | Number of calibration samples | 128 |
| `--seq_len` | Sequence length for calibration | 512 |
| `--batch_size` | Batch size for calibration | 1 |
| `--exclude_layers` | Layers to exclude from quantization | Auto-detected |
| `--exclude_last_n_layers` | Number of last transformer layers to exclude | 1 |
| `--aggressive_exclusion` | Use aggressive layer exclusion | False |
| `--device` | Device to use (cuda, cpu) | cuda |
| `--memory_efficient` | Use memory-efficient file-to-file quantization | False |
| `--layerwise` | Use layer-wise quantization for large models | False |
| `--skip_evaluation` | Skip perplexity evaluation | False |
| `--trust_remote_code` | Trust remote code for model loading | True |

## Quantization Schemes

| Scheme | Description | Recommended For |
|--------|-------------|-----------------|
| `int4` | INT4 weight-only (128 group size) | Best quality/size tradeoff |
| `int4_64` | INT4 weight-only (64 group size) | Higher quality, slightly larger |
| `int4_32` | INT4 weight-only (32 group size) | Highest quality, larger |
| `int8` | INT8 weight + activation | More compression, may degrade quality |
| `fp8` | FP8 quantization | FP8-compatible hardware |
| `mxfp4` | MX FP4 quantization | Experimental |
| `uint4` | UINT4 weight-only | Alternative to INT4 |

## Supported Models

- Llama (2, 3, 3.1, 3.2)
- Qwen (2, 2.5, 3)
- Mistral / Mixtral
- Phi (2, 3)
- DeepSeek
- Gemma
- And other models supported by AMD Quark

## Documentation

- [Examples and Experiment Results](docs/examples.md)

## License

MIT License. This tool uses AMD Quark for quantization. See Quark's license for details.
