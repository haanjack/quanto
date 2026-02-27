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
- **NVIDIA and AMD ROCm Support**: Works with both CUDA and ROCm backends

## Installation

### Option 1: Using Pre-built Docker Images (Recommended)

Pre-built images include Quark and all dependencies, eliminating compilation time.

#### NVIDIA CUDA
```bash
# Pull or build
docker build -f Dockerfile.nvidia.prebuilt -t haanjack/quanto:26.02-cuda .

# Run
docker run --gpus all \
    -v ~/models:/models \
    -v ~/datasets:/datasets \
    -v $(pwd):/workspace \
    -w /workspace \
    haanjack/quanto:26.02-cuda \
    python3 -m quanto --model_path /models/meta-llama/Meta-Llama-3-8B --output_dir ./quantized --precision int4
```

#### AMD ROCm
```bash
# Pull or build
docker build -f Dockerfile.rocm.prebuilt -t haanjack/quanto:26.02-rocm .

# Run
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -v ~/models:/models \
    -v ~/datasets:/datasets \
    -v $(pwd):/workspace \
    -w /workspace \
    haanjack/quanto:26.02-rocm \
    python3 -m quanto --model_path /models/meta-llama/Meta-Llama-3-8B --output_dir ./quantized --precision int4
```

### Option 2: Using Docker (Build from Source)

#### NVIDIA CUDA
```bash
docker build -f Dockerfile.nvidia -t quanto:nvidia .
docker run --gpus all -v ~/models:/models -v $(pwd):/workspace -w /workspace quanto:nvidia
```

#### AMD ROCm
```bash
docker build -f Dockerfile.rocm -t quanto:rocm .
docker run --device=/dev/kfd --device=/dev/dri --group-add video -v ~/models:/models -v $(pwd):/workspace -w /workspace quanto:rocm
```

### Option 3: Install from Source

```bash
# Install Quark (from source)
pip install /path/to/quark

# Install dependencies (NVIDIA)
pip install -r requirements-nvidia.txt

# Install dependencies (ROCm - first install PyTorch with ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements-rocm.txt
```

## Usage

### Basic Usage

```bash
python3 -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4
```

### Layer-wise Quantization (For Large Models)

For models larger than GPU memory, use `--layerwise` to quantize one layer at a time on GPU while keeping weights on CPU:

```bash
python3 -m quanto \
    --model_path /models/qwen/qwen3-32b \
    --output_dir ./quantized/qwen3-32b-int4 \
    --precision int4 \
    --layerwise \
    --skip_evaluation
```

This approach:
- Loads model weights to CPU/disk
- Moves one layer at a time to GPU for quantization
- Performs all quantization computation on GPU
- Saves quantized weights back to CPU/disk
- Uses only ~1GB GPU memory per layer

### With Custom Calibration Data

```bash
python3 -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4 \
    --calibration_data /datasets/calibration-data \
    --num_calib_samples 256
```

### Using HuggingFace Dataset for Calibration

```bash
python3 -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4 \
    --calibration_data wikitext \
    --num_calib_samples 512
```

### Memory Efficient Mode

For systems with limited GPU memory:

```bash
python3 -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4 \
    --memory_efficient
```

### Custom Layer Exclusion

```bash
python3 -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./quantized \
    --precision int4 \
    --exclude_layers lm_head model.layers.31.*
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to the model (local or HuggingFace) | Required |
| `--output_dir` | Output directory for quantized model | Required |
| `--precision` | Quantization precision (int8, int4, int4_64, int4_32, fp8, mxfp4, mxfp6, uint4) | int4 |
| `--calibration_data` | Calibration dataset (HuggingFace name or local path) | pileval |
| `--num_calib_samples` | Number of calibration samples | 512 |
| `--seq_len` | Sequence length for calibration | 512 |
| `--batch_size` | Batch size for calibration | 1 |
| `--exclude_layers` | Layers to exclude from quantization | Auto-detected |
| `--exclude_last_n_layers` | Number of last transformer layers to exclude | 1 |
| `--aggressive_exclusion` | Use aggressive layer exclusion | False |
| `--device` | Device to use (cuda, cpu) | cuda |
| `--memory_efficient` | Use memory-efficient file-to-file quantization | False |
| `--layerwise` | Use layer-wise quantization for large models (GPU-only compute) | False |
| `--skip_evaluation` | Skip perplexity evaluation | False |
| `--trust_remote_code` | Trust remote code for model loading | False |

## Output

The tool creates a quantized model in the output directory with:

- `quantized_model/` - The quantized model files
- `quantization_result.json` - Results including perplexity scores and timing

Example `quantization_result.json`:

```json
{
  "success": true,
  "output_dir": "./quantized/Meta-Llama-3-8B-INT4",
  "original_ppl": 6.137,
  "quantized_ppl": 6.883,
  "ppl_change": 0.746,
  "exclude_layers_used": ["lm_head", "model.layers.31.*"],
  "model_type": "llama",
  "quant_scheme": "int4_wo_128",
  "timing": {
    "model_loading": 2.89,
    "evaluation": 86.93,
    "quantization": 10.53,
    "export": 63.83
  }
}
```

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
- Mistral
- Phi
- And other models supported by AMD Quark

## License

This project is licensed under the [MIT License](LICENSE).

This tool uses AMD Quark for quantization. See [Quark's license](https://github.com/AMD-AIG-AIMA/Quark) for details.
