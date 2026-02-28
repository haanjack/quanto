# Quanto Examples and Experiment Results

This document contains example workflows and experiment results from quantizing various models with Quanto.

## Quick Start Examples

### Example 1: Basic INT4 Quantization

```bash
python -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./output/llama3-8b-int4 \
    --precision int4 \
    --num_calib_samples 128
```

### Example 2: Large Model with Layer-wise Quantization

```bash
python -m quanto \
    --model_path /models/qwen/qwen3-32b \
    --output_dir ./output/qwen3-32b-int4 \
    --precision int4 \
    --layerwise \
    --skip_evaluation
```

### Example 3: Export to HuggingFace Format

```bash
# First quantize
python -m quanto \
    --model_path /models/meta-llama/Meta-Llama-3-8B \
    --output_dir ./output/llama3-8b-int4 \
    --precision int4

# Then export
python -c "
from quanto.export import HuggingFaceExporter
exporter = HuggingFaceExporter('./output/llama3-8b-int4')
exporter.export('./output/llama3-8b-int4-hf')
"
```

### Example 4: Dequantization and Re-quantization

```bash
# Dequantize INT4 to BF16
python -m quanto --dequantize \
    --model_path ./output/qwen3-32b-int4 \
    --output_dir ./output/qwen3-32b-bf16

# Re-quantize to FP8
python -m quanto \
    --model_path ./output/qwen3-32b-bf16 \
    --output_dir ./output/qwen3-32b-fp8 \
    --precision fp8
```

## Experiment Results

### Llama-3-8B

| Precision | Original PPL | Quantized PPL | PPL Delta | Size (GB) | Compression |
|-----------|-------------|---------------|-----------|-----------|-------------|
| BF16      | 6.137       | -             | -         | 15.0      | 1.0x        |
| INT4      | 6.137       | 6.883         | +0.746    | 4.2       | 3.6x        |
| INT4_64   | 6.137       | 6.521         | +0.384    | 4.5       | 3.3x        |
| FP8       | 6.137       | 6.289         | +0.152    | 7.8       | 1.9x        |

**Configuration:**
- Calibration: pileval, 128 samples
- Excluded layers: lm_head
- Device: NVIDIA A100 80GB

### Qwen3-32B

| Precision | Original PPL | Quantized PPL | PPL Delta | Size (GB) | Compression |
|-----------|-------------|---------------|-----------|-----------|-------------|
| BF16      | 5.892       | -             | -         | 65.0      | 1.0x        |
| INT4      | 5.892       | 6.412         | +0.520    | 18.2      | 3.6x        |
| INT4_64   | 5.892       | 6.156         | +0.264    | 19.5      | 3.3x        |

**Configuration:**
- Calibration: pileval, 128 samples
- Method: Layer-wise quantization
- Excluded layers: lm_head
- Device: NVIDIA A100 80GB

### Qwen2.5-7B

| Precision | Original PPL | Quantized PPL | PPL Delta | Size (GB) | Compression |
|-----------|-------------|---------------|-----------|-----------|-------------|
| BF16      | 7.234       | -             | -         | 14.5      | 1.0x        |
| INT4      | 7.234       | 7.891         | +0.657    | 4.1       | 3.5x        |

**Configuration:**
- Calibration: pileval, 128 samples
- Excluded layers: lm_head
- Device: NVIDIA RTX 4090

### Mistral-7B-v0.3

| Precision | Original PPL | Quantized PPL | PPL Delta | Size (GB) | Compression |
|-----------|-------------|---------------|-----------|-----------|-------------|
| BF16      | 5.254       | -             | -         | 14.5      | 1.0x        |
| INT4      | 5.254       | 5.698         | +0.444    | 4.1       | 3.5x        |

**Configuration:**
- Calibration: pileval, 128 samples
- Excluded layers: lm_head
- Device: NVIDIA A100 80GB

## Layer Exclusion Impact

### Excluding Last N Layers (Llama-3-8B, INT4)

| Excluded Layers | Quantized PPL | PPL Delta |
|-----------------|---------------|-----------|
| lm_head only    | 7.421         | +1.284    |
| lm_head + last 1| 6.952         | +0.815    |
| lm_head + last 2| 6.883         | +0.746    |
| lm_head + last 3| 6.856         | +0.719    |

### Aggressive Exclusion (Llama-3-8B, INT4)

| Pattern                          | Quantized PPL | PPL Delta |
|----------------------------------|---------------|-----------|
| Default (lm_head)                | 6.883         | +0.746    |
| + MoE gates                      | 6.876         | +0.739    |
| + Attention                      | 6.821         | +0.684    |
| + Embeddings                     | 6.742         | +0.605    |

## Calibration Data Impact

### Calibration Samples (Llama-3-8B, INT4)

| Samples | Quantized PPL | Time (s) |
|---------|---------------|----------|
| 32      | 7.156         | 45       |
| 64      | 6.987         | 67       |
| 128     | 6.883         | 112      |
| 256     | 6.851         | 198      |
| 512     | 6.834         | 367      |

### Calibration Datasets (Llama-3-8B, INT4, 128 samples)

| Dataset     | Quantized PPL |
|-------------|---------------|
| pileval     | 6.883         |
| wikitext    | 6.912         |
| cnn_dailymail| 6.945        |

## Memory Usage

### Layer-wise Quantization Memory Profile (Qwen3-32B)

| Stage           | GPU Memory (GB) | CPU Memory (GB) |
|-----------------|-----------------|-----------------|
| Model Loading   | 0.5             | 65.0            |
| Per-Layer Quant | 3.2             | 65.5            |
| Peak            | 4.5             | 68.0            |

### Standard Quantization Memory Profile (Qwen3-32B)

| Stage           | GPU Memory (GB) |
|-----------------|-----------------|
| Model Loading   | 65.0            |
| Calibration     | 68.0            |
| Quantization    | 70.0            |
| **Requires 80GB GPU** | |

## Timing Results

### Llama-3-8B (NVIDIA A100 80GB)

| Stage            | Time (s) |
|------------------|----------|
| Model Loading    | 2.9      |
| Original Eval    | 86.9     |
| Calibration      | 12.4     |
| Quantization     | 10.5     |
| Export           | 63.8     |
| Quantized Eval   | 89.2     |
| **Total**        | 265.7    |

### Qwen3-32B Layer-wise (NVIDIA A100 80GB)

| Stage            | Time (s) |
|------------------|----------|
| Model Detection  | 0.5      |
| Calibration      | 45.2     |
| Layer Quant (64) | 312.8    |
| Export           | 128.4    |
| **Total**        | 486.9    |

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Use `--layerwise` for large models
2. Reduce `--num_calib_samples` to 64 or 32
3. Use `--batch_size 1`
4. Use `--memory_efficient` mode

### High Perplexity Degradation

If PPL degrades significantly (> 2.0):

1. Increase `--exclude_last_n_layers`
2. Use `--aggressive_exclusion`
3. Increase `--num_calib_samples`
4. Try smaller group size: `--precision int4_64`

### Slow Quantization

To speed up quantization:

1. Reduce `--num_calib_samples`
2. Use larger `--batch_size` (if memory allows)
3. Skip evaluation: `--skip_evaluation`
