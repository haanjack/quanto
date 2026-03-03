#!/bin/bash
# End-to-End Test Script for UnifiedQuantizer
#
# Usage:
#   ./scripts/run_e2e_tests.sh [cuda|rocm] [test_numbers]
#
# Examples:
#   ./scripts/run_e2e_tests.sh cuda        # Run all CUDA tests (1-8)
#   ./scripts/run_e2e_tests.sh cuda 1,2,3  # Run CUDA tests 1, 2, 3
#   ./scripts/run_e2e_tests.sh rocm        # Run all ROCm tests (1-6)
#
# Prerequisites:
#   - Docker installed
#   - Models downloaded to ~/models/
#   - Build image first:
#       docker build -f docker/Dockerfile.cuda -t quanto:cuda .
#       docker build -f docker/Dockerfile.rocm -t quanto:rocm .

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/test_outputs"
LOG_DIR="${OUTPUT_DIR}/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Parse arguments
PLATFORM="${1:-cuda}"
shift || true
TEST_RANGE="$1"

log_info "============================================"
log_info "  Quanto End-to-End Test Suite"
log_info "============================================"
log_info "Platform: $PLATFORM"
log_info "Project directory: $PROJECT_DIR"
log_info "Output directory: $OUTPUT_DIR"
log_info "============================================"

# Select Docker image based on platform
if [ "$PLATFORM" == "cuda" ]; then
    DOCKER_IMAGE="quanto:cuda"
    DOCKERFILE="docker/Dockerfile.cuda"
    EXTRA_ARGS="--gpus all"
elif [ "$PLATFORM" == "rocm" ]; then
    DOCKER_IMAGE="quanto:rocm"
    DOCKERFILE="docker/Dockerfile.rocm"
    EXTRA_ARGS="--device /dev/kfd --device /dev/dri --group-add video"
else
    log_error "Unknown platform: $PLATFORM. Use 'cuda' or 'rocm'."
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    log_warn "Docker image $DOCKER_IMAGE not found."
    log_info "Building image from $DOCKERFILE..."
    docker build -f "$PROJECT_DIR/$DOCKERFILE" -t "$DOCKER_IMAGE" "$PROJECT_DIR"
fi

# Docker run command template
docker_run() {
    local test_name="$1"
    shift
    docker run --rm \
        $EXTRA_ARGS \
        -v "$PROJECT_DIR:/workspace:rw" \
        -v "$HOME/models:/models:ro" \
        -v "$OUTPUT_DIR:/outputs:rw" \
        -w /workspace \
        -e PYTHONPATH=/workspace/src \
        "$DOCKER_IMAGE" \
        "$@"
}

# Test functions

test_1_llama3_int4_quantize() {
    log_test "Test 1: Quantizing Llama-3-8B to INT4..."

    local output_dir="/outputs/llama3-8b-int4"
    local log_file="$LOG_DIR/test1_quantize.log"

    docker_run "test1" python -c "
from quanto import UnifiedQuantizer, UnifiedConfig
import json
import sys

config = UnifiedConfig(
    model_path='/models/meta-llama/Meta-Llama-3-8B',
    output_dir='$output_dir',
    precision='int4',
    memory_strategy='auto',
    pack_int4=True,
    num_calib_samples=128,
    skip_evaluation=True,
)

quantizer = UnifiedQuantizer(config)
result = quantizer.run()

print('Result:', json.dumps(result.to_dict(), indent=2))
if not result.success:
    print('TEST 1 FAILED: Quantization failed!', file=sys.stderr)
    sys.exit(1)
print('TEST 1 PASSED: Llama-3-8B INT4 quantization successful')
" 2>&1 | tee "$log_file"

    log_info "Test 1 completed. Output: $OUTPUT_DIR/llama3-8b-int4"
}

test_2_llama3_load_test() {
    log_test "Test 2: Loading Llama-3-8B INT4 checkpoint..."

    local checkpoint_dir="/outputs/llama3-8b-int4"
    local log_file="$LOG_DIR/test2_load.log"

    docker_run "test2" python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quark.torch import import_model_from_safetensors
import sys

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$checkpoint_dir')

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    '/models/meta-llama/Meta-Llama-3-8B',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

print('Importing quantized weights...')
model = import_model_from_safetensors(model, '$checkpoint_dir')

# Simple generation test
prompt = 'The capital of France is'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

print(f'Input: {prompt}')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Output: {generated}')

# Basic validation
if 'Paris' not in generated and 'paris' not in generated.lower():
    print(f'TEST 2 FAILED: Expected Paris in output, got: {generated}', file=sys.stderr)
    sys.exit(1)

print('TEST 2 PASSED: Llama-3-8B INT4 checkpoint loaded and generates expected output')
" 2>&1 | tee "$log_file"

    log_info "Test 2 completed."
}

test_3_llama3_mmlu_hf() {
    log_test "Test 3: Llama-3-8B INT4 MMLU (5-shot) with HuggingFace backend..."

    local checkpoint_dir="$OUTPUT_DIR/llama3-8b-int4"
    local log_file="$LOG_DIR/test3_mmlu_hf.log"

    docker_run "test3" lm_eval \
        --model hf \
        --model_args "pretrained=$checkpoint_dir,dtype=float16,trust_remote_code=True" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 4 \
        --output_path "/outputs/llama3_mmlu_hf" \
        2>&1 | tee "$log_file"

    log_info "Test 3 completed. Results in: $OUTPUT_DIR/llama3_mmlu_hf"
}

test_4_qwen32_int4_quantize() {
    log_test "Test 4: Quantizing Qwen3-32B to INT4..."

    local output_dir="/outputs/qwen3-32b-int4"
    local log_file="$LOG_DIR/test4_quantize.log"

    docker_run "test4" python -c "
from quanto import UnifiedQuantizer, UnifiedConfig
import json
import sys

config = UnifiedConfig(
    model_path='/models/qwen/qwen3-32b',
    output_dir='$output_dir',
    precision='int4',
    memory_strategy='lazy',  # Use lazy for large models
    pack_int4=True,
    num_calib_samples=128,
    skip_evaluation=True,
)

quantizer = UnifiedQuantizer(config)
result = quantizer.run()

print('Result:', json.dumps(result.to_dict(), indent=2))
if not result.success:
    print('TEST 4 FAILED: Quantization failed!', file=sys.stderr)
    sys.exit(1)
print('TEST 4 PASSED: Qwen3-32B INT4 quantization successful')
" 2>&1 | tee "$log_file"

    log_info "Test 4 completed. Output: $OUTPUT_DIR/qwen3-32b-int4"
}

test_5_qwen32_load_test() {
    log_test "Test 5: Verifying Qwen3-32B INT4 checkpoint structure..."

    local checkpoint_dir="/outputs/qwen3-32b-int4"
    local log_file="$LOG_DIR/test5_load.log"

    docker_run "test5" python -c "
import json
from pathlib import Path
from safetensors import safe_open

checkpoint = Path('$checkpoint_dir')

# Verify required files exist
required_files = ['config.json', 'tokenizer.json', 'model.safetensors.index.json']
for f in required_files:
    if not (checkpoint / f).exists():
        raise FileNotFoundError(f'Missing required file: {f}')
print(f'✓ All required files present')

# Verify config has quantization_config
with open(checkpoint / 'config.json') as f:
    config = json.load(f)
if 'quantization_config' not in config:
    raise ValueError('Missing quantization_config in config.json')
print(f'✓ quantization_config present')
print(f'  quant_method: {config[\"quantization_config\"].get(\"quant_method\")}')
print(f'  dtype: {config[\"quantization_config\"][\"global_quant_config\"][\"weight\"][\"dtype\"]}')

# Verify safetensors index
with open(checkpoint / 'model.safetensors.index.json') as f:
    index = json.load(f)
weight_map = index.get('weight_map', {})
print(f'✓ Weight index contains {len(weight_map)} tensors')

# Check for packed weights (INT4 format)
packed_count = sum(1 for k in weight_map if '.packed' in k)
scale_count = sum(1 for k in weight_map if '.scale' in k)
zero_point_count = sum(1 for k in weight_map if '.zero_point' in k)
print(f'  Packed weights: {packed_count}')
print(f'  Scale tensors: {scale_count}')
print(f'  Zero point tensors: {zero_point_count}')

# Verify we can open one shard
shard_files = set(weight_map.values())
first_shard = checkpoint / list(shard_files)[0]
with safe_open(first_shard, framework='pt') as f:
    keys = list(f.keys())
print(f'✓ Can read safetensors shard: {len(keys)} tensors in {list(shard_files)[0]}')

print('')
print('TEST 5 PASSED: Qwen3-32B INT4 checkpoint structure verified')
" 2>&1 | tee "$log_file"

    log_info "Test 5 completed."
}

test_6_qwen32_mmlu_vllm() {
    log_test "Test 6: Qwen3-32B INT4 MMLU (5-shot) with vLLM backend..."

    local checkpoint_dir="/outputs/qwen3-32b-int4"
    local log_file="$LOG_DIR/test6_mmlu_vllm.log"

    docker_run "test6" lm_eval \
        --model vllm \
        --model_args "pretrained=$checkpoint_dir,dtype=float16,trust_remote_code=True,tensor_parallel_size=1" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "/outputs/qwen32_mmlu_vllm" \
        2>&1 | tee "$log_file"

    log_info "Test 6 completed. Results in: $OUTPUT_DIR/qwen32_mmlu_vllm"
}

test_7_qwen32_int8_quantize() {
    log_test "Test 7: Quantizing Qwen3-32B to INT8..."

    local output_dir="/outputs/qwen3-32b-int8"
    local log_file="$LOG_DIR/test7_quantize.log"

    docker_run "test7" python -c "
from quanto import UnifiedQuantizer, UnifiedConfig
import json
import sys

config = UnifiedConfig(
    model_path='/models/qwen/qwen3-32b',
    output_dir='$output_dir',
    precision='int8',
    memory_strategy='lazy',
    num_calib_samples=128,
    skip_evaluation=True,
)

quantizer = UnifiedQuantizer(config)
result = quantizer.run()

print('Result:', json.dumps(result.to_dict(), indent=2))
if not result.success:
    print('TEST 7 FAILED: Quantization failed!', file=sys.stderr)
    sys.exit(1)
print('TEST 7 PASSED: Qwen3-32B INT8 quantization successful')
" 2>&1 | tee "$log_file"

    log_info "Test 7 completed. Output: $OUTPUT_DIR/qwen3-32b-int8"
}

test_8_qwen32_mmlu_vllm_tp2() {
    log_test "Test 8: Qwen3-32B INT8 MMLU (5-shot) with vLLM backend (TP=2)..."

    local checkpoint_dir="/outputs/qwen3-32b-int8"
    local log_file="$LOG_DIR/test8_mmlu_vllm_tp2.log"

    docker_run "test8" lm_eval \
        --model vllm \
        --model_args "pretrained=$checkpoint_dir,dtype=float16,trust_remote_code=True,tensor_parallel_size=2" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "/outputs/qwen32_mmlu_vllm_tp2" \
        2>&1 | tee "$log_file"

    log_info "Test 8 completed. Results in: $OUTPUT_DIR/qwen32_mmlu_vllm_tp2"
}

# Run tests based on platform and test number
run_cuda_tests() {
    local test_range="$1"

    if [ -z "$test_range" ]; then
        # Run all tests
        log_info "Running all CUDA tests (1-8)..."
        test_1_llama3_int4_quantize
        test_2_llama3_load_test
        test_3_llama3_mmlu_hf
        test_4_qwen32_int4_quantize
        test_5_qwen32_load_test
        test_6_qwen32_mmlu_vllm
        test_7_qwen32_int8_quantize
        test_8_qwen32_mmlu_vllm_tp2
    else
        # Run specific tests
        IFS=',' read -ra TESTS <<< "$test_range"
        for t in "${TESTS[@]}"; do
            case $t in
                1) test_1_llama3_int4_quantize ;;
                2) test_2_llama3_load_test ;;
                3) test_3_llama3_mmlu_hf ;;
                4) test_4_qwen32_int4_quantize ;;
                5) test_5_qwen32_load_test ;;
                6) test_6_qwen32_mmlu_vllm ;;
                7) test_7_qwen32_int8_quantize ;;
                8) test_8_qwen32_mmlu_vllm_tp2 ;;
                *) log_error "Unknown test number: $t" ;;
            esac
        done
    fi
}

run_rocm_tests() {
    local test_range="$1"

    if [ -z "$test_range" ]; then
        # Run all ROCm tests (1-6 only)
        log_info "Running all ROCm tests (1-6)..."
        test_1_llama3_int4_quantize
        test_2_llama3_load_test
        test_3_llama3_mmlu_hf
        test_4_qwen32_int4_quantize
        test_5_qwen32_load_test
        test_6_qwen32_mmlu_vllm
    else
        # Run specific tests
        IFS=',' read -ra TESTS <<< "$test_range"
        for t in "${TESTS[@]}"; do
            case $t in
                1) test_1_llama3_int4_quantize ;;
                2) test_2_llama3_load_test ;;
                3) test_3_llama3_mmlu_hf ;;
                4) test_4_qwen32_int4_quantize ;;
                5) test_5_qwen32_load_test ;;
                6) test_6_qwen32_mmlu_vllm ;;
                *) log_error "Unknown test number: $t (ROCm supports tests 1-6)" ;;
            esac
        done
    fi
}

# Main execution
if [ "$PLATFORM" == "cuda" ]; then
    run_cuda_tests "$TEST_RANGE"
else
    run_rocm_tests "$TEST_RANGE"
fi

log_info "============================================"
log_info "  All tests completed!"
log_info "============================================"
log_info "Logs available in: $LOG_DIR"
log_info "Outputs available in: $OUTPUT_DIR"
