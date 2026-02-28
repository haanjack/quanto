#!/bin/bash
# =============================================================================
# Quanto Docker Test Runner
# =============================================================================
# This script runs quantization tests using Docker images for both
# NVIDIA CUDA and AMD ROCm GPUs.
#
# Usage:
#   ./scripts/run_tests.sh --gpu nvidia --test quantize_int4
#   ./scripts/run_tests.sh --gpu rocm --test all
#   ./scripts/run_tests.sh --gpu nvidia --test dequantize
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
MODELS_DIR="${HOME}/models"

# Default values
GPU_TYPE="nvidia"
TEST_TYPE="all"
BUILD_IMAGE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --gpu {nvidia|rocm}    GPU type to use (default: nvidia)
    --test {all|quantize_int4|quantize_int8|dequantize|hf_verify}
                           Test to run (default: all)
    --build                Build Docker image before running
    --models-dir PATH      Path to models directory (default: ~/models)
    --output-dir PATH      Output directory (default: ./outputs)
    -h, --help             Show this help message

Tests:
    quantize_int4    - Quantize Llama3.1-8B and Qwen3-32B to INT4
    quantize_int8    - Quantize Llama3.1-8B to INT8
    dequantize       - Dequantize INT4 model back to BF16
    hf_verify        - Verify HuggingFace Transformers compatibility
    all              - Run all tests

Examples:
    $0 --gpu nvidia --test quantize_int4
    $0 --gpu rocm --test all --build
    $0 --gpu nvidia --test dequantize
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --test)
            TEST_TYPE="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate GPU type
if [[ "$GPU_TYPE" != "nvidia" && "$GPU_TYPE" != "rocm" ]]; then
    echo -e "${RED}Error: GPU type must be 'nvidia' or 'rocm'${NC}"
    exit 1
fi

# Set Docker image and runtime based on GPU type
if [[ "$GPU_TYPE" == "nvidia" ]]; then
    DOCKERFILE="docker/Dockerfile.nvidia.prebuilt"
    IMAGE_NAME="quanto:cuda-test"
    DOCKER_RUNTIME="--gpus all"
else
    DOCKERFILE="docker/Dockerfile.rocm.prebuilt"
    IMAGE_NAME="quanto:rocm-test"
    DOCKER_RUNTIME="--device=/dev/kfd --device=/dev/dri --group-add video"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Docker image if requested
if [[ "$BUILD_IMAGE" == true ]]; then
    echo -e "${YELLOW}Building Docker image: $IMAGE_NAME${NC}"
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${RED}Error: Docker image '$IMAGE_NAME' not found. Use --build to create it.${NC}"
    exit 1
fi

echo -e "${GREEN}=== Quanto Docker Test Runner ===${NC}"
echo "GPU Type: $GPU_TYPE"
echo "Test: $TEST_TYPE"
echo "Models: $MODELS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Image: $IMAGE_NAME"
echo ""

# Docker run command template
run_docker() {
    local script="$1"
    docker run --rm $DOCKER_RUNTIME \
        -v "$PROJECT_DIR:/workspace" \
        -v "$MODELS_DIR:/models:ro" \
        -v "$OUTPUT_DIR:/output" \
        -w /workspace \
        "$IMAGE_NAME" \
        python "$script"
}

# Test functions
test_quantize_int4_llama() {
    echo -e "${YELLOW}[Test] Quantizing Llama3.1-8B-Instruct to INT4...${NC}"
    run_docker "tests/docker/test_quantize_int4_llama.py"
}

test_quantize_int4_qwen() {
    echo -e "${YELLOW}[Test] Quantizing Qwen3-32B to INT4...${NC}"
    run_docker "tests/docker/test_quantize_int4_qwen.py"
}

test_quantize_int8_llama() {
    echo -e "${YELLOW}[Test] Quantizing Llama3.1-8B-Instruct to INT8...${NC}"
    run_docker "tests/docker/test_quantize_int8_llama.py"
}

test_dequantize_llama() {
    echo -e "${YELLOW}[Test] Dequantizing Llama3.1-8B INT4 to BF16...${NC}"
    run_docker "tests/docker/test_dequantize_llama.py"
}

test_hf_verify() {
    echo -e "${YELLOW}[Test] Verifying HuggingFace Transformers compatibility...${NC}"
    run_docker "tests/docker/test_hf_verify.py"
}

# Run tests
case $TEST_TYPE in
    quantize_int4)
        test_quantize_int4_llama
        test_quantize_int4_qwen
        ;;
    quantize_int8)
        test_quantize_int8_llama
        ;;
    dequantize)
        test_dequantize_llama
        ;;
    hf_verify)
        test_hf_verify
        ;;
    all)
        test_quantize_int4_llama
        test_quantize_int4_qwen
        test_quantize_int8_llama
        test_dequantize_llama
        test_hf_verify
        ;;
    *)
        echo -e "${RED}Unknown test: $TEST_TYPE${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== All tests completed ===${NC}"
