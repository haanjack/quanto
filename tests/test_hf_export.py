#!/usr/bin/env python3
"""
Test exported HuggingFace model with perplexity evaluation.
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from quark.contrib.llm_eval import ppl_eval
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def test_exported_model(model_path: str, device: str = "cuda:0"):
    """Test the exported HuggingFace model with PPL evaluation."""
    print(f"Testing exported model: {model_path}")
    print(f"Device: {device}")

    model_path = Path(model_path)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    # Load config
    print("Loading config...")
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    # Check if model has quantization config
    quant_config = getattr(config, "quantization_config", None)
    if quant_config:
        print(f"Quantization config found: {quant_config.get('quant_method', 'unknown')}")

    # Load model - with quantization support
    print("Loading model...")
    start_time = time.time()

    try:
        # Try loading with quantization
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        print(f"Model loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Print model info
    print(f"Model type: {type(model).__name__}")
    print(f"Model device: {model.device}")

    # Load wikitext-2 test data
    print("\nLoading wikitext-2 test data...")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Evaluate perplexity
    print("Evaluating perplexity...")
    start_time = time.time()
    ppl = ppl_eval(model, testenc, str(model.device))
    eval_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Perplexity: {ppl.item():.4f}")
    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"{'=' * 60}")

    return ppl.item()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test exported HF model with PPL")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/hanjack/workspace/amd/quantization/output/qwen3-32b-int4-hf",
        help="Path to exported model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )

    args = parser.parse_args()

    test_exported_model(args.model_path, args.device)


if __name__ == "__main__":
    main()
