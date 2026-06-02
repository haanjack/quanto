"""
Quanto: Runtime patches for AMD Quark compatibility.

Applies targeted monkey-patches to Quark without requiring a fork:
- Registers new model templates (e.g. deepseek_v4) via the public
  LLMTemplate.register_template() API.
- Patches _recover_fp8_weights() to handle DeepSeek-V4's sibling
  ".scale" key convention in addition to Quark's expected "_scale_inv".

Call apply_patches() once before any Quark API is used. It is idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def _register_deepseek_v4_template() -> None:
    from quark.torch import LLMTemplate

    if "deepseek_v4" in LLMTemplate.list_available():
        return

    # DeepSeek-V4 uses CSA+HCA attention (not MLA):
    #   - Query: low-rank wq_a / wq_b projections
    #   - Key-Value: combined wkv (MQA)
    # FFN experts: ffn.experts.N.w1/w2/w3, shared: ffn.shared_experts.*
    # Hyper-connection residuals (hc_*) and router gates (ffn.gate) are excluded.
    template = LLMTemplate(
        model_type="deepseek_v4",
        kv_layers_name=["*attn.wkv"],
        q_layer_name=["*attn.wq_a", "*attn.wq_b"],
        exclude_layers_name=[
            "head",
            "embed",
            "*attn*",       # All attention layers (CSA/HCA are sensitive)
            "*ffn.gate",    # MoE router gates
            "*norm*",       # RMS norms
            "*hc_*",        # Hyper-connection residual parameters
        ],
    )
    LLMTemplate.register_template(template)
    logger.debug("Registered deepseek_v4 LLMTemplate")


def _patch_recover_fp8_weights() -> None:
    """
    Patch quark's _recover_fp8_weights to also handle the DeepSeek-V4
    weight naming convention where the scale tensor is stored as a sibling
    key "X.scale" rather than the suffix "X.weight_scale_inv" that Quark
    expects by default.
    """
    import quark.torch.quantization.file2file_quantization as f2f

    if getattr(f2f._recover_fp8_weights, "_quanto_patched", False):
        return

    _original = f2f._recover_fp8_weights

    def _patched_recover_fp8_weights(
        safetensor_path,
        device,
        weight_map=None,
        scale_inv_cache=None,
    ):
        import os
        import torch
        from safetensors import safe_open

        # Re-import module-level helpers via the module reference so we stay
        # consistent with whatever version of Quark is installed.
        _is_linear = f2f._is_linear_weight_tensor
        _dequant_fp8 = f2f._weight_dequant_fp8
        _empty_cache = f2f._empty_cache_if_cuda

        def _dequant_dsv4(weight: torch.Tensor, scale: torch.Tensor, dev) -> torch.Tensor:
            """
            Dequantize a DeepSeek-V4 weight tensor to bfloat16.

            DeepSeek-V4 uses two formats:
            - FP8 E4M3 weight + FP32 scale (Base model attention):
                standard block-FP8, handled by Quark's Triton kernel.
            - FP8 E4M3 weight + E8M0 scale (Instruct model attention):
                convert E8M0 -> float32 first, then same Triton kernel.
            - INT8 weight + E8M0 scale (Instruct model experts, packed MXFP4):
                each INT8 byte holds two FP4 E2M1 values; use
                upcast_from_mxfp to expand to bfloat16.
            """
            if weight.dtype == torch.float8_e4m3fn:
                # FP8 E4M3 weight — normalise scale to float32 for the kernel
                if scale.dtype != torch.float32:
                    # E8M0: exponent stored as uint8, value = 2^(e - 127)
                    scale = (
                        2.0 ** (scale.view(torch.uint8).to(torch.float32) - 127.0)
                    ).contiguous()
                return _dequant_fp8(weight.contiguous(), scale.contiguous())

            elif weight.dtype == torch.int8:
                # Packed MXFP4: 2 × FP4 E2M1 per int8 byte, E8M0 block scale
                # Axis=-1: 32 FP4 elements per scale block along columns
                try:
                    from quark.torch.kernel.mx.triton import upcast_from_mxfp
                    return upcast_from_mxfp(
                        weight.view(torch.uint8).to(dev),
                        scale.view(torch.uint8).to(dev),
                        dtype=torch.bfloat16,
                        axis=-1,
                        BLOCK_QUANT_DIM=32,
                    ).contiguous()
                except Exception:
                    from quark.torch.kernel.mx.triton import upcast_from_mxfp_torch
                    return upcast_from_mxfp_torch(
                        weight.view(torch.uint8),
                        scale.view(torch.uint8),
                        target_dtype=torch.bfloat16,
                        axis=-1,
                    ).contiguous()

            else:
                raise ValueError(
                    f"Unsupported weight dtype for DeepSeek-V4 dequantization: "
                    f"{weight.dtype}"
                )

        recovered_tensors: dict = {}
        fp8_weight_count = 0
        device_str = str(device)

        with safe_open(safetensor_path, framework="pt", device=device_str) as f:
            all_keys = set(f.keys())

            for weight_name in all_keys:
                # Skip upstream _scale_inv tensors
                if weight_name.endswith("_scale_inv"):
                    continue
                # Skip DeepSeek-V4 sibling .scale tensors
                if (
                    weight_name.endswith(".scale")
                    and weight_name[:-6] + ".weight" in all_keys
                ):
                    continue

                scale_inv_name = f"{weight_name}_scale_inv"
                # DeepSeek-V4 sibling scale: "layers.N.X.weight" -> "layers.N.X.scale"
                dsv4_scale_name = (
                    weight_name[:-7] + ".scale"
                    if weight_name.endswith(".weight")
                    else None
                )

                if _is_linear(weight_name):
                    if scale_inv_name in all_keys:
                        weight = f.get_tensor(weight_name)
                        scale_inv = f.get_tensor(scale_inv_name)
                        recovered_tensors[weight_name] = _dequant(weight, scale_inv)
                        fp8_weight_count += 1
                        del weight, scale_inv
                        _empty_cache(device)
                    elif dsv4_scale_name is not None and dsv4_scale_name in all_keys:
                        weight = f.get_tensor(weight_name)
                        scale = f.get_tensor(dsv4_scale_name)
                        recovered_tensors[weight_name] = _dequant_dsv4(
                            weight, scale, device
                        )
                        fp8_weight_count += 1
                        del weight, scale
                        _empty_cache(device)
                    elif scale_inv_cache is not None and scale_inv_name in scale_inv_cache:
                        weight = f.get_tensor(weight_name)
                        scale_inv = scale_inv_cache[scale_inv_name].to(device)
                        recovered_tensors[weight_name] = _dequant(weight, scale_inv)
                        fp8_weight_count += 1
                        del weight, scale_inv
                        _empty_cache(device)
                    elif weight_map is not None and scale_inv_name not in weight_map:
                        recovered_tensors[weight_name] = f.get_tensor(weight_name)
                    else:
                        raise ValueError(
                            f"FP8 weight '{weight_name}' found in "
                            f"'{os.path.basename(safetensor_path)}' but its "
                            f"scale_inv '{scale_inv_name}' is not in the same "
                            f"file and not found in pre-loaded cache. Please "
                            f"ensure model.safetensors.index.json exists and "
                            f"is correct."
                        )
                else:
                    recovered_tensors[weight_name] = f.get_tensor(weight_name)

        f2f.logger.info(
            f"Dequantized {fp8_weight_count} FP8 weights, "
            f"total tensors: {len(recovered_tensors)}"
        )
        return recovered_tensors

    _patched_recover_fp8_weights._quanto_patched = True
    f2f._recover_fp8_weights = _patched_recover_fp8_weights
    logger.debug("Patched quark _recover_fp8_weights for DeepSeek-V4 .scale key support")


def apply_patches() -> None:
    """
    Apply all Quark compatibility patches. Safe to call multiple times.
    """
    global _patches_applied
    if _patches_applied:
        return

    try:
        _register_deepseek_v4_template()
        _patch_recover_fp8_weights()
        _patches_applied = True
        logger.debug("Quark compatibility patches applied")
    except ImportError:
        # Quark not installed — patches will apply when it becomes available
        pass
