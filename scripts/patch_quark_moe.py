"""
Patch vLLM's quark_moe.py to use convert_weight_to_mxfp4_moe_kernel_format
instead of convert_gpt_oss_weight_to_mxfp4_moe_kernel_format.

The GPT-OSS version assumes interleaved gate/up rows which doesn't match
HuggingFace's contiguous block format used by Qwen3.5 MoE.
"""
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src) as f:
    content = f.read()

# 1. Add import for convert_weight_to_mxfp4_moe_kernel_format
content = content.replace(
    "    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,\n",
    "    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,\n"
    "    convert_weight_to_mxfp4_moe_kernel_format,\n",
)

# 2. Replace the function call in _setup_kernel
content = content.replace(
    "            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(\n",
    "            convert_weight_to_mxfp4_moe_kernel_format(\n",
)

with open(dst, "w") as f:
    f.write(content)

# Verify
with open(dst) as f:
    lines = f.readlines()

found_import = any("convert_weight_to_mxfp4_moe_kernel_format," in l for l in lines)
found_call = any("convert_weight_to_mxfp4_moe_kernel_format(" in l for l in lines)
print(f"Import added: {found_import}")
print(f"Call replaced: {found_call}")
print(f"Total lines: {len(lines)}")
