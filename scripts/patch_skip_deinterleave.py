"""
Patch vLLM's quark_moe.py to skip the de-interleave step in
convert_gpt_oss_weight_to_mxfp4_moe_kernel_format for AITER backend.

The de-interleave assumes GPT-OSS interleaved layout [gate0,up0,gate1,up1,...]
but Qwen3.5 MoE loads weights in contiguous [gate_all|up_all] format via
_load_w13 w1/w3 sharding. Skipping de-interleave preserves correct layout.

We patch the oracle/mxfp4.py file directly.
"""
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src) as f:
    lines = f.readlines()

# Find and comment out the de-interleave block in AITER_MXFP4_BF16 section
# of convert_gpt_oss_weight_to_mxfp4_moe_kernel_format
#
# Pattern to find (AITER section):
#   w13_weight.view(torch.uint8).copy_(
#       w13_weight.data.view(torch.uint8)
#       .view(e, n // 2, 2, k)
#       .permute(0, 2, 1, 3)
#       ...
#   )
#   w13_weight_scale.data = (
#       w13_weight_scale.data.view(e, n // 2, 2, -1)
#       .permute(0, 2, 1, 3)
#       ...
#   )

in_gpt_oss_func = False
in_aiter_block = False
skip_deinterleave = False
paren_depth = 0
modified_lines = 0

output_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Track which function we're in
    if "def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format" in line:
        in_gpt_oss_func = True
    elif in_gpt_oss_func and line.strip().startswith("def "):
        in_gpt_oss_func = False

    # Find AITER_MXFP4_BF16 block within gpt_oss function
    if in_gpt_oss_func and "Mxfp4MoeBackend.AITER_MXFP4_BF16" in line:
        in_aiter_block = True

    # Find de-interleave: w13_weight.view(torch.uint8).copy_(
    if (in_aiter_block and not skip_deinterleave
            and "w13_weight.view(torch.uint8).copy_(" in line):
        # Comment out this statement and the next one (w13_weight_scale.data = ...)
        skip_deinterleave = True
        # Find the end of w13_weight.view(...).copy_(...) block
        block_lines = []
        paren_depth = line.count("(") - line.count(")")
        block_lines.append(line)
        j = i + 1
        while paren_depth > 0 and j < len(lines):
            paren_depth += lines[j].count("(") - lines[j].count(")")
            block_lines.append(lines[j])
            j += 1

        # Also grab the w13_weight_scale.data block right after
        while j < len(lines) and lines[j].strip() == "":
            block_lines.append(lines[j])
            j += 1
        if j < len(lines) and "w13_weight_scale.data" in lines[j]:
            paren_depth = lines[j].count("(") - lines[j].count(")")
            block_lines.append(lines[j])
            k = j + 1
            while paren_depth > 0 and k < len(lines):
                paren_depth += lines[k].count("(") - lines[k].count(")")
                block_lines.append(lines[k])
                k += 1
            j = k

        # Comment out all the block lines
        output_lines.append(
            "        # [QUANTO PATCH] Skip de-interleave: weights are already\n"
        )
        output_lines.append(
            "        # in contiguous [gate_all|up_all] format from _load_w13.\n"
        )
        for bl in block_lines:
            output_lines.append("        # " + bl.lstrip())
        modified_lines += len(block_lines)
        i = j
        continue

    # Also skip bias de-interleave if present
    if (in_aiter_block and skip_deinterleave
            and "w13_bias" in line and ".view(-1, n // 2, 2)" in line):
        block_lines = []
        paren_depth = line.count("(") - line.count(")")
        block_lines.append(line)
        j = i + 1
        while paren_depth > 0 and j < len(lines):
            paren_depth += lines[j].count("(") - lines[j].count(")")
            block_lines.append(lines[j])
            j += 1
        output_lines.append(
            "        # [QUANTO PATCH] Skip bias de-interleave\n"
        )
        for bl in block_lines:
            output_lines.append("        # " + bl.lstrip())
        modified_lines += len(block_lines)
        i = j
        continue

    output_lines.append(line)
    i += 1

with open(dst, "w") as f:
    f.writelines(output_lines)

print(f"Modified {modified_lines} lines (de-interleave commented out)")
print(f"Output: {dst}")
