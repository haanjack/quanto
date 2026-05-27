"""QAT fine-tuning with HuggingFace Trainer and metric bridging callbacks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch.quantization.tensor_quantize import FrozenScaledFakeQuantize

logger = logging.getLogger(__name__)


def precompute_quantized_weights(model):
    """Free BF16 weights by pre-computing quantized integers.

    After freeze, each QuantLinear still holds the original BF16 weight (~16GB).
    This function:
    1. Computes quantized_int = round(weight / scale).clamp(quant_min, quant_max)
    2. Stores quantized_int as an int8 buffer (halves memory)
    3. Patches the quantizer forward to dequant from int8 * scale
    4. Replaces the BF16 weight with a tiny dummy, freeing ~8GB GPU memory

    The STE gradient to scale flows naturally through autograd's broadcasting.
    """
    from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

    freed_bytes = 0
    patched = 0

    for _name, mod in model.named_modules():
        if not isinstance(mod, QuantLinear):
            continue
        if not hasattr(mod, "_weight_quantizer") or mod._weight_quantizer is None:
            continue

        wq = mod._weight_quantizer
        if not isinstance(wq, FrozenScaledFakeQuantize):
            continue

        weight = mod.weight.data  # BF16 [out, in]
        scale = wq.scale.data  # [out, num_groups]
        group_size = wq.group_size
        quant_min = wq.quant_min
        quant_max = wq.quant_max
        # Compute quantized integers: round(W / S).clamp(min, max)
        with torch.no_grad():
            if group_size and group_size > 0 and weight.ndim == 2:
                num_groups = weight.shape[1] // group_size
                w_reshaped = weight.reshape(weight.shape[0], num_groups, group_size)
                s_expanded = scale.reshape(weight.shape[0], num_groups, 1)
                q_int = torch.round(w_reshaped / s_expanded).clamp(quant_min, quant_max)
                q_int = q_int.reshape(weight.shape)
            else:
                q_int = torch.round(weight / scale).clamp(quant_min, quant_max)

        # Store as int8 buffer
        q_int = q_int.to(torch.int8)
        freed_bytes += weight.numel() * (weight.element_size() - q_int.element_size())

        wq.register_buffer("quantized_int", q_int)
        wq.group_size_stored = group_size
        wq.orig_out_features = weight.shape[0]
        wq.orig_in_features = weight.shape[1]

        # Patch forward: dequant from int8 * scale (standard autograd handles grad)
        wq.forward = lambda X, _self=wq: _dequant_forward(_self, X)
        wq.frozen_params = False

        # Make scale trainable
        scale_val = wq.scale
        delattr(wq, "scale")
        wq.scale = nn.Parameter(scale_val, requires_grad=True)

        # Replace BF16 weight with tiny dummy to free memory
        freed_bytes += weight.numel() * weight.element_size()
        mod.weight = nn.Parameter(
            torch.empty(1, 1, device=weight.device, dtype=weight.dtype),
            requires_grad=False,
        )
        patched += 1

    torch.cuda.empty_cache()
    freed_gb = freed_bytes / 1e9
    logger.info(f"Pre-computed quantized weights for {patched} layers, freed {freed_gb:.2f}GB")
    return freed_gb


def _dequant_forward(self, X):
    """Dequantize from pre-computed int8 * scale. Gradient flows to scale via autograd."""
    qi = self.quantized_int.to(self.scale.dtype)
    if self.group_size_stored and self.group_size_stored > 0 and qi.ndim == 2:
        num_groups = qi.shape[1] // self.group_size_stored
        qi = qi.reshape(qi.shape[0], num_groups, self.group_size_stored)
        scale_exp = self.scale.reshape(qi.shape[0], num_groups, 1)
        result = qi * scale_exp
        return result.reshape(self.orig_out_features, self.orig_in_features)
    return qi * self.scale


class MetricBridgeCallback(TrainerCallback):
    """Bridges HF Trainer eval metrics to the generic MetricCallback interface."""

    def __init__(self, metric_callback):
        self.metric_callback = metric_callback

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            filtered = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.metric_callback.report(filtered, step=state.global_step)


class QATSaveCallback(TrainerCallback):
    """
    Callback to handle model saving for QAT models.

    Disables Trainer's default checkpointing (which calls state_dict/load,
    breaking on FrozenScaledFakeQuantize's resize_) and instead saves
    only the scale parameters manually.
    """

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Save only scale parameters
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

        scale_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                scale_state[name] = param.data.cpu()

        save_path = os.path.join(checkpoint_dir, "scales.pt")
        torch.save(scale_state, save_path)
        logger.info(f"Saved {len(scale_state)} trainable parameters to {save_path}")


def train_qat(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.0,
    gradient_checkpointing: bool = True,
    output_dir: str = "./qat_trial",
    only_train_scaling_factor: bool = False,
    precision: str = "int4",
    metric_callback=None,
):
    """
    Run QAT fine-tuning with HuggingFace Trainer.

    Args:
        model: Model with fake quantization modules.
        tokenizer: HuggingFace tokenizer.
        train_dataset: Training dataset (MixedDataset or similar).
        eval_dataset: Evaluation dataset.
        learning_rate: Learning rate.
        num_epochs: Number of training epochs.
        per_device_batch_size: Batch size per device.
        weight_decay: Weight decay.
        warmup_ratio: Warmup ratio.
        gradient_checkpointing: Enable gradient checkpointing.
        output_dir: Directory for checkpoints.
        only_train_scaling_factor: Only train quantization scales (EfficientQAT Stage 2).
        precision: "int4" or "mxfp4".

    Returns:
        Trainer instance after training.
    """
    if only_train_scaling_factor and precision == "int4":
        print(f"[QAT] Before precompute: {torch.cuda.memory_allocated() / 1e9:.2f}GB", flush=True)
        freed = precompute_quantized_weights(model)
        print(
            f"[QAT] After precompute: {torch.cuda.memory_allocated() / 1e9:.2f}GB (freed={freed:.2f}GB)",
            flush=True,
        )
        # Freeze all params except quantization scales created by precompute
        for name, p in model.named_parameters():
            if "scale" not in name:
                p.requires_grad_(False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[QAT] {trainable} trainable params", flush=True)

    callbacks = []
    if metric_callback is not None:
        callbacks.append(MetricBridgeCallback(metric_callback))

    # Use save_strategy="no" to avoid the FrozenScaledFakeQuantize._load_from_state_dict
    # crash when scale is an nn.Parameter (cannot resize variables that require_grad).
    # Instead, we use QATSaveCallback to save only the trainable scale parameters.
    callbacks.append(QATSaveCallback())

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        logging_strategy="epoch",
        report_to="none",
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    trainer.train()

    return trainer
