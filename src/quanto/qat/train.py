"""
QAT fine-tuning with HuggingFace Trainer.

Wraps HF Trainer with a Ray Tune reporting callback
for per-epoch metric reporting during hyperparameter search.
"""

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


class RayReportCallback(TrainerCallback):
    """Reports eval_loss to Ray Tune after each evaluation epoch."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        try:
            import ray.tune

            if metrics and "eval_loss" in metrics:
                ray.tune.report(eval_loss=metrics["eval_loss"])
        except ImportError:
            pass


def set_scaling_factor_trainable(model):
    """
    EfficientQAT Stage 2: freeze all parameters except scaling factors.

    Converts FrozenScaledFakeQuantize.scale from buffer to nn.Parameter.
    Only works for INT4 (FrozenScaledFakeQuantize). MXFP4 is skipped.
    """
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    # Enable input gradient flow for embeddings
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Convert scale buffers to trainable parameters
    for name, module in model.named_modules():
        if isinstance(module, FrozenScaledFakeQuantize):
            _convert_scale_to_parameter(model, name, module)

    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after set_scaling_factor_trainable: {trainable_count}")


def _convert_scale_to_parameter(model, module_name: str, module: FrozenScaledFakeQuantize):
    """Convert a FrozenScaledFakeQuantize's scale buffer to a trainable parameter."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    attr_name = parts[-1]
    submodule = getattr(parent, attr_name)

    scale_value = submodule.scale
    delattr(submodule, "scale")
    submodule.scale = nn.Parameter(scale_value, requires_grad=True)


def train_qat(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    per_device_batch_size: int = 2,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.0,
    gradient_checkpointing: bool = True,
    output_dir: str = "./qat_trial",
    only_train_scaling_factor: bool = False,
    precision: str = "int4",
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
        set_scaling_factor_trainable(model)

    callbacks = [RayReportCallback()]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="epoch",
        report_to="none",
        gradient_checkpointing=gradient_checkpointing,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    trainer.train()

    return trainer
