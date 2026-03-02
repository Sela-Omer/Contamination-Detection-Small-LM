"""LoRA fine-tuning trainer with sanity checks for the contamination detection pipeline."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from contamination_detection.config import LoRAConfig, TrainingConfig
from contamination_detection.training.model_loader import save_checkpoint

logger = logging.getLogger("contamination_detection.training.trainer")


class CausalLMDataset(TorchDataset):
    """Simple dataset that tokenizes text prompts for causal language modelling."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        # For causal LM, labels == input_ids; the model shifts internally.
        # Mask padding tokens in labels with -100 so they don't contribute to loss.
        labels = item["input_ids"].clone()
        labels[labels == self.encodings["input_ids"][0][-1]] = -100  # rough; see below
        # More precise: mask wherever attention_mask is 0
        labels[item["attention_mask"] == 0] = -100
        item["labels"] = labels
        return item


@dataclass
class TrainingResult:
    """Lightweight container for training outcomes."""

    output_dir: str
    final_loss: float
    initial_loss: float
    loss_history: List[float]
    sanity_passed: bool


def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_texts: Sequence[str],
    training_config: TrainingConfig,
    lora_config: LoRAConfig | None = None,
    output_dir: str | None = None,
    max_length: int = 256,
    eval_texts: Sequence[str] | None = None,
) -> TrainingResult:
    """Fine-tune a (PEFT-wrapped) model on the given texts.

    Args:
        model: A PEFT model returned by ``load_pythia_with_lora``.
        tokenizer: Matching tokenizer.
        train_texts: Training prompts (already formatted).
        training_config: Hyperparameters for the Trainer.
        lora_config: Optional LoRA config to save alongside checkpoint.
        output_dir: Where to write the final checkpoint. Falls back to
            ``training_config.output_dir``.
        max_length: Max token length for tokenisation.
        eval_texts: Optional evaluation texts for validation loss tracking.

    Returns:
        A ``TrainingResult`` with loss history and sanity-check outcome.
    """
    out = output_dir or training_config.output_dir
    os.makedirs(out, exist_ok=True)

    train_dataset = CausalLMDataset(train_texts, tokenizer, max_length=max_length)

    eval_dataset = None
    if eval_texts is not None and len(eval_texts) > 0:
        eval_dataset = CausalLMDataset(eval_texts, tokenizer, max_length=max_length)

    # Auto-detect GPU and precision
    use_cuda = torch.cuda.is_available()
    # LoRA models work fine with fp16; full fine-tune needs bf16 on A100s
    # (fp16 GradScaler fails with "Attempting to unscale FP16 gradients")
    is_peft = hasattr(model, "peft_config")
    if use_cuda:
        if is_peft:
            use_fp16, use_bf16 = True, False
        else:
            use_fp16, use_bf16 = False, True
    else:
        use_fp16, use_bf16 = False, False

    training_args = TrainingArguments(
        output_dir=out,
        overwrite_output_dir=True,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_strategy="no",  # we save manually at the end
        seed=training_config.seed,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",  # disable wandb / tensorboard from Trainer
        no_cuda=not use_cuda,
        use_mps_device=False,
        dataloader_pin_memory=use_cuda,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info(
        f"Starting fine-tuning: {len(train_texts)} examples, "
        f"epochs={training_config.num_epochs}, lr={training_config.learning_rate}, "
        f"batch={training_config.batch_size}, grad_accum={training_config.gradient_accumulation_steps}"
    )

    train_result = trainer.train()

    # Extract loss history from trainer log history
    loss_history: List[float] = [
        entry["loss"]
        for entry in trainer.state.log_history
        if "loss" in entry
    ]

    initial_loss = loss_history[0] if loss_history else float("inf")
    final_loss = loss_history[-1] if loss_history else float("inf")

    logger.info(f"Training complete. Initial loss={initial_loss:.4f}, final loss={final_loss:.4f}")

    # Sanity check
    sanity_passed = check_training_sanity(initial_loss, final_loss)

    # Save final checkpoint
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=out,
        training_config=training_config,
        lora_config=lora_config,
        extra_metadata={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "sanity_passed": sanity_passed,
        },
    )

    return TrainingResult(
        output_dir=out,
        final_loss=final_loss,
        initial_loss=initial_loss,
        loss_history=loss_history,
        sanity_passed=sanity_passed,
    )


def check_training_sanity(initial_loss: float, final_loss: float) -> bool:
    """Verify that training loss decreased.

    Args:
        initial_loss: Loss at the first logging step.
        final_loss: Loss at the last logging step.

    Returns:
        True if final_loss < initial_loss, False otherwise.
    """
    if final_loss >= initial_loss:
        logger.warning(
            f"Training sanity check FAILED: final loss ({final_loss:.4f}) "
            f"did not decrease below initial loss ({initial_loss:.4f}). "
            "This may indicate a training configuration issue."
        )
        return False

    logger.info(
        f"Training sanity check passed: loss decreased from "
        f"{initial_loss:.4f} → {final_loss:.4f}"
    )
    return True
