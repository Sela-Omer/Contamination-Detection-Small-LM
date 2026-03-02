"""Pythia model loader with LoRA adapter attachment and checkpoint management."""

import json
import logging
import os
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from contamination_detection.config import LoRAConfig, TrainingConfig

logger = logging.getLogger("contamination_detection.training.model_loader")

# Canonical Pythia model names on HuggingFace
PYTHIA_MODELS = {
    "70m": "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1b": "EleutherAI/pythia-1b",
}


def load_pythia_with_lora(
    model_name: str,
    lora_config: LoRAConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a Pythia model from HuggingFace and attach LoRA adapters.

    Args:
        model_name: Either a short name (e.g. "70m") or full HF name
            (e.g. "EleutherAI/pythia-70m").
        lora_config: LoRA hyperparameters from the project config.

    Returns:
        A tuple of (peft_model, tokenizer).

    Raises:
        RuntimeError: If the model cannot be loaded (e.g. OOM).
    """
    # Resolve short names like "70m" → full HF identifier
    hf_name = PYTHIA_MODELS.get(model_name.lower(), model_name)
    logger.info(f"Loading base model: {hf_name}")

    # Pick dtype and device automatically: use GPU + float16 when available
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        logger.info(f"CUDA available — loading {hf_name} in float16 with device_map='auto'")
    else:
        dtype = torch.float32
        device_map = "cpu"
        logger.info(f"No CUDA — loading {hf_name} in float32 on CPU")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"OOM loading {hf_name}. "
            f"Available memory: {torch.cuda.mem_get_info() if torch.cuda.is_available() else 'N/A (CPU only)'}. "
            "Try a smaller model or reduce batch size."
        )
        raise
    except Exception as exc:
        logger.error(f"Failed to load model {hf_name}: {exc}")
        raise

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    # Pythia tokenizer has no pad token by default; use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    # Attach LoRA adapters
    peft_lora_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, peft_lora_config)

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA attached to {hf_name}: "
        f"trainable={trainable:,} / total={total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return peft_model, tokenizer

def load_pythia_full(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a Pythia model for full fine-tuning (no LoRA, all params trainable).

    Args:
        model_name: Either a short name (e.g. "70m") or full HF name.

    Returns:
        A tuple of (model, tokenizer).
    """
    hf_name = PYTHIA_MODELS.get(model_name.lower(), model_name)
    logger.info(f"Loading base model for full fine-tuning: {hf_name}")

    # Full fine-tune MUST use float32 master weights.
    # bf16 mixed precision in the Trainer handles the forward/backward casting.
    # Loading in float16 causes grad_norm=nan and loss=0.
    if torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float32, device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Full fine-tune {hf_name}: trainable={trainable:,} / total={total:,} (100%)")

    return base_model, tokenizer




def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    training_config: TrainingConfig | None = None,
    lora_config: LoRAConfig | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """Save LoRA adapter weights, tokenizer, and training metadata.

    Args:
        model: The PEFT model (only LoRA weights are saved).
        tokenizer: The tokenizer to save alongside.
        output_dir: Directory to write checkpoint files into.
        training_config: Optional training hyperparameters to record.
        lora_config: Optional LoRA hyperparameters to record.
        extra_metadata: Any additional key-value pairs to persist.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights — LoRA adapter or full model
    is_peft = hasattr(model, "save_pretrained") and hasattr(model, "peft_config")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Record hyperparameters alongside the checkpoint
    metadata: dict = {"is_peft": is_peft}
    if training_config is not None:
        metadata["training"] = {
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "num_epochs": training_config.num_epochs,
            "warmup_ratio": training_config.warmup_ratio,
            "seed": training_config.seed,
            "logging_steps": training_config.logging_steps,
        }
    if lora_config is not None:
        metadata["lora"] = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": list(lora_config.target_modules),
        }
    if extra_metadata:
        metadata.update(extra_metadata)

    if metadata:
        meta_path = os.path.join(output_dir, "training_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Checkpoint saved to {output_dir} (metadata: {list(metadata.keys())})")
    else:
        logger.info(f"Checkpoint saved to {output_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    base_model_name: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, dict]:
    """Load a LoRA checkpoint from disk.

    Args:
        checkpoint_dir: Path to the saved checkpoint directory.
        base_model_name: HF model name for the base model. If None, reads
            from the adapter config stored in the checkpoint.

    Returns:
        A tuple of (peft_model, tokenizer, metadata_dict).
    """
    # Load metadata if present
    meta_path = os.path.join(checkpoint_dir, "training_metadata.json")
    metadata: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    # Determine base model name from adapter config if not provided
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)

    if base_model_name is None:
        if is_peft:
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", None)
        else:
            # Full model checkpoint — read from config.json
            config_path = os.path.join(checkpoint_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                base_model_name = cfg.get("_name_or_path", None)
        if base_model_name is None:
            raise ValueError(
                "Cannot determine base model name. "
                "Provide base_model_name or ensure checkpoint has adapter_config.json or config.json."
            )

    # Resolve short names
    hf_name = PYTHIA_MODELS.get(base_model_name.lower(), base_model_name)

    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        logger.info(f"Loading LoRA checkpoint from {checkpoint_dir} (base: {hf_name})")
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map=device_map,
        )
        base_model.config.pad_token_id = tokenizer.eos_token_id
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        logger.info(f"LoRA checkpoint restored from {checkpoint_dir}")
        return peft_model, tokenizer, metadata
    else:
        logger.info(f"Loading full model checkpoint from {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=dtype, device_map=device_map,
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Full checkpoint restored from {checkpoint_dir}")
        return model, tokenizer, metadata
