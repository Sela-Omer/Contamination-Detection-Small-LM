"""Unit tests for the fine-tuning engine (model loader, trainer, checkpoint management)."""

import json
import os
import tempfile

import pytest
import torch

from contamination_detection.config import LoRAConfig, TrainingConfig
from contamination_detection.training.model_loader import (
    load_checkpoint,
    load_pythia_with_lora,
    save_checkpoint,
)
from contamination_detection.training.trainer import (
    CausalLMDataset,
    check_training_sanity,
    fine_tune,
)

# Use Pythia-70M for all tests (smallest model, ~150 MB)
MODEL_NAME = "EleutherAI/pythia-70m"
DEFAULT_LORA = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query_key_value"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load Pythia-70M with LoRA once for the entire test module."""
    model, tokenizer = load_pythia_with_lora(MODEL_NAME, DEFAULT_LORA)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 4.1  Model loading & LoRA attachment
# ---------------------------------------------------------------------------

class TestModelLoader:
    """Tests for Pythia model loading and LoRA adapter attachment."""

    def test_load_pythia_70m(self, model_and_tokenizer):
        """Test that Pythia-70M loads successfully with LoRA."""
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_lora_reduces_trainable_params(self, model_and_tokenizer):
        """LoRA should make only a small fraction of params trainable."""
        model, _ = model_and_tokenizer
        trainable, total = model.get_nb_trainable_parameters()
        assert trainable > 0, "No trainable parameters found"
        assert trainable < total, "LoRA should freeze most parameters"
        # Typically < 1-2 % for rank-8 LoRA
        ratio = trainable / total
        assert ratio < 0.05, f"Trainable ratio {ratio:.4f} unexpectedly high"

    def test_tokenizer_has_pad_token(self, model_and_tokenizer):
        """Pythia tokenizer should have pad_token set to eos_token."""
        _, tokenizer = model_and_tokenizer
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token == tokenizer.eos_token

    def test_short_name_resolution(self):
        """Short names like '70m' should resolve to full HF identifiers."""
        model, tokenizer = load_pythia_with_lora("70m", DEFAULT_LORA)
        assert model is not None
        assert tokenizer is not None


# ---------------------------------------------------------------------------
# 4.4  Checkpoint save / load round-trip
# ---------------------------------------------------------------------------

class TestCheckpointManagement:
    """Tests for checkpoint save and load."""

    def test_save_load_roundtrip(self, model_and_tokenizer):
        """Saving then loading a checkpoint should restore LoRA weights."""
        model, tokenizer = model_and_tokenizer
        training_cfg = TrainingConfig(seed=123, learning_rate=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=tmpdir,
                training_config=training_cfg,
                lora_config=DEFAULT_LORA,
                extra_metadata={"test_key": "test_value"},
            )

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "adapter_config.json"))
            assert os.path.exists(os.path.join(tmpdir, "training_metadata.json"))

            # Load checkpoint
            loaded_model, loaded_tokenizer, metadata = load_checkpoint(tmpdir)

            assert loaded_model is not None
            assert loaded_tokenizer is not None
            assert loaded_tokenizer.pad_token is not None

            # Metadata round-trip
            assert metadata["training"]["seed"] == 123
            assert metadata["training"]["learning_rate"] == 1e-4
            assert metadata["lora"]["r"] == 8
            assert metadata["test_key"] == "test_value"

    def test_metadata_json_written(self, model_and_tokenizer):
        """Training metadata should be persisted as JSON."""
        model, tokenizer = model_and_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=tmpdir,
                training_config=TrainingConfig(),
                lora_config=DEFAULT_LORA,
            )
            meta_path = os.path.join(tmpdir, "training_metadata.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                data = json.load(f)
            assert "training" in data
            assert "lora" in data


# ---------------------------------------------------------------------------
# 4.3  Training sanity check
# ---------------------------------------------------------------------------

class TestSanityCheck:
    """Tests for the training sanity check function."""

    def test_loss_decreased_passes(self):
        assert check_training_sanity(initial_loss=5.0, final_loss=3.0) is True

    def test_loss_not_decreased_fails(self):
        assert check_training_sanity(initial_loss=3.0, final_loss=5.0) is False

    def test_equal_loss_fails(self):
        assert check_training_sanity(initial_loss=3.0, final_loss=3.0) is False


# ---------------------------------------------------------------------------
# 4.2 + 4.3  Training a few steps on tiny data
# ---------------------------------------------------------------------------

class TestFineTuning:
    """Test actual fine-tuning on a tiny dataset."""

    def test_few_steps_loss_decreases(self):
        """Fine-tune Pythia-70M for 2 steps on 5 examples; loss should decrease."""
        lora_cfg = LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["query_key_value"])
        model, tokenizer = load_pythia_with_lora(MODEL_NAME, lora_cfg)

        # Tiny training data — repeat a simple pattern so the model can memorise
        train_texts = [
            "Question: What is 1+1?\nAnswer: 2",
            "Question: What is 2+2?\nAnswer: 4",
            "Question: What is 3+3?\nAnswer: 6",
            "Question: What is 4+4?\nAnswer: 8",
            "Question: What is 5+5?\nAnswer: 10",
        ]

        training_cfg = TrainingConfig(
            learning_rate=5e-4,
            batch_size=2,
            gradient_accumulation_steps=1,
            num_epochs=3,
            warmup_ratio=0.0,
            seed=42,
            logging_steps=1,
            output_dir="",  # will be overridden
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fine_tune(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                training_config=training_cfg,
                lora_config=lora_cfg,
                output_dir=tmpdir,
                max_length=64,
            )

            assert len(result.loss_history) > 0, "No loss entries recorded"
            assert result.final_loss < result.initial_loss, (
                f"Loss did not decrease: {result.initial_loss:.4f} → {result.final_loss:.4f}"
            )
            assert result.sanity_passed is True

            # Checkpoint files should exist
            assert os.path.exists(os.path.join(tmpdir, "adapter_config.json"))
            assert os.path.exists(os.path.join(tmpdir, "training_metadata.json"))


# ---------------------------------------------------------------------------
# CausalLMDataset
# ---------------------------------------------------------------------------

class TestCausalLMDataset:
    """Tests for the tokenisation dataset wrapper."""

    def test_dataset_length(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        texts = ["Hello world", "Foo bar baz"]
        ds = CausalLMDataset(texts, tokenizer, max_length=32)
        assert len(ds) == 2

    def test_dataset_item_keys(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        ds = CausalLMDataset(["test"], tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_labels_mask_padding(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        ds = CausalLMDataset(["hi"], tokenizer, max_length=32)
        item = ds[0]
        # Where attention_mask is 0, labels should be -100
        pad_positions = (item["attention_mask"] == 0)
        if pad_positions.any():
            assert (item["labels"][pad_positions] == -100).all()
