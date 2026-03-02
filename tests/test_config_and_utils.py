"""Tests for configuration loading/validation and utility functions."""

import os
import tempfile

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from contamination_detection.config import (
    DataConfig,
    DetectionConfig,
    ExperimentConfig,
    LoRAConfig,
    SamplingConfig,
    TrainingConfig,
)
from contamination_detection.utils import (
    record_dependency_versions,
    set_global_seed,
    timer,
)


# ── Hydra config loading tests ──────────────────────────────────────────


class TestHydraConfigLoading:
    """Test that Hydra loads configs correctly from YAML files."""

    def test_load_default_config(self):
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
            assert cfg.seed == 42
            assert "EleutherAI/pythia-70m" in cfg.model_sizes
            assert cfg.data.train_ratio == 0.6

    def test_data_defaults(self):
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
            assert cfg.data.contamination_ratio == 0.2
            assert cfg.data.eval_ratio == 0.2
            assert "QASC" in cfg.data.datasets

    def test_sampling_defaults(self):
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
            assert cfg.sampling.n_samples == 20
            assert cfg.sampling.temperature == 1.0

    def test_detection_defaults(self):
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
            assert cfg.detection.alpha == 0.1
            assert cfg.detection.xi == 0.5


# ── Config validation tests ──────────────────────────────────────────────


class TestConfigValidation:
    """Test that config validation catches invalid values."""

    def test_data_config_valid(self):
        cfg = DataConfig()
        cfg.validate()  # should not raise

    def test_data_config_ratios_exceed_one(self):
        cfg = DataConfig(train_ratio=0.5, contamination_ratio=0.4, eval_ratio=0.3)
        with pytest.raises(ValueError, match="exceeds 1.0"):
            cfg.validate()

    def test_data_config_negative_ratio(self):
        cfg = DataConfig(train_ratio=-0.1)
        with pytest.raises(ValueError, match="out of range"):
            cfg.validate()

    def test_lora_config_valid(self):
        cfg = LoRAConfig()
        cfg.validate()

    def test_lora_config_invalid_rank(self):
        cfg = LoRAConfig(r=0)
        with pytest.raises(ValueError, match="rank"):
            cfg.validate()

    def test_lora_config_invalid_dropout(self):
        cfg = LoRAConfig(lora_dropout=1.5)
        with pytest.raises(ValueError, match="lora_dropout"):
            cfg.validate()

    def test_training_config_valid(self):
        cfg = TrainingConfig()
        cfg.validate()

    def test_training_config_invalid_lr(self):
        cfg = TrainingConfig(learning_rate=-1e-4)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_sampling_config_valid(self):
        cfg = SamplingConfig()
        cfg.validate()

    def test_sampling_config_invalid_n_samples(self):
        cfg = SamplingConfig(n_samples=0)
        with pytest.raises(ValueError, match="n_samples"):
            cfg.validate()

    def test_sampling_config_invalid_temperature(self):
        cfg = SamplingConfig(temperature=-0.5)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_detection_config_valid(self):
        cfg = DetectionConfig()
        cfg.validate()

    def test_detection_config_invalid_alpha(self):
        cfg = DetectionConfig(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_detection_config_invalid_xi(self):
        cfg = DetectionConfig(xi=1.5)
        with pytest.raises(ValueError, match="xi"):
            cfg.validate()

    def test_experiment_config_validates_all(self):
        cfg = ExperimentConfig()
        cfg.validate()  # should not raise

    def test_experiment_config_propagates_error(self):
        cfg = ExperimentConfig(data=DataConfig(train_ratio=0.9, contamination_ratio=0.5, eval_ratio=0.5))
        with pytest.raises(ValueError):
            cfg.validate()


# ── Seed setting tests ───────────────────────────────────────────────────


class TestSeedSetting:
    """Test that global seed setting produces deterministic behavior."""

    def test_numpy_determinism(self):
        set_global_seed(123)
        a = np.random.rand(10)
        set_global_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_determinism(self):
        set_global_seed(456)
        a = torch.randn(10)
        set_global_seed(456)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_python_random_determinism(self):
        import random

        set_global_seed(789)
        a = [random.random() for _ in range(10)]
        set_global_seed(789)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_different_seeds_differ(self):
        set_global_seed(1)
        a = np.random.rand(10)
        set_global_seed(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


# ── Version recording tests ─────────────────────────────────────────────


class TestVersionRecording:
    """Test that dependency version recording captures key packages."""

    def test_records_key_packages(self):
        versions = record_dependency_versions()
        assert "torch" in versions
        assert "transformers" in versions
        assert "numpy" in versions
        assert "peft" in versions

    def test_versions_are_strings(self):
        versions = record_dependency_versions()
        for pkg, ver in versions.items():
            assert isinstance(ver, str)

    def test_torch_version_not_unknown(self):
        versions = record_dependency_versions()
        assert versions["torch"] != "unknown"
        assert versions["torch"] != "not installed"


# ── Timer tests ──────────────────────────────────────────────────────────


class TestTimer:
    """Test the timing context manager."""

    def test_timer_runs(self, capsys):
        with timer("test op"):
            pass
        captured = capsys.readouterr()
        assert "test op completed in" in captured.out
