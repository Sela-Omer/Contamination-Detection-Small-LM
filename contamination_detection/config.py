"""Structured configuration dataclasses with validation for the contamination detection pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    datasets: List[str] = field(default_factory=lambda: ["QASC", "StrategyQA"])
    train_ratio: float = 0.6
    contamination_ratio: float = 0.2
    eval_ratio: float = 0.2
    seed: int = 42
    cache_dir: str = "outputs/data"

    def validate(self) -> None:
        total = self.train_ratio + self.contamination_ratio + self.eval_ratio
        if total > 1.0:
            raise ValueError(
                f"Split ratios sum to {total:.2f}, which exceeds 1.0. "
                f"train={self.train_ratio}, contamination={self.contamination_ratio}, "
                f"eval={self.eval_ratio}. Each ratio must be in [0, 1] and their sum <= 1.0."
            )
        for name, val in [
            ("train_ratio", self.train_ratio),
            ("contamination_ratio", self.contamination_ratio),
            ("eval_ratio", self.eval_ratio),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name}={val} is out of range [0, 1].")


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])


    def validate(self) -> None:
        if self.r < 1:
            raise ValueError(f"LoRA rank r={self.r} must be >= 1.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha={self.lora_alpha} must be >= 1.")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout={self.lora_dropout} must be in [0, 1).")


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    seed: int = 42
    logging_steps: int = 10
    output_dir: str = "outputs/training"
    contamination_epochs: List[int] = field(default_factory=lambda: [0, 1, 5, 10])

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate={self.learning_rate} must be > 0.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size={self.batch_size} must be >= 1.")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs={self.num_epochs} must be >= 1.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio={self.warmup_ratio} must be in [0, 1).")


@dataclass
class SamplingConfig:
    """Configuration for output sampling in CDD."""
    n_samples: int = 50
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 100
    seed: int = 42

    def validate(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"n_samples={self.n_samples} must be >= 1.")
        if self.temperature <= 0:
            raise ValueError(f"temperature={self.temperature} must be > 0.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens={self.max_new_tokens} must be >= 1.")


@dataclass
class DetectionConfig:
    """Configuration for CDD detection thresholds."""
    alpha: float = 0.05
    xi: float = 0.01
    alpha_range: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25])

    def validate(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha={self.alpha} must be in (0, 1].")
        if not 0.0 <= self.xi <= 1.0:
            raise ValueError(f"xi={self.xi} must be in [0, 1].")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    model_sizes: List[str] = field(
        default_factory=lambda: [
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
        ]
    )
    output_dir: str = "outputs"
    experiment_name: str = "cdd_contamination_detection"
    seed: int = 42

    def validate(self) -> None:
        """Validate all sub-configs."""
        self.data.validate()
        self.lora.validate()
        self.training.validate()
        self.sampling.validate()
        self.detection.validate()
