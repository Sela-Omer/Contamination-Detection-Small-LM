#!/usr/bin/env python
"""Main entry point for running contamination detection experiments.

Uses Hydra for configuration management. Supports running the full pipeline
or individual stages via command-line overrides.

Usage:
    # Full pipeline
    python scripts/run_experiment.py

    # Single stage
    python scripts/run_experiment.py stage=data_preparation

    # Override config values
    python scripts/run_experiment.py training.num_epochs=5 sampling.n_samples=10

    # Specific model sizes
    python scripts/run_experiment.py model_sizes='[EleutherAI/pythia-70m]'

Requirements: 22.1, 23.1, 23.4
"""

import logging
import sys
import os

# Add project root to path so imports work when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig, OmegaConf

from contamination_detection.config import (
    DataConfig,
    DetectionConfig,
    ExperimentConfig,
    LoRAConfig,
    SamplingConfig,
    TrainingConfig,
)
from contamination_detection.orchestrator import ExperimentOrchestrator, PIPELINE_STAGES

logger = logging.getLogger("contamination_detection.scripts.run_experiment")


def _build_config(cfg: DictConfig) -> ExperimentConfig:
    """Build an ExperimentConfig from a Hydra DictConfig."""
    data = DataConfig(
        datasets=list(cfg.get("data", {}).get("datasets", ["QASC", "StrategyQA"])),
        train_ratio=cfg.get("data", {}).get("train_ratio", 0.6),
        contamination_ratio=cfg.get("data", {}).get("contamination_ratio", 0.2),
        eval_ratio=cfg.get("data", {}).get("eval_ratio", 0.2),
        seed=cfg.get("data", {}).get("seed", 42),
        cache_dir=cfg.get("data", {}).get("cache_dir", "outputs/data"),
    )

    lora = LoRAConfig(
        r=cfg.get("lora", {}).get("r", 8),
        lora_alpha=cfg.get("lora", {}).get("lora_alpha", 16),
        lora_dropout=cfg.get("lora", {}).get("lora_dropout", 0.1),
        target_modules=list(cfg.get("lora", {}).get("target_modules", ["query_key_value"])),
    )

    training = TrainingConfig(
        learning_rate=cfg.get("training", {}).get("learning_rate", 2e-4),
        batch_size=cfg.get("training", {}).get("batch_size", 8),
        gradient_accumulation_steps=cfg.get("training", {}).get("gradient_accumulation_steps", 4),
        num_epochs=cfg.get("training", {}).get("num_epochs", 3),
        warmup_ratio=cfg.get("training", {}).get("warmup_ratio", 0.1),
        seed=cfg.get("training", {}).get("seed", 42),
        logging_steps=cfg.get("training", {}).get("logging_steps", 10),
        contamination_epochs=list(cfg.get("training", {}).get("contamination_epochs", [0, 1, 5, 10])),
    )

    sampling = SamplingConfig(
        n_samples=cfg.get("sampling", {}).get("n_samples", 20),
        temperature=cfg.get("sampling", {}).get("temperature", 1.0),
        top_k=cfg.get("sampling", {}).get("top_k", 50),
        top_p=cfg.get("sampling", {}).get("top_p", 0.95),
        max_new_tokens=cfg.get("sampling", {}).get("max_new_tokens", 100),
        seed=cfg.get("sampling", {}).get("seed", 42),
    )

    detection = DetectionConfig(
        alpha=cfg.get("detection", {}).get("alpha", 0.1),
        xi=cfg.get("detection", {}).get("xi", 0.5),
    )

    model_sizes = list(cfg.get("model_sizes", [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
    ]))

    return ExperimentConfig(
        data=data,
        lora=lora,
        training=training,
        sampling=sampling,
        detection=detection,
        model_sizes=model_sizes,
        output_dir=cfg.get("output_dir", "outputs"),
        experiment_name=cfg.get("experiment_name", "cdd_contamination_detection"),
        seed=cfg.get("seed", 42),
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the contamination detection experiment."""
    print(OmegaConf.to_yaml(cfg))

    config = _build_config(cfg)
    stage = cfg.get("stage", None)
    resume = cfg.get("resume", None)

    orchestrator = ExperimentOrchestrator(
        config=config,
        resume_from=resume,
    )

    if stage:
        if stage not in PIPELINE_STAGES:
            # Allow short aliases
            alias_map = {
                "data": "data_preparation",
                "train": "fine_tuning",
                "sample": "sampling",
                "detect": "detection",
                "eval": "evaluation",
                "viz": "visualization",
            }
            stage = alias_map.get(stage, stage)

        if stage not in PIPELINE_STAGES:
            logger.error(f"Unknown stage '{stage}'. Valid: {PIPELINE_STAGES}")
            sys.exit(1)

        logger.info(f"Running single stage: {stage}")
        orchestrator.run_stage(stage)
    else:
        logger.info("Running full pipeline")
        orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()
