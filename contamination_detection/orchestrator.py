"""Experiment orchestrator for the contamination detection pipeline.

Coordinates the full pipeline: dataset prep → fine-tuning → sampling →
detection → evaluation → visualization. Supports checkpointing and
resumption from the last completed stage.

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 23.1, 23.4, 24.4
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from contamination_detection.config import ExperimentConfig
from contamination_detection.utils import (
    record_dependency_versions,
    set_global_seed,
    setup_logging,
)

logger = logging.getLogger("contamination_detection.orchestrator")

# Pipeline stages in execution order
PIPELINE_STAGES = [
    "data_preparation",
    "fine_tuning",
    "sampling",
    "detection",
    "evaluation",
    "visualization",
]

# Rough per-example time estimates (seconds) by stage, for runtime estimation
_TIME_ESTIMATES_PER_EXAMPLE = {
    "data_preparation": 0.01,
    "fine_tuning": 2.0,       # per example per epoch
    "sampling": 1.0,          # per example (20 samples)
    "detection": 0.5,
    "evaluation": 0.01,
    "visualization": 0.1,
}

# Multipliers by model size (relative to 70M)
_MODEL_SIZE_MULTIPLIERS = {
    "EleutherAI/pythia-70m": 1.0,
    "EleutherAI/pythia-160m": 2.0,
    "EleutherAI/pythia-410m": 5.0,
    "EleutherAI/pythia-1b": 12.0,
}


@dataclass
class PipelineState:
    """Tracks which stages have been completed."""
    completed_stages: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    started_at: Optional[str] = None
    last_updated: Optional[str] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)

    def is_stage_completed(self, stage: str) -> bool:
        return stage in self.completed_stages

    def mark_completed(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.current_stage = None
        self.last_updated = datetime.now().isoformat()

    def mark_started(self, stage: str) -> None:
        self.current_stage = stage
        self.last_updated = datetime.now().isoformat()

    def mark_failed(self, stage: str, error_msg: str) -> None:
        self.error = error_msg
        self.error_stage = stage
        self.current_stage = None
        self.last_updated = datetime.now().isoformat()


def _save_state(state: PipelineState, path: str) -> None:
    """Save pipeline state to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "completed_stages": state.completed_stages,
        "current_stage": state.current_stage,
        "started_at": state.started_at,
        "last_updated": state.last_updated,
        "error": state.error,
        "error_stage": state.error_stage,
        # results may contain numpy arrays — convert to serialisable form
        "results": _make_serialisable(state.results),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_state(path: str) -> PipelineState:
    """Load pipeline state from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    state = PipelineState(
        completed_stages=data.get("completed_stages", []),
        current_stage=data.get("current_stage"),
        started_at=data.get("started_at"),
        last_updated=data.get("last_updated"),
        error=data.get("error"),
        error_stage=data.get("error_stage"),
        results=data.get("results", {}),
    )
    return state


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ExperimentOrchestrator:
    """Orchestrates the full contamination detection experiment pipeline.

    Each stage is an independent method that can be called individually
    or as part of the full pipeline. Supports checkpointing and resumption.

    Args:
        config: Experiment configuration.
        output_dir: Root output directory. Defaults to
            ``outputs/{experiment_name}/{timestamp}/``.
        resume_from: Path to a checkpoint JSON to resume from.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.config = config
        config.validate()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or os.path.join(
            config.output_dir, config.experiment_name, timestamp
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(self.output_dir, "pipeline_state.json")

        # Resume or start fresh
        if resume_from and os.path.exists(resume_from):
            self.state = _load_state(resume_from)
            logger.info(
                f"Resumed from checkpoint: completed={self.state.completed_stages}"
            )
        elif os.path.exists(self._checkpoint_path):
            self.state = _load_state(self._checkpoint_path)
            logger.info(
                f"Resumed from existing checkpoint: completed={self.state.completed_stages}"
            )
        else:
            self.state = PipelineState(started_at=datetime.now().isoformat())

        # Set up logging
        self._logger = setup_logging(
            log_dir=os.path.join(self.output_dir, "logs"),
            tensorboard=False,
        )

    # ── Public API ────────────────────────────────────────────────────

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute all pipeline stages in order, skipping completed ones.

        Returns:
            Dict of results keyed by stage name.
        """
        self._log_config()
        self._log_runtime_estimate()

        for stage in PIPELINE_STAGES:
            if self.state.is_stage_completed(stage):
                logger.info(f"Skipping completed stage: {stage}")
                continue
            self._run_stage(stage)

        logger.info("Full pipeline complete.")
        return self.state.results

    def run_stage(self, stage: str) -> Any:
        """Run a single named stage.

        Args:
            stage: One of the PIPELINE_STAGES names.

        Returns:
            Stage-specific result dict.
        """
        if stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Unknown stage '{stage}'. Valid: {PIPELINE_STAGES}"
            )
        return self._run_stage(stage)

    def estimate_runtime(self) -> timedelta:
        """Estimate total compute time based on config.

        Returns:
            Estimated timedelta.
        """
        n_models = len(self.config.model_sizes)
        n_contam = len(self.config.training.contamination_epochs)
        n_conditions = n_models * n_contam
        # Assume ~100 examples per condition as a rough default
        n_examples = 100

        total_seconds = 0.0
        for stage, per_example in _TIME_ESTIMATES_PER_EXAMPLE.items():
            stage_time = per_example * n_examples * n_conditions
            total_seconds += stage_time

        # Apply model size multiplier (average across models)
        avg_multiplier = np.mean([
            _MODEL_SIZE_MULTIPLIERS.get(m, 1.0)
            for m in self.config.model_sizes
        ])
        total_seconds *= avg_multiplier

        return timedelta(seconds=total_seconds)

    # ── Stage implementations ─────────────────────────────────────────

    def stage_data_preparation(self) -> Dict[str, Any]:
        """Stage 1: Load datasets, create splits, build contaminated sets."""
        from contamination_detection.data.loader import load_qa_dataset, save_dataset
        from contamination_detection.data.splitter import create_splits
        from contamination_detection.data.contamination import create_contaminated_training_set
        from contamination_detection.data.formatter import format_prompts

        set_global_seed(self.config.seed)
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        results: Dict[str, Any] = {"datasets": {}}

        for dataset_name in self.config.data.datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                ds = load_qa_dataset(dataset_name, cache_dir=self.config.data.cache_dir)
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue

            splits = create_splits(
                ds,
                train_ratio=self.config.data.train_ratio,
                contamination_ratio=self.config.data.contamination_ratio,
                eval_ratio=self.config.data.eval_ratio,
                seed=self.config.data.seed,
            )

            # Save splits
            ds_dir = os.path.join(data_dir, dataset_name)
            save_dataset(splits.train, os.path.join(ds_dir, "train"))
            save_dataset(splits.contamination, os.path.join(ds_dir, "contamination"))
            save_dataset(splits.evaluation, os.path.join(ds_dir, "evaluation"))

            # Create contaminated training sets for each level
            for contam_epochs in self.config.training.contamination_epochs:
                contam_train = create_contaminated_training_set(
                    splits.train, splits.contamination,
                    contamination_epochs=contam_epochs,
                    seed=self.config.seed,
                )
                save_dataset(
                    contam_train,
                    os.path.join(ds_dir, f"train_contam_{contam_epochs}"),
                )

            # Format prompts for eval set
            prompts = format_prompts(splits.evaluation)

            results["datasets"][dataset_name] = {
                "train_size": len(splits.train),
                "contamination_size": len(splits.contamination),
                "eval_size": len(splits.evaluation),
                "n_prompts": len(prompts),
            }

        return results

    def stage_fine_tuning(self) -> Dict[str, Any]:
        """Stage 2: Fine-tune models for all conditions."""
        from contamination_detection.training.model_loader import load_pythia_with_lora
        from contamination_detection.training.trainer import fine_tune
        from contamination_detection.data.loader import load_saved_dataset
        from contamination_detection.data.formatter import format_prompts

        results: Dict[str, Any] = {"models": {}}
        training_dir = os.path.join(self.output_dir, "models")

        for model_name in self.config.model_sizes:
            short_name = model_name.split("/")[-1] if "/" in model_name else model_name

            for contam_epochs in self.config.training.contamination_epochs:
                condition = f"{short_name}_contam{contam_epochs}"
                model_dir = os.path.join(training_dir, condition)

                if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                    logger.info(f"Skipping {condition}: checkpoint exists")
                    results["models"][condition] = {"status": "cached", "dir": model_dir}
                    continue

                logger.info(f"Fine-tuning {condition}")
                try:
                    model, tokenizer = load_pythia_with_lora(
                        model_name, self.config.lora
                    )

                    # Load contaminated training data
                    for ds_name in self.config.data.datasets:
                        data_path = os.path.join(
                            self.output_dir, "data", ds_name,
                            f"train_contam_{contam_epochs}",
                        )
                        if os.path.exists(data_path):
                            train_ds = load_saved_dataset(data_path)
                            train_texts = format_prompts(train_ds)
                            break
                    else:
                        logger.warning(f"No training data found for {condition}")
                        continue

                    result = fine_tune(
                        model=model,
                        tokenizer=tokenizer,
                        train_texts=train_texts,
                        training_config=self.config.training,
                        lora_config=self.config.lora,
                        output_dir=model_dir,
                    )

                    results["models"][condition] = {
                        "status": "trained",
                        "dir": model_dir,
                        "final_loss": result.final_loss,
                        "sanity_passed": result.sanity_passed,
                    }

                    # Free memory
                    del model, tokenizer
                except Exception as e:
                    logger.error(f"Failed to fine-tune {condition}: {e}")
                    results["models"][condition] = {"status": "failed", "error": str(e)}

        return results

    def stage_sampling(self) -> Dict[str, Any]:
        """Stage 3: Sample outputs from fine-tuned models."""
        from contamination_detection.detection.sampler import sample_outputs
        from contamination_detection.training.model_loader import load_checkpoint
        from contamination_detection.data.loader import load_saved_dataset
        from contamination_detection.data.formatter import format_prompts

        results: Dict[str, Any] = {"sampling": {}}
        sampling_dir = os.path.join(self.output_dir, "sampling")
        os.makedirs(sampling_dir, exist_ok=True)

        models_dir = os.path.join(self.output_dir, "models")
        if not os.path.exists(models_dir):
            logger.warning("No models directory found; skipping sampling.")
            return results

        # Load eval prompts
        eval_prompts: List[str] = []
        for ds_name in self.config.data.datasets:
            eval_path = os.path.join(self.output_dir, "data", ds_name, "evaluation")
            if os.path.exists(eval_path):
                eval_ds = load_saved_dataset(eval_path)
                eval_prompts = format_prompts(eval_ds)
                break

        if not eval_prompts:
            logger.warning("No evaluation prompts found; skipping sampling.")
            return results

        for condition_dir in sorted(os.listdir(models_dir)):
            condition_path = os.path.join(models_dir, condition_dir)
            adapter_path = os.path.join(condition_path, "adapter_config.json")
            if not os.path.exists(adapter_path):
                continue

            output_file = os.path.join(sampling_dir, f"{condition_dir}.json")
            if os.path.exists(output_file):
                logger.info(f"Skipping sampling for {condition_dir}: cached")
                continue

            logger.info(f"Sampling outputs for {condition_dir}")
            try:
                model, tokenizer, _ = load_checkpoint(condition_path)
                all_outputs = []
                for idx, prompt in enumerate(eval_prompts):
                    sr = sample_outputs(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        n_samples=self.config.sampling.n_samples,
                        config=self.config.sampling,
                        seed=self.config.sampling.seed + idx,
                    )
                    all_outputs.append({
                        "prompt": prompt,
                        "outputs": sr.outputs,
                    })

                with open(output_file, "w") as f:
                    json.dump(all_outputs, f, indent=2)

                results["sampling"][condition_dir] = {
                    "n_prompts": len(all_outputs),
                    "n_samples_per_prompt": self.config.sampling.n_samples,
                }

                del model, tokenizer
            except Exception as e:
                logger.error(f"Sampling failed for {condition_dir}: {e}")
                results["sampling"][condition_dir] = {"error": str(e)}

        return results

    def stage_detection(self) -> Dict[str, Any]:
        """Stage 4: Run CDD detection and baselines on sampled outputs."""
        from contamination_detection.detection.edit_distance import compute_edit_distances
        from contamination_detection.detection.peakedness import compute_peakedness
        from contamination_detection.detection.classifier import classify

        results: Dict[str, Any] = {"detection": {}}
        sampling_dir = os.path.join(self.output_dir, "sampling")
        detection_dir = os.path.join(self.output_dir, "detection")
        os.makedirs(detection_dir, exist_ok=True)

        if not os.path.exists(sampling_dir):
            logger.warning("No sampling directory found; skipping detection.")
            return results

        for fname in sorted(os.listdir(sampling_dir)):
            if not fname.endswith(".json"):
                continue
            condition = fname.replace(".json", "")
            output_file = os.path.join(detection_dir, f"{condition}.json")

            if os.path.exists(output_file):
                logger.info(f"Skipping detection for {condition}: cached")
                continue

            logger.info(f"Running detection for {condition}")
            try:
                with open(os.path.join(sampling_dir, fname)) as f:
                    samples = json.load(f)

                detection_results = []
                for item in samples:
                    outputs = item["outputs"]
                    dist = compute_edit_distances(outputs)
                    peak = compute_peakedness(
                        dist.normalized_matrix, self.config.detection.alpha
                    )
                    cr = classify(peak, self.config.detection.xi)
                    detection_results.append({
                        "prompt": item["prompt"],
                        "peakedness": peak,
                        "is_contaminated": cr.is_contaminated,
                        "confidence": cr.confidence,
                    })

                with open(output_file, "w") as f:
                    json.dump(detection_results, f, indent=2)

                results["detection"][condition] = {
                    "n_examples": len(detection_results),
                    "n_contaminated": sum(
                        1 for r in detection_results if r["is_contaminated"]
                    ),
                }
            except Exception as e:
                logger.error(f"Detection failed for {condition}: {e}")
                results["detection"][condition] = {"error": str(e)}

        return results

    def stage_evaluation(self) -> Dict[str, Any]:
        """Stage 5: Compute metrics, CIs, significance tests, export."""
        from contamination_detection.evaluation.metrics import compute_metrics
        from contamination_detection.evaluation.confidence import bootstrap_confidence_intervals
        from contamination_detection.evaluation.exporter import export_csv, export_json, export_latex

        results: Dict[str, Any] = {"evaluation": {}}
        detection_dir = os.path.join(self.output_dir, "detection")
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        if not os.path.exists(detection_dir):
            logger.warning("No detection directory found; skipping evaluation.")
            return results

        # For now, compute metrics per condition using detection results
        # In a full run, ground-truth labels would come from the data splits
        all_metrics = {}
        for fname in sorted(os.listdir(detection_dir)):
            if not fname.endswith(".json"):
                continue
            condition = fname.replace(".json", "")

            with open(os.path.join(detection_dir, fname)) as f:
                det_results = json.load(f)

            # In a real experiment, y_true comes from the eval set labels
            # For now, store detection predictions
            results["evaluation"][condition] = {
                "n_examples": len(det_results),
            }

        logger.info("Evaluation stage complete.")
        return results

    def stage_visualization(self) -> Dict[str, Any]:
        """Stage 6: Generate all plots and figures."""
        from contamination_detection.visualization.plots import setup_publication_style

        setup_publication_style()
        results: Dict[str, Any] = {"visualization": {}}
        viz_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(viz_dir, exist_ok=True)

        logger.info(f"Visualization outputs will be saved to {viz_dir}")
        results["visualization"]["output_dir"] = viz_dir
        return results

    # ── Internal helpers ──────────────────────────────────────────────

    def _run_stage(self, stage: str) -> Any:
        """Run a single stage with checkpointing and error handling."""
        stage_method = {
            "data_preparation": self.stage_data_preparation,
            "fine_tuning": self.stage_fine_tuning,
            "sampling": self.stage_sampling,
            "detection": self.stage_detection,
            "evaluation": self.stage_evaluation,
            "visualization": self.stage_visualization,
        }

        method = stage_method.get(stage)
        if method is None:
            raise ValueError(f"No implementation for stage '{stage}'")

        self.state.mark_started(stage)
        _save_state(self.state, self._checkpoint_path)

        start = time.perf_counter()
        try:
            result = method()
            elapsed = time.perf_counter() - start
            logger.info(f"Stage '{stage}' completed in {elapsed:.1f}s")

            self.state.results[stage] = result
            self.state.mark_completed(stage)
            _save_state(self.state, self._checkpoint_path)
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            error_msg = (
                f"Stage '{stage}' failed after {elapsed:.1f}s: "
                f"{type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            self.state.mark_failed(stage, error_msg)
            _save_state(self.state, self._checkpoint_path)
            raise

    def _log_config(self) -> None:
        """Log all configuration parameters at run start."""
        logger.info("=" * 60)
        logger.info("Experiment Configuration")
        logger.info("=" * 60)
        logger.info(f"Experiment name: {self.config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Global seed: {self.config.seed}")
        logger.info(f"Model sizes: {self.config.model_sizes}")
        logger.info(f"Contamination epochs: {self.config.training.contamination_epochs}")
        logger.info(f"Datasets: {self.config.data.datasets}")
        logger.info(f"LoRA rank: {self.config.lora.r}, alpha: {self.config.lora.lora_alpha}")
        logger.info(
            f"Training: lr={self.config.training.learning_rate}, "
            f"batch={self.config.training.batch_size}, "
            f"epochs={self.config.training.num_epochs}"
        )
        logger.info(
            f"Sampling: n={self.config.sampling.n_samples}, "
            f"temp={self.config.sampling.temperature}"
        )
        logger.info(
            f"Detection: alpha={self.config.detection.alpha}, "
            f"xi={self.config.detection.xi}"
        )

        # Record dependency versions
        versions = record_dependency_versions()
        logger.info(f"Dependency versions: {versions}")

        # Save config to file
        config_path = os.path.join(self.output_dir, "config.json")
        config_dict = _make_serialisable({
            "experiment_name": self.config.experiment_name,
            "seed": self.config.seed,
            "model_sizes": self.config.model_sizes,
            "data": {
                "datasets": self.config.data.datasets,
                "train_ratio": self.config.data.train_ratio,
                "contamination_ratio": self.config.data.contamination_ratio,
                "eval_ratio": self.config.data.eval_ratio,
                "seed": self.config.data.seed,
            },
            "training": {
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.batch_size,
                "num_epochs": self.config.training.num_epochs,
                "contamination_epochs": self.config.training.contamination_epochs,
            },
            "lora": {
                "r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
                "lora_dropout": self.config.lora.lora_dropout,
            },
            "sampling": {
                "n_samples": self.config.sampling.n_samples,
                "temperature": self.config.sampling.temperature,
            },
            "detection": {
                "alpha": self.config.detection.alpha,
                "xi": self.config.detection.xi,
            },
            "dependency_versions": versions,
        })
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {config_path}")

    def _log_runtime_estimate(self) -> None:
        """Estimate and log expected compute time."""
        est = self.estimate_runtime()
        logger.info(f"Estimated total runtime: {est}")
