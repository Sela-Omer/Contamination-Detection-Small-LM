"""Tests for the experiment orchestrator.

Includes:
- Property test for checkpoint resumption equivalence (13.3)
- Integration test for full pipeline on tiny data (13.5)

Requirements: 22.1–22.5
"""

import json
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from contamination_detection.config import (
    DataConfig,
    DetectionConfig,
    ExperimentConfig,
    LoRAConfig,
    SamplingConfig,
    TrainingConfig,
)
from contamination_detection.orchestrator import (
    ExperimentOrchestrator,
    PipelineState,
    PIPELINE_STAGES,
    _load_state,
    _save_state,
)


# ── Unit tests for PipelineState and checkpointing ───────────────────


class TestPipelineState:
    """Test pipeline state management."""

    def test_initial_state(self):
        state = PipelineState()
        assert state.completed_stages == []
        assert state.current_stage is None
        assert state.error is None

    def test_mark_completed(self):
        state = PipelineState()
        state.mark_completed("data_preparation")
        assert state.is_stage_completed("data_preparation")
        assert not state.is_stage_completed("fine_tuning")

    def test_mark_started(self):
        state = PipelineState()
        state.mark_started("fine_tuning")
        assert state.current_stage == "fine_tuning"

    def test_mark_failed(self):
        state = PipelineState()
        state.mark_failed("detection", "OOM error")
        assert state.error == "OOM error"
        assert state.error_stage == "detection"
        assert state.current_stage is None

    def test_idempotent_completion(self):
        state = PipelineState()
        state.mark_completed("data_preparation")
        state.mark_completed("data_preparation")
        assert state.completed_stages.count("data_preparation") == 1


class TestCheckpointSaveLoad:
    """Test checkpoint serialization round-trip."""

    def test_save_load_roundtrip(self):
        state = PipelineState()
        state.mark_completed("data_preparation")
        state.mark_completed("fine_tuning")
        state.results["data_preparation"] = {"n_examples": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            _save_state(state, path)

            loaded = _load_state(path)
            assert loaded.completed_stages == ["data_preparation", "fine_tuning"]
            assert loaded.results["data_preparation"]["n_examples"] == 100

    def test_save_load_with_numpy(self):
        """Numpy types should be serialised correctly."""
        state = PipelineState()
        state.results["test"] = {
            "array": np.array([1.0, 2.0, 3.0]),
            "int": np.int64(42),
            "float": np.float64(3.14),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            _save_state(state, path)

            loaded = _load_state(path)
            assert loaded.results["test"]["array"] == [1.0, 2.0, 3.0]
            assert loaded.results["test"]["int"] == 42
            assert abs(loaded.results["test"]["float"] - 3.14) < 1e-10

    def test_save_load_with_error(self):
        state = PipelineState()
        state.mark_failed("sampling", "GPU OOM")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            _save_state(state, path)

            loaded = _load_state(path)
            assert loaded.error == "GPU OOM"
            assert loaded.error_stage == "sampling"


class TestOrchestratorInit:
    """Test orchestrator initialization and config logging."""

    def _make_config(self, **overrides) -> ExperimentConfig:
        config = ExperimentConfig(
            model_sizes=["EleutherAI/pythia-70m"],
            training=TrainingConfig(
                contamination_epochs=[0],
                num_epochs=1,
                batch_size=2,
            ),
            data=DataConfig(datasets=["QASC"]),
        )
        return config

    def test_creates_output_dir(self):
        config = self._make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_run")
            orch = ExperimentOrchestrator(config, output_dir=out)
            assert os.path.isdir(out)

    def test_resume_from_checkpoint(self):
        config = self._make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a checkpoint
            state = PipelineState()
            state.mark_completed("data_preparation")
            ckpt_path = os.path.join(tmpdir, "ckpt.json")
            _save_state(state, ckpt_path)

            out = os.path.join(tmpdir, "resumed")
            orch = ExperimentOrchestrator(config, output_dir=out, resume_from=ckpt_path)
            assert orch.state.is_stage_completed("data_preparation")

    def test_runtime_estimate(self):
        config = self._make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "est")
            orch = ExperimentOrchestrator(config, output_dir=out)
            est = orch.estimate_runtime()
            assert est.total_seconds() > 0

    def test_invalid_stage_raises(self):
        config = self._make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "inv")
            orch = ExperimentOrchestrator(config, output_dir=out)
            with pytest.raises(ValueError, match="Unknown stage"):
                orch.run_stage("nonexistent_stage")


# ── 13.3 Property test: Checkpoint Resumption Equivalence ────────────


class TestCheckpointResumptionProperty:
    """Property 12: Checkpoint Resumption Equivalence.

    **Validates: Requirements 22.2**

    For any set of completed stages, saving and loading a checkpoint
    should produce an equivalent state.
    """

    @given(
        completed=st.lists(
            st.sampled_from(PIPELINE_STAGES),
            min_size=0,
            max_size=len(PIPELINE_STAGES),
            unique=True,
        ),
        has_error=st.booleans(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_checkpoint_roundtrip_preserves_state(self, completed, has_error, tmp_path):
        """Saving then loading a checkpoint preserves all state fields.

        **Validates: Requirements 22.2**
        """
        state = PipelineState(started_at="2025-01-01T00:00:00")
        for stage in completed:
            state.mark_completed(stage)

        if has_error and completed:
            state.mark_failed(completed[-1], "test error")

        # Add some results
        for stage in completed:
            state.results[stage] = {"n": len(completed)}

        path = str(tmp_path / "state.json")
        _save_state(state, path)
        loaded = _load_state(path)

        # Verify equivalence
        assert set(loaded.completed_stages) == set(state.completed_stages)
        assert loaded.error == state.error
        assert loaded.error_stage == state.error_stage
        for stage in completed:
            assert loaded.results.get(stage) == state.results.get(stage)


# ── 13.5 Integration test: Full pipeline on tiny data ────────────────


@pytest.mark.slow
class TestFullPipelineIntegration:
    """End-to-end integration test on tiny dataset with Pythia-70M.

    This test is slow (downloads model, fine-tunes, samples) — run with:
        pytest -m slow tests/test_orchestrator.py

    Requirements: 22.1–22.5
    """

    def test_full_pipeline_tiny(self, tmp_path):
        """Run full pipeline on 10 examples with Pythia-70M, 1 epoch, 2 steps."""
        config = ExperimentConfig(
            model_sizes=["EleutherAI/pythia-70m"],
            data=DataConfig(
                datasets=["QASC"],
                train_ratio=0.5,
                contamination_ratio=0.25,
                eval_ratio=0.25,
                seed=42,
            ),
            lora=LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.0),
            training=TrainingConfig(
                learning_rate=5e-4,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_epochs=1,
                warmup_ratio=0.0,
                seed=42,
                logging_steps=1,
                contamination_epochs=[0, 1],
            ),
            sampling=SamplingConfig(
                n_samples=3,
                temperature=1.0,
                max_new_tokens=20,
                seed=42,
            ),
            detection=DetectionConfig(alpha=0.1, xi=0.5),
            output_dir=str(tmp_path),
            experiment_name="integration_test",
            seed=42,
        )

        orch = ExperimentOrchestrator(config, output_dir=str(tmp_path / "run"))
        results = orch.run_full_pipeline()

        # Verify all stages completed
        for stage in PIPELINE_STAGES:
            assert orch.state.is_stage_completed(stage), f"Stage {stage} not completed"

        # Verify output directory structure
        assert os.path.exists(os.path.join(orch.output_dir, "pipeline_state.json"))
        assert os.path.exists(os.path.join(orch.output_dir, "config.json"))

        # Verify no errors
        assert orch.state.error is None
