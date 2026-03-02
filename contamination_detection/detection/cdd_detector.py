"""CDD detector facade — unified interface for contamination detection.

Implements the correct CDD algorithm from Dong et al. (2024):
  1. Generate 1 greedy sample + N temperature samples per prompt
  2. Tokenize with the model's BPE tokenizer
  3. Compute edit distance of each temperature sample vs the greedy reference (star topology)
  4. Peakedness = proportion of distances <= alpha * max_length
  5. Classify: peakedness > xi → contaminated
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from contamination_detection.config import DetectionConfig, SamplingConfig
from contamination_detection.detection.sampler import sample_outputs_cdd
from contamination_detection.detection.edit_distance import (
    compute_edit_distances_star,
    compute_peakedness,
)
from contamination_detection.detection.classifier import (
    ClassificationResult,
    classify,
)

logger = logging.getLogger("contamination_detection.detection.cdd_detector")


@dataclass
class DetectionResult:
    """Detection result for a single prompt."""
    prompt: str
    is_contaminated: bool
    confidence: float
    peakedness: float
    metadata: Dict = field(default_factory=dict)


def detect(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    sampling_config: Optional[SamplingConfig] = None,
    detection_config: Optional[DetectionConfig] = None,
    seed: int = 42,
    model_name: str = "",
    max_token_length: int = 100,
) -> List[DetectionResult]:
    """Run the full CDD pipeline on a list of prompts.

    For each prompt:
      1. Generate 1 greedy + N temperature samples
      2. Compute BPE-level edit distances (star: greedy vs each sample)
      3. Peakedness = proportion of distances <= alpha * max_length
      4. Classify: peakedness > xi → contaminated

    Args:
        model: A HuggingFace causal-LM.
        tokenizer: Matching tokenizer (used for BPE tokenization).
        prompts: Input prompts to evaluate.
        sampling_config: Sampling hyper-parameters.
        detection_config: Detection thresholds (alpha, xi).
        seed: Base random seed.
        model_name: Optional identifier for logging.
        max_token_length: Truncate token sequences to this length (paper: 100).

    Returns:
        List of DetectionResult.
    """
    if sampling_config is None:
        sampling_config = SamplingConfig()
    if detection_config is None:
        detection_config = DetectionConfig()

    results: List[DetectionResult] = []

    for idx, prompt in enumerate(prompts):
        try:
            # 1. Sample: 1 greedy + N temperature
            sr = sample_outputs_cdd(
                prompt=prompt, model=model, tokenizer=tokenizer,
                n_samples=sampling_config.n_samples, config=sampling_config,
                seed=seed + idx * sampling_config.n_samples, model_name=model_name,
            )

            # 2. Edit distances (star topology, BPE tokens)
            dist = compute_edit_distances_star(
                greedy_tokens=sr.greedy_tokens,
                sample_token_lists=sr.sample_token_lists,
                max_token_length=max_token_length,
            )

            # 3. Peakedness
            peak = compute_peakedness(
                dist.distances, dist.max_length, detection_config.alpha
            )

            # 4. Classify
            cr: ClassificationResult = classify(peak, detection_config.xi)

            results.append(DetectionResult(
                prompt=prompt,
                is_contaminated=cr.is_contaminated,
                confidence=cr.confidence,
                peakedness=peak,
                metadata={
                    "n_samples": sampling_config.n_samples,
                    "alpha": detection_config.alpha,
                    "xi": detection_config.xi,
                    "max_length": dist.max_length,
                    "edit_distance_summary": dist.summary,
                },
            ))
        except Exception:
            logger.exception(f"Failed to process prompt {idx}: {prompt[:80]!r}")
            continue

    logger.info(f"CDD detection complete: {len(results)}/{len(prompts)} prompts processed")
    return results
