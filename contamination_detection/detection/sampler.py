"""Output sampler for CDD contamination detection.

Implements the CDD paper's sampling strategy:
  1. One GREEDY sample (temperature=0, do_sample=False) as the reference
  2. N TEMPERATURE samples (temperature=0.8, do_sample=True)
  3. Returns raw token IDs (for BPE-level edit distance) plus decoded text

This matches the CDD reference: task['greedy_sample'] + task['samples'].
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from contamination_detection.config import SamplingConfig

logger = logging.getLogger("contamination_detection.detection.sampler")

_MAX_CONTEXT_TOKENS: Dict[str, int] = {
    "EleutherAI/pythia-70m": 512,
    "EleutherAI/pythia-160m": 512,
    "EleutherAI/pythia-410m": 512,
    "EleutherAI/pythia-1b": 512,
}
_DEFAULT_MAX_CONTEXT = 512
_MAX_RETRIES = 3


@dataclass
class CDDSamplingResult:
    """Result of CDD sampling for a single prompt.

    Contains both the greedy reference and temperature samples,
    with raw token IDs for edit distance computation.
    """
    prompt: str
    greedy_text: str
    greedy_tokens: List[int]
    sample_texts: List[str]
    sample_token_lists: List[List[int]]
    seed: int
    config: SamplingConfig
    metadata: Dict = field(default_factory=dict)


def _get_max_context(model: PreTrainedModel) -> int:
    name = getattr(model.config, "_name_or_path", "")
    return _MAX_CONTEXT_TOKENS.get(name, _DEFAULT_MAX_CONTEXT)


def sample_outputs_cdd(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 50,
    config: Optional[SamplingConfig] = None,
    seed: int = 42,
    model_name: str = "",
) -> CDDSamplingResult:
    """Generate 1 greedy + N temperature samples for CDD detection.

    Matches the CDD paper: greedy (temp=0) as reference, then N samples
    at temperature=0.8. Returns raw token IDs for BPE-level edit distance.

    Args:
        prompt: Input text to condition generation on.
        model: A HuggingFace causal-LM.
        tokenizer: Matching tokenizer (used for BPE tokenization).
        n_samples: Number of temperature samples (paper default: 50).
        config: Sampling hyper-parameters.
        seed: Base random seed.
        model_name: Optional identifier for logging.

    Returns:
        A CDDSamplingResult with greedy + temperature samples and token IDs.
    """
    if config is None:
        config = SamplingConfig()

    max_ctx = _get_max_context(model)
    max_new = min(config.max_new_tokens, max_ctx)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_ctx - max_new,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    model.eval()

    # 1. Greedy sample (temperature=0, no sampling)
    with torch.no_grad():
        greedy_out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=False, max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
        )
    greedy_new = greedy_out[0, prompt_len:].tolist()
    greedy_text = tokenizer.decode(greedy_new, skip_special_tokens=True).strip()

    # 2. Temperature samples
    sample_texts: List[str] = []
    sample_token_lists: List[List[int]] = []

    for i in range(n_samples):
        sample_seed = seed + i
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                do_sample=True,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = gen_out[0, prompt_len:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Retry on empty
        retries = 0
        while not new_tokens and retries < _MAX_RETRIES:
            retries += 1
            retry_seed = sample_seed + n_samples * (retries + 1)
            torch.manual_seed(retry_seed)
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    do_sample=True, temperature=config.temperature,
                    top_k=config.top_k, top_p=config.top_p,
                    max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen_out[0, prompt_len:].tolist()
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sample_texts.append(text)
        sample_token_lists.append(new_tokens)

    return CDDSamplingResult(
        prompt=prompt,
        greedy_text=greedy_text,
        greedy_tokens=greedy_new,
        sample_texts=sample_texts,
        sample_token_lists=sample_token_lists,
        seed=seed,
        config=config,
    )


# Keep backward-compatible wrapper for tests
@dataclass
class SamplingResult:
    prompt: str
    outputs: List[str]
    model_name: str
    seed: int
    config: SamplingConfig
    metadata: Dict = field(default_factory=dict)


def sample_outputs(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 20,
    config: Optional[SamplingConfig] = None,
    seed: int = 42,
    model_name: str = "",
) -> SamplingResult:
    """Legacy wrapper: generate N temperature samples (no greedy).

    Kept for backward compatibility with existing tests.
    """
    if config is None:
        config = SamplingConfig()

    max_ctx = _get_max_context(model)
    max_new = min(config.max_new_tokens, max_ctx)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_ctx - max_new,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    outputs: List[str] = []

    for i in range(n_samples):
        sample_seed = seed + i
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                do_sample=True, temperature=config.temperature,
                top_k=config.top_k, top_p=config.top_p,
                max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)

    return SamplingResult(
        prompt=prompt, outputs=outputs, model_name=model_name,
        seed=seed, config=config,
    )
