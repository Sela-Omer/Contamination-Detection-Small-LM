"""Prompt formatting for QA examples targeting decoder-only models.

Supports GSM8K, QASC, and StrategyQA.

Two formatters:
  - format_prompts: question-only prompts for CDD detection/sampling
  - format_training_texts: question+answer pairs for fine-tuning
"""

import logging
from typing import Dict, List, Union

from datasets import Dataset

logger = logging.getLogger("contamination_detection")


def _extract_question(example: Dict) -> str:
    """Pull the question string from a single example dict."""
    if "question" in example:
        return str(example["question"])
    if "formatted_question" in example:
        return str(example["formatted_question"])
    raise KeyError(
        f"Cannot find a question field in example keys: {list(example.keys())}"
    )


def _extract_answer(example: Dict) -> str:
    """Pull the answer string from a single example dict.

    GSM8K: 'answer' field contains step-by-step solution ending with #### <number>
    QASC: 'answerKey' is just a letter; 'combinedfact' is the full answer
    StrategyQA: 'answer' is boolean
    """
    if "answer" in example:
        return str(example["answer"])
    if "combinedfact" in example:
        return str(example["combinedfact"])
    if "answerKey" in example:
        return str(example["answerKey"])
    raise KeyError(
        f"Cannot find an answer field in example keys: {list(example.keys())}"
    )


def format_prompts(
    examples: Union[Dataset, List[Dict]],
    dataset_name: str = "auto",
) -> List[str]:
    """Format examples as ``Question: {question}\\nAnswer:`` prompts (no answer).

    Used for CDD detection — the model generates completions from this prompt.
    """
    prompts: List[str] = []
    for ex in examples:
        question = _extract_question(ex)
        prompts.append(f"Question: {question}\nAnswer:")
    return prompts


def format_training_texts(
    examples: Union[Dataset, List[Dict]],
    dataset_name: str = "auto",
) -> List[str]:
    """Format examples as ``Question: {question}\\nAnswer: {answer}`` for training.

    The model learns to produce the answer given the question, which is what
    creates the memorization signal that CDD detects.
    """
    texts: List[str] = []
    for ex in examples:
        question = _extract_question(ex)
        answer = _extract_answer(ex)
        texts.append(f"Question: {question}\nAnswer: {answer}")
    return texts
