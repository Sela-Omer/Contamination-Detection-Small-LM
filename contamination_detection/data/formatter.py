"""Prompt formatting for QA/code examples targeting decoder-only models.

Supports GSM8K, QASC, StrategyQA, HumanEval, and ARC-Challenge.

Two formatters:
  - format_prompts: question/prompt-only for CDD detection/sampling
  - format_training_texts: question+answer pairs for fine-tuning
"""

import logging
from typing import Dict, List, Union

from datasets import Dataset

logger = logging.getLogger("contamination_detection")


def _extract_question(example: Dict) -> str:
    """Pull the question/prompt string from a single example dict."""
    # HumanEval: 'prompt' field contains the function signature + docstring
    if "prompt" in example and "canonical_solution" in example:
        return str(example["prompt"])
    # MATH: 'problem' field
    if "problem" in example and "solution" in example:
        return str(example["problem"])
    # Standard QA datasets
    if "question" in example:
        return str(example["question"])
    if "formatted_question" in example:
        return str(example["formatted_question"])
    raise KeyError(
        f"Cannot find a question field in example keys: {list(example.keys())}"
    )


def _extract_answer(example: Dict) -> str:
    """Pull the answer string from a single example dict.

    GSM8K: 'answer' field with step-by-step solution
    HumanEval: 'canonical_solution' field with code
    ARC: 'answerKey' + 'choices' to build full answer text
    QASC: 'combinedfact' or 'answerKey'
    StrategyQA: 'answer' is boolean
    """
    # HumanEval
    if "canonical_solution" in example:
        return str(example["canonical_solution"])
    # MATH
    if "solution" in example and "problem" in example:
        return str(example["solution"])
    # ARC: build answer from choices + answerKey
    if "choices" in example and "answerKey" in example:
        choices = example["choices"]
        key = str(example["answerKey"])
        if isinstance(choices, dict) and "label" in choices and "text" in choices:
            for label, text in zip(choices["label"], choices["text"]):
                if str(label) == key:
                    return f"{key}. {text}"
        return key
    # GSM8K and general
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
    """Format examples as prompts for CDD detection (no answer).

    HumanEval: uses the function signature directly as the prompt.
    QA datasets: uses ``Question: {question}\\nAnswer:`` format.
    """
    prompts: List[str] = []
    for ex in examples:
        # HumanEval: prompt is already a complete function signature
        if "prompt" in ex and "canonical_solution" in ex:
            prompts.append(str(ex["prompt"]))
        # MATH: use problem field
        elif "problem" in ex and "solution" in ex:
            prompts.append(f"Problem: {ex['problem']}\nSolution:")
        else:
            question = _extract_question(ex)
            prompts.append(f"Question: {question}\nAnswer:")
    return prompts


def format_training_texts(
    examples: Union[Dataset, List[Dict]],
    dataset_name: str = "auto",
) -> List[str]:
    """Format examples as prompt+answer for training.

    HumanEval: function signature + canonical solution.
    QA datasets: ``Question: {question}\\nAnswer: {answer}`` format.
    """
    texts: List[str] = []
    for ex in examples:
        if "prompt" in ex and "canonical_solution" in ex:
            texts.append(str(ex["prompt"]) + str(ex["canonical_solution"]))
        elif "problem" in ex and "solution" in ex:
            texts.append(f"Problem: {ex['problem']}\nSolution: {ex['solution']}")
        else:
            question = _extract_question(ex)
            answer = _extract_answer(ex)
            texts.append(f"Question: {question}\nAnswer: {answer}")
    return texts
