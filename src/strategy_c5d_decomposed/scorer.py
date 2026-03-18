"""Scorer for Strategy C5d: Rubric Decomposition."""

import logging

from src.common.data_loader import load_train_3way
from src.common.openrouter import call_with_retry
from src.strategy_c4_smart_examples.example_selector import SmartExampleSelector
from src.strategy_c5d_decomposed.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)

_selector: SmartExampleSelector | None = None
_seed: int = 42
_n_boundary: int = 1
_n_similar: int = 1


def configure(seed: int = 42, n_boundary: int = 1, n_similar: int = 1):
    """Configure the scorer. Fewer examples by default since the prompt is more structured."""
    global _selector, _seed, _n_boundary, _n_similar
    _seed = seed
    _n_boundary = n_boundary
    _n_similar = n_similar
    _selector = None  # reset


def _ensure_loaded():
    global _selector
    if _selector is not None:
        return
    train = load_train_3way()
    _selector = SmartExampleSelector(train, seed=_seed)


def score_sample(sample: dict) -> dict:
    _ensure_loaded()
    examples = _selector.get_examples(sample, n_boundary=_n_boundary, n_similar=_n_similar)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=sample["question"],
        answer=sample["answer"],
        rubric=sample["rubric"],
        examples=examples,
    )

    raw_response = call_with_retry(system_prompt, user_prompt)
    result = parse_response(raw_response)
    result["n_examples"] = len(examples)
    return result
