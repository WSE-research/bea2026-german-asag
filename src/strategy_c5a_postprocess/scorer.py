"""Scorer for Strategy C5a: Post-Processing + Anti-Examples."""

import logging
import re

from src.common.data_loader import load_train_3way
from src.common.openrouter import call_with_retry
from src.strategy_c4_smart_examples.example_selector import SmartExampleSelector
from src.strategy_c5a_postprocess.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)

_selector: SmartExampleSelector | None = None
_seed: int = 42
_n_boundary: int = 2
_n_similar: int = 1

# Patterns for non-answers
_EMPTY_PATTERN = re.compile(r'^[\s?!.\-]*$')
_REFUSE_PHRASES = [
    "keine ahnung",
    "weiß nicht",
    "weis nicht",
    "keine antwort",
    "kein plan",
    "kp",
    "ka",
]


def configure(seed: int = 42, n_boundary: int = 2, n_similar: int = 1):
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


def _should_override_incorrect(answer: str) -> bool:
    """Check if the answer should be overridden to Incorrect via post-processing rules."""
    stripped = answer.strip()

    # Rule 1: Very short answers (< 15 characters)
    if len(stripped) < 15:
        return True

    # Rule 2: Empty/punctuation-only answers
    if _EMPTY_PATTERN.match(stripped):
        return True

    # Rule 3: Refusal phrases
    lower = stripped.lower()
    for phrase in _REFUSE_PHRASES:
        if phrase in lower:
            return True

    return False


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

    # Post-processing: override to Incorrect for short/empty/refusal answers
    if _should_override_incorrect(sample["answer"]):
        if result["score"] != "Incorrect":
            logger.info(
                "Post-processing override for %s: '%s' -> Incorrect (was %s)",
                sample["id"],
                sample["answer"][:50],
                result["score"],
            )
        result["overridden"] = result["score"] != "Incorrect"
        result["score"] = "Incorrect"
    else:
        result["overridden"] = False

    return result
