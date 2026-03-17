"""
Scorer for Strategy C: Rubric + Few-Shot Examples.

Loads per-question few-shot examples from the training data and includes
them in the prompt alongside the rubric. Examples are balanced across
score levels and selected with a fixed seed for reproducibility.

Critical: when scoring training data (dev evaluation), the current sample
is excluded from its own few-shot examples to prevent data leakage.
"""

import logging
import random

from src.common.openrouter import call_with_retry
from src.common.data_loader import load_train_3way
from src.strategy_c_rubric_fewshot.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)

# Score-level ordering for consistent example presentation
_LABEL_ORDER = ["Correct", "Partially correct", "Incorrect"]

# Module-level cache for training data and pre-computed examples
_train_data: list[dict] | None = None
_question_examples: dict[str, list[dict]] = {}  # question_id -> list of examples
_examples_per_label: int = 2
_seed: int = 42


def _build_question_examples(
    data: list[dict],
    n_per_label: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Pre-compute per-question example pools from training data.

    For each question, selects up to ``n_per_label`` examples per score
    level using a deterministic seed. Each example dict includes the
    sample ``id`` so individual samples can be excluded at scoring time.

    Args:
        data: Full training data (list of sample dicts).
        n_per_label: Number of examples to select per label.
        seed: Random seed for reproducible selection.

    Returns:
        Dict mapping question_id to list of example dicts, each containing
        ``id``, ``answer``, and ``score``.
    """
    # Group samples by question_id, then by label
    by_question: dict[str, dict[str, list[dict]]] = {}
    for sample in data:
        qid = sample["question_id"]
        label = sample["score"]
        by_question.setdefault(qid, {}).setdefault(label, []).append(sample)

    result: dict[str, list[dict]] = {}
    rng = random.Random(seed)

    for qid, label_groups in by_question.items():
        examples: list[dict] = []
        for label in _LABEL_ORDER:
            samples = label_groups.get(label, [])
            picked = rng.sample(samples, min(n_per_label, len(samples)))
            for s in picked:
                examples.append({
                    "id": s["id"],
                    "answer": s["answer"],
                    "score": s["score"],
                })
        # Sort by label order for consistent prompt construction
        label_rank = {label: i for i, label in enumerate(_LABEL_ORDER)}
        examples.sort(key=lambda ex: label_rank.get(ex["score"], 99))
        result[qid] = examples

    logger.info(
        "Built example pools for %d questions (%d per label, seed=%d)",
        len(result),
        n_per_label,
        seed,
    )
    return result


def configure(examples_per_label: int = 2, seed: int = 42) -> None:
    """Configure the scorer parameters before first use.

    Must be called before ``score_sample`` if non-default settings are
    desired. Calling this resets the cache so examples are recomputed.

    Args:
        examples_per_label: Number of examples to include per score level.
        seed: Random seed for reproducible example selection.
    """
    global _train_data, _question_examples, _examples_per_label, _seed
    _examples_per_label = examples_per_label
    _seed = seed
    # Reset cache to force recomputation with new params
    _train_data = None
    _question_examples = {}


def _ensure_examples_loaded() -> None:
    """Load and cache training examples on first call."""
    global _train_data, _question_examples
    if _train_data is not None:
        return
    _train_data = load_train_3way()
    _question_examples = _build_question_examples(
        _train_data,
        n_per_label=_examples_per_label,
        seed=_seed,
    )


def score_sample(sample: dict) -> dict:
    """Score a single ALICE sample using rubric + few-shot examples.

    Loads examples for this question from training data (cached).
    When scoring training data, the current sample is excluded from
    its own few-shot examples to prevent data leakage.

    Args:
        sample: A sample dict with at least ``id``, ``question_id``,
            ``question``, ``answer``, and ``rubric`` keys.

    Returns:
        Dict with ``"score"`` (str), ``"confidence"`` (float or None),
        and ``"n_examples"`` (int, the number of few-shot examples used).
    """
    _ensure_examples_loaded()

    question_id = sample["question_id"]
    sample_id = sample["id"]

    # Get cached examples, excluding the current sample to avoid leakage
    all_examples = _question_examples.get(question_id, [])
    examples = [ex for ex in all_examples if ex["id"] != sample_id]

    if not examples:
        logger.warning(
            "No few-shot examples available for question %s (sample %s)",
            question_id,
            sample_id,
        )

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

    logger.debug(
        "Scored sample %s: score=%s, confidence=%s, n_examples=%d",
        sample_id,
        result["score"],
        result["confidence"],
        result["n_examples"],
    )

    return result
