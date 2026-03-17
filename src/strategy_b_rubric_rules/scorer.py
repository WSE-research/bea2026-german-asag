"""
Single-sample scorer for Strategy B (Rubric + Strict Rules).

Takes an ALICE dataset sample and returns a scored result dict
by calling the LLM via OpenRouter with the rubric + strict rules prompt.
"""

import logging

from src.common.openrouter import call_with_retry
from src.strategy_b_rubric_rules.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)


def score_sample(sample: dict) -> dict:
    """
    Score a single ALICE sample using the rubric + strict rules strategy.

    Args:
        sample: Dict with keys "id", "question_id", "question", "answer",
                "rubric" (dict with "Correct", "Partially correct", "Incorrect"),
                and optionally "score" (ground-truth label for evaluation).

    Returns:
        Dict with keys:
            - "id": Sample ID.
            - "question_id": Question ID.
            - "score": Predicted label ("Correct", "Partially correct", or "Incorrect").
            - "confidence": Float confidence value or None.
            - "true_label": Ground-truth label if available, else None.
            - "error": Error message string if the LLM call failed, else None.
    """
    sample_id = sample["id"]
    question_id = sample.get("question_id", "")
    true_label = sample.get("score")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=sample["question"],
        answer=sample["answer"],
        rubric=sample["rubric"],
    )

    try:
        raw_response = call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = parse_response(raw_response)

        return {
            "id": sample_id,
            "question_id": question_id,
            "score": parsed["score"],
            "confidence": parsed["confidence"],
            "true_label": true_label,
            "error": None,
        }

    except Exception as exc:
        logger.error("Failed to score sample %s: %s", sample_id, exc)
        return {
            "id": sample_id,
            "question_id": question_id,
            "score": "Incorrect",
            "confidence": None,
            "true_label": true_label,
            "error": str(exc),
        }
