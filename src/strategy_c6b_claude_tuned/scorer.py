"""Scorer for Strategy C6b: Claude-optimized prompt with adaptive difficulty.

Same adaptive difficulty logic as C5c/C6, but with a Claude-specific prompt
that uses XML tags, English system prompt, and explicit JSON-only output
instructions following Anthropic's prompting best practices.
"""

import logging
import time
from collections import Counter
from math import log2

from src.common.data_loader import load_train_3way
from src.common.openrouter import call_with_retry
from src.strategy_c4_smart_examples.example_selector import SmartExampleSelector
from src.strategy_c6b_claude_tuned.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)

_selector: SmartExampleSelector | None = None
_seed: int = 42
_difficulty: dict | None = None

TIER_CONFIG = {
    "easy": (1, 2),
    "medium": (2, 2),
    "hard": (3, 2),
}


def configure(seed: int = 42):
    global _selector, _seed, _difficulty
    _seed = seed
    _selector = None
    _difficulty = None


def _compute_difficulty(train_data: list[dict]) -> dict:
    scores_by_q: dict[str, list[str]] = {}
    for s in train_data:
        scores_by_q.setdefault(s["question_id"], []).append(s["score"])

    difficulty = {}
    for qid, scores in scores_by_q.items():
        label_counts = Counter(scores)
        total = sum(label_counts.values())
        dominant_pct = max(label_counts.values()) / total

        entropy = 0.0
        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2(p)

        if dominant_pct > 0.60:
            tier = "easy"
        elif dominant_pct < 0.40:
            tier = "hard"
        else:
            tier = "medium"

        difficulty[qid] = {
            "tier": tier,
            "dominant_pct": round(dominant_pct, 4),
            "entropy": round(entropy, 4),
        }

    tier_counts = Counter(d["tier"] for d in difficulty.values())
    logger.info(
        "Difficulty tiers: easy=%d, medium=%d, hard=%d (total %d questions)",
        tier_counts.get("easy", 0),
        tier_counts.get("medium", 0),
        tier_counts.get("hard", 0),
        len(difficulty),
    )
    return difficulty


def _ensure_loaded():
    global _selector, _difficulty
    if _selector is not None:
        return
    train = load_train_3way()
    _selector = SmartExampleSelector(train, seed=_seed)
    _difficulty = _compute_difficulty(train)


def get_difficulty() -> dict | None:
    return _difficulty


def score_sample(sample: dict) -> dict:
    _ensure_loaded()

    qid = sample["question_id"]
    q_diff = _difficulty.get(qid, {"tier": "medium", "dominant_pct": 0.5, "entropy": 1.0})
    tier = q_diff["tier"]
    n_boundary, n_similar = TIER_CONFIG[tier]

    examples = _selector.get_examples(sample, n_boundary=n_boundary, n_similar=n_similar)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=sample["question"],
        answer=sample["answer"],
        rubric=sample["rubric"],
        examples=examples,
    )

    t_start = time.monotonic()
    raw_response = call_with_retry(system_prompt, user_prompt, max_tokens=256, timeout=60.0)
    t_elapsed = time.monotonic() - t_start

    inline_meta = raw_response.pop("_metadata", {})
    result = parse_response(raw_response)

    result["n_examples"] = len(examples)
    result["difficulty_tier"] = tier
    result["n_boundary"] = n_boundary
    result["n_similar"] = n_similar
    result["prompt_tokens"] = inline_meta.get("prompt_tokens")
    result["completion_tokens"] = inline_meta.get("completion_tokens")
    result["total_tokens"] = inline_meta.get("total_tokens")
    result["total_cost_usd"] = inline_meta.get("cost")
    result["generation_id"] = inline_meta.get("generation_id")
    result["model_used"] = inline_meta.get("model")
    result["reasoning_tokens"] = inline_meta.get("reasoning_tokens")
    result["cached_tokens"] = inline_meta.get("cached_tokens")
    result["wall_clock_seconds"] = round(t_elapsed, 3)
    result["system_prompt_chars"] = len(system_prompt)
    result["user_prompt_chars"] = len(user_prompt)

    return result
