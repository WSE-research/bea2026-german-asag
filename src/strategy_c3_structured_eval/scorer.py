"""Scorer for Strategy C3: Structured Evaluation with Few-Shot."""

import logging
import random

from src.common.openrouter import call_with_retry
from src.common.data_loader import load_train_3way
from src.strategy_c3_structured_eval.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)

_LABEL_ORDER = ["Correct", "Partially correct", "Incorrect"]

_train_data: list[dict] | None = None
_question_examples: dict[str, list[dict]] = {}
_examples_per_label: int = 2
_seed: int = 42


def _build_question_examples(data, n_per_label, seed):
    by_question: dict[str, dict[str, list[dict]]] = {}
    for sample in data:
        qid = sample["question_id"]
        label = sample["score"]
        by_question.setdefault(qid, {}).setdefault(label, []).append(sample)

    result = {}
    rng = random.Random(seed)

    for qid, label_groups in by_question.items():
        examples = []
        for label in _LABEL_ORDER:
            samples = label_groups.get(label, [])
            picked = rng.sample(samples, min(n_per_label, len(samples)))
            for s in picked:
                examples.append({"id": s["id"], "answer": s["answer"], "score": s["score"]})
        label_rank = {label: i for i, label in enumerate(_LABEL_ORDER)}
        examples.sort(key=lambda ex: label_rank.get(ex["score"], 99))
        result[qid] = examples

    logger.info("Built example pools for %d questions (%d per label, seed=%d)", len(result), n_per_label, seed)
    return result


def configure(examples_per_label: int = 2, seed: int = 42):
    global _train_data, _question_examples, _examples_per_label, _seed
    _examples_per_label = examples_per_label
    _seed = seed
    _train_data = None
    _question_examples = {}


def _ensure_examples_loaded():
    global _train_data, _question_examples
    if _train_data is not None:
        return
    _train_data = load_train_3way()
    _question_examples = _build_question_examples(_train_data, _examples_per_label, _seed)


def score_sample(sample: dict) -> dict:
    _ensure_examples_loaded()

    question_id = sample["question_id"]
    sample_id = sample["id"]

    all_examples = _question_examples.get(question_id, [])
    examples = [ex for ex in all_examples if ex["id"] != sample_id]

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=sample["question"],
        answer=sample["answer"],
        rubric=sample["rubric"],
        examples=examples,
    )

    # Allow more tokens for structured output (criteria lists + score)
    raw_response = call_with_retry(system_prompt, user_prompt, max_tokens=500)
    result = parse_response(raw_response)
    result["n_examples"] = len(examples)
    return result
