"""
Data loading utilities for the ALICE-LP German ASAG dataset.

Provides functions to load training/trial splits, extract question metadata
with rubrics, and sample labeled examples per question for few-shot prompts.
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: two levels up from this file (src/common/data_loader.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
TRAIN_PATH = DATA_DIR / "ALICE_LP_train_3way__v2.json"
TRIAL_PATH = DATA_DIR / "ALICE_LP_trial_3way__v2.json"


def load_alice_data(path: str | Path) -> list[dict]:
    """Load an ALICE-LP JSON file.

    Args:
        path: Absolute or relative path to the JSON file.

    Returns:
        List of sample dicts, each containing at least:
        ``id``, ``question_id``, ``question``, ``answer``, ``rubric``, ``score``.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d samples from %s", len(data), path.name)
    return data


def load_train_3way() -> list[dict]:
    """Load the 3-way training split from the standard location.

    Returns:
        List of training sample dicts.
    """
    return load_alice_data(TRAIN_PATH)


def load_trial_3way() -> list[dict]:
    """Load the 3-way trial split from the standard location.

    Returns:
        List of trial sample dicts.
    """
    return load_alice_data(TRIAL_PATH)


def get_questions(data: list[dict]) -> dict[str, dict]:
    """Extract unique questions with their rubrics and grouped samples.

    Args:
        data: List of sample dicts (as returned by ``load_alice_data``).

    Returns:
        Dict mapping ``question_id`` to::

            {
                "text": str,           # the question text
                "rubric": dict,        # {label: description}
                "samples": list[dict], # all samples for this question
            }
    """
    questions: dict[str, dict] = {}

    for sample in data:
        qid = sample["question_id"]
        if qid not in questions:
            questions[qid] = {
                "text": sample["question"],
                "rubric": sample.get("rubric", {}),
                "samples": [],
            }
        questions[qid]["samples"].append(sample)

    logger.info(
        "Extracted %d unique questions from %d samples",
        len(questions),
        len(data),
    )
    return questions


def get_question_examples(
    data: list[dict],
    question_id: str,
    n: int = 3,
    per_label: bool = True,
) -> list[dict]:
    """Get example answers for a specific question, suitable for few-shot prompts.

    Args:
        data: List of sample dicts.
        question_id: The question to retrieve examples for.
        n: Number of examples to return. If ``per_label`` is True, returns up
           to ``n`` examples *per label* (up to 3n total for 3-way).
        per_label: If True, stratify by label so each class is represented.

    Returns:
        List of ``{"answer": str, "score": str}`` dicts. If ``per_label`` is
        True, the list contains up to ``n`` examples per label, shuffled.
        Otherwise, ``n`` random examples from the question.
    """
    question_samples = [s for s in data if s["question_id"] == question_id]

    if not question_samples:
        logger.warning("No samples found for question_id=%s", question_id)
        return []

    if per_label:
        by_label: dict[str, list[dict]] = {}
        for sample in question_samples:
            label = sample["score"]
            by_label.setdefault(label, []).append(sample)

        examples: list[dict] = []
        for label, samples in by_label.items():
            picked = random.sample(samples, min(n, len(samples)))
            for s in picked:
                examples.append({"answer": s["answer"], "score": s["score"]})

        random.shuffle(examples)
    else:
        picked = random.sample(question_samples, min(n, len(question_samples)))
        examples = [{"answer": s["answer"], "score": s["score"]} for s in picked]

    logger.debug(
        "Selected %d examples for question %s (per_label=%s)",
        len(examples),
        question_id,
        per_label,
    )
    return examples
