"""
Score a single sample across multiple models and combine via majority vote.

Uses the C2 (tuned fewshot) prompt strategy but swaps the model for each call.
"""
import logging
from collections import Counter
from src.common.openrouter import call_with_retry, call_openrouter
from src.common.data_loader import load_train_3way
from src.strategy_c2_fewshot_tuned.prompt import build_system_prompt, build_user_prompt, parse_response

# Reuse C2's example selection
from src.strategy_c2_fewshot_tuned.scorer import (
    configure as c2_configure,
    _ensure_examples_loaded,
    _question_examples,
)

logger = logging.getLogger(__name__)

# Default ensemble models — all cheap, all latest versions
DEFAULT_MODELS = [
    "google/gemini-3-flash-preview",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-30b-a3b",
    "google/gemma-3-27b-it",
]

def configure(models: list[str] | None = None, examples_per_label: int = 3, seed: int = 42):
    """Configure ensemble models and example selection."""
    global _models
    _models = models or DEFAULT_MODELS
    c2_configure(examples_per_label=examples_per_label, seed=seed)

_models = DEFAULT_MODELS

def score_sample_single_model(sample: dict, model: str, examples: list[dict]) -> dict:
    """Score with a specific model. Returns {"model": ..., "score": ..., "confidence": ..., "error": ...}"""
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=sample["question"],
        answer=sample["answer"],
        rubric=sample["rubric"],
        examples=examples,
    )
    try:
        raw = call_with_retry(system_prompt, user_prompt, model=model)
        result = parse_response(raw)
        return {"model": model, "score": result["score"], "confidence": result.get("confidence"), "error": None}
    except Exception as e:
        logger.warning("Model %s failed on %s: %s", model, sample["id"], e)
        return {"model": model, "score": None, "confidence": None, "error": str(e)}

def score_sample_ensemble(sample: dict) -> dict:
    """
    Score a sample across all ensemble models. Returns majority-voted prediction.

    Returns dict with:
    - "score": majority-voted label
    - "confidence": fraction of models that agreed
    - "per_model": list of per-model results
    - "agreement": number of distinct scores across models
    - "n_examples": number of examples used
    """
    _ensure_examples_loaded()

    # Get examples (same for all models — fair comparison)
    all_examples = _question_examples.get(sample["question_id"], [])
    examples = [ex for ex in all_examples if ex["id"] != sample["id"]]

    # Score with each model
    per_model = []
    for model in _models:
        result = score_sample_single_model(sample, model, examples)
        per_model.append(result)

    # Majority vote (only from successful predictions)
    valid_scores = [r["score"] for r in per_model if r["score"] is not None]

    if not valid_scores:
        return {
            "score": "Incorrect",  # fallback
            "confidence": 0.0,
            "per_model": per_model,
            "agreement": 0,
            "n_examples": len(examples),
            "n_models": len(_models),
            "n_valid": 0,
        }

    vote_counts = Counter(valid_scores)
    winner, winner_count = vote_counts.most_common(1)[0]

    return {
        "score": winner,
        "confidence": winner_count / len(valid_scores),
        "per_model": per_model,
        "agreement": len(set(valid_scores)),
        "n_examples": len(examples),
        "n_models": len(_models),
        "n_valid": len(valid_scores),
    }
