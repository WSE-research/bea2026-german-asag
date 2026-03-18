"""
Batch scoring runner for the BEA 2026 German ASAG shared task.

Provides concurrent execution of a scorer function across all samples,
with JSONL-based crash-safe output, resume support, progress logging,
and cost estimation. Designed for use with the OpenRouter API client.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Default cost assumptions (per 1M tokens) — Gemini Flash pricing
DEFAULT_INPUT_COST_PER_M = 0.10   # $/1M input tokens
DEFAULT_OUTPUT_COST_PER_M = 0.40  # $/1M output tokens


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> list[dict]:
    """Load a JSONL results file.

    Each line in the file should be a valid JSON object. Blank lines and
    lines that fail to parse are skipped with a warning.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dicts.
    """
    path = Path(path)
    if not path.exists():
        return []

    results = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL at line %d in %s", line_num, path)

    return results


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record as one line to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_m: float = DEFAULT_INPUT_COST_PER_M,
    output_cost_per_m: float = DEFAULT_OUTPUT_COST_PER_M,
) -> float:
    """Estimate cost in USD from token counts."""
    return (input_tokens / 1_000_000 * input_cost_per_m +
            output_tokens / 1_000_000 * output_cost_per_m)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    data: list[dict],
    scorer_fn: Callable[[dict], dict],
    output_path: str | Path,
    max_workers: int = 5,
    resume: bool = True,
    progress_every: int = 50,
    input_cost_per_m: float = DEFAULT_INPUT_COST_PER_M,
    output_cost_per_m: float = DEFAULT_OUTPUT_COST_PER_M,
) -> list[dict]:
    """Score all samples concurrently and write results as JSONL.

    Args:
        data: List of sample dicts. Each must have an ``"id"`` key.
        scorer_fn: Callable that takes a sample dict and returns a result dict
            with at least ``{"id": str, "question_id": str, "score": str}``.
            May also include ``"confidence"``, ``"feedback"``, ``"raw_response"``,
            ``"error"``, ``"input_tokens"``, ``"output_tokens"``.
        output_path: Path for the JSONL output file. Created if absent.
        max_workers: Number of concurrent threads.
        resume: If True and ``output_path`` exists, skip already-scored IDs.
        progress_every: Log progress every N completed samples.
        input_cost_per_m: Cost per 1M input tokens (USD).
        output_cost_per_m: Cost per 1M output tokens (USD).

    Returns:
        List of all result dicts (including previously completed ones on resume).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results and determine which IDs to skip
    existing_results: list[dict] = []
    done_ids: set[str] = set()

    if resume and output_path.exists():
        existing_results = load_results(output_path)
        done_ids = {r["id"] for r in existing_results}
        logger.info("Resuming: %d samples already scored", len(done_ids))

    remaining = [s for s in data if s["id"] not in done_ids]
    total = len(data)
    skipped = total - len(remaining)

    if not remaining:
        logger.info("All %d samples already scored. Nothing to do.", total)
        return existing_results

    logger.info(
        "Batch run: %d total, %d skipped (resume), %d to score, %d workers",
        total,
        skipped,
        len(remaining),
        max_workers,
    )

    # Tracking counters
    completed = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    new_results: list[dict] = []
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(scorer_fn, sample): sample
            for sample in remaining
        }

        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            sample_id = sample["id"]

            try:
                result = future.result()
            except Exception as exc:
                errors += 1
                logger.error("Error scoring sample %s: %s", sample_id, exc)
                result = {
                    "id": sample_id,
                    "question_id": sample.get("question_id", ""),
                    "score": "",
                    "error": str(exc),
                }

            # Accumulate token counts if provided
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)

            # Write immediately for crash safety
            _append_jsonl(output_path, result)
            new_results.append(result)
            completed += 1

            # Progress logging
            if completed % progress_every == 0 or completed == len(remaining):
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                cost = _estimate_cost(
                    total_input_tokens,
                    total_output_tokens,
                    input_cost_per_m,
                    output_cost_per_m,
                )
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.1f samples/s | "
                    "%.0fs elapsed | %d errors | ~$%.4f cost",
                    completed + skipped,
                    total,
                    (completed + skipped) / total * 100,
                    rate,
                    elapsed,
                    errors,
                    cost,
                )

    # Final summary
    elapsed = time.monotonic() - start_time
    cost = _estimate_cost(
        total_input_tokens,
        total_output_tokens,
        input_cost_per_m,
        output_cost_per_m,
    )
    logger.info(
        "Batch complete: %d scored, %d errors, %.1fs elapsed, ~$%.4f estimated cost",
        completed,
        errors,
        elapsed,
        cost,
    )

    return existing_results + new_results


# ---------------------------------------------------------------------------
# Submission compiler
# ---------------------------------------------------------------------------

def compile_submission(
    results_path: str | Path,
    output_path: str | Path,
    track: str = "3way",
) -> None:
    """Compile JSONL results into a submission JSON file.

    Args:
        results_path: Path to the JSONL results file.
        output_path: Path for the output submission JSON.
        track: ``"3way"`` (default) or ``"2way"``. For 2-way, maps
               ``"Partially correct"`` to ``"Incorrect"``.

    Raises:
        ValueError: If ``track`` is not ``"3way"`` or ``"2way"``.
        FileNotFoundError: If ``results_path`` does not exist.
    """
    if track not in ("3way", "2way"):
        raise ValueError(f"track must be '3way' or '2way', got '{track}'")

    results_path = Path(results_path)
    output_path = Path(output_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)

    submission = []
    for r in results:
        score = r.get("score", "")

        # Skip entries with errors (no valid score)
        if not score:
            logger.warning("Skipping sample %s: no score", r.get("id", "?"))
            continue

        if track == "2way" and score == "Partially correct":
            score = "Incorrect"

        submission.append({
            "id": r["id"],
            "question_id": r["question_id"],
            "score": score,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info(
        "Compiled submission: %d entries -> %s (track=%s)",
        len(submission),
        output_path,
        track,
    )


def compile_submission_from_predictions(
    predictions: list[dict],
    output_path: str | Path,
    track: str = "3way",
    score_key: str = "predicted_score",
) -> None:
    """Compile a list of prediction dicts into a submission JSON file.

    Works with the prediction format used by C2/C3/C4/ensemble strategies.

    Args:
        predictions: List of prediction dicts with ``id``, ``question_id``,
            and a score field (default key: ``predicted_score``).
        output_path: Path for the output submission JSON.
        track: ``"3way"`` (default) or ``"2way"``.
        score_key: Key name for the predicted score in each dict.
    """
    if track not in ("3way", "2way"):
        raise ValueError(f"track must be '3way' or '2way', got '{track}'")

    output_path = Path(output_path)
    submission = []
    for r in predictions:
        score = r.get(score_key, "")
        if not score:
            logger.warning("Skipping sample %s: no score", r.get("id", "?"))
            continue
        if track == "2way" and score == "Partially correct":
            score = "Incorrect"
        submission.append({
            "id": r["id"],
            "question_id": r["question_id"],
            "score": score,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info(
        "Compiled submission: %d entries -> %s (track=%s)",
        len(submission),
        output_path,
        track,
    )
