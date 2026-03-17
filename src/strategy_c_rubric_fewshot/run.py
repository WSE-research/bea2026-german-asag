"""
Strategy C: Rubric + Few-Shot Examples Scoring

Scores student answers using the textual rubric AND concrete labeled
examples from the same question drawn from the training set. For each
answer, the prompt includes up to N examples per score level (default 2,
for 6 total) alongside the rubric, enabling the LLM to calibrate its
scoring against established benchmarks.

Usage:
    python -m src.strategy_c_rubric_fewshot.run [--split train|trial] [--workers 5] [--limit N] [--examples-per-label 2] [--seed 42]

Examples:
    # Score the trial split with defaults (2 examples per label, 5 workers)
    python -m src.strategy_c_rubric_fewshot.run --split trial

    # Score first 50 training samples with 3 examples per label
    python -m src.strategy_c_rubric_fewshot.run --split train --limit 50 --examples-per-label 3

    # Reproducible run with custom seed
    python -m src.strategy_c_rubric_fewshot.run --split trial --seed 123
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.common.data_loader import load_train_3way, load_trial_3way
from src.common.openrouter import get_model
from src.strategy_c_rubric_fewshot.scorer import configure, score_sample

# Project root: three levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_c"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the run."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def score_one(sample: dict) -> dict:
    """Score a single sample and return the result with metadata.

    Wraps ``score_sample`` with error handling so a single failure
    does not abort the entire run.

    Returns:
        Dict with ``id``, ``question_id``, ``predicted_score``,
        ``confidence``, ``n_examples``, ``gold_score`` (if available),
        and ``error`` (if scoring failed).
    """
    result = {
        "id": sample["id"],
        "question_id": sample["question_id"],
        "predicted_score": None,
        "confidence": None,
        "n_examples": 0,
        "gold_score": sample.get("score"),
        "error": None,
    }
    try:
        pred = score_sample(sample)
        result["predicted_score"] = pred["score"]
        result["confidence"] = pred["confidence"]
        result["n_examples"] = pred.get("n_examples", 0)
    except Exception as exc:
        result["error"] = str(exc)
        logging.getLogger(__name__).error(
            "Error scoring sample %s: %s", sample["id"], exc
        )
    return result


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics from scored results.

    Calculates accuracy, per-class precision/recall/F1, macro-averaged
    F1, weighted F1, and Quadratic Weighted Kappa (QWK).

    Args:
        results: List of result dicts with ``predicted_score`` and ``gold_score``.

    Returns:
        Dict with computed metrics.
    """
    # Filter to results that have both gold and predicted scores
    valid = [r for r in results if r["predicted_score"] and r["gold_score"]]
    if not valid:
        return {"error": "No valid predictions to evaluate"}

    gold = [r["gold_score"] for r in valid]
    pred = [r["predicted_score"] for r in valid]

    labels = ["Correct", "Partially correct", "Incorrect"]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Accuracy
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    accuracy = correct / len(valid)

    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }

    # Macro F1
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / len(labels)

    # Weighted F1
    total_support = sum(pc["support"] for pc in per_class.values())
    weighted_f1 = sum(
        pc["f1"] * pc["support"] / total_support
        for pc in per_class.values()
    ) if total_support > 0 else 0.0

    # Quadratic Weighted Kappa (QWK)
    n = len(labels)
    # Build confusion matrix
    conf_matrix = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        gi = label_to_idx.get(g, 0)
        pi = label_to_idx.get(p, 0)
        conf_matrix[gi][pi] += 1

    total = len(valid)
    # Weight matrix (quadratic)
    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            weights[i][j] = ((i - j) ** 2) / ((n - 1) ** 2)

    # Expected matrix (outer product of marginals)
    row_sums = [sum(conf_matrix[i]) for i in range(n)]
    col_sums = [sum(conf_matrix[i][j] for i in range(n)) for j in range(n)]

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            expected = row_sums[i] * col_sums[j] / total if total > 0 else 0
            numerator += weights[i][j] * conf_matrix[i][j]
            denominator += weights[i][j] * expected

    qwk = 1 - (numerator / denominator) if denominator > 0 else 0.0

    return {
        "n_scored": len(valid),
        "n_errors": len(results) - len(valid),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "qwk": round(qwk, 4),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy C: Rubric + Few-Shot Examples Scoring"
    )
    parser.add_argument(
        "--split",
        choices=["train", "trial"],
        default="trial",
        help="Which data split to score (default: trial)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers for API calls (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=2,
        help="Number of few-shot examples per score level (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible example selection (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Strategy C: Rubric + Few-Shot Examples Scoring")
    print("=" * 70)
    print(f"  Split:              {args.split}")
    print(f"  Workers:            {args.workers}")
    print(f"  Limit:              {args.limit or 'all'}")
    print(f"  Examples per label: {args.examples_per_label}")
    print(f"  Seed:               {args.seed}")
    print(f"  Model:              {get_model()}")
    print("=" * 70)
    print()

    # ── Configure scorer ────────────────────────────────────────────────
    configure(examples_per_label=args.examples_per_label, seed=args.seed)

    # ── Load data ───────────────────────────────────────────────────────
    logger.info("Loading %s data...", args.split)
    if args.split == "train":
        data = load_train_3way()
    else:
        data = load_trial_3way()

    if args.limit:
        data = data[: args.limit]

    logger.info("Scoring %d samples with %d workers", len(data), args.workers)

    # ── Score samples ───────────────────────────────────────────────────
    results: list[dict] = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, sample): sample for sample in data}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress reporting every 25 samples
            if completed % 25 == 0 or completed == len(data):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                errors = sum(1 for r in results if r["error"])
                print(
                    f"  [{completed:>5d}/{len(data)}] "
                    f"{rate:.1f} samples/s | "
                    f"errors: {errors} | "
                    f"elapsed: {elapsed:.1f}s"
                )

    elapsed_total = time.time() - t_start

    # ── Sort results by original order ──────────────────────────────────
    id_order = {sample["id"]: i for i, sample in enumerate(data)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    # ── Evaluate ────────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    print()
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    if "error" not in metrics:
        print(f"  Samples scored:    {metrics['n_scored']}")
        print(f"  Errors:            {metrics['n_errors']}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
        print(f"  QWK:               {metrics['qwk']:.4f}")
        print()
        print("  Per-class metrics:")
        for label in ["Correct", "Partially correct", "Incorrect"]:
            pc = metrics["per_class"][label]
            print(
                f"    {label:>20s}:  P={pc['precision']:.3f}  "
                f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
                f"(n={pc['support']})"
            )
    else:
        print(f"  Error: {metrics['error']}")
    print(f"\n  Total time: {elapsed_total:.1f}s")
    print("=" * 70)

    # ── Save results ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = get_model().replace("/", "_")

    # Predictions file
    predictions_path = RESULTS_DIR / f"predictions_{args.split}_{model_short}_{timestamp}.json"
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Predictions saved to %s", predictions_path)

    # Metrics file
    run_meta = {
        "strategy": "C: Rubric + Few-Shot",
        "split": args.split,
        "model": get_model(),
        "examples_per_label": args.examples_per_label,
        "seed": args.seed,
        "n_samples": len(data),
        "n_workers": args.workers,
        "elapsed_seconds": round(elapsed_total, 2),
        "timestamp": timestamp,
        "metrics": metrics,
    }
    metrics_path = RESULTS_DIR / f"metrics_{args.split}_{model_short}_{timestamp}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    print(f"\n  Predictions: {predictions_path}")
    print(f"  Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
