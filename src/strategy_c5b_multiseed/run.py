"""
Strategy C5b: Multi-Seed Majority Vote (Self-Consistency)

Runs C4's scorer 3 times with different random seeds, then majority-votes
the per-sample predictions. This is majority voting across runs of the same
model with different example selections — NOT learned ensemble aggregation.
Because ``configure()`` resets the internal SmartExampleSelector, each seed
requires a full sequential pass over all samples before moving to the next seed.

Usage:
    python -m src.strategy_c5b_multiseed.run [--split trial|train|test] \
        [--workers 5] [--limit N] [--offset N] \
        [--n-boundary 2] [--n-similar 2] [--verbose]
"""

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.common.batch_runner import compile_submission_from_predictions
from src.common.data_loader import load_test_3way, load_train_3way, load_trial_3way
from src.common.openrouter import get_model
from src.strategy_c4_smart_examples.scorer import configure, score_sample

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_c5b"

SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_one(sample: dict) -> dict:
    """Score a single sample using the currently-configured C4 scorer."""
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
        logger.error("Error scoring %s: %s", sample["id"], exc)
    return result


def run_seed_pass(
    seed: int,
    data: list[dict],
    n_boundary: int,
    n_similar: int,
    workers: int,
) -> dict[str, dict]:
    """Run a full scoring pass over *data* with the given seed.

    Returns a dict mapping sample-id -> result dict.
    """
    configure(seed=seed, n_boundary=n_boundary, n_similar=n_similar)

    results: dict[str, dict] = {}
    completed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_one, s): s for s in data}
        for future in as_completed(futures):
            res = future.result()
            results[res["id"]] = res
            completed += 1
            if completed % 25 == 0 or completed == len(data):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                errors = sum(1 for r in results.values() if r["error"])
                print(
                    f"    [{completed:>5d}/{len(data)}] {rate:.1f} samples/s "
                    f"| errors: {errors} | elapsed: {elapsed:.1f}s"
                )

    return results


def majority_vote(
    per_seed_results: dict[int, dict[str, dict]],
    sample_ids: list[str],
) -> list[dict]:
    """Majority-vote across seed passes.

    Returns a list of voted prediction dicts (one per sample, ordered by
    *sample_ids*) with per-seed detail embedded.
    """
    voted_results: list[dict] = []
    for sid in sample_ids:
        seed_preds: dict[int, str | None] = {}
        seed_details: dict[str, dict] = {}
        for seed in SEEDS:
            res = per_seed_results[seed].get(sid, {})
            seed_preds[seed] = res.get("predicted_score")
            seed_details[str(seed)] = {
                "predicted_score": res.get("predicted_score"),
                "confidence": res.get("confidence"),
                "n_examples": res.get("n_examples", 0),
                "error": res.get("error"),
            }

        # Majority vote (only over non-None predictions)
        valid_preds = [p for p in seed_preds.values() if p is not None]
        if len(valid_preds) >= 2:
            counter = Counter(valid_preds)
            most_common_score, most_common_count = counter.most_common(1)[0]
            if most_common_count >= 2:
                voted_score = most_common_score
                agreement = "majority" if most_common_count == 2 else "unanimous"
            else:
                # All 3 disagree — tiebreaker: seed 42
                voted_score = seed_preds[42]
                agreement = "split"
        elif len(valid_preds) == 1:
            voted_score = valid_preds[0]
            agreement = "single"
        else:
            voted_score = None
            agreement = "none"

        # Check if all 3 agree
        if len(valid_preds) == 3 and len(set(valid_preds)) == 1:
            agreement = "unanimous"

        # Grab gold from any seed's result
        any_res = per_seed_results[SEEDS[0]].get(sid, {})
        voted_results.append({
            "id": sid,
            "question_id": any_res.get("question_id"),
            "predicted_score": voted_score,
            "gold_score": any_res.get("gold_score"),
            "agreement": agreement,
            "per_seed": seed_details,
        })

    return voted_results


def compute_metrics(results: list[dict]) -> dict:
    """Compute classification metrics (copied from C4 for self-containment)."""
    valid = [r for r in results if r["predicted_score"] and r["gold_score"]]
    if not valid:
        return {"error": "No valid predictions"}

    gold = [r["gold_score"] for r in valid]
    pred = [r["predicted_score"] for r in valid]
    labels = ["Correct", "Partially correct", "Incorrect"]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    accuracy = correct / len(valid)

    per_class = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }

    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / len(labels)
    total_support = sum(pc["support"] for pc in per_class.values())
    weighted_f1 = (
        sum(pc["f1"] * pc["support"] / total_support for pc in per_class.values())
        if total_support > 0
        else 0.0
    )

    n = len(labels)
    conf_matrix = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        conf_matrix[label_to_idx.get(g, 0)][label_to_idx.get(p, 0)] += 1

    weights = [
        [((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)
    ]
    row_sums = [sum(conf_matrix[i]) for i in range(n)]
    col_sums = [sum(conf_matrix[i][j] for i in range(n)) for j in range(n)]

    numerator = sum(
        weights[i][j] * conf_matrix[i][j] for i in range(n) for j in range(n)
    )
    denominator = sum(
        weights[i][j] * row_sums[i] * col_sums[j] / len(valid)
        for i in range(n)
        for j in range(n)
    )
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


def compute_agreement_stats(voted_results: list[dict]) -> dict:
    """Count agreement categories across the multi-seed vote."""
    counts: Counter = Counter()
    for item in voted_results:
        counts[item["agreement"]] += 1
    total = len(voted_results)
    return {
        "total": total,
        "unanimous": counts.get("unanimous", 0),
        "majority": counts.get("majority", 0),
        "split": counts.get("split", 0),
        "single": counts.get("single", 0),
        "none": counts.get("none", 0),
        "unanimous_pct": round(counts.get("unanimous", 0) / total * 100, 1) if total else 0,
        "majority_pct": round(counts.get("majority", 0) / total * 100, 1) if total else 0,
        "split_pct": round(counts.get("split", 0) / total * 100, 1) if total else 0,
    }


# ---------------------------------------------------------------------------
# Per-seed metrics helper
# ---------------------------------------------------------------------------

def results_to_list(seed_results: dict[str, dict], sample_ids: list[str]) -> list[dict]:
    """Convert a seed's results dict into an ordered list for metrics."""
    return [seed_results[sid] for sid in sample_ids if sid in seed_results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategy C5b: Multi-Seed Majority Vote"
    )
    parser.add_argument("--split", choices=["train", "trial", "test"], default="trial")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument(
        "--n-boundary", type=int, default=2,
        help="Boundary examples per adjacent boundary",
    )
    parser.add_argument(
        "--n-similar", type=int, default=2,
        help="Similar examples per label via TF-IDF",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("  Strategy C5b: Multi-Seed Majority Vote")
    print("=" * 70)
    print(
        f"  Split: {args.split} | Workers: {args.workers} "
        f"| Offset: {args.offset} | Limit: {args.limit or 'all'}"
    )
    print(
        f"  Boundary: {args.n_boundary} | Similar: {args.n_similar} "
        f"| Seeds: {SEEDS} | Model: {get_model()}"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading %s data...", args.split)
    if args.split == "test":
        data = load_test_3way()
    elif args.split == "trial":
        data = load_trial_3way()
    else:
        data = load_train_3way()
    data = data[args.offset:]
    if args.limit:
        data = data[: args.limit]

    sample_ids = [s["id"] for s in data]
    logger.info("Scoring %d samples (offset=%d)", len(data), args.offset)

    # ------------------------------------------------------------------
    # Run 3 seed passes sequentially
    # ------------------------------------------------------------------
    per_seed_results: dict[int, dict[str, dict]] = {}
    t_start = time.time()

    for seed in SEEDS:
        print(f"\n{'─' * 70}")
        print(f"  Seed {seed} — scoring {len(data)} samples")
        print(f"{'─' * 70}")
        per_seed_results[seed] = run_seed_pass(
            seed=seed,
            data=data,
            n_boundary=args.n_boundary,
            n_similar=args.n_similar,
            workers=args.workers,
        )

    elapsed_total = time.time() - t_start

    # ------------------------------------------------------------------
    # Per-seed metrics
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  Per-Seed Metrics")
    print("=" * 70)
    per_seed_metrics: dict[str, dict] = {}
    for seed in SEEDS:
        ordered = results_to_list(per_seed_results[seed], sample_ids)
        m = compute_metrics(ordered)
        per_seed_metrics[str(seed)] = m
        if "error" not in m:
            print(
                f"  Seed {seed:>3d}: QWK={m['qwk']:.4f} | "
                f"Acc={m['accuracy']:.4f} | wF1={m['weighted_f1']:.4f} | "
                f"Scored={m['n_scored']} Errors={m['n_errors']}"
            )
        else:
            print(f"  Seed {seed:>3d}: {m['error']}")

    # ------------------------------------------------------------------
    # Majority vote across seeds
    # ------------------------------------------------------------------
    voted = majority_vote(per_seed_results, sample_ids)
    vote_metrics = compute_metrics(voted)
    agreement_stats = compute_agreement_stats(voted)

    print(f"\n{'=' * 70}")
    print("  Majority Vote Results (multi-seed)")
    print("=" * 70)
    if "error" not in vote_metrics:
        print(
            f"  QWK: {vote_metrics['qwk']:.4f} | "
            f"Acc: {vote_metrics['accuracy']:.4f} | "
            f"wF1: {vote_metrics['weighted_f1']:.4f}"
        )
        print(
            f"  Scored: {vote_metrics['n_scored']} | "
            f"Errors: {vote_metrics['n_errors']}"
        )
        for label in ["Correct", "Partially correct", "Incorrect"]:
            pc = vote_metrics["per_class"][label]
            print(
                f"    {label:>20s}: P={pc['precision']:.3f} "
                f"R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})"
            )
    else:
        print(f"  Error: {vote_metrics['error']}")

    print(f"\n  Agreement:")
    print(
        f"    Unanimous: {agreement_stats['unanimous']} "
        f"({agreement_stats['unanimous_pct']}%)"
    )
    print(
        f"    Majority:  {agreement_stats['majority']} "
        f"({agreement_stats['majority_pct']}%)"
    )
    print(
        f"    Split:     {agreement_stats['split']} "
        f"({agreement_stats['split_pct']}%)"
    )
    print(f"  Total time: {elapsed_total:.1f}s")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Predictions (voted results with per-seed detail)
    pred_path = RESULTS_DIR / f"predictions_{args.split}_voted_{timestamp}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(voted, f, ensure_ascii=False, indent=2)

    # Metrics
    meta = {
        "strategy": "C5b: Multi-Seed Majority Vote",
        "split": args.split,
        "model": get_model(),
        "seeds": SEEDS,
        "offset": args.offset,
        "n_boundary": args.n_boundary,
        "n_similar": args.n_similar,
        "n_samples": len(data),
        "elapsed_seconds": round(elapsed_total, 2),
        "vote_metrics": vote_metrics,
        "per_seed_metrics": per_seed_metrics,
        "agreement": agreement_stats,
    }
    metrics_path = RESULTS_DIR / f"metrics_{args.split}_voted_{timestamp}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Submission files
    for track in ("3way", "2way"):
        sub_path = RESULTS_DIR / f"submission_{args.split}_{track}_{timestamp}.json"
        compile_submission_from_predictions(voted, sub_path, track=track)

    print(f"\n  Predictions: {pred_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Submissions: {RESULTS_DIR / f'submission_{args.split}_*_{timestamp}.json'}")


if __name__ == "__main__":
    main()
