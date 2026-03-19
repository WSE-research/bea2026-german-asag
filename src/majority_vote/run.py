"""
Multi-Model Majority Vote Scoring

Runs C2 prompt across multiple cheap LLMs and combines via majority vote
(simple plurality voting, not learned ensemble aggregation).

Usage:
    python -m src.majority_vote.run [--split trial] [--workers 2] [--limit N] [--models model1,model2,...]
"""

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.common.data_loader import load_train_3way, load_trial_3way, load_test_3way
from src.common.batch_runner import compile_submission_from_predictions
from src.majority_vote.multi_model_scorer import (
    configure,
    score_sample_majority_vote,
    DEFAULT_MODELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "majority_vote"


def score_one(sample: dict) -> dict:
    """Score a single sample with the full majority vote."""
    result = {
        "id": sample["id"],
        "question_id": sample["question_id"],
        "gold_score": sample.get("score"),
        "predicted_score": None,
        "confidence": None,
        "agreement": None,
        "n_examples": 0,
        "n_models": 0,
        "n_valid": 0,
        "per_model": [],
        "error": None,
    }
    try:
        pred = score_sample_majority_vote(sample)
        result["predicted_score"] = pred["score"]
        result["confidence"] = pred["confidence"]
        result["agreement"] = pred["agreement"]
        result["n_examples"] = pred["n_examples"]
        result["n_models"] = pred["n_models"]
        result["n_valid"] = pred["n_valid"]
        result["per_model"] = pred["per_model"]
    except Exception as exc:
        result["error"] = str(exc)
        logging.getLogger(__name__).error("Error scoring %s: %s", sample["id"], exc)
    return result


def compute_metrics(results: list[dict], pred_key: str = "predicted_score") -> dict:
    """Compute QWK, accuracy, and per-class metrics.

    Args:
        results: List of result dicts.
        pred_key: Key to use as the predicted score (allows reuse for per-model metrics).
    """
    valid = [r for r in results if r.get(pred_key) and r.get("gold_score")]
    if not valid:
        return {"error": "No valid predictions"}

    gold = [r["gold_score"] for r in valid]
    pred = [r[pred_key] for r in valid]
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
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
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

    weights = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    row_sums = [sum(conf_matrix[i]) for i in range(n)]
    col_sums = [sum(conf_matrix[i][j] for i in range(n)) for j in range(n)]

    numerator = sum(weights[i][j] * conf_matrix[i][j] for i in range(n) for j in range(n))
    denominator = sum(
        weights[i][j] * row_sums[i] * col_sums[j] / len(valid) for i in range(n) for j in range(n)
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


def compute_per_model_metrics(results: list[dict], models: list[str]) -> dict[str, dict]:
    """Compute metrics for each individual model from the per_model data."""
    per_model_metrics = {}

    for model in models:
        # Build virtual result rows: one per sample, using this model's prediction
        virtual_results = []
        for r in results:
            model_pred = None
            for pm in r.get("per_model", []):
                if pm["model"] == model and pm["score"] is not None:
                    model_pred = pm["score"]
                    break
            virtual_results.append({
                "gold_score": r.get("gold_score"),
                "predicted_score": model_pred,
            })

        per_model_metrics[model] = compute_metrics(virtual_results)

    return per_model_metrics


def compute_agreement_stats(results: list[dict]) -> dict:
    """Compute agreement statistics across the multi-model vote."""
    valid = [r for r in results if r.get("predicted_score")]
    if not valid:
        return {}

    agreements = [r["agreement"] for r in valid if r.get("agreement") is not None]
    confidences = [r["confidence"] for r in valid if r.get("confidence") is not None]

    unanimous = sum(1 for a in agreements if a == 1)
    total = len(agreements)

    return {
        "n_samples": total,
        "unanimous": unanimous,
        "unanimous_pct": round(unanimous / total * 100, 1) if total > 0 else 0.0,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "agreement_distribution": dict(Counter(agreements)),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Majority Vote Scoring")
    parser.add_argument("--split", choices=["train", "trial", "test"], default="trial")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model list (overrides defaults)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Parse models
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    active_models = models or DEFAULT_MODELS

    print("=" * 70)
    print("  Multi-Model Majority Vote Scoring")
    print("=" * 70)
    print(f"  Split: {args.split} | Workers: {args.workers} | Offset: {args.offset} | Limit: {args.limit or 'all'}")
    print(f"  Examples/label: {args.examples_per_label} | Seed: {args.seed}")
    print(f"  Models ({len(active_models)}):")
    for m in active_models:
        print(f"    - {m}")
    print("=" * 70)

    configure(models=models, examples_per_label=args.examples_per_label, seed=args.seed)

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

    logger.info("Scoring %d samples (offset=%d) with %d models each", len(data), args.offset, len(active_models))

    results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, s): s for s in data}
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(data):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                errors = sum(1 for r in results if r["error"])
                print(
                    f"  [{completed:>5d}/{len(data)}] {rate:.1f} samples/s | "
                    f"errors: {errors} | elapsed: {elapsed:.1f}s"
                )

    elapsed_total = time.time() - t_start
    id_order = {s["id"]: i for i, s in enumerate(data)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    # --- Compute all metrics ---
    ensemble_metrics = compute_metrics(results)
    per_model_metrics = compute_per_model_metrics(results, active_models)
    agreement_stats = compute_agreement_stats(results)

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("  Model Comparison")
    print("=" * 70)
    print(f"  {'Source':<40s}  {'QWK':>6s}  {'Acc':>6s}  {'wF1':>6s}  {'N':>4s}")
    print("  " + "-" * 66)

    for model in active_models:
        m = per_model_metrics.get(model, {})
        if "error" in m:
            print(f"  {model:<40s}  {'ERR':>6s}  {'ERR':>6s}  {'ERR':>6s}  {'0':>4s}")
        else:
            print(
                f"  {model:<40s}  {m['qwk']:>6.4f}  {m['accuracy']:>6.4f}  "
                f"{m['weighted_f1']:>6.4f}  {m['n_scored']:>4d}"
            )

    print("  " + "-" * 66)
    if "error" not in ensemble_metrics:
        print(
            f"  {'>> MAJORITY VOTE (multi-model) <<':<40s}  {ensemble_metrics['qwk']:>6.4f}  "
            f"{ensemble_metrics['accuracy']:>6.4f}  {ensemble_metrics['weighted_f1']:>6.4f}  "
            f"{ensemble_metrics['n_scored']:>4d}"
        )
    else:
        print(f"  {'>> MAJORITY VOTE <<':<40s}  Error: {ensemble_metrics['error']}")

    # --- Print majority vote detail ---
    print("\n" + "=" * 70)
    print("  Majority Vote Results (Detail)")
    print("=" * 70)
    if "error" not in ensemble_metrics:
        print(
            f"  QWK: {ensemble_metrics['qwk']:.4f} | Acc: {ensemble_metrics['accuracy']:.4f} | "
            f"wF1: {ensemble_metrics['weighted_f1']:.4f}"
        )
        print(f"  Scored: {ensemble_metrics['n_scored']} | Errors: {ensemble_metrics['n_errors']}")
        for label in ["Correct", "Partially correct", "Incorrect"]:
            pc = ensemble_metrics["per_class"][label]
            print(
                f"    {label:>20s}: P={pc['precision']:.3f} R={pc['recall']:.3f} "
                f"F1={pc['f1']:.3f} (n={pc['support']})"
            )
    else:
        print(f"  Error: {ensemble_metrics['error']}")

    # --- Print agreement stats ---
    print("\n" + "=" * 70)
    print("  Agreement Statistics")
    print("=" * 70)
    if agreement_stats:
        print(f"  Unanimous agreement: {agreement_stats['unanimous']}/{agreement_stats['n_samples']} ({agreement_stats['unanimous_pct']}%)")
        print(f"  Mean vote confidence: {agreement_stats['mean_confidence']:.4f}")
        dist = agreement_stats["agreement_distribution"]
        for n_distinct, count in sorted(dist.items()):
            print(f"    {n_distinct} distinct score(s): {count} samples")

    print(f"\n  Time: {elapsed_total:.1f}s")
    print("=" * 70)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save full predictions (with per-model detail)
    pred_path = RESULTS_DIR / f"predictions_{args.split}_ensemble_{timestamp}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save per-model individual predictions for analysis
    per_model_preds = {}
    for model in active_models:
        model_results = []
        for r in results:
            model_pred = None
            for pm in r.get("per_model", []):
                if pm["model"] == model:
                    model_pred = pm["score"]
                    break
            model_results.append({
                "id": r["id"],
                "question_id": r["question_id"],
                "gold_score": r["gold_score"],
                "predicted_score": model_pred,
            })
        per_model_preds[model] = model_results

    per_model_path = RESULTS_DIR / f"per_model_predictions_{args.split}_{timestamp}.json"
    with per_model_path.open("w", encoding="utf-8") as f:
        json.dump(per_model_preds, f, ensure_ascii=False, indent=2)

    # Save metrics
    meta = {
        "strategy": "Multi-Model Majority Vote",
        "split": args.split,
        "models": active_models,
        "n_models": len(active_models),
        "offset": args.offset,
        "n_samples": len(data),
        "examples_per_label": args.examples_per_label,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_total, 2),
        "ensemble_metrics": ensemble_metrics,
        "per_model_metrics": per_model_metrics,
        "agreement_stats": agreement_stats,
    }
    metrics_path = RESULTS_DIR / f"metrics_{args.split}_ensemble_{timestamp}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Compile submission files
    for track in ("3way", "2way"):
        sub_path = RESULTS_DIR / f"submission_{args.split}_{track}_{timestamp}.json"
        compile_submission_from_predictions(results, sub_path, track=track)

    print(f"\n  Predictions:       {pred_path}")
    print(f"  Per-model preds:   {per_model_path}")
    print(f"  Metrics:           {metrics_path}")
    print(f"  Submissions:       {RESULTS_DIR / f'submission_{args.split}_*.json'}")


if __name__ == "__main__":
    main()
