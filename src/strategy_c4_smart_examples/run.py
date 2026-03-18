"""
Strategy C4: Smart Example Selection

Combines boundary-focused examples (hardest cases near decision boundaries)
with TF-IDF similarity retrieval (most similar training answers per label).

Usage:
    python -m src.strategy_c4_smart_examples.run [--split train|trial] [--workers 5] [--limit N] [--n-boundary 2] [--n-similar 1] [--seed 42] [--offset N]
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.common.data_loader import load_train_3way, load_trial_3way, load_test_3way
from src.common.batch_runner import compile_submission_from_predictions
from src.common.openrouter import get_model
from src.strategy_c4_smart_examples.scorer import configure, score_sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_c4"


def score_one(sample: dict) -> dict:
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
        logging.getLogger(__name__).error("Error scoring %s: %s", sample["id"], exc)
    return result


def compute_metrics(results: list[dict]) -> dict:
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
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": sum(1 for g in gold if g == label)}

    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / len(labels)
    total_support = sum(pc["support"] for pc in per_class.values())
    weighted_f1 = sum(pc["f1"] * pc["support"] / total_support for pc in per_class.values()) if total_support > 0 else 0.0

    n = len(labels)
    conf_matrix = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        conf_matrix[label_to_idx.get(g, 0)][label_to_idx.get(p, 0)] += 1

    weights = [[(( i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    row_sums = [sum(conf_matrix[i]) for i in range(n)]
    col_sums = [sum(conf_matrix[i][j] for i in range(n)) for j in range(n)]

    numerator = sum(weights[i][j] * conf_matrix[i][j] for i in range(n) for j in range(n))
    denominator = sum(weights[i][j] * row_sums[i] * col_sums[j] / len(valid) for i in range(n) for j in range(n))
    qwk = 1 - (numerator / denominator) if denominator > 0 else 0.0

    return {"n_scored": len(valid), "n_errors": len(results) - len(valid),
            "accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4), "qwk": round(qwk, 4), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description="Strategy C4: Smart Example Selection")
    parser.add_argument("--split", choices=["train", "trial", "test"], default="trial")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--n-boundary", type=int, default=2, help="Boundary examples per adjacent boundary")
    parser.add_argument("--n-similar", type=int, default=1, help="Similar examples per label via TF-IDF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("  Strategy C4: Smart Example Selection")
    print("=" * 70)
    print(f"  Split: {args.split} | Workers: {args.workers} | Offset: {args.offset} | Limit: {args.limit or 'all'}")
    print(f"  Boundary: {args.n_boundary} | Similar: {args.n_similar} | Seed: {args.seed} | Model: {get_model()}")
    print("=" * 70)

    configure(seed=args.seed, n_boundary=args.n_boundary, n_similar=args.n_similar)

    logger.info("Loading %s data...", args.split)
    if args.split == "test":
        data = load_test_3way()
    elif args.split == "trial":
        data = load_trial_3way()
    else:
        data = load_train_3way()
    data = data[args.offset:]
    if args.limit:
        data = data[:args.limit]

    logger.info("Scoring %d samples (offset=%d)", len(data), args.offset)

    results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, s): s for s in data}
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(data):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                errors = sum(1 for r in results if r["error"])
                print(f"  [{completed:>5d}/{len(data)}] {rate:.1f} samples/s | errors: {errors} | elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    id_order = {s["id"]: i for i, s in enumerate(data)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    metrics = compute_metrics(results)

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    if "error" not in metrics:
        print(f"  QWK: {metrics['qwk']:.4f} | Acc: {metrics['accuracy']:.4f} | wF1: {metrics['weighted_f1']:.4f}")
        print(f"  Scored: {metrics['n_scored']} | Errors: {metrics['n_errors']}")
        for label in ["Correct", "Partially correct", "Incorrect"]:
            pc = metrics["per_class"][label]
            print(f"    {label:>20s}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")
    else:
        print(f"  Error: {metrics['error']}")
    print(f"  Time: {elapsed_total:.1f}s")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = get_model().replace("/", "_")

    pred_path = RESULTS_DIR / f"predictions_{args.split}_{model_short}_{timestamp}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    meta = {"strategy": "C4: Smart Example Selection", "split": args.split, "model": get_model(),
            "offset": args.offset, "n_boundary": args.n_boundary, "n_similar": args.n_similar,
            "n_samples": len(data), "elapsed_seconds": round(elapsed_total, 2),
            "metrics": metrics}
    metrics_path = RESULTS_DIR / f"metrics_{args.split}_{model_short}_{timestamp}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Compile submission files
    for track in ("3way", "2way"):
        sub_path = RESULTS_DIR / f"submission_{args.split}_{track}_{timestamp}.json"
        compile_submission_from_predictions(results, sub_path, track=track)

    print(f"\n  Predictions: {pred_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Submissions: {RESULTS_DIR / f'submission_{args.split}_*.json'}")


if __name__ == "__main__":
    main()
