"""
Strategy C6: Claude Sonnet with Adaptive Difficulty + Full Metadata Logging

Same scoring logic as C5c (adaptive difficulty, smart example selection),
but targeting Claude Sonnet 4.6 via OpenRouter with comprehensive per-request
metadata capture for scientific reproducibility.

Usage:
    OPENROUTER_MODEL=anthropic/claude-sonnet-4.6 python -m src.strategy_c6_claude_sonnet.run --split trial --limit 10 --workers 1 --seed 42 --verbose
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
from src.common.openrouter import get_model
from src.strategy_c6_claude_sonnet.scorer import configure, score_sample, get_difficulty

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_c6"


def score_one(sample: dict) -> dict:
    result = {
        "id": sample["id"],
        "question_id": sample["question_id"],
        "predicted_score": None,
        "confidence": None,
        "n_examples": 0,
        "difficulty_tier": None,
        "n_boundary": None,
        "n_similar": None,
        "gold_score": sample.get("score"),
        "error": None,
        # Metadata fields (populated on success)
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "inline_cost": None,
        "total_cost_usd": None,
        "generation_id": None,
        "model_used": None,
        "provider_name": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "latency_ms": None,
        "generation_time_ms": None,
        "wall_clock_seconds": None,
        "native_tokens_prompt": None,
        "native_tokens_completion": None,
        "native_tokens_reasoning": None,
        "native_tokens_cached": None,
        "finish_reason": None,
        "cache_discount": None,
        "system_prompt_chars": None,
        "user_prompt_chars": None,
    }
    try:
        pred = score_sample(sample)
        result["predicted_score"] = pred["score"]
        result["confidence"] = pred["confidence"]
        result["n_examples"] = pred.get("n_examples", 0)
        result["difficulty_tier"] = pred.get("difficulty_tier")
        result["n_boundary"] = pred.get("n_boundary")
        result["n_similar"] = pred.get("n_similar")
        # Copy all metadata fields
        for key in [
            "prompt_tokens", "completion_tokens", "total_tokens",
            "inline_cost", "total_cost_usd", "generation_id", "model_used",
            "provider_name", "reasoning_tokens", "cached_tokens",
            "latency_ms", "generation_time_ms", "wall_clock_seconds",
            "native_tokens_prompt", "native_tokens_completion",
            "native_tokens_reasoning", "native_tokens_cached",
            "finish_reason", "cache_discount",
            "system_prompt_chars", "user_prompt_chars",
        ]:
            if key in pred:
                result[key] = pred[key]
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

    weights = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    row_sums = [sum(conf_matrix[i]) for i in range(n)]
    col_sums = [sum(conf_matrix[i][j] for i in range(n)) for j in range(n)]

    numerator = sum(weights[i][j] * conf_matrix[i][j] for i in range(n) for j in range(n))
    denominator = sum(weights[i][j] * row_sums[i] * col_sums[j] / len(valid) for i in range(n) for j in range(n))
    qwk = 1 - (numerator / denominator) if denominator > 0 else 0.0

    return {
        "n_scored": len(valid), "n_errors": len(results) - len(valid),
        "accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4), "qwk": round(qwk, 4),
        "per_class": per_class, "confusion_matrix": conf_matrix,
    }


def compute_cost_summary(results: list[dict]) -> dict:
    """Aggregate cost and token metadata across all results."""
    scored = [r for r in results if not r["error"]]
    if not scored:
        return {}

    total_cost = sum(r.get("total_cost_usd") or 0 for r in scored)
    total_prompt = sum(r.get("prompt_tokens") or 0 for r in scored)
    total_completion = sum(r.get("completion_tokens") or 0 for r in scored)
    total_reasoning = sum(r.get("native_tokens_reasoning") or 0 for r in scored)
    total_cached = sum(r.get("native_tokens_cached") or 0 for r in scored)
    wall_times = [r.get("wall_clock_seconds") or 0 for r in scored]
    latencies = [r.get("latency_ms") for r in scored if r.get("latency_ms") is not None]
    gen_times = [r.get("generation_time_ms") for r in scored if r.get("generation_time_ms") is not None]

    providers = Counter(r.get("provider_name") for r in scored if r.get("provider_name"))
    models = Counter(r.get("model_used") for r in scored if r.get("model_used"))
    finish_reasons = Counter(r.get("finish_reason") for r in scored if r.get("finish_reason"))

    n = len(scored)
    return {
        "n_scored": n,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_sample_usd": round(total_cost / n, 6) if n > 0 else 0,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "avg_prompt_tokens": round(total_prompt / n, 1) if n > 0 else 0,
        "avg_completion_tokens": round(total_completion / n, 1) if n > 0 else 0,
        "total_reasoning_tokens": total_reasoning,
        "total_cached_tokens": total_cached,
        "avg_wall_clock_seconds": round(sum(wall_times) / n, 3) if n > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "avg_generation_time_ms": round(sum(gen_times) / len(gen_times), 1) if gen_times else None,
        "providers": dict(providers),
        "models": dict(models),
        "finish_reasons": dict(finish_reasons),
    }


def main():
    parser = argparse.ArgumentParser(description="Strategy C6: Claude Sonnet + Adaptive Difficulty")
    parser.add_argument("--split", choices=["train", "trial", "test"], default="trial")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    model = get_model()
    print("=" * 70)
    print("  Strategy C6: Claude Sonnet + Adaptive Difficulty")
    print("=" * 70)
    print(f"  Split: {args.split} | Workers: {args.workers} | Offset: {args.offset} | Limit: {args.limit or 'all'}")
    print(f"  Seed: {args.seed} | Model: {model}")
    print(f"  Adaptive n_boundary/n_similar per question difficulty")
    print(f"  Full metadata logging enabled (tokens, cost, latency, provider)")
    print("=" * 70)

    if "claude" not in model.lower() and "anthropic" not in model.lower():
        print(f"\n  WARNING: Model '{model}' does not appear to be Claude Sonnet.")
        print(f"  Expected: anthropic/claude-sonnet-4.6")
        print(f"  Set OPENROUTER_MODEL=anthropic/claude-sonnet-4.6 to use Claude Sonnet.\n")

    configure(seed=args.seed)

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
            r = future.result()
            results.append(r)
            completed += 1

            # Per-sample logging
            cost_str = f"${r.get('total_cost_usd', 0) or 0:.4f}" if r.get("total_cost_usd") else "pending"
            tokens_str = f"{r.get('prompt_tokens', '?')}/{r.get('completion_tokens', '?')}"
            reason_str = f" reason={r.get('native_tokens_reasoning', 0)}" if r.get("native_tokens_reasoning") else ""
            status = "OK" if not r["error"] else f"ERR: {r['error'][:60]}"
            pred_label = r.get("predicted_score") or "N/A"

            if completed <= 20 or completed % 25 == 0 or completed == len(data):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                errors = sum(1 for r2 in results if r2["error"])
                running_cost = sum(r2.get("total_cost_usd") or 0 for r2 in results)
                print(
                    f"  [{completed:>5d}/{len(data)}] {rate:.1f}/s | "
                    f"tokens={tokens_str}{reason_str} | cost={cost_str} | "
                    f"running=${running_cost:.4f} | errors={errors} | "
                    f"{pred_label:>20s} | {status}"
                )

    elapsed_total = time.time() - t_start
    id_order = {s["id"]: i for i, s in enumerate(data)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    metrics = compute_metrics(results)
    cost_summary = compute_cost_summary(results)

    print("\n" + "=" * 70)
    print("  Scoring Results")
    print("=" * 70)
    if "error" not in metrics:
        print(f"  QWK: {metrics['qwk']:.4f} | Acc: {metrics['accuracy']:.4f} | wF1: {metrics['weighted_f1']:.4f}")
        print(f"  Scored: {metrics['n_scored']} | Errors: {metrics['n_errors']}")
        for label in ["Correct", "Partially correct", "Incorrect"]:
            pc = metrics["per_class"][label]
            print(f"    {label:>20s}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")
        if metrics.get("confusion_matrix"):
            print(f"\n  Confusion Matrix (rows=gold, cols=pred):")
            print(f"  {'':>20s}  {'Correct':>10s} {'Part.corr':>10s} {'Incorrect':>10s}")
            for i, label in enumerate(["Correct", "Partially correct", "Incorrect"]):
                row = metrics["confusion_matrix"][i]
                print(f"  {label:>20s}  {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")
    else:
        print(f"  Error: {metrics['error']}")

    print(f"\n  Time: {elapsed_total:.1f}s")

    print("\n" + "=" * 70)
    print("  Cost & Metadata Summary")
    print("=" * 70)
    if cost_summary:
        print(f"  Total cost:           ${cost_summary['total_cost_usd']:.4f}")
        print(f"  Cost per sample:      ${cost_summary['cost_per_sample_usd']:.6f}")
        print(f"  Prompt tokens:        {cost_summary['total_prompt_tokens']:,} (avg {cost_summary['avg_prompt_tokens']:.0f}/sample)")
        print(f"  Completion tokens:    {cost_summary['total_completion_tokens']:,} (avg {cost_summary['avg_completion_tokens']:.0f}/sample)")
        if cost_summary["total_reasoning_tokens"]:
            print(f"  Reasoning tokens:     {cost_summary['total_reasoning_tokens']:,} (WARNING: reasoning active!)")
        if cost_summary["total_cached_tokens"]:
            print(f"  Cached tokens:        {cost_summary['total_cached_tokens']:,}")
        print(f"  Avg wall clock:       {cost_summary['avg_wall_clock_seconds']:.2f}s/sample")
        if cost_summary.get("avg_latency_ms"):
            print(f"  Avg latency:          {cost_summary['avg_latency_ms']:.0f}ms")
        if cost_summary.get("avg_generation_time_ms"):
            print(f"  Avg generation time:  {cost_summary['avg_generation_time_ms']:.0f}ms")
        print(f"  Providers:            {cost_summary['providers']}")
        print(f"  Models:               {cost_summary['models']}")
        print(f"  Finish reasons:       {cost_summary['finish_reasons']}")

    # Difficulty tier distribution
    difficulty = get_difficulty()
    if difficulty:
        tier_counts = Counter(d["tier"] for d in difficulty.values())
        print("\n  Difficulty Tier Distribution (from training data):")
        for tier in ("easy", "medium", "hard"):
            count = tier_counts.get(tier, 0)
            pct = count / len(difficulty) * 100 if difficulty else 0
            n_b, n_s = {"easy": (1, 2), "medium": (2, 2), "hard": (3, 2)}[tier]
            print(f"    {tier:>6s}: {count:>3d} questions ({pct:5.1f}%) -> n_boundary={n_b}, n_similar={n_s}")

    print("=" * 70)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = model.replace("/", "_")

    pred_path = RESULTS_DIR / f"predictions_{args.split}_{model_short}_{timestamp}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    meta = {
        "strategy": "C6: Claude Sonnet + Adaptive Difficulty",
        "strategy_base": "C5c (adaptive difficulty) + C4 (smart examples)",
        "split": args.split,
        "model": model,
        "model_expected": "anthropic/claude-sonnet-4.6",
        "offset": args.offset,
        "seed": args.seed,
        "workers": args.workers,
        "limit": args.limit,
        "tier_config": {"easy": {"n_boundary": 1, "n_similar": 2},
                        "medium": {"n_boundary": 2, "n_similar": 2},
                        "hard": {"n_boundary": 3, "n_similar": 2}},
        "difficulty_distribution": dict(Counter(d["tier"] for d in difficulty.values())) if difficulty else {},
        "n_samples": len(data),
        "elapsed_seconds": round(elapsed_total, 2),
        "metrics": metrics,
        "cost_summary": cost_summary,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = RESULTS_DIR / f"metrics_{args.split}_{model_short}_{timestamp}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Compile submission files (only if gold labels available or test split)
    for track in ("3way", "2way"):
        sub_path = RESULTS_DIR / f"submission_{args.split}_{track}_{timestamp}.json"
        compile_submission_from_predictions(results, sub_path, track=track)

    print(f"\n  Predictions: {pred_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Submissions: {RESULTS_DIR / f'submission_{args.split}_*.json'}")


if __name__ == "__main__":
    main()
