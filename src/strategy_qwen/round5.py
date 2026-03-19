"""
Round 5: Run best variants on full 827 trial set.
Q26 = QWK 0.732 on 200 samples — need full trial confirmation.
Also run Q27 and Q23 for comparison.
Then try Q29: Q26 + self-consistency.
"""
import json
import time
from collections import Counter
from src.strategy_qwen.runner import (
    load_data, run_variant, call_model, parse_score,
    LABELS, RESULTS_DIR, compute_metrics, MODEL
)
from src.strategy_qwen.round4 import q26_best_of_breed, q27_adaptive, ROUND4_VARIANTS


def run_selfconsistency_variant(variant_name, build_fn, trial, train, limit, n_votes=3):
    """Run a variant with self-consistency (majority vote over n_votes runs)."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            messages = build_fn(sample, train)
            votes = []
            for _ in range(n_votes):
                resp = call_model(messages, temperature=0.5)
                score = parse_score(resp)
                if score in LABELS:
                    votes.append(score)
            if len(votes) >= 2:
                vote_counts = Counter(votes)
                pred = vote_counts.most_common(1)[0][0]
                golds.append(sample["score"])
                preds.append(pred)
                raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": pred, "votes": votes})
            else:
                errors += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  {variant_name} [{i+1}/{limit}] {(i+1)/elapsed:.1f} s/s | err={errors}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": variant_name, "model": MODEL})
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_{variant_name}_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_{variant_name}_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    pc = metrics.get("per_class", {})
    cr = pc.get("Correct", {}).get("R", 0)
    pr = pc.get("Partially correct", {}).get("R", 0)
    ir = pc.get("Incorrect", {}).get("R", 0)
    print(f"\n  {variant_name}: QWK={metrics.get('qwk', 0):.3f} Acc={metrics.get('accuracy', 0):.1%} Err={errors}")
    print(f"    Cor_R={cr:.2f} Par_R={pr:.2f} Inc_R={ir:.2f} Time={elapsed:.0f}s")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=827, help="827 for full trial set")
    args = parser.parse_args()

    train, trial = load_data()
    limit = min(args.limit, len(trial))
    results = []

    # Q26 full trial
    print(f"\n>>> Q26 best-of-breed ({limit} samples) <<<")
    m = run_variant("q26_full", q26_best_of_breed, trial, train, limit=limit)
    results.append(m)

    # Q27 full trial
    print(f"\n>>> Q27 adaptive ({limit} samples) <<<")
    m = run_variant("q27_full", q27_adaptive, trial, train, limit=limit)
    results.append(m)

    # Q29: Q26 with self-consistency (3 votes)
    print(f"\n>>> Q29 Q26+self-consistency ({limit} samples, 3 votes) <<<")
    m = run_selfconsistency_variant("q29_q26_sc3", q26_best_of_breed, trial, train, limit=limit, n_votes=3)
    results.append(m)

    # Print comparison
    print(f"\n{'='*75}")
    print(f"  ROUND 5: FULL TRIAL RESULTS ({limit} samples)")
    print(f"{'='*75}")
    for m in sorted(results, key=lambda x: -x.get("qwk", 0)):
        pc = m.get("per_class", {})
        cr = pc.get("Correct", {}).get("R", 0)
        pr = pc.get("Partially correct", {}).get("R", 0)
        ir = pc.get("Incorrect", {}).get("R", 0)
        print(f"  {m['variant']:<28s} QWK={m.get('qwk',0):.3f} Acc={m.get('accuracy',0):.1%} Cor_R={cr:.2f} Par_R={pr:.2f} Inc_R={ir:.2f} Time={m.get('elapsed_s',0):.0f}s")
    print(f"\n  Reference: Q12 full trial  = QWK 0.712")
    print(f"  Reference: Gemini Flash C5c = QWK 0.748")
    print(f"{'='*75}")
