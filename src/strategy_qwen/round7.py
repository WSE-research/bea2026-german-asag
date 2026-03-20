"""
Round 7: Q26 at optimal temperature (0.1) on full trial set.
Temp sweep found: temp=0.1 → QWK 0.744 (200 samples), nearly matching Gemini (0.748).
Also try Q29 self-consistency at temp=0.3 (diverse votes).
"""
import json
import time
from collections import Counter
from src.strategy_qwen.runner import (
    load_data, call_model, parse_score,
    LABELS, RESULTS_DIR, compute_metrics, MODEL
)
from src.strategy_qwen.round4 import q26_best_of_breed


def run_q26_temp(trial, train, limit=827, temperature=0.1):
    """Q26 at specific temperature on full trial set."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            messages = q26_best_of_breed(sample, train)
            resp = call_model(messages, temperature=temperature)
            score = parse_score(resp)
            if score not in LABELS:
                errors += 1
                raw_results.append({"id": sample["id"], "error": f"Invalid: {score}", "raw": resp[:200]})
                continue
            golds.append(sample["score"])
            preds.append(score)
            raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": score})
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{limit}] {(i+1)/elapsed:.1f} s/s | err={errors}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1),
                    "variant": f"q32_q26_temp{temperature}_full",
                    "model": MODEL, "temperature": temperature})

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"q32_q26_temp{temperature}_full"
    with open(RESULTS_DIR / f"metrics_{fname}_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_{fname}_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    pc = metrics.get("per_class", {})
    print(f"\n  Q32 Q26 temp={temperature} FULL ({len(golds)} scored):")
    print(f"    QWK={metrics.get('qwk', 0):.4f} Acc={metrics.get('accuracy', 0):.1%} Err={errors} Time={elapsed:.0f}s")
    for label in LABELS:
        s = pc.get(label, {})
        print(f"    {label}: P={s.get('P',0):.3f} R={s.get('R',0):.3f} F1={s.get('F1',0):.3f} (n={s.get('support',0)})")
    return metrics


def run_q33_sc_diverse(trial, train, limit=827, n_votes=5, temps=[0.0, 0.1, 0.2, 0.3, 0.5]):
    """Self-consistency with diverse temperatures. Each vote at a different temp."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            messages = q26_best_of_breed(sample, train)
            votes = []
            for temp in temps[:n_votes]:
                resp = call_model(messages, temperature=temp)
                score = parse_score(resp)
                if score in LABELS:
                    votes.append(score)
            if len(votes) >= 3:
                vote_counts = Counter(votes)
                pred = vote_counts.most_common(1)[0][0]
                golds.append(sample["score"])
                preds.append(pred)
                raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": pred, "votes": votes})
            else:
                errors += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  Q33 [{i+1}/{limit}] {(i+1)/elapsed:.2f} s/s | err={errors}")
        except Exception as e:
            errors += 1

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1),
                    "variant": "q33_sc_diverse_5vote", "model": MODEL})

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q33_sc_diverse_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_q33_sc_diverse_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Q33 diverse SC: QWK={metrics.get('qwk', 0):.4f} Acc={metrics.get('accuracy', 0):.1%} Err={errors} Time={elapsed:.0f}s")
    return metrics


if __name__ == "__main__":
    train, trial = load_data()

    # Run Q26 at temp=0.1 on full trial
    print(f"\n>>> Q32: Q26 at temp=0.1 (full 827 trial) <<<")
    m1 = run_q26_temp(trial, train, limit=827, temperature=0.1)

    # Run Q26 at temp=0.0 on full trial
    print(f"\n>>> Q32b: Q26 at temp=0.0 (full 827 trial) <<<")
    m2 = run_q26_temp(trial, train, limit=827, temperature=0.0)

    # Self-consistency with diverse temps (full trial)
    print(f"\n>>> Q33: Diverse self-consistency, 5 votes (full 827 trial) <<<")
    m3 = run_q33_sc_diverse(trial, train, limit=827)

    # Final comparison
    print(f"\n{'='*75}")
    print(f"  ROUND 7: FINAL RESULTS (827 samples)")
    print(f"{'='*75}")
    for m in sorted([m1, m2, m3], key=lambda x: -x.get("qwk", 0)):
        print(f"  {m['variant']:<35s} QWK={m.get('qwk',0):.4f} Acc={m.get('accuracy',0):.1%} Time={m.get('elapsed_s',0):.0f}s")
    print(f"\n  Previous best: Q29 (sc3)          QWK=0.7250")
    print(f"  Target:        Gemini Flash C5c    QWK=0.7480")
    print(f"{'='*75}")
