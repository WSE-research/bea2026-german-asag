"""
Round 6: Q26 with thinking mode enabled.
Also try Q30: ensemble Gemini predictions + Qwen predictions (if both available).
And Q31: Q26 with temperature sweep to find optimal temp.
"""
import json
import time
from pathlib import Path
from src.strategy_qwen.prompting.runner import (
    load_data, call_model, parse_score,
    LABELS, LABEL_MAP, RESULTS_DIR, compute_metrics, MODEL
)
from src.strategy_qwen.prompting.round4 import q26_best_of_breed
import httpx


def run_q26_thinking(trial, train, limit=827):
    """Q26 with Qwen3.5 thinking mode enabled. Slower but potentially better."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            messages = q26_best_of_breed(sample, train)
            # Call with thinking enabled and higher max_tokens
            resp = call_model(messages, max_tokens=2000, enable_thinking=True)
            score = parse_score(resp)
            if score not in LABELS:
                errors += 1
                raw_results.append({"id": sample["id"], "error": f"Invalid: {score}", "raw": resp[:200]})
                continue
            golds.append(sample["score"])
            preds.append(score)
            raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": score})
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  Q26_think [{i+1}/{limit}] {(i+1)/elapsed:.2f} s/s | err={errors}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": "q30_q26_thinking", "model": MODEL})
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q30_q26_thinking_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_q30_q26_thinking_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    pc = metrics.get("per_class", {})
    print(f"\n  Q30 Q26+thinking: QWK={metrics.get('qwk', 0):.3f} Acc={metrics.get('accuracy', 0):.1%} Err={errors} Time={elapsed:.0f}s")
    for label in LABELS:
        s = pc.get(label, {})
        print(f"    {label}: P={s.get('P',0):.3f} R={s.get('R',0):.3f} F1={s.get('F1',0):.3f}")
    return metrics


def run_temp_sweep(trial, train, limit=200):
    """Test Q26 at different temperatures."""
    results = []
    for temp in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        samples = trial[:limit]
        golds, preds = [], []
        errors = 0
        start = time.time()

        for sample in samples:
            try:
                messages = q26_best_of_breed(sample, train)
                resp = call_model(messages, temperature=temp)
                score = parse_score(resp)
                if score in LABELS:
                    golds.append(sample["score"])
                    preds.append(score)
                else:
                    errors += 1
            except:
                errors += 1

        elapsed = time.time() - start
        metrics = compute_metrics(golds, preds) if golds else {}
        metrics.update({"errors": errors, "variant": f"q31_temp_{temp}", "model": MODEL,
                        "elapsed_s": round(elapsed, 1), "temperature": temp})

        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(RESULTS_DIR / f"metrics_q31_temp{temp}_{ts}.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"  temp={temp:.1f}: QWK={metrics.get('qwk', 0):.3f} Acc={metrics.get('accuracy', 0):.1%} Err={errors}")
        results.append(metrics)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["thinking", "tempsweep", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train, trial = load_data()

    if args.mode in ("tempsweep", "all"):
        limit = args.limit or 200
        print(f"\n>>> Temperature sweep ({limit} samples) <<<")
        run_temp_sweep(trial, train, limit=limit)

    if args.mode in ("thinking", "all"):
        limit = args.limit or 827
        print(f"\n>>> Q26 + thinking mode ({limit} samples) <<<")
        run_q26_thinking(trial, train, limit=limit)
