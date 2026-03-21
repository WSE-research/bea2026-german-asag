"""
Round 8: Post-processing calibration experiments.

Error analysis shows the model gravitates to "Partially correct" (59.6% of errors).
Idea: Use training data to learn per-question label distributions, then
post-process LLM predictions to match expected distributions.

Also: Try having the model output a confidence score and using it for
borderline reclassification.
"""
import json
import time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from src.strategy_qwen.prompting.runner import (
    load_data, call_model, parse_score,
    LABELS, LABEL_MAP, RESULTS_DIR, compute_metrics, MODEL
)
from src.strategy_qwen.prompting.round4 import q26_best_of_breed


def _get_train_label_dist(train):
    """Get per-question label distribution from training data."""
    dist = defaultdict(Counter)
    for s in train:
        dist[s["question_id"]][s["score"]] += 1
    # Normalize
    result = {}
    for qid, counts in dist.items():
        total = sum(counts.values())
        result[qid] = {label: counts.get(label, 0) / total for label in LABELS}
    return result


def run_q34_confidence(trial, train, limit=827):
    """Q26 but ask for confidence, use it for reclassification."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            # Modified Q26 that asks for confidence
            from src.strategy_qwen.prompting.round4 import _get_smart_examples, _get_difficulty
            tier, _ = _get_difficulty(sample["question_id"], train)
            n_sim = {"easy": 1, "medium": 2, "hard": 3}[tier]

            system = (
                "Du bist ein Bewertungssystem für Schülerantworten.\n\n"
                "METHODE:\n"
                "1. Lies die Rubrik und die bewerteten Referenzantworten.\n"
                "2. Prüfe ZUERST: Erfüllt die Antwort ALLE Correct-Kriterien? → Correct\n"
                "3. Prüfe DANN: Erfüllt die Antwort KEINE der Kriterien? → Incorrect\n"
                "4. NUR wenn weder Correct noch Incorrect klar zutrifft → Partially correct\n\n"
                "WICHTIG: Vergleiche die Antwort direkt mit den Referenzantworten.\n\n"
                '{"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}'
            )
            rubric = sample["rubric"]
            examples = _get_smart_examples(sample, train, n_similar=n_sim)
            ex_text = "\n".join(f'[{ex["score"]}]: {ex["answer"]}' for ex in examples)
            user = (
                f"Frage: {sample['question']}\n\n"
                f"Rubrik:\n- Correct: {rubric['Correct']}\n"
                f"- Partially correct: {rubric['Partially correct']}\n"
                f"- Incorrect: {rubric['Incorrect']}\n\n"
                f"Referenzantworten:\n{ex_text}\n\n"
                f"Schülerantwort: {sample['answer']}"
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            resp = call_model(messages)

            try:
                obj = json.loads(resp.strip())
                score = obj.get("score")
                confidence = float(obj.get("confidence", 0.5))
            except:
                score = parse_score(resp)
                confidence = 0.5

            if score not in LABELS:
                errors += 1
                continue

            # Post-processing: if Partially correct with low confidence,
            # look at per-question distribution to decide
            if score == "Partially correct" and confidence < 0.6:
                q_dist = _get_train_label_dist(train).get(sample["question_id"], {})
                # If this question rarely has Partially correct answers, reclassify
                pc_rate = q_dist.get("Partially correct", 0.33)
                if pc_rate < 0.25:
                    # Question is mostly Correct/Incorrect — flip to the more common one
                    if q_dist.get("Correct", 0) > q_dist.get("Incorrect", 0):
                        score = "Correct"
                    else:
                        score = "Incorrect"

            golds.append(sample["score"])
            preds.append(score)
            raw_results.append({
                "id": sample["id"], "gold": sample["score"], "pred": score,
                "confidence": confidence, "match": score == sample["score"]
            })

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  Q34 [{i+1}/{limit}] {(i+1)/elapsed:.1f} s/s | err={errors}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": "q34_confidence_calibration", "model": MODEL})

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q34_confidence_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_q34_confidence_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    pc = metrics.get("per_class", {})
    print(f"\n  Q34 confidence calibration: QWK={metrics.get('qwk', 0):.4f} Acc={metrics.get('accuracy', 0):.1%} Err={errors} Time={elapsed:.0f}s")
    for label in LABELS:
        s = pc.get(label, {})
        print(f"    {label}: P={s.get('P',0):.3f} R={s.get('R',0):.3f} F1={s.get('F1',0):.3f}")
    return metrics


def run_q35_post_calibrate(trial, train, limit=827):
    """Take Q26 predictions and post-calibrate using training label distributions.
    Offline recalibration — no additional LLM calls needed."""

    # Load Q26 full trial predictions
    pred_files = sorted(Path('/home/jgwozdz/bea26/bea2026-german-asag/results/strategy_qwen').glob('predictions_q26_full_*.json'))
    if not pred_files:
        print("  No Q26 full predictions found!")
        return {}

    q26_preds = json.load(open(pred_files[-1]))
    trial_by_id = {s['id']: s for s in trial}
    q_dist = _get_train_label_dist(train)

    # Strategy: if model predicts "Partially correct" but the question's
    # training data shows very low Partially correct rate, reclassify
    # based on answer length relative to correct/incorrect averages
    q_avg_len = defaultdict(lambda: defaultdict(list))
    for s in train:
        q_avg_len[s["question_id"]][s["score"]].append(len(s["answer"]))

    golds, preds = [], []
    reclassified = 0

    for p in q26_preds:
        if "gold" not in p:
            continue
        sample = trial_by_id.get(p["id"])
        if not sample:
            continue

        gold = p["gold"]
        pred = p["pred"]
        qid = sample["question_id"]

        # Post-calibration: reclassify low-confidence "Partially correct"
        if pred == "Partially correct":
            dist = q_dist.get(qid, {})
            pc_rate = dist.get("Partially correct", 0.33)

            if pc_rate < 0.30:
                # This question rarely has "Partially correct" answers
                # Use answer length heuristic
                lens = q_avg_len[qid]
                avg_correct = np.mean(lens.get("Correct", [200])) if lens.get("Correct") else 200
                avg_incorrect = np.mean(lens.get("Incorrect", [100])) if lens.get("Incorrect") else 100

                ans_len = len(sample["answer"])
                if ans_len > (avg_correct + avg_incorrect) / 2:
                    pred = "Correct"
                else:
                    pred = "Incorrect"
                reclassified += 1

        golds.append(gold)
        preds.append(pred)

    metrics = compute_metrics(golds, preds)
    metrics.update({"variant": "q35_post_calibrate", "model": MODEL, "reclassified": reclassified,
                    "scored": len(golds), "errors": 0})

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q35_post_calibrate_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pc = metrics.get("per_class", {})
    print(f"\n  Q35 post-calibrate (offline, {reclassified} reclassified):")
    print(f"    QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.1%}")
    for label in LABELS:
        s = pc.get(label, {})
        print(f"    {label}: P={s.get('P',0):.3f} R={s.get('R',0):.3f} F1={s.get('F1',0):.3f}")
    return metrics


if __name__ == "__main__":
    train, trial = load_data()

    # Q35: Offline post-calibration (no LLM calls, instant)
    print(f"\n>>> Q35: Post-calibration of Q26 predictions <<<")
    m1 = run_q35_post_calibrate(trial, train)

    # Q34: Confidence-based calibration (full trial)
    print(f"\n>>> Q34: Confidence calibration (827 samples) <<<")
    m2 = run_q34_confidence(trial, train, limit=827)

    print(f"\n{'='*70}")
    print(f"  ROUND 8: CALIBRATION EXPERIMENTS")
    print(f"{'='*70}")
    for m in [m1, m2]:
        if m:
            print(f"  {m['variant']:<35s} QWK={m.get('qwk',0):.4f} Acc={m.get('accuracy',0):.1%}")
    print(f"  Previous best: Q26              QWK=0.7191")
    print(f"  Target:        Gemini C5c       QWK=0.7480")
    print(f"{'='*70}")
