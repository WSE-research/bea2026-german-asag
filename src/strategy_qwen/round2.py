"""
Round 2: Targeted experiments building on Q1-Q5 findings.

Key insights from round 1:
- German prompt > English for Qwen3.5 (Q1=0.510 > Q3=0.480)
- Few-shot examples help (Q5=0.539 > Q1=0.510)
- "Partially correct" massively over-predicted (recall 0.68-0.85)
- CoT is too slow and barely helps

This round focuses on:
- Q9:  German + few-shot (best of both)
- Q10: German + few-shot + anti-lenient rules
- Q11: German + 2 examples/label + strict boundary rules
- Q12: German + smart TF-IDF examples + calibration
- Q13: Two-pass: first classify Correct/not, then Partial/Incorrect
- Q14: Rubric reformulation — make model compare answer to rubric point by point
- Q15: Few-shot with NEGATIVE examples highlighted (anti-patterns for "Partially correct")
"""
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.strategy_qwen.runner import load_data, run_variant, call_model, parse_score, LABELS, RESULTS_DIR

# Cache for examples
_examples_cache = {}
_tfidf_cache = {}


def _get_examples(question_id, train, n_per_label=1):
    if question_id not in _examples_cache:
        by_label = defaultdict(list)
        for s in train:
            if s["question_id"] == question_id:
                by_label[s["score"]].append(s)
        _examples_cache[question_id] = by_label
    by_label = _examples_cache[question_id]
    examples = []
    for label in LABELS:
        candidates = by_label.get(label, [])
        if candidates:
            selected = random.sample(candidates, min(n_per_label, len(candidates)))
            examples.extend(selected)
    return examples


def _get_smart_examples(sample, train, n_similar=2, n_boundary=1):
    qid = sample["question_id"]
    if qid not in _tfidf_cache:
        q_samples = [s for s in train if s["question_id"] == qid]
        if not q_samples:
            return []
        answers = [s["answer"] for s in q_samples]
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(answers)
        _tfidf_cache[qid] = (q_samples, vectorizer, tfidf_matrix)
    q_samples, vectorizer, tfidf_matrix = _tfidf_cache[qid]
    query_vec = vectorizer.transform([sample["answer"]])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    examples = []
    seen = set()
    for label in LABELS:
        idxs = [i for i, s in enumerate(q_samples) if s["score"] == label]
        if not idxs:
            continue
        scored = sorted([(i, similarities[i]) for i in idxs], key=lambda x: -x[1])
        for idx, _ in scored[:n_similar]:
            if q_samples[idx]["id"] not in seen:
                examples.append(q_samples[idx])
                seen.add(q_samples[idx]["id"])
    return examples


# ============================================================
# Q9: German + few-shot (1 per label)
# ============================================================
def q9_german_fewshot(sample, train):
    system = (
        "Du bist ein präzises Bewertungssystem für Schülerantworten. "
        "Bewerte die Antwort anhand der Rubrik und der bewerteten Beispiele. "
        'Antworte ausschließlich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples(sample["question_id"], train, n_per_label=1)
    ex_text = "\n".join(
        f'Beispiel (Bewertung: {ex["score"]}): {ex["answer"]}'
        for ex in examples
    )
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Bewertete Beispiele:\n{ex_text}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q10: German + few-shot + anti-lenient calibration
# ============================================================
def q10_german_fewshot_strict(sample, train):
    system = (
        "Du bist ein STRENGES Bewertungssystem für Schülerantworten.\n\n"
        "WICHTIGE KALIBRIERUNGSREGELN:\n"
        "1. 'Partially correct' ist SELTEN — vergib es NUR wenn die Antwort "
        "eindeutig teilweise richtige Elemente enthält.\n"
        "2. Wenn die Antwort vage, ungenau oder nur ansatzweise richtig ist → 'Incorrect'.\n"
        "3. Wenn die Antwort die wesentlichen Punkte der Correct-Rubrik abdeckt → 'Correct'.\n"
        "4. Im Zweifel zwischen Partially correct und einer anderen Bewertung: "
        "wähle die andere Bewertung.\n\n"
        'Antworte ausschließlich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples(sample["question_id"], train, n_per_label=1)
    ex_text = "\n".join(
        f'Beispiel (Bewertung: {ex["score"]}): {ex["answer"]}'
        for ex in examples
    )
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Bewertete Beispiele:\n{ex_text}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q11: German + 2 examples/label + strict boundary rules
# ============================================================
def q11_german_2ex_strict(sample, train):
    system = (
        "Du bist ein Bewertungssystem für Schülerantworten. "
        "Studiere die bewerteten Beispiele sorgfältig und bewerte die neue Antwort.\n\n"
        "ENTSCHEIDUNGSHILFE:\n"
        "- Vergleiche die Antwort zuerst mit den Correct-Beispielen. Deckt sie die gleichen Punkte ab? → Correct\n"
        "- Vergleiche mit den Incorrect-Beispielen. Fehlen die gleichen Kernpunkte? → Incorrect\n"
        "- NUR wenn die Antwort klar zwischen den beiden liegt → Partially correct\n\n"
        'Antworte mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples(sample["question_id"], train, n_per_label=2)
    ex_text = "\n".join(
        f'[{ex["score"]}]: {ex["answer"]}'
        for ex in examples
    )
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Rubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Bewertete Beispiele:\n{ex_text}\n\n"
        f"Neue Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q12: German + smart TF-IDF examples + calibration
# ============================================================
def q12_german_smart_strict(sample, train):
    system = (
        "Du bist ein Bewertungssystem für Schülerantworten. "
        "Dir werden ähnliche, bereits bewertete Antworten als Referenz gegeben.\n\n"
        "VORGEHEN:\n"
        "1. Vergleiche die Schülerantwort mit den Referenzantworten.\n"
        "2. Welcher bewerteten Antwort ist sie am ähnlichsten?\n"
        "3. Vergib die gleiche Bewertung wie die ähnlichste Referenz.\n\n"
        "WICHTIG: Bevorzuge klare Entscheidungen (Correct/Incorrect) vor 'Partially correct'.\n\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_smart_examples(sample, train, n_similar=2, n_boundary=0)
    ex_text = "\n".join(
        f'[{ex["score"]}]: {ex["answer"]}'
        for ex in examples
    )
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Rubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Ähnliche bewertete Antworten:\n{ex_text}\n\n"
        f"Neue Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q13: Two-pass binary cascade
# ============================================================
def q13_binary_cascade(sample, train):
    """First decide Correct vs not-Correct, then Partially vs Incorrect."""
    rubric = sample["rubric"]
    examples = _get_examples(sample["question_id"], train, n_per_label=1)

    # Pass 1: Is it Correct?
    system1 = (
        "Du bewertest Schülerantworten. Entscheide NUR: Ist die Antwort vollständig korrekt?\n"
        '{"correct": true | false}'
    )
    correct_exs = [ex for ex in examples if ex["score"] == "Correct"]
    incorrect_exs = [ex for ex in examples if ex["score"] != "Correct"]
    ex_text = ""
    for ex in correct_exs:
        ex_text += f"[Korrekt]: {ex['answer']}\n"
    for ex in incorrect_exs[:1]:
        ex_text += f"[Nicht korrekt]: {ex['answer']}\n"

    user1 = (
        f"Frage: {sample['question']}\n"
        f"Correct-Kriterium: {rubric['Correct']}\n\n"
        f"Referenzen:\n{ex_text}\n"
        f"Antwort: {sample['answer']}"
    )
    resp1 = call_model([{"role": "system", "content": system1}, {"role": "user", "content": user1}])
    try:
        is_correct = json.loads(resp1).get("correct", False)
    except:
        is_correct = "true" in resp1.lower()

    if is_correct:
        return None  # Signal: return "Correct"

    # Pass 2: Partially correct or Incorrect?
    system2 = (
        "Die Antwort ist NICHT vollständig korrekt. Entscheide: "
        "Ist sie teilweise korrekt oder komplett falsch?\n"
        '{"score": "Partially correct" | "Incorrect"}'
    )
    user2 = (
        f"Frage: {sample['question']}\n"
        f"Partially correct: {rubric['Partially correct']}\n"
        f"Incorrect: {rubric['Incorrect']}\n\n"
        f"Antwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system2}, {"role": "user", "content": user2}]


# ============================================================
# Q14: Point-by-point rubric comparison
# ============================================================
def q14_point_by_point(sample, train):
    system = (
        "Du bist ein Bewertungssystem. Vergleiche die Schülerantwort Punkt für Punkt "
        "mit den Rubrik-Kriterien.\n\n"
        "Für jedes Kriterium in der Correct-Rubrik, prüfe ob die Antwort es erfüllt. "
        "Zähle dann:\n"
        "- Alle Kriterien erfüllt → Correct\n"
        "- Kein Kriterium erfüllt → Incorrect\n"
        "- Einige erfüllt → Partially correct\n\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    user = (
        f"Frage: {sample['question']}\n\n"
        f"KRITERIEN für 'Correct':\n{rubric['Correct']}\n\n"
        f"KRITERIEN für 'Partially correct':\n{rubric['Partially correct']}\n\n"
        f"KRITERIEN für 'Incorrect':\n{rubric['Incorrect']}\n\n"
        f"Schülerantwort:\n{sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q15: Few-shot + negative examples for "Partially correct"
# ============================================================
def q15_negative_calibration(sample, train):
    """Show examples where answers LOOK partially correct but are actually Incorrect."""
    system = (
        "Du bist ein Bewertungssystem. WARNUNG: Viele Antworten sehen teilweise richtig aus, "
        "sind aber tatsächlich falsch. Sei besonders streng bei 'Partially correct'.\n\n"
        "Typische FALLEN:\n"
        "- Antwort wiederholt die Frage ohne inhaltlichen Beitrag → Incorrect\n"
        "- Antwort nennt ein Stichwort aber erklärt es falsch → Incorrect\n"
        "- Antwort ist sehr kurz und vage → Incorrect\n"
        "- Antwort enthält falsche Kausalzusammenhänge → Incorrect\n\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples(sample["question_id"], train, n_per_label=2)
    ex_text = "\n".join(
        f'[{ex["score"]}]: {ex["answer"]}'
        for ex in examples
    )
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Rubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Bewertete Referenzen:\n{ex_text}\n\n"
        f"Neue Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# VARIANTS registry
# ============================================================
ROUND2_VARIANTS = {
    "q9_german_fewshot": q9_german_fewshot,
    "q10_german_fewshot_strict": q10_german_fewshot_strict,
    "q11_german_2ex_strict": q11_german_2ex_strict,
    "q12_german_smart_strict": q12_german_smart_strict,
    "q14_point_by_point": q14_point_by_point,
    "q15_negative_calibration": q15_negative_calibration,
}
# Q13 is special (multi-pass) — handled separately


def run_q13(trial, train, limit=100):
    """Special runner for the two-pass binary cascade (Q13)."""
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            result = q13_binary_cascade(sample, train)
            if result is None:
                # Pass 1 said Correct
                pred = "Correct"
            else:
                resp = call_model(result)
                score = parse_score(resp)
                if score not in LABELS:
                    errors += 1
                    raw_results.append({"id": sample["id"], "error": f"Invalid: {score}"})
                    continue
                pred = score

            golds.append(sample["score"])
            preds.append(pred)
            raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": pred, "match": pred == sample["score"]})

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"  Q13 [{i+1}/{limit}] {(i+1)/elapsed:.1f} s/s")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    from src.strategy_qwen.runner import compute_metrics, MODEL
    metrics = compute_metrics(golds, preds)
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "samples_per_s": round(len(samples)/elapsed, 2),
                    "variant": "q13_binary_cascade", "model": MODEL})

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q13_binary_cascade_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_q13_binary_cascade_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Q13 binary_cascade: QWK={metrics['qwk']:.3f} Acc={metrics['accuracy']:.1%} Err={errors} Time={elapsed:.0f}s")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--variants", type=str, default=None)
    args = parser.parse_args()

    train, trial = load_data()

    if args.variants:
        names = [v.strip() for v in args.variants.split(",")]
    else:
        names = list(ROUND2_VARIANTS.keys()) + ["q13_binary_cascade"]

    results = []
    for name in names:
        if name == "q13_binary_cascade":
            print(f"\n>>> Q13 binary cascade ({args.limit} samples) <<<")
            m = run_q13(trial, train, limit=args.limit)
            results.append(m)
        elif name in ROUND2_VARIANTS:
            print(f"\n>>> {name} ({args.limit} samples) <<<")
            m = run_variant(name, ROUND2_VARIANTS[name], trial, train, limit=args.limit)
            results.append(m)
        else:
            print(f"Unknown: {name}")

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  ROUND 2 COMPARISON (Qwen3.5-27B-FP8)")
    print(f"{'='*70}")
    print(f"  {'Variant':<28s} {'QWK':>6s} {'Acc':>6s} {'Err':>4s} {'Cor_R':>6s} {'Par_R':>6s} {'Inc_R':>6s}")
    for m in results:
        pc = m.get("per_class", {})
        print(f"  {m['variant']:<28s} {m.get('qwk',0):>6.3f} {m.get('accuracy',0):>5.1%} {m['errors']:>4d} "
              f"{pc.get('Correct',{}).get('R',0):>6.2f} {pc.get('Partially correct',{}).get('R',0):>6.2f} {pc.get('Incorrect',{}).get('R',0):>6.2f}")
    print(f"{'='*70}")
    print(f"\n  Reference: Q5 english_fewshot = QWK 0.539 (round 1 best)")
    print(f"  Reference: Gemini Flash C5c  = QWK 0.748 (overall best)")
