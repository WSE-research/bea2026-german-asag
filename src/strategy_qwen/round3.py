"""
Round 3: Optimize Q12 (smart examples + strict calibration).

Q12 = QWK 0.634 on 100 samples. Now we:
1. Try different n_similar / n_boundary combinations
2. Test different calibration instruction phrasings
3. Run the best on full 827 trial set
4. Try ensemble of Q10+Q12 (strict + smart)

Key: Q12's strength is TF-IDF example selection + "prefer clear decisions" rule.
"""
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.strategy_qwen.runner import load_data, run_variant, call_model, parse_score, LABELS, RESULTS_DIR, compute_metrics, MODEL

_tfidf_cache = {}
_examples_cache = {}


def _get_smart_examples(sample, train, n_similar=2, n_boundary=0):
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
        # Boundary examples: least similar within the label
        if n_boundary > 0:
            for idx, _ in scored[-n_boundary:]:
                if q_samples[idx]["id"] not in seen:
                    examples.append(q_samples[idx])
                    seen.add(q_samples[idx]["id"])
    return examples


def make_q12_variant(n_similar, n_boundary, system_extra=""):
    """Factory for Q12 variants with different params."""
    def build(sample, train):
        base_system = (
            "Du bist ein Bewertungssystem für Schülerantworten. "
            "Dir werden ähnliche, bereits bewertete Antworten als Referenz gegeben.\n\n"
            "VORGEHEN:\n"
            "1. Vergleiche die Schülerantwort mit den Referenzantworten.\n"
            "2. Welcher bewerteten Antwort ist sie am ähnlichsten?\n"
            "3. Vergib die gleiche Bewertung wie die ähnlichste Referenz.\n\n"
            "WICHTIG: Bevorzuge klare Entscheidungen (Correct/Incorrect) vor 'Partially correct'.\n"
        )
        if system_extra:
            base_system += system_extra + "\n"
        base_system += '{"score": "Correct" | "Partially correct" | "Incorrect"}'

        rubric = sample["rubric"]
        examples = _get_smart_examples(sample, train, n_similar=n_similar, n_boundary=n_boundary)
        ex_text = "\n".join(f'[{ex["score"]}]: {ex["answer"]}' for ex in examples)
        user = (
            f"Frage: {sample['question']}\n\n"
            f"Rubrik:\n"
            f"- Correct: {rubric['Correct']}\n"
            f"- Partially correct: {rubric['Partially correct']}\n"
            f"- Incorrect: {rubric['Incorrect']}\n\n"
            f"Ähnliche bewertete Antworten:\n{ex_text}\n\n"
            f"Neue Schülerantwort: {sample['answer']}"
        )
        return [{"role": "system", "content": base_system}, {"role": "user", "content": user}]
    return build


# Variants with different example counts
ROUND3_VARIANTS = {
    "q16_smart_1sim": make_q12_variant(1, 0),
    "q17_smart_3sim": make_q12_variant(3, 0),
    "q18_smart_2sim_1bound": make_q12_variant(2, 1),
    "q19_smart_3sim_1bound": make_q12_variant(3, 1),
    "q20_smart_2sim_extreme": make_q12_variant(2, 0,
        "ZUSÄTZLICHE REGEL: 'Partially correct' nur vergeben wenn die Antwort "
        "EXAKT die Partially-correct-Rubrik-Kriterien erfüllt. "
        "Alle anderen Fälle sind entweder Correct oder Incorrect."
    ),
    "q21_smart_2sim_compare": make_q12_variant(2, 0,
        "ENTSCHEIDUNGSMETHODE: Ordne die Schülerantwort der Referenzantwort zu, "
        "der sie inhaltlich am ähnlichsten ist. Übernimm deren Bewertung."
    ),
    "q22_smart_2sim_rubric_first": make_q12_variant(2, 0,
        "REIHENFOLGE: Prüfe ZUERST ob die Correct-Rubrik erfüllt ist. "
        "Falls ja → Correct. Falls nein, prüfe die Incorrect-Rubrik. "
        "Falls sie zutrifft → Incorrect. Nur wenn weder Correct noch Incorrect "
        "klar zutrifft → Partially correct."
    ),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--full", action="store_true", help="Run best variant on full 827 trial set")
    args = parser.parse_args()

    train, trial = load_data()

    if args.full:
        # Run Q12 original on full trial set
        print(f"\n>>> Running Q12 on FULL trial set ({len(trial)} samples) <<<\n")
        q12_fn = make_q12_variant(2, 0)
        m = run_variant("q12_full_trial", q12_fn, trial, train, limit=len(trial))
        print(f"\n  FULL TRIAL: QWK={m['qwk']:.3f} Acc={m['accuracy']:.1%} Err={m['errors']}\n")
    else:
        if args.variants:
            names = [v.strip() for v in args.variants.split(",")]
        else:
            names = list(ROUND3_VARIANTS.keys())

        results = []
        for name in names:
            if name in ROUND3_VARIANTS:
                print(f"\n>>> {name} ({args.limit} samples) <<<")
                m = run_variant(name, ROUND3_VARIANTS[name], trial, train, limit=args.limit)
                results.append(m)

        # Comparison
        print(f"\n{'='*75}")
        print(f"  ROUND 3: Q12 OPTIMIZATION (Qwen3.5-27B-FP8, {args.limit} samples)")
        print(f"{'='*75}")
        for m in sorted(results, key=lambda x: -x.get('qwk', 0)):
            pc = m.get('per_class', {})
            cr = pc.get('Correct', {}).get('R', 0)
            pr = pc.get('Partially correct', {}).get('R', 0)
            ir = pc.get('Incorrect', {}).get('R', 0)
            print(f"  {m['variant']:<28s} QWK={m['qwk']:.3f} Acc={m['accuracy']:.1%} Cor_R={cr:.2f} Par_R={pr:.2f} Inc_R={ir:.2f}")
        print(f"\n  Reference: Q12 original = QWK 0.634")
        print(f"  Reference: Gemini C5c   = QWK 0.748")
        print(f"{'='*75}")
