"""
Round 4: Creative experiments beyond prompt engineering.

Q12 caps around QWK 0.63 with prompt-only approaches.
New ideas:
- Q23: Self-consistency (3 runs, majority vote)
- Q24: Calibrated scoring with per-question label priors from training data
- Q25: Hybrid: TF-IDF kNN classifier + LLM for uncertain cases
- Q26: German + smart examples + rubric-first + strict (combine best elements)
- Q27: Adaptive examples like Gemini C5c (vary count by difficulty)
- Q28: Multi-turn conversation (first analyze, then score)
"""
import json
import random
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
from src.strategy_qwen.runner import (
    load_data, run_variant, call_model, parse_score,
    LABELS, LABEL_MAP, RESULTS_DIR, compute_metrics, MODEL
)

_tfidf_cache = {}


def _get_smart_examples(sample, train, n_similar=2):
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


def _get_difficulty(question_id, train):
    scores = [s["score"] for s in train if s["question_id"] == question_id]
    if not scores:
        return "medium", 0.33
    counts = Counter(scores)
    total = sum(counts.values())
    dominant_pct = max(counts.values()) / total
    if dominant_pct > 0.60:
        return "easy", dominant_pct
    elif dominant_pct < 0.40:
        return "hard", dominant_pct
    else:
        return "medium", dominant_pct


# ============================================================
# Q23: Self-consistency (3 runs, majority vote)
# ============================================================
def run_q23_selfconsistency(trial, train, limit=100):
    """Run Q12 three times at temp=0.5, take majority vote."""
    from src.strategy_qwen.round3 import make_q12_variant
    build_fn = make_q12_variant(2, 0)
    samples = trial[:limit]
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(samples):
        try:
            messages = build_fn(sample, train)
            votes = []
            for _ in range(3):
                resp = call_model(messages, temperature=0.5)
                score = parse_score(resp)
                if score in LABELS:
                    votes.append(score)
            if len(votes) >= 2:
                # Majority vote
                vote_counts = Counter(votes)
                pred = vote_counts.most_common(1)[0][0]
                golds.append(sample["score"])
                preds.append(pred)
                raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": pred, "votes": votes})
            else:
                errors += 1
                raw_results.append({"id": sample["id"], "error": f"Only {len(votes)} valid votes"})

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"  Q23 [{i+1}/{limit}] {(i+1)/elapsed:.1f} s/s")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": "q23_selfconsistency", "model": MODEL})
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q23_selfconsistency_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_q23_selfconsistency_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    return metrics


# ============================================================
# Q25: Pure TF-IDF kNN baseline (no LLM needed!)
# ============================================================
def run_q25_knn(trial, train, limit=100, k=5):
    """Pure TF-IDF kNN classifier. No LLM calls at all."""
    samples = trial[:limit]
    golds, preds = [], []

    # Group training data by question
    train_by_q = defaultdict(list)
    for s in train:
        train_by_q[s["question_id"]].append(s)

    start = time.time()
    for i, sample in enumerate(samples):
        qid = sample["question_id"]
        q_train = train_by_q.get(qid, [])
        if not q_train:
            preds.append("Partially correct")  # fallback
            golds.append(sample["score"])
            continue

        # TF-IDF similarity
        answers = [s["answer"] for s in q_train] + [sample["answer"]]
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(answers)
        query_vec = tfidf_matrix[-1]
        train_vecs = tfidf_matrix[:-1]
        sims = cosine_similarity(query_vec, train_vecs).flatten()

        # Top-k neighbors
        top_k_idx = np.argsort(sims)[-k:]
        neighbor_labels = [q_train[idx]["score"] for idx in top_k_idx]
        vote = Counter(neighbor_labels).most_common(1)[0][0]

        golds.append(sample["score"])
        preds.append(vote)

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds)
    metrics.update({"errors": 0, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": f"q25_knn_k{k}", "model": "TF-IDF_kNN"})
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q25_knn_k{k}_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


# ============================================================
# Q26: Best-of-breed (combine all winning elements)
# ============================================================
def q26_best_of_breed(sample, train):
    """Combine: German + smart TF-IDF examples + rubric-first + strict + adaptive count."""
    tier, _ = _get_difficulty(sample["question_id"], train)
    n_sim = {"easy": 1, "medium": 2, "hard": 3}[tier]

    system = (
        "Du bist ein Bewertungssystem für Schülerantworten.\n\n"
        "METHODE:\n"
        "1. Lies die Rubrik und die bewerteten Referenzantworten.\n"
        "2. Prüfe ZUERST: Erfüllt die Antwort ALLE Correct-Kriterien? → Correct\n"
        "3. Prüfe DANN: Erfüllt die Antwort KEINE der Kriterien? → Incorrect\n"
        "4. NUR wenn weder Correct noch Incorrect klar zutrifft → Partially correct\n\n"
        "WICHTIG: Vergleiche die Antwort direkt mit den Referenzantworten. "
        "Welcher Referenz ist sie am ähnlichsten? Vergib die gleiche Bewertung.\n\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_smart_examples(sample, train, n_similar=n_sim)
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
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q27: Adaptive n_similar by difficulty (like Gemini C5c)
# ============================================================
def q27_adaptive(sample, train):
    """Adaptive example count by question difficulty, like C5c for Gemini."""
    tier, dominant_pct = _get_difficulty(sample["question_id"], train)
    config = {
        "easy":   {"n_similar": 1, "instruction": "Diese Frage ist relativ einfach. Die meisten Antworten sind klar richtig oder falsch."},
        "medium": {"n_similar": 2, "instruction": "Diese Frage hat eine ausgewogene Schwierigkeit."},
        "hard":   {"n_similar": 3, "instruction": "Diese Frage ist schwierig. Achte besonders auf Nuancen in der Antwort."},
    }
    cfg = config[tier]

    system = (
        f"Du bist ein Bewertungssystem für Schülerantworten.\n"
        f"{cfg['instruction']}\n\n"
        "Vergleiche die Antwort mit den Referenzantworten und der Rubrik. "
        "Bevorzuge klare Entscheidungen (Correct/Incorrect) vor 'Partially correct'.\n\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_smart_examples(sample, train, n_similar=cfg["n_similar"])
    ex_text = "\n".join(f'[{ex["score"]}]: {ex["answer"]}' for ex in examples)
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Rubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Referenzantworten:\n{ex_text}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q28: Hybrid kNN+LLM (kNN first, LLM for uncertain)
# ============================================================
def run_q28_hybrid(trial, train, limit=100, k=7, confidence_threshold=0.6):
    """kNN with confidence. If confident → use kNN, else → use LLM."""
    from src.strategy_qwen.round3 import make_q12_variant
    build_fn = make_q12_variant(2, 0)
    samples = trial[:limit]
    golds, preds = [], []
    knn_used, llm_used = 0, 0
    errors = 0
    raw_results = []

    train_by_q = defaultdict(list)
    for s in train:
        train_by_q[s["question_id"]].append(s)

    start = time.time()
    for i, sample in enumerate(samples):
        try:
            qid = sample["question_id"]
            q_train = train_by_q.get(qid, [])

            # kNN prediction with confidence
            if q_train:
                answers = [s["answer"] for s in q_train] + [sample["answer"]]
                vectorizer = TfidfVectorizer(max_features=5000)
                tfidf_matrix = vectorizer.fit_transform(answers)
                sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                top_k_idx = np.argsort(sims)[-k:]
                neighbor_labels = [q_train[idx]["score"] for idx in top_k_idx]
                vote_counts = Counter(neighbor_labels)
                total_votes = sum(vote_counts.values())
                best_label, best_count = vote_counts.most_common(1)[0]
                confidence = best_count / total_votes

                if confidence >= confidence_threshold:
                    pred = best_label
                    knn_used += 1
                else:
                    # Low confidence → use LLM
                    messages = build_fn(sample, train)
                    resp = call_model(messages)
                    pred = parse_score(resp)
                    if pred not in LABELS:
                        pred = best_label  # fallback to kNN
                    llm_used += 1
            else:
                messages = build_fn(sample, train)
                resp = call_model(messages)
                pred = parse_score(resp)
                llm_used += 1

            if pred in LABELS:
                golds.append(sample["score"])
                preds.append(pred)
                raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": pred})
            else:
                errors += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"  Q28 [{i+1}/{limit}] kNN={knn_used} LLM={llm_used}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {}
    metrics.update({"errors": errors, "total": len(samples), "scored": len(golds),
                    "elapsed_s": round(elapsed, 1), "variant": "q28_hybrid_knn_llm",
                    "model": MODEL, "knn_used": knn_used, "llm_used": llm_used})
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q28_hybrid_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


ROUND4_VARIANTS = {
    "q26_best_of_breed": q26_best_of_breed,
    "q27_adaptive": q27_adaptive,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    train, trial = load_data()
    results = []

    # Standard variants
    for name, fn in ROUND4_VARIANTS.items():
        print(f"\n>>> {name} ({args.limit} samples) <<<")
        m = run_variant(name, fn, trial, train, limit=args.limit)
        results.append(m)

    # Self-consistency
    print(f"\n>>> Q23 self-consistency ({args.limit} samples, 3 votes each) <<<")
    m = run_q23_selfconsistency(trial, train, limit=args.limit)
    results.append(m)

    # kNN baselines
    for k in [3, 5, 7, 11]:
        print(f"\n>>> Q25 kNN k={k} ({args.limit} samples) <<<")
        m = run_q25_knn(trial, train, limit=args.limit, k=k)
        results.append(m)

    # Hybrid kNN+LLM
    print(f"\n>>> Q28 hybrid kNN+LLM ({args.limit} samples) <<<")
    m = run_q28_hybrid(trial, train, limit=args.limit)
    results.append(m)

    # Comparison
    print(f"\n{'='*75}")
    print(f"  ROUND 4: CREATIVE EXPERIMENTS (100 samples)")
    print(f"{'='*75}")
    for m in sorted(results, key=lambda x: -x.get('qwk', 0)):
        pc = m.get('per_class', {})
        cr = pc.get('Correct', {}).get('R', 0)
        pr = pc.get('Partially correct', {}).get('R', 0)
        ir = pc.get('Incorrect', {}).get('R', 0)
        extra = ""
        if "knn_used" in m:
            extra = f" kNN={m['knn_used']} LLM={m['llm_used']}"
        print(f"  {m['variant']:<28s} QWK={m.get('qwk',0):.3f} Acc={m.get('accuracy',0):.1%} {m['model']:<20s}{extra}")
    print(f"\n  Reference: Q12 = QWK 0.634 | Gemini C5c = QWK 0.748")
    print(f"{'='*75}")
