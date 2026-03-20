"""
Round 9: Methods prepared while fine-tuning runs on GPU.

9a. BM25 example selection (replaces TF-IDF in Q26)
9b. Weighted kNN+LLM ensemble using existing Q26 predictions
9c. Per-question accuracy analysis + specialized strategies
"""
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
TRAIN_FILE = DATA_DIR / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = DATA_DIR / "ALICE_LP_trial_3way__v2.json"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}


def load_data():
    with open(TRAIN_FILE) as f:
        train = json.load(f)
    with open(TRIAL_FILE) as f:
        trial = json.load(f)
    return train, trial


def compute_metrics(golds, preds):
    g = [LABEL_MAP[x] for x in golds]
    p = [LABEL_MAP[x] for x in preds]
    qwk = cohen_kappa_score(g, p, weights="quadratic")
    acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)
    per_class = {}
    for label in LABELS:
        tp = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl == label)
        fp = sum(1 for gl, pl in zip(golds, preds) if gl != label and pl == label)
        fn = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[label] = {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3)}
    return {"qwk": round(qwk, 4), "accuracy": round(acc, 4), "per_class": per_class}


# ============================================================
# 9a. BM25 example selection
# ============================================================

class BM25ExampleSelector:
    """BM25-based example selection. Handles term frequency saturation better than TF-IDF."""

    def __init__(self, train, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self._cache = {}
        self._train_by_q = defaultdict(list)
        for s in train:
            self._train_by_q[s["question_id"]].append(s)

    def _build_index(self, qid):
        if qid in self._cache:
            return self._cache[qid]
        samples = self._train_by_q.get(qid, [])
        if not samples:
            self._cache[qid] = None
            return None
        docs = [s["answer"].lower().split() for s in samples]
        doc_lens = [len(d) for d in docs]
        avgdl = np.mean(doc_lens)
        N = len(docs)
        df = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        idf = {}
        for term, freq in df.items():
            idf[term] = np.log((N - freq + 0.5) / (freq + 0.5) + 1)
        self._cache[qid] = (samples, docs, doc_lens, avgdl, idf)
        return self._cache[qid]

    def score_doc(self, query_tokens, doc_tokens, doc_len, avgdl, idf):
        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in idf:
                continue
            term_tf = tf.get(term, 0)
            numerator = idf[term] * term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += numerator / denominator
        return score

    def get_examples(self, sample, n_similar=2):
        qid = sample["question_id"]
        index = self._build_index(qid)
        if index is None:
            return []
        samples, docs, doc_lens, avgdl, idf = index
        query = sample["answer"].lower().split()
        scores = []
        for i, (doc, doc_len) in enumerate(zip(docs, doc_lens)):
            s = self.score_doc(query, doc, doc_len, avgdl, idf)
            scores.append((i, s))
        examples = []
        seen = set()
        for label in LABELS:
            label_scores = [(i, s) for i, s in scores if samples[i]["score"] == label]
            label_scores.sort(key=lambda x: -x[1])
            for idx, _ in label_scores[:n_similar]:
                if samples[idx]["id"] not in seen:
                    examples.append(samples[idx])
                    seen.add(samples[idx]["id"])
        return examples


# ============================================================
# 9b. Weighted kNN+LLM ensemble (NO GPU needed)
# ============================================================

def run_weighted_ensemble(train, trial):
    """Combine kNN and Q26 LLM predictions using per-question kNN accuracy."""
    trial_by_id = {s["id"]: s for s in trial}
    train_by_q = defaultdict(list)
    for s in train:
        train_by_q[s["question_id"]].append(s)

    # Load Q26 predictions
    pred_files = sorted(RESULTS_DIR.glob("predictions_q26_full_*.json"))
    if not pred_files:
        print("  No Q26 predictions found!")
        return None
    q26_preds = {p["id"]: p for p in json.load(open(pred_files[-1])) if "pred" in p}

    # Per-question kNN accuracy (leave-one-out on training data)
    q_knn_acc = {}
    for qid, q_samples in train_by_q.items():
        if len(q_samples) < 5:
            q_knn_acc[qid] = 0.5
            continue
        answers = [s["answer"] for s in q_samples]
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(answers)
        correct = 0
        for i in range(len(q_samples)):
            sims = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix).flatten()
            sims[i] = -1
            top_k = np.argsort(sims)[-7:]
            votes = [q_samples[j]["score"] for j in top_k]
            pred = Counter(votes).most_common(1)[0][0]
            if pred == q_samples[i]["score"]:
                correct += 1
        q_knn_acc[qid] = correct / len(q_samples)

    # Ensemble
    golds, preds = [], []
    knn_wins, llm_wins, agree = 0, 0, 0
    for sample in trial:
        if sample["id"] not in q26_preds:
            continue
        q26_pred = q26_preds[sample["id"]]["pred"]
        qid = sample["question_id"]
        knn_acc = q_knn_acc.get(qid, 0.5)

        q_train = train_by_q.get(qid, [])
        if q_train:
            answers = [s["answer"] for s in q_train] + [sample["answer"]]
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(answers)
            sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
            top_k = np.argsort(sims)[-7:]
            votes = [q_train[j]["score"] for j in top_k]
            knn_pred = Counter(votes).most_common(1)[0][0]
        else:
            knn_pred = q26_pred

        if q26_pred == knn_pred:
            final = q26_pred
            agree += 1
        elif knn_acc > 0.75:
            final = knn_pred
            knn_wins += 1
        else:
            final = q26_pred
            llm_wins += 1

        golds.append(sample["score"])
        preds.append(final)

    metrics = compute_metrics(golds, preds)
    metrics.update({"variant": "q37_weighted_ensemble", "scored": len(golds),
                    "model": "kNN+Q26_weighted", "agree": agree,
                    "knn_wins": knn_wins, "llm_wins": llm_wins})

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_q37_weighted_ensemble_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Q37 weighted ensemble: QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.1%}")
    print(f"  Agree: {agree}, kNN wins: {knn_wins}, LLM wins: {llm_wins}")
    return metrics


# ============================================================
# 9c. Per-question accuracy analysis
# ============================================================

def analyze_per_question(train, trial):
    """Analyze Q26 accuracy per question, identify hard questions."""
    trial_by_id = {s["id"]: s for s in trial}
    pred_files = sorted(RESULTS_DIR.glob("predictions_q26_full_*.json"))
    if not pred_files:
        print("  No Q26 predictions found!")
        return
    q26_preds = json.load(open(pred_files[-1]))

    q_results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    for p in q26_preds:
        if "gold" not in p:
            continue
        sample = trial_by_id.get(p["id"])
        if not sample:
            continue
        qid = sample["question_id"]
        q_results[qid]["total"] += 1
        if p.get("match"):
            q_results[qid]["correct"] += 1
        else:
            q_results[qid]["errors"].append(f"{p['gold']}->{p['pred']}")

    q_acc = []
    for qid, stats in q_results.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        q_acc.append((qid, acc, stats["total"], stats["errors"]))
    q_acc.sort(key=lambda x: x[1])

    print(f"\n  Per-question accuracy (Q26, 827 trial):")
    print(f"  {'QID':<12s} {'Acc':>5s} {'N':>4s} {'Top errors'}")
    print(f"  {'-'*60}")

    hard_questions = []
    for qid, acc, n, errors in q_acc[:15]:
        err_summary = Counter(errors).most_common(2)
        err_str = ", ".join(f"{e}({c})" for e, c in err_summary)
        print(f"  {qid[:10]:<12s} {acc:>4.0%} {n:>4d}  {err_str}")
        if acc < 0.55:
            hard_questions.append(qid)

    easy_count = len([q for q, a, n, e in q_acc if a > 0.85])
    print(f"\n  Hard questions (acc<55%): {len(hard_questions)}")
    print(f"  Easy questions (acc>85%): {easy_count}")

    analysis = {
        "per_question": [{"qid": qid, "accuracy": acc, "n": n, "errors": errors}
                         for qid, acc, n, errors in q_acc],
        "hard_questions": hard_questions,
    }
    with open(RESULTS_DIR / "per_question_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    return hard_questions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ensemble", "analysis", "all"], default="all")
    args = parser.parse_args()

    train, trial = load_data()

    if args.mode in ("ensemble", "all"):
        print("\n>>> Q37: Weighted kNN+LLM ensemble (no GPU needed) <<<")
        run_weighted_ensemble(train, trial)

    if args.mode in ("analysis", "all"):
        print("\n>>> Per-question accuracy analysis <<<")
        analyze_per_question(train, trial)

    print("\nDone.")
