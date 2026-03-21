"""
Stacking ensemble: train a meta-learner on predictions from multiple models.

Uses cross-validated predictions from:
1. Fine-tuned 32B (train-only) predictions on trial
2. Q26 prompt (Qwen3.5-27B) predictions on trial  
3. kNN (TF-IDF k=7) predictions on trial

Features for each sample:
- Model 1 prediction (one-hot)
- Model 2 prediction (one-hot)
- kNN prediction (one-hot)
- kNN confidence (max vote fraction)
- Answer length
- Question difficulty (dominant label fraction from training)

Meta-learner: logistic regression or gradient boosting, trained via
leave-one-question-out cross-validation on trial set.
"""
import json, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}


def load_predictions(pattern):
    """Load the latest prediction file matching pattern."""
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        return None
    return {p["id"]: p for p in json.load(open(files[-1])) if "pred" in p}


def main():
    with open(TRAIN_FILE) as f:
        train = json.load(f)
    with open(TRIAL_FILE) as f:
        trial = json.load(f)
    
    trial_by_id = {s["id"]: s for s in trial}
    train_by_q = defaultdict(list)
    for s in train:
        train_by_q[s["question_id"]].append(s)
    
    # Load predictions from different models
    ft32_preds = load_predictions("predictions_finetune_32b_trial_*.json")
    q26_preds = load_predictions("predictions_q26_full_*.json")
    
    if not ft32_preds:
        print("Missing 32B fine-tuned predictions!")
        return
    if not q26_preds:
        print("Missing Q26 predictions!")
        return
    
    print(f"32B FT predictions: {len(ft32_preds)}")
    print(f"Q26 predictions: {len(q26_preds)}")
    
    # Build kNN predictions
    print("Building kNN predictions...")
    knn_preds = {}
    knn_confs = {}
    for sample in trial:
        qid = sample["question_id"]
        q_train = train_by_q.get(qid, [])
        if not q_train:
            knn_preds[sample["id"]] = "Partially correct"
            knn_confs[sample["id"]] = 0.33
            continue
        answers = [s["answer"] for s in q_train] + [sample["answer"]]
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(answers)
        sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
        top_k = np.argsort(sims)[-7:]
        votes = [q_train[j]["score"] for j in top_k]
        vote_counts = Counter(votes)
        knn_preds[sample["id"]] = vote_counts.most_common(1)[0][0]
        knn_confs[sample["id"]] = vote_counts.most_common(1)[0][1] / 7
    
    # Question difficulty
    q_difficulty = {}
    for qid, samples in train_by_q.items():
        counts = Counter(s["score"] for s in samples)
        total = sum(counts.values())
        q_difficulty[qid] = max(counts.values()) / total
    
    # Build feature matrix
    common_ids = set(ft32_preds.keys()) & set(q26_preds.keys()) & set(knn_preds.keys())
    print(f"Common samples: {len(common_ids)}")
    
    X, y, groups = [], [], []
    for sid in sorted(common_ids):
        sample = trial_by_id[sid]
        ft_pred = LABEL_MAP.get(ft32_preds[sid]["pred"], 1)
        q26_pred = LABEL_MAP.get(q26_preds[sid]["pred"], 1)
        knn_pred = LABEL_MAP.get(knn_preds[sid], 1)
        knn_conf = knn_confs[sid]
        ans_len = len(sample["answer"])
        q_diff = q_difficulty.get(sample["question_id"], 0.33)
        
        # One-hot encode predictions
        ft_oh = [1 if ft_pred == i else 0 for i in range(3)]
        q26_oh = [1 if q26_pred == i else 0 for i in range(3)]
        knn_oh = [1 if knn_pred == i else 0 for i in range(3)]
        
        features = ft_oh + q26_oh + knn_oh + [knn_conf, ans_len / 1000, q_diff]
        X.append(features)
        y.append(LABEL_MAP[sample["score"]])
        groups.append(sample["question_id"])
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    # Cross-validated stacking
    print("\n=== Cross-Validated Stacking ===")
    
    for name, clf in [
        ("LogReg", LogisticRegression(max_iter=1000, multi_class="multinomial")),
        ("GBM", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]:
        # Leave-one-question-out CV
        logo = LeaveOneGroupOut()
        all_preds = np.zeros(len(y))
        
        for train_idx, test_idx in logo.split(X, y, groups):
            clf.fit(X[train_idx], y[train_idx])
            all_preds[test_idx] = clf.predict(X[test_idx])
        
        qwk = cohen_kappa_score(y, all_preds, weights="quadratic")
        acc = np.mean(y == all_preds)
        print(f"  {name}: QWK={qwk:.4f} Acc={acc:.1%}")
    
    # Simple majority vote baseline
    majority_preds = []
    for sid in sorted(common_ids):
        votes = [
            ft32_preds[sid]["pred"],
            q26_preds[sid]["pred"],
            knn_preds[sid],
        ]
        majority_preds.append(LABEL_MAP[Counter(votes).most_common(1)[0][0]])
    majority_preds = np.array(majority_preds)
    qwk_maj = cohen_kappa_score(y, majority_preds, weights="quadratic")
    acc_maj = np.mean(y == majority_preds)
    print(f"  Majority vote: QWK={qwk_maj:.4f} Acc={acc_maj:.1%}")
    
    # Individual model baselines
    for name, pred_dict in [("32B FT", ft32_preds), ("Q26", q26_preds), ("kNN", knn_preds)]:
        model_preds = [LABEL_MAP.get(pred_dict[sid] if isinstance(pred_dict[sid], str) else pred_dict[sid].get("pred", "Partially correct"), 1) for sid in sorted(common_ids)]
        qwk_m = cohen_kappa_score(y, model_preds, weights="quadratic")
        print(f"  {name} alone: QWK={qwk_m:.4f}")
    
    print(f"\n  Reference: Gemini Flash C5c = QWK 0.748")
    
    # Save
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"experiment": "stacking_ensemble", "models_used": ["32B_FT", "Q26", "kNN"]}
    with open(RESULTS_DIR / f"metrics_stacking_ensemble_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
