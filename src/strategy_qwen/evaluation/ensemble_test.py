"""
Stacking ensemble on test data.
Trains LogReg meta-learner on trial predictions from 5 models,
applies to test data predictions, generates CodaBench submissions.
"""
import json, time, numpy as np, zipfile
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import httpx

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
LABELS = ["Correct", "Partially correct", "Incorrect"]
LM = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

def load_json(path):
    with open(path) as f:
        return json.load(f)

def score_7b_via_vllm(test_data, output_name):
    """Score test data with 7B via vLLM (must be running on port 8081)."""
    predictions = {}
    errors = 0
    start = time.time()
    for i, sample in enumerate(test_data):
        r = sample["rubric"]
        user_msg = ("Frage: " + sample["question"] + "\n\nBewertungsrubrik:\n"
                    "- Correct: " + r["Correct"] + "\n"
                    "- Partially correct: " + r["Partially correct"] + "\n"
                    "- Incorrect: " + r["Incorrect"] + "\n\n"
                    "Schuelerantwort: " + sample["answer"])
        try:
            resp = httpx.post("http://localhost:8081/v1/chat/completions", json={
                "model": "finetuned",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_msg}],
                "max_tokens": 50, "temperature": 0.1,
            }, timeout=60.0)
            content = resp.json()["choices"][0]["message"]["content"].strip()
            try:
                score = json.loads(content).get("score")
            except json.JSONDecodeError:
                brace = content.rfind("{")
                try:
                    score = json.loads(content[brace:]).get("score") if brace >= 0 else None
                except (json.JSONDecodeError, TypeError):
                    score = None
            if score not in LABELS:
                score = "Partially correct"
                errors += 1
        except Exception:
            score = "Partially correct"
            errors += 1
        predictions[sample["id"]] = score
        if (i + 1) % 500 == 0:
            print(f"  7B [{i+1}/{len(test_data)}] {(i+1)/(time.time()-start):.1f} s/s | err={errors}")
    print(f"  7B {output_name}: {len(predictions)} scored, {errors} errors, {time.time()-start:.0f}s")
    return predictions

def get_knn_predictions(test_data, train_data, k=7):
    """Pure TF-IDF kNN predictions."""
    train_by_q = defaultdict(list)
    for s in train_data:
        train_by_q[s["question_id"]].append(s)
    predictions = {}
    confs = {}
    for sample in test_data:
        q_train = train_by_q.get(sample["question_id"], [])
        if not q_train:
            predictions[sample["id"]] = "Partially correct"
            confs[sample["id"]] = 0.33
            continue
        answers = [s["answer"] for s in q_train] + [sample["answer"]]
        vec = TfidfVectorizer(max_features=5000)
        mat = vec.fit_transform(answers)
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        top_k = np.argsort(sims)[-k:]
        votes = [q_train[j]["score"] for j in top_k]
        vc = Counter(votes)
        predictions[sample["id"]] = vc.most_common(1)[0][0]
        confs[sample["id"]] = vc.most_common(1)[0][1] / k
    return predictions, confs

def load_submission_as_preds(path):
    """Load a submission JSON as {id: score} dict."""
    data = load_json(path)
    return {d["id"]: d["score"] for d in data}

def build_features(sample_ids, model_preds, knn_confs, test_data_by_id, train_data):
    """Build feature matrix for stacking."""
    train_by_q = defaultdict(list)
    for s in train_data:
        train_by_q[s["question_id"]].append(s)
    q_diff = {}
    for qid, samps in train_by_q.items():
        counts = Counter(s["score"] for s in samps)
        q_diff[qid] = max(counts.values()) / sum(counts.values())

    X = []
    for sid in sample_ids:
        sample = test_data_by_id[sid]
        feats = []
        for model_name in model_preds:
            pred = LM.get(model_preds[model_name].get(sid, "Partially correct"), 1)
            feats.extend([1 if pred == i else 0 for i in range(3)])
        feats.append(knn_confs.get(sid, 0.33))
        feats.append(len(sample["answer"]) / 1000)
        feats.append(q_diff.get(sample["question_id"], 0.33))
        X.append(feats)
    return np.array(X)

def main():
    print("=== Loading data ===")
    train = load_json(DATA_DIR / "ALICE_LP_train_3way__v2.json")
    trial = load_json(DATA_DIR / "ALICE_LP_trial_3way__v2.json")
    test_seen = load_json(DATA_DIR / "3way_unseen_answers_eval.json")
    test_unseen = load_json(DATA_DIR / "3way_unseen_questions_eval.json")
    all_train = train + trial

    for track_name, test_data in [("unseen_answers", test_seen), ("unseen_questions", test_unseen)]:
        print(f"\n=== Track: {track_name} ({len(test_data)} samples) ===")
        test_by_id = {s["id"]: s for s in test_data}

        # Step 1: Score with 7B
        print("Step 1: Scoring with 7B via vLLM...")
        preds_7b = score_7b_via_vllm(test_data, track_name)

        # Step 2: kNN predictions
        print("Step 2: kNN predictions...")
        preds_knn, confs_knn = get_knn_predictions(test_data, all_train)

        # Step 3: Load existing predictions
        print("Step 3: Loading existing model predictions...")
        # 32B
        preds_32b = load_submission_as_preds(
            SUBMISSIONS_DIR / f"32b_ft_{track_name}_3way.json")
        # 14B
        sub_14b_files = sorted(RESULTS_DIR.glob(f"submission_test_14b_{track_name}_*.json"))
        preds_14b = load_submission_as_preds(sub_14b_files[-1]) if sub_14b_files else {}
        # Gemini
        gemini_files = sorted((PROJECT_ROOT / "results" / "strategy_c5c").glob("submission_test_3way_*.json"))
        if track_name == "unseen_answers":
            preds_gemini = load_submission_as_preds(gemini_files[0]) if gemini_files else {}
        else:
            preds_gemini = load_submission_as_preds(gemini_files[-1]) if len(gemini_files) > 1 else {}

        print(f"  32B: {len(preds_32b)}, 14B: {len(preds_14b)}, 7B: {len(preds_7b)}, Gemini: {len(preds_gemini)}, kNN: {len(preds_knn)}")

        model_preds = {"32b": preds_32b, "14b": preds_14b, "7b": preds_7b, "gemini": preds_gemini, "knn": preds_knn}

        # Step 4: Train stacking on TRIAL data
        print("Step 4: Training stacking meta-learner on trial data...")
        # Get trial predictions from each model (using existing results)
        trial_preds = {}
        # 32B trial
        ft32_files = sorted(RESULTS_DIR.glob("predictions_confidence_32b_*.json"))
        if ft32_files:
            trial_preds["32b"] = {p["id"]: p["pred"] for p in load_json(ft32_files[-1]) if "pred" in p}
        # 14B trial
        ft14_files = sorted(RESULTS_DIR.glob("predictions_finetune_14b_trial_*.json"))
        if ft14_files:
            trial_preds["14b"] = {p["id"]: p["pred"] for p in load_json(ft14_files[-1]) if "pred" in p}
        # 7B trial
        ft7_files = sorted(RESULTS_DIR.glob("predictions_finetune_trainonly_trial_*.json"))
        if ft7_files:
            trial_preds["7b"] = {p["id"]: p["pred"] for p in load_json(ft7_files[-1]) if "pred" in p}
        # Gemini/Q26 trial
        q26_files = sorted(RESULTS_DIR.glob("predictions_q26_full_*.json"))
        if q26_files:
            trial_preds["gemini"] = {p["id"]: p["pred"] for p in load_json(q26_files[-1]) if "pred" in p}
        # kNN trial
        trial_knn, trial_knn_confs = get_knn_predictions(trial, train)
        trial_preds["knn"] = trial_knn

        trial_by_id = {s["id"]: s for s in trial}
        common_trial = set.intersection(*[set(v.keys()) for v in trial_preds.values()])
        print(f"  Common trial samples for training: {len(common_trial)}")

        trial_ids = sorted(common_trial)
        X_train = build_features(trial_ids, trial_preds, trial_knn_confs, trial_by_id, train)
        y_train = np.array([LM[trial_by_id[sid]["score"]] for sid in trial_ids])

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        print(f"  Meta-learner trained on {len(X_train)} samples")

        # Step 5: Apply to test
        print("Step 5: Generating ensemble predictions...")
        common_test = set.intersection(*[set(v.keys()) for v in model_preds.values()])
        print(f"  Common test samples: {len(common_test)}")

        test_ids = sorted(common_test)
        X_test = build_features(test_ids, model_preds, confs_knn, test_by_id, all_train)
        y_pred = clf.predict(X_test)

        # Build submission
        id_to_pred = {sid: LABELS[p] for sid, p in zip(test_ids, y_pred)}
        # Fill any missing with 32B prediction as fallback
        for s in test_data:
            if s["id"] not in id_to_pred:
                id_to_pred[s["id"]] = preds_32b.get(s["id"], "Partially correct")

        sub_3way = [{"id": s["id"], "question_id": s["question_id"], "score": id_to_pred[s["id"]]} for s in test_data]
        sub_2way = [{"id": s["id"], "question_id": s["question_id"],
                     "score": "Incorrect" if id_to_pred[s["id"]] in ("Incorrect", "Partially correct") else "Correct"}
                    for s in test_data]

        # Save
        with open(SUBMISSIONS_DIR / f"ensemble_{track_name}_3way.json", "w") as f:
            json.dump(sub_3way, f, ensure_ascii=False)
        with open(SUBMISSIONS_DIR / f"ensemble_{track_name}_2way.json", "w") as f:
            json.dump(sub_2way, f, ensure_ascii=False)

        # CodaBench zips
        cb_dir = SUBMISSIONS_DIR / "codabench"
        track_num_3 = "1" if track_name == "unseen_answers" else "3"
        track_num_2 = "2" if track_name == "unseen_answers" else "4"

        with open("/tmp/submission.json", "w") as f:
            json.dump(sub_3way, f, ensure_ascii=False)
        with zipfile.ZipFile(str(cb_dir / f"track{track_num_3}_3way_ensemble.zip"), "w") as z:
            z.write("/tmp/submission.json", "submission.json")

        with open("/tmp/submission.json", "w") as f:
            json.dump(sub_2way, f, ensure_ascii=False)
        with zipfile.ZipFile(str(cb_dir / f"track{track_num_2}_2way_ensemble.zip"), "w") as z:
            z.write("/tmp/submission.json", "submission.json")

        dist = Counter(id_to_pred.values())
        print(f"  Saved: ensemble_{track_name}_3way/2way + CodaBench zips")
        print(f"  Distribution: {dict(dist)}")

    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
