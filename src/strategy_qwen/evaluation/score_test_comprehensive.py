"""
Comprehensive test set scoring pipeline.
Runs when test data drops (expected 2026-03-21).

Models to score:
1. Fine-tuned 32B (all data) — via direct PEFT inference
2. Fine-tuned 14B (all data) — via vLLM merged model  
3. Fine-tuned 7B (all data) — via vLLM merged model
4. Stacking ensemble — LogReg over models 1-3 + Q26 + kNN

Usage:
    python -m src.strategy_qwen.score_test_comprehensive \
        --test-3way data/raw/3way/ALICE_LP_test_3way.json \
        --test-unseen data/raw/3way/ALICE_LP_test_unseen_3way.json
"""
import json, time, torch, numpy as np, argparse
from pathlib import Path
from collections import Counter, defaultdict
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'


def score_with_peft(test_data, model_name, adapter_path, device_map="auto"):
    """Score with a PEFT adapter model."""
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictions = {}
    start = time.time()
    for i, sample in enumerate(test_data):
        r = sample["rubric"]
        user_msg = ("Frage: " + sample["question"] + "\n\nBewertungsrubrik:\n" +
                    "- Correct: " + r["Correct"] + "\n" +
                    "- Partially correct: " + r["Partially correct"] + "\n" +
                    "- Incorrect: " + r["Incorrect"] + "\n\n" +
                    "Schuelerantwort: " + sample["answer"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            score = json.loads(response).get("score")
        except:
            brace = response.rfind("{")
            try: score = json.loads(response[brace:]).get("score") if brace >= 0 else None
            except: score = None
        predictions[sample["id"]] = score if score in LABELS else "Partially correct"
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(test_data)}] {(i+1)/(time.time()-start):.1f} s/s")

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"  Scored {len(predictions)} in {time.time()-start:.0f}s")
    return predictions


def score_with_knn(test_data, train, k=7):
    """Score with TF-IDF kNN."""
    train_by_q = defaultdict(list)
    for s in train: train_by_q[s["question_id"]].append(s)
    predictions = {}
    confidences = {}
    for sample in test_data:
        q_train = train_by_q.get(sample["question_id"], [])
        if not q_train:
            predictions[sample["id"]] = "Partially correct"
            confidences[sample["id"]] = 0.33
            continue
        answers = [s["answer"] for s in q_train] + [sample["answer"]]
        vec = TfidfVectorizer(max_features=5000)
        mat = vec.fit_transform(answers)
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        top_k = np.argsort(sims)[-k:]
        votes = [q_train[j]["score"] for j in top_k]
        vc = Counter(votes)
        predictions[sample["id"]] = vc.most_common(1)[0][0]
        confidences[sample["id"]] = vc.most_common(1)[0][1] / k
    return predictions, confidences


def save_submission(predictions, test_data, name):
    """Save 3-way and 2-way submission files."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    # 3-way
    sub_3way = [{"id": s["id"], "question_id": s["question_id"], "score": predictions.get(s["id"], "Partially correct")} for s in test_data]
    path_3way = SUBMISSION_DIR / f"{name}_3way_{ts}.json"
    with open(path_3way, "w") as f: json.dump(sub_3way, f, indent=2, ensure_ascii=False)
    # 2-way
    sub_2way = [{"id": s["id"], "question_id": s["question_id"], "score": "Incorrect" if predictions.get(s["id"]) in ("Incorrect", "Partially correct") else "Correct"} for s in test_data]
    path_2way = SUBMISSION_DIR / f"{name}_2way_{ts}.json"
    with open(path_2way, "w") as f: json.dump(sub_2way, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path_3way.name}, {path_2way.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-3way", required=True, help="Test file (seen questions)")
    parser.add_argument("--test-unseen", default=None, help="Test file (unseen questions)")
    args = parser.parse_args()

    with open(TRAIN_FILE) as f: train = json.load(f)
    with open(TRIAL_FILE) as f: trial = json.load(f)
    all_train = train + trial

    for test_file, track_name in [(args.test_3way, "seen"), (args.test_unseen, "unseen")]:
        if not test_file: continue
        print(f"\n=== Scoring {track_name} track: {test_file} ===")
        with open(test_file) as f: test_data = json.load(f)
        print(f"  {len(test_data)} test samples")

        # Model 1: 32B fine-tuned (all data)
        adapter_32b = PROJECT_ROOT / "models" / "qwen25-32b-lora-alldata" / "checkpoints" / "checkpoint-1482"
        if adapter_32b.exists():
            print("\n  Model 1: 32B fine-tuned (all data)")
            preds_32b = score_with_peft(test_data, "Qwen/Qwen2.5-32B-Instruct", adapter_32b, device_map="auto")
            save_submission(preds_32b, test_data, f"32b_ft_{track_name}")
        else:
            print("  32B adapter not found, skipping")
            preds_32b = None

        # Model 2: 14B fine-tuned (all data)
        adapter_14b = PROJECT_ROOT / "models" / "qwen25-14b-lora-alldata" / "checkpoints"
        ckpts_14b = sorted(adapter_14b.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if adapter_14b.exists() else []
        if ckpts_14b:
            print("\n  Model 2: 14B fine-tuned (all data)")
            preds_14b = score_with_peft(test_data, "Qwen/Qwen2.5-14B-Instruct", ckpts_14b[-1], device_map={"": 0})
            save_submission(preds_14b, test_data, f"14b_ft_{track_name}")
        else:
            print("  14B adapter not found, skipping")
            preds_14b = None

        # Model 3: kNN
        print("\n  Model 3: kNN (TF-IDF k=7)")
        preds_knn, confs_knn = score_with_knn(test_data, all_train)
        save_submission(preds_knn, test_data, f"knn_{track_name}")

        # Ensemble (if we have at least 2 models)
        available_preds = {k: v for k, v in [("32b", preds_32b), ("14b", preds_14b), ("knn", preds_knn)] if v}
        if len(available_preds) >= 2:
            print(f"\n  Majority vote ensemble ({len(available_preds)} models)")
            ensemble_preds = {}
            for sample in test_data:
                sid = sample["id"]
                votes = [available_preds[m][sid] for m in available_preds]
                ensemble_preds[sid] = Counter(votes).most_common(1)[0][0]
            save_submission(ensemble_preds, test_data, f"ensemble_{track_name}")

        print(f"\n  {track_name} track done!")

    print("\nAll done! Check submissions/ folder.")

if __name__ == "__main__":
    main()
