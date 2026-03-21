"""
Score trial/test set using the fine-tuned Qwen2.5-7B model.
The fine-tuned model needs only a minimal prompt (no few-shot, no rules)
since the scoring behavior is learned from training data.
"""
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import httpx

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"

VLLM_URL = "http://localhost:8081/v1/chat/completions"
MODEL = "finetune"  # or the merged model path

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "Du bist ein Bewertungssystem fuer Schuelerantworten. "
    "Bewerte die Antwort anhand der Rubrik. "
    'Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
)


def score_dataset(input_file, output_name, model_name=None):
    with open(input_file) as f:
        data = json.load(f)

    model = model_name or MODEL
    golds, preds = [], []
    errors = 0
    raw_results = []
    start = time.time()

    for i, sample in enumerate(data):
        rubric = sample["rubric"]
        user_msg = (
            f"Frage: {sample['question']}\n\n"
            f"Bewertungsrubrik:\n"
            f"- Correct: {rubric['Correct']}\n"
            f"- Partially correct: {rubric['Partially correct']}\n"
            f"- Incorrect: {rubric['Incorrect']}\n\n"
            f"Schuelerantwort: {sample['answer']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            resp = httpx.post(
                VLLM_URL,
                json={"model": model, "messages": messages,
                      "max_tokens": 50, "temperature": 0.1},
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            try:
                obj = json.loads(content)
                score = obj.get("score")
            except json.JSONDecodeError:
                brace = content.rfind("{")
                try:
                    score = json.loads(content[brace:]).get("score") if brace >= 0 else None
                except (json.JSONDecodeError, TypeError):
                    score = None

            if score in LABELS:
                if "score" in sample:  # has gold label
                    golds.append(sample["score"])
                preds.append(score)
                raw_results.append({"id": sample["id"], "question_id": sample["question_id"],
                                    "pred": score, "gold": sample.get("score")})
            else:
                errors += 1
                raw_results.append({"id": sample["id"], "error": f"Invalid: {score}"})
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(data)}] {(i+1)/elapsed:.1f} s/s | err={errors}")

    elapsed = time.time() - start

    # Metrics (only if gold labels available)
    if golds:
        g = [LABEL_MAP[x] for x in golds]
        p = [LABEL_MAP[x] for x in preds[:len(golds)]]
        qwk = cohen_kappa_score(g, p, weights="quadratic")
        acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)
        print(f"\n  {output_name}: QWK={qwk:.4f} Acc={acc:.1%} Err={errors} Time={elapsed:.0f}s")
    else:
        qwk = None
        acc = None
        print(f"\n  {output_name}: Scored {len(preds)} samples (no gold labels) Err={errors}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"variant": output_name, "model": model, "qwk": qwk, "accuracy": acc,
               "errors": errors, "scored": len(preds), "total": len(data), "elapsed_s": round(elapsed, 1)}
    with open(RESULTS_DIR / f"metrics_{output_name}_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"predictions_{output_name}_{ts}.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # Also save submission format
    submission = [{"id": r["id"], "question_id": r["question_id"], "score": r["pred"]}
                  for r in raw_results if "pred" in r]
    with open(RESULTS_DIR / f"submission_{output_name}_{ts}.json", "w") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON (trial or test)")
    parser.add_argument("--name", default="finetune_scoring", help="Output name prefix")
    parser.add_argument("--model", default=None, help="Model name for vLLM (default: auto-detect)")
    args = parser.parse_args()

    score_dataset(args.input, args.name, args.model)
