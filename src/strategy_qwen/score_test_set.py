"""
BEA26 Test Set Scoring Pipeline

Runs all best models on the test set and produces submission files.
Usage:
    python -m src.strategy_qwen.score_test_set --test-file data/raw/3way/ALICE_LP_test_3way.json

Models to run:
1. Fine-tuned Qwen2.5-7B (train+trial) via vLLM — our best
2. Qwen3.5-27B Q26 prompt engineering via vLLM
3. Gemini Flash C5c via OpenRouter (needs API key)
"""
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Correct", "Partially correct", "Incorrect"]


def score_with_vllm(test_file, model_name, served_name, output_prefix, system_prompt=None):
    """Score test set with a vLLM-served model."""
    import httpx

    VLLM_URL = "http://localhost:8081/v1/chat/completions"
    DEFAULT_SYSTEM = (
        'Du bist ein Bewertungssystem fuer Schuelerantworten. '
        'Bewerte die Antwort anhand der Rubrik. '
        'Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    system = system_prompt or DEFAULT_SYSTEM

    with open(test_file) as f:
        data = json.load(f)

    predictions = []
    errors = 0
    start = time.time()

    for i, sample in enumerate(data):
        r = sample["rubric"]
        user_msg = (
            "Frage: " + sample["question"] + "\n\n"
            "Bewertungsrubrik:\n"
            "- Correct: " + r["Correct"] + "\n"
            "- Partially correct: " + r["Partially correct"] + "\n"
            "- Incorrect: " + r["Incorrect"] + "\n\n"
            "Schuelerantwort: " + sample["answer"]
        )

        try:
            resp = httpx.post(VLLM_URL, json={
                "model": served_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            }, timeout=60.0)
            resp.raise_for_status()
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
                score = "Partially correct"  # safe fallback for submission
                errors += 1

            predictions.append({
                "id": sample["id"],
                "question_id": sample["question_id"],
                "score": score,
            })
        except Exception as e:
            errors += 1
            predictions.append({
                "id": sample["id"],
                "question_id": sample["question_id"],
                "score": "Partially correct",  # fallback
            })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(data)}] {(i+1)/elapsed:.1f} s/s | err={errors}")

    elapsed = time.time() - start
    print(f"  {output_prefix}: {len(predictions)} scored, {errors} errors, {elapsed:.0f}s")

    # Save submission (3-way)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_3way = SUBMISSION_DIR / f"{output_prefix}_3way_{ts}.json"
    with open(sub_3way, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Save submission (2-way: Partially correct -> Incorrect)
    preds_2way = []
    for p in predictions:
        score_2way = "Incorrect" if p["score"] in ("Incorrect", "Partially correct") else "Correct"
        preds_2way.append({"id": p["id"], "question_id": p["question_id"], "score": score_2way})

    sub_2way = SUBMISSION_DIR / f"{output_prefix}_2way_{ts}.json"
    with open(sub_2way, "w") as f:
        json.dump(preds_2way, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {sub_3way.name}, {sub_2way.name}")
    return predictions


def score_with_gemini(test_file, output_prefix):
    """Score test set with Gemini Flash C5c via OpenRouter."""
    # This uses the existing C5c strategy
    print(f"  To score with Gemini, run:")
    print(f"  python -m src.strategy_c5c_adaptive.run --split test --workers 5")
    print(f"  (requires OPENROUTER_API_KEY in .env)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEA26 Test Set Scoring Pipeline")
    parser.add_argument("--test-file", required=True, help="Path to test JSON")
    parser.add_argument("--model", choices=["finetuned", "q26", "gemini", "all"], default="all")
    parser.add_argument("--served-name", default="finetuned", help="vLLM served model name")
    args = parser.parse_args()

    print(f"BEA26 Test Set Scoring — {datetime.now().isoformat()}")
    print(f"Test file: {args.test_file}")

    if args.model in ("finetuned", "all"):
        print(f"\n=== Fine-tuned Qwen2.5-7B ===")
        score_with_vllm(args.test_file, "Qwen2.5-7B-QLoRA", args.served_name,
                        "finetuned_qwen25_7b")

    if args.model in ("gemini", "all"):
        print(f"\n=== Gemini Flash C5c ===")
        score_with_gemini(args.test_file, "gemini_c5c")

    print(f"\nDone. Submissions in: {SUBMISSION_DIR}")
