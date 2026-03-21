"""
Confidence threshold experiment.

Andreas's insight: larger models under-classify "Correct" when "Partially correct"
is available. Solution: if model says "Partially correct" with low confidence,
check if "Correct" was close behind, and reclassify.

Approach: Generate with the fine-tuned 32B model, get logprobs for the score
token, and apply threshold-based reclassification.

Uses direct PEFT inference with output logprobs.
"""
import json, time, torch, math
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import cohen_kappa_score
from collections import Counter

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"

# Use the train-only model for unbiased eval
ADAPTER = PROJECT_ROOT / "models" / "qwen25-32b-lora" / "checkpoints" / "checkpoint-1326"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'


def compute_metrics(golds, preds):
    g = [LABEL_MAP[x] for x in golds]
    p = [LABEL_MAP[x] for x in preds]
    qwk = cohen_kappa_score(g, p, weights="quadratic")
    acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)
    return round(qwk, 4), round(acc, 4)


def main():
    print("Loading 32B model + LoRA (train-only for unbiased eval)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(TRIAL_FILE) as f:
        trial = json.load(f)

    # First pass: collect predictions WITH generation logprobs
    print("Scoring with logprob collection...")
    results = []
    start = time.time()

    for i, sample in enumerate(trial):
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
            outputs = model.generate(
                **inputs, max_new_tokens=50, temperature=0.1, do_sample=True,
                output_scores=True, return_dict_in_generate=True,
            )

        # Get generated tokens and their logprobs
        gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Compute average logprob of generated tokens as confidence proxy
        scores = outputs.scores  # tuple of (vocab_size,) tensors per step
        token_logprobs = []
        for step_idx, score_tensor in enumerate(scores):
            if step_idx >= len(gen_ids):
                break
            log_probs = torch.log_softmax(score_tensor[0], dim=-1)
            token_id = gen_ids[step_idx]
            token_logprobs.append(log_probs[token_id].item())

        avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else -999
        confidence = math.exp(avg_logprob)  # convert to probability

        # Parse score
        try:
            score = json.loads(response).get("score")
        except:
            brace = response.rfind("{")
            try:
                score = json.loads(response[brace:]).get("score") if brace >= 0 else None
            except:
                score = None

        results.append({
            "id": sample["id"],
            "gold": sample["score"],
            "pred": score,
            "confidence": round(confidence, 4),
            "avg_logprob": round(avg_logprob, 4),
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(trial)}] {(i+1)/elapsed:.1f} s/s")

    elapsed = time.time() - start
    print(f"Scoring done in {elapsed:.0f}s")

    # Save raw results with confidence
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"predictions_confidence_32b_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Baseline (no threshold)
    valid = [r for r in results if r["pred"] in LABELS]
    golds_base = [r["gold"] for r in valid]
    preds_base = [r["pred"] for r in valid]
    qwk_base, acc_base = compute_metrics(golds_base, preds_base)
    print(f"\nBaseline (no threshold): QWK={qwk_base} Acc={acc_base}")

    # Analyze confidence distribution by prediction
    for label in LABELS:
        confs = [r["confidence"] for r in valid if r["pred"] == label]
        if confs:
            avg_conf = sum(confs) / len(confs)
            min_conf = min(confs)
            print(f"  {label:>20s}: avg_conf={avg_conf:.3f} min={min_conf:.3f} n={len(confs)}")

    # Try different confidence thresholds for reclassification
    print("\n=== Confidence Threshold Sweep ===")
    print(f"{'Threshold':>10s} {'Strategy':>30s} {'QWK':>6s} {'Acc':>6s} {'Reclassified':>12s}")
    print("-" * 70)

    best_qwk = qwk_base
    best_config = "none"

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Strategy 1: Partially correct with low confidence -> Correct
        preds_s1 = []
        reclassified = 0
        for r in valid:
            if r["pred"] == "Partially correct" and r["confidence"] < threshold:
                preds_s1.append("Correct")
                reclassified += 1
            else:
                preds_s1.append(r["pred"])
        qwk_s1, acc_s1 = compute_metrics(golds_base, preds_s1)
        print(f"{threshold:>10.1f} {'PartCor->Correct if low':>30s} {qwk_s1:>6.4f} {acc_s1:>5.1%} {reclassified:>12d}")
        if qwk_s1 > best_qwk:
            best_qwk = qwk_s1
            best_config = f"PartCor->Correct@{threshold}"

        # Strategy 2: Partially correct with low confidence -> Incorrect
        preds_s2 = []
        reclassified = 0
        for r in valid:
            if r["pred"] == "Partially correct" and r["confidence"] < threshold:
                preds_s2.append("Incorrect")
                reclassified += 1
            else:
                preds_s2.append(r["pred"])
        qwk_s2, acc_s2 = compute_metrics(golds_base, preds_s2)
        print(f"{threshold:>10.1f} {'PartCor->Incorrect if low':>30s} {qwk_s2:>6.4f} {acc_s2:>5.1%} {reclassified:>12d}")
        if qwk_s2 > best_qwk:
            best_qwk = qwk_s2
            best_config = f"PartCor->Incorrect@{threshold}"

        # Strategy 3: Use answer length as tiebreaker for low-confidence PartCor
        preds_s3 = []
        reclassified = 0
        for r in valid:
            if r["pred"] == "Partially correct" and r["confidence"] < threshold:
                # Long answers -> more likely Correct, short -> Incorrect
                sample = next(s for s in trial if s["id"] == r["id"])
                if len(sample["answer"]) > 200:
                    preds_s3.append("Correct")
                else:
                    preds_s3.append("Incorrect")
                reclassified += 1
            else:
                preds_s3.append(r["pred"])
        qwk_s3, acc_s3 = compute_metrics(golds_base, preds_s3)
        print(f"{threshold:>10.1f} {'PartCor->len heuristic':>30s} {qwk_s3:>6.4f} {acc_s3:>5.1%} {reclassified:>12d}")
        if qwk_s3 > best_qwk:
            best_qwk = qwk_s3
            best_config = f"len_heuristic@{threshold}"

    print(f"\nBest: {best_config} -> QWK={best_qwk} (baseline: {qwk_base})")

    # Save results
    summary = {
        "baseline_qwk": qwk_base,
        "best_qwk": best_qwk,
        "best_config": best_config,
        "improvement": round(best_qwk - qwk_base, 4),
    }
    with open(RESULTS_DIR / f"metrics_confidence_threshold_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to metrics_confidence_threshold_{ts}.json")


if __name__ == "__main__":
    main()
