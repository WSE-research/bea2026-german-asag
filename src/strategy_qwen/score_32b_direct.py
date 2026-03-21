"""Score trial with 32B fine-tuned model using direct PEFT inference (no vLLM)."""
import json, time, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"
ADAPTER = PROJECT_ROOT / "models" / "qwen25-32b-lora" / "checkpoints" / "checkpoint-1326"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

print("Loading base model on 2 GPUs...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

with open(TRIAL_FILE) as f:
    trial = json.load(f)

golds, preds = [], []
errors = 0
start = time.time()

for i, sample in enumerate(trial):
    r = sample["rubric"]
    user_msg = ("Frage: " + sample["question"] + "\n\nBewertungsrubrik:\n" +
                "- Correct: " + r["Correct"] + "\n" +
                "- Partially correct: " + r["Partially correct"] + "\n" +
                "- Incorrect: " + r["Incorrect"] + "\n\n" +
                "Schuelerantwort: " + sample["answer"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    try:
        score = json.loads(response).get("score")
    except json.JSONDecodeError:
        brace = response.rfind("{")
        try:
            score = json.loads(response[brace:]).get("score") if brace >= 0 else None
        except (json.JSONDecodeError, TypeError):
            score = None
    
    if score in LABELS:
        golds.append(sample["score"])
        preds.append(score)
    else:
        errors += 1
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(trial)}] {(i+1)/elapsed:.1f} s/s | err={errors}")

elapsed = time.time() - start
g = [LABEL_MAP[x] for x in golds]
p = [LABEL_MAP[x] for x in preds]
qwk = cohen_kappa_score(g, p, weights="quadratic")
acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)

print(f"\n  32B fine-tuned (train-only): QWK={qwk:.4f} Acc={acc:.1%} Err={errors} Time={elapsed:.0f}s")
print(f"  vs 14B: QWK=0.753")
print(f"  vs Gemini: QWK=0.748")

results = {"variant": "finetune_32b_trial", "qwk": round(qwk, 4), "accuracy": round(acc, 4),
           "errors": errors, "scored": len(golds), "total": len(trial), "elapsed_s": round(elapsed, 1)}
ts = time.strftime("%Y%m%d_%H%M%S")
with open(RESULTS_DIR / f"metrics_finetune_32b_trial_{ts}.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
