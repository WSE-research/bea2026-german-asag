"""Score ALL test tracks with 32B all-data submission model."""
import json, time, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
ADAPTER = PROJECT_ROOT / "models" / "qwen25-32b-lora-alldata" / "checkpoints" / "checkpoint-1482"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
LABELS = ["Correct", "Partially correct", "Incorrect"]
SP = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

print("Loading 32B + adapter...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()
tokenizer = AutoModelForCausalLM  # placeholder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def score_file(input_path, output_name):
    with open(input_path) as f:
        data = json.load(f)
    print(f"Scoring {output_name}: {len(data)} samples")
    
    predictions = []
    errors = 0
    start = time.time()
    
    for i, sample in enumerate(data):
        r = sample["rubric"]
        if isinstance(r, str):
            import ast
            r = ast.literal_eval(r)
        user_msg = ("Frage: " + sample["question"] + "\n\nBewertungsrubrik:\n"
                    "- Correct: " + r["Correct"] + "\n"
                    "- Partially correct: " + r["Partially correct"] + "\n"
                    "- Incorrect: " + r["Incorrect"] + "\n\n"
                    "Schuelerantwort: " + sample["answer"])
        msgs = [{"role": "system", "content": SP}, {"role": "user", "content": user_msg}]
        inp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inp, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            score = json.loads(resp).get("score")
        except json.JSONDecodeError:
            brace = resp.rfind("{")
            try:
                score = json.loads(resp[brace:]).get("score") if brace >= 0 else None
            except (json.JSONDecodeError, TypeError):
                score = None
        if score not in LABELS:
            score = "Partially correct"
            errors += 1
        predictions.append({"id": sample["id"], "question_id": sample["question_id"], "score": score})
        if (i+1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(data)}] {(i+1)/elapsed:.1f} s/s | err={errors}")
    
    elapsed = time.time() - start
    print(f"  Done: {len(predictions)} scored, {errors} errors, {elapsed:.0f}s")
    
    # 3-way
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    with open(sub_dir / f"{output_name}_3way.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # 2-way
    preds_2way = [{"id": p["id"], "question_id": p["question_id"],
                   "score": "Incorrect" if p["score"] in ("Incorrect", "Partially correct") else "Correct"}
                  for p in predictions]
    with open(sub_dir / f"{output_name}_2way.json", "w") as f:
        json.dump(preds_2way, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved 3-way and 2-way submissions")
    return predictions

# Score all tracks
score_file("data/raw/3way/3way_unseen_answers_eval.json", "32b_ft_unseen_answers")
score_file("data/raw/3way/3way_unseen_questions_eval.json", "32b_ft_unseen_questions")
print("\nAll test tracks scored!")
