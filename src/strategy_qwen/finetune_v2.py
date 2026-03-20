"""
QLoRA fine-tuning of Qwen2.5-7B-Instruct — using HuggingFace TRL + PEFT directly.
(Unsloth had compatibility issues with current TRL version)
"""
import json
import os
import time
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-7b-qlora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = DATA_DIR / "ALICE_LP_trial_3way__v2.json"

SYSTEM_PROMPT = (
    "Du bist ein Bewertungssystem fuer Schuelerantworten. "
    "Bewerte die Antwort anhand der Rubrik. "
    'Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def prepare_data():
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    with open(TRIAL_FILE) as f:
        trial_data = json.load(f)

    all_data = train_data + trial_data
    print(f"Total: {len(all_data)} (train={len(train_data)}, trial={len(trial_data)})")

    chat_data = []
    for s in all_data:
        rubric = s["rubric"]
        user_msg = (
            f"Frage: {s['question']}\n\n"
            f"Bewertungsrubrik:\n"
            f"- Correct: {rubric['Correct']}\n"
            f"- Partially correct: {rubric['Partially correct']}\n"
            f"- Incorrect: {rubric['Incorrect']}\n\n"
            f"Schuelerantwort: {s['answer']}"
        )
        chat_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps({"score": s["score"]}, ensure_ascii=False)},
            ]
        })

    jsonl_path = OUTPUT_DIR / "train_chat.jsonl"
    with open(jsonl_path, "w") as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {jsonl_path}")
    return jsonl_path


def finetune(jsonl_path):
    print("\n=== Loading model with 4-bit quantization ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("\n=== Configuring LoRA ===")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat)
    print(f"Dataset: {len(dataset)} samples")

    # Training
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )

    print("\n=== Starting training ===")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed/60:.1f} min ===")

    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved to {adapter_path}")
    return adapter_path


def run_evaluation(adapter_path):
    from peft import PeftModel
    from sklearn.metrics import cohen_kappa_score

    print("\n=== Loading fine-tuned model ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    with open(TRIAL_FILE) as f:
        trial_data = json.load(f)

    LABELS = ["Correct", "Partially correct", "Incorrect"]
    LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
    golds, preds = [], []
    errors = 0
    start = time.time()

    for i, sample in enumerate(trial_data):
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
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        try:
            obj = json.loads(response)
            score = obj.get("score")
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
            elapsed_so_far = time.time() - start
            print(f"  [{i+1}/{len(trial_data)}] {(i+1)/elapsed_so_far:.1f} s/s | err={errors}")

    elapsed = time.time() - start
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

    print(f"\n{'='*60}")
    print(f"  FINE-TUNED Qwen2.5-7B QLoRA ({len(golds)} scored)")
    print(f"{'='*60}")
    print(f"  QWK: {qwk:.4f}  Acc: {acc:.1%}  Err: {errors}")
    for label in LABELS:
        s = per_class[label]
        print(f"  {label:>20s}: P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f}")
    print(f"\n  vs Gemini C5c:     QWK=0.748")
    print(f"  vs Qwen3.5 Q26:   QWK=0.721")
    print(f"  NOTE: Trial was in training set!")
    print(f"{'='*60}")

    results = {
        "qwk": round(qwk, 4), "accuracy": round(acc, 4), "errors": errors,
        "scored": len(golds), "total": len(trial_data),
        "elapsed_s": round(elapsed, 1), "per_class": per_class,
        "model": "Qwen2.5-7B-Instruct-QLoRA", "variant": "finetune_v2",
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / "strategy_qwen"
    with open(results_dir / f"metrics_finetune_v2_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    print(f"Started: {datetime.now().isoformat()}")
    jsonl_path = prepare_data()
    adapter_path = finetune(jsonl_path)
    run_evaluation(adapter_path)
    print(f"Completed: {datetime.now().isoformat()}")
