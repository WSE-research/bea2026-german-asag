"""
QLoRA fine-tuning of Qwen2.5-7B-Instruct on BEA26 German ASAG training data.

Uses Unsloth for 2x faster training and 60% less VRAM.
Target: QWK > 0.748 (beating Gemini Flash C5c).
Reference: CHiL(L)Grader achieved QWK 0.80+ with fine-tuned Qwen-7B.
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. Prepare training data
# ============================================================

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-7b-qlora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = DATA_DIR / "ALICE_LP_trial_3way__v2.json"

# System prompt — keep it simple and consistent
SYSTEM_PROMPT = (
    "Du bist ein Bewertungssystem für Schülerantworten. "
    "Bewerte die Antwort anhand der Rubrik. "
    'Antworte ausschließlich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'
)


def build_chat_message(sample):
    """Convert a training sample to chat format."""
    rubric = sample["rubric"]
    user_msg = (
        f"Frage: {sample['question']}\n\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    assistant_msg = json.dumps({"score": sample["score"]}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def prepare_data():
    """Convert training data to chat JSONL format."""
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    with open(TRIAL_FILE) as f:
        trial_data = json.load(f)

    # Use both train + trial for fine-tuning (task rules allow it)
    all_data = train_data + trial_data
    print(f"Total samples: {len(all_data)} (train={len(train_data)}, trial={len(trial_data)})")

    # Convert to chat format
    chat_data = [build_chat_message(s) for s in all_data]

    # Save as JSONL
    jsonl_path = OUTPUT_DIR / "train_chat.jsonl"
    with open(jsonl_path, "w") as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(chat_data)} samples to {jsonl_path}")
    return jsonl_path, len(chat_data)


# ============================================================
# 2. Fine-tune with Unsloth
# ============================================================

def finetune(jsonl_path, num_samples):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print("\n=== Loading Qwen2.5-7B-Instruct with 4-bit quantization ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    print("\n=== Applying LoRA adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    print(f"\nDataset: {len(dataset)} samples")

    # Apply chat template
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Training arguments
    num_epochs = 3
    batch_size = 4
    grad_accum = 4  # effective batch = 16
    steps_per_epoch = num_samples // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs

    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} x {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: 2e-4")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=25,
            save_strategy="epoch",
            seed=42,
            report_to="none",
        ),
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("\n=== Starting fine-tuning ===")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\n=== Training complete in {elapsed/60:.1f} minutes ===")

    # Save the LoRA adapter
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    return adapter_path


# ============================================================
# 3. Evaluate on trial set
# ============================================================

def evaluate(adapter_path):
    from unsloth import FastLanguageModel
    from sklearn.metrics import cohen_kappa_score
    import re

    print("\n=== Loading fine-tuned model for evaluation ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    with open(TRIAL_FILE) as f:
        trial_data = json.load(f)

    # We trained on trial too, so this is in-sample for trial.
    # But we still evaluate to verify the model learned correctly.
    # The REAL test is on the unseen test set (released 2026-03-21).

    LABELS = ["Correct", "Partially correct", "Incorrect"]
    LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

    golds, preds = [], []
    errors = 0

    print(f"Evaluating on {len(trial_data)} trial samples...")
    start = time.time()

    for i, sample in enumerate(trial_data):
        rubric = sample["rubric"]
        user_msg = (
            f"Frage: {sample['question']}\n\n"
            f"Bewertungsrubrik:\n"
            f"- Correct: {rubric['Correct']}\n"
            f"- Partially correct: {rubric['Partially correct']}\n"
            f"- Incorrect: {rubric['Incorrect']}\n\n"
            f"Schülerantwort: {sample['answer']}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Parse score
        try:
            obj = json.loads(response)
            score = obj.get("score")
        except json.JSONDecodeError:
            # Try to find JSON in response
            brace = response.rfind("{")
            if brace >= 0:
                try:
                    obj = json.loads(response[brace:])
                    score = obj.get("score")
                except:
                    score = None
            else:
                score = None

        if score in LABELS:
            golds.append(sample["score"])
            preds.append(score)
        else:
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(trial_data)}] {(i+1)/elapsed:.1f} samples/s | errors: {errors}")

    elapsed = time.time() - start

    # Compute metrics
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
    print(f"  FINE-TUNED Qwen2.5-7B — Trial Set ({len(golds)} scored)")
    print(f"{'='*60}")
    print(f"  QWK: {qwk:.4f}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Errors: {errors}/{len(trial_data)}")
    print(f"  Time: {elapsed:.0f}s ({len(trial_data)/elapsed:.1f} samples/s)")
    for label in LABELS:
        s = per_class[label]
        print(f"  {label:>20s}: P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f}")
    print(f"\n  vs Gemini Flash C5c: QWK=0.748")
    print(f"  vs Qwen3.5-27B Q26: QWK=0.721")
    print(f"{'='*60}")

    # Save results
    results = {
        "qwk": round(qwk, 4), "accuracy": round(acc, 4), "errors": errors,
        "scored": len(golds), "total": len(trial_data),
        "elapsed_s": round(elapsed, 1), "per_class": per_class,
        "model": "Qwen2.5-7B-Instruct-QLoRA", "variant": "finetune_v1",
        "adapter_path": str(adapter_path),
        "note": "Trial set was included in training — this is in-sample eval. Real test on 2026-03-21."
    }
    results_dir = PROJECT_ROOT / "results" / "strategy_qwen"
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(results_dir / f"metrics_finetune_v1_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"BEA26 Qwen2.5-7B QLoRA Fine-Tuning")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"=" * 60)

    # Step 1: Prepare data
    jsonl_path, num_samples = prepare_data()

    # Step 2: Fine-tune
    adapter_path = finetune(jsonl_path, num_samples)

    # Step 3: Evaluate
    evaluate(adapter_path)

    print(f"\nCompleted: {datetime.now().isoformat()}")
