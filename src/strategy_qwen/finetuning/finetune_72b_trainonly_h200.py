"""72B fine-tune on H200: QLoRA (4-bit NF4) + LoRA r=32, TRAIN ONLY (no trial), 3 epochs.

Unbiased evaluation: trains on 7,072 training samples only, then scores the
827 trial samples and computes QWK + accuracy to get a true out-of-sample metric.
"""
import json, time, torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-72b-lora-trainonly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}


def build_user_message(sample):
    r = sample["rubric"]
    return (
        "Frage: " + sample["question"] + "\n\n"
        "Bewertungsrubrik:\n"
        "- Correct: " + r["Correct"] + "\n"
        "- Partially correct: " + r["Partially correct"] + "\n"
        "- Incorrect: " + r["Incorrect"] + "\n\n"
        "Schuelerantwort: " + sample["answer"]
    )


def train():
    print("=" * 60)
    print("72B H200 FINE-TUNE — QLoRA NF4 + LoRA r=32, TRAIN ONLY, 3 EPOCHS")
    print("=" * 60)
    print("Started:", datetime.now().isoformat())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load ONLY training data (no trial!)
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    print(f"Train-only data: {len(train_data)} samples (NO trial)")

    # Verify trial NOT included
    with open(TRIAL_FILE) as f:
        trial_data = json.load(f)
    train_ids = {s["id"] for s in train_data}
    trial_ids = {s["id"] for s in trial_data}
    overlap = train_ids & trial_ids
    assert len(overlap) == 0, f"DATA LEAK: {len(overlap)} trial IDs in training data!"
    print(f"Trial data: {len(trial_data)} samples (held out for evaluation)")

    # Prepare chat format for training
    chat_data = []
    for s in train_data:
        user_msg = build_user_message(s)
        chat_data.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps({"score": s["score"]}, ensure_ascii=False)},
        ]})

    jsonl_path = OUTPUT_DIR / "train_chat.jsonl"
    with open(jsonl_path, "w") as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading {MODEL_NAME} in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # LoRA config — same as alldata recipe
    lora_config = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # Prepare dataset
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    dataset = dataset.map(format_chat)
    print(f"Dataset: {len(dataset)} samples")

    # Training config — identical to alldata
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
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
        model=model, train_dataset=dataset, args=training_args, peft_config=lora_config,
    )

    print(f"Training 72B-QLoRA: r=32, all targets, 3 epochs, lr=2e-4, grad_ckpt=True")
    print(f"GPU memory before train: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"Training done in {train_elapsed/60:.1f} min ({train_elapsed/3600:.1f} h)")

    # Save final adapter
    adapter_dir = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapter to {adapter_dir}")

    # Report checkpoints
    checkpoints = sorted(
        (OUTPUT_DIR / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    for cp in checkpoints:
        print(f"  Checkpoint: {cp.name}")

    # ===== PHASE 2: Score trial set for unbiased evaluation =====
    print("\n" + "=" * 60)
    print("PHASE 2: SCORING 827 TRIAL SAMPLES (OUT-OF-SAMPLE)")
    print("=" * 60)

    # Free trainer memory, reload model fresh with adapter
    del trainer
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(5)

    print("Reloading model with trained adapter for inference...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    print(f"Model loaded for inference. VRAM: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    golds, preds = [], []
    errors = 0
    raw_results = []
    score_start = time.time()

    for i, sample in enumerate(trial_data):
        user_msg = build_user_message(sample)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
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

        golds.append(sample["score"])
        preds.append(score)
        raw_results.append({
            "id": sample["id"],
            "question_id": sample["question_id"],
            "pred": score,
            "gold": sample["score"],
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - score_start
            g = [LABEL_MAP[x] for x in golds]
            p = [LABEL_MAP[x] for x in preds]
            qwk = cohen_kappa_score(g, p, weights="quadratic")
            acc = accuracy_score(golds, preds)
            print(f"  [{i+1}/{len(trial_data)}] QWK={qwk:.4f} Acc={acc:.1%} Err={errors} Speed={elapsed/(i+1):.2f}s/sample")

    score_elapsed = time.time() - score_start

    # Final metrics
    g = [LABEL_MAP[x] for x in golds]
    p = [LABEL_MAP[x] for x in preds]
    qwk = cohen_kappa_score(g, p, weights="quadratic")
    acc = accuracy_score(golds, preds)

    print(f"\n{'='*60}")
    print(f"72B QLoRA TRIAL RESULTS (OUT-OF-SAMPLE, train-only)")
    print(f"{'='*60}")
    print(f"QWK:           {qwk:.4f}")
    print(f"Accuracy:      {acc:.1%}")
    print(f"Errors:        {errors}/{len(trial_data)}")
    print(f"Train time:    {train_elapsed:.0f}s ({train_elapsed/3600:.1f}h)")
    print(f"Score time:    {score_elapsed:.0f}s ({score_elapsed/len(trial_data):.2f}s/sample)")
    print(f"Predictions:   {Counter(preds)}")
    print(f"Gold labels:   {Counter(golds)}")

    # Save metrics
    metrics = {
        "model": "72B-QLoRA-trainonly",
        "variant": "trainonly_trial_unbiased",
        "train_samples": len(train_data),
        "trial_samples": len(trial_data),
        "qwk": round(qwk, 6),
        "accuracy": round(acc, 6),
        "errors": errors,
        "train_elapsed_s": round(train_elapsed, 1),
        "score_elapsed_s": round(score_elapsed, 1),
        "config": {
            "base_model": MODEL_NAME,
            "quantization": "NF4",
            "lora_r": 32,
            "lora_alpha": 32,
            "lr": 2e-4,
            "epochs": 3,
            "batch_size": "1x16",
            "max_length": 2048,
        },
        "timestamp": datetime.now().isoformat(),
    }

    metrics_path = RESULTS_DIR / "metrics_72b_trainonly_trial.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # Save predictions
    ts = time.strftime("%Y%m%d_%H%M%S")
    preds_path = RESULTS_DIR / f"predictions_72b_trainonly_trial_{ts}.json"
    with open(preds_path, "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {preds_path}")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    train()
