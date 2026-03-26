"""72B fine-tune on H200: AWQ + LoRA r=32, all data (train+trial), 3 epochs.

Matches the successful 32B-alldata config but at 72B scale.
H200 NVL has 150 GB VRAM — AWQ model uses ~38 GB, leaving ~112 GB for training.
"""
import transformers.loss.loss_utils as lu
import torch

# Patch loss for device safety (same as finetune_72b_awq.py)
orig_fce = lu.fixed_cross_entropy
def patched_fce(logits, labels, n, ignore_index=-100, **kw):
    labels = labels.to(logits.device)
    if isinstance(n, torch.Tensor): n = n.to(logits.device)
    return orig_fce(logits, labels, n, ignore_index, **kw)
lu.fixed_cross_entropy = patched_fce

import json, time
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-72b-lora-alldata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_trial_3way__v2.json"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

def main():
    print("=" * 60)
    print("72B H200 FINE-TUNE — AWQ + LoRA r=32, ALL DATA, 3 EPOCHS")
    print("=" * 60)
    print("Started:", datetime.now().isoformat())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load ALL data (train + trial)
    with open(TRAIN_FILE) as f:
        data = json.load(f)
    with open(TRIAL_FILE) as f:
        data += json.load(f)
    print(f"ALL data: {len(data)} samples (train + trial)")

    # Prepare chat format
    chat_data = []
    for s in data:
        r = s["rubric"]
        user_msg = ("Frage: " + s["question"] + "\n\nBewertungsrubrik:\n" +
                    "- Correct: " + r["Correct"] + "\n" +
                    "- Partially correct: " + r["Partially correct"] + "\n" +
                    "- Incorrect: " + r["Incorrect"] + "\n\n" +
                    "Schuelerantwort: " + s["answer"])
        chat_data.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps({"score": s["score"]}, ensure_ascii=False)},
        ]})

    jsonl_path = OUTPUT_DIR / "train_chat.jsonl"
    with open(jsonl_path, "w") as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16,
    )
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # LoRA config — matching the successful 32B recipe
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

    # Training config — generous thanks to H200 headroom
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
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

    print(f"Training 72B-AWQ: r=32, all targets, 3 epochs, lr=2e-4, grad_ckpt=True")
    print(f"GPU memory before train: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")

    # Save final adapter
    trainer.save_model(str(OUTPUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
    print(f"Saved adapter to {OUTPUT_DIR / 'adapter'}")

    # Report checkpoints
    checkpoints = sorted(
        (OUTPUT_DIR / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    for cp in checkpoints:
        print(f"  Checkpoint: {cp.name}")
    print(f"SUBMISSION MODEL: {checkpoints[-1] if checkpoints else 'NONE'}")
    print("Completed:", datetime.now().isoformat())

if __name__ == "__main__":
    main()
