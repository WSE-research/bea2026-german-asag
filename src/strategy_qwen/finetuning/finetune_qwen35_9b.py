"""Fine-tune Qwen3.5-9B with bf16 LoRA on 2 GPUs. Train-only for unbiased eval."""
import json, time, torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen35-9b-lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
MODEL_NAME = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

def main():
    print("Started:", datetime.now().isoformat())
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    print("Train-only samples:", len(train_data))

    chat_data = []
    for s in train_data:
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

    print("Loading Qwen3.5-9B in bf16 on 2 GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

    print("GPU 0:", round(torch.cuda.memory_allocated(0)/1e9, 1), "GB")
    print("GPU 1:", round(torch.cuda.memory_allocated(1)/1e9, 1), "GB")

    lora_config = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False)  # Qwen3.5 thinking mode OFF
        return {"text": text}
    dataset = dataset.map(format_chat)
    print("Dataset:", len(dataset))

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, peft_config=lora_config)

    print("Training: Qwen3.5-9B, rank=32, 3 epochs, lr=2e-4, batch=2x8")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print("Done in", round(elapsed/60, 1), "min")

    checkpoints = sorted((OUTPUT_DIR / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last_ckpt = checkpoints[-1] if checkpoints else None
    print("Checkpoint:", last_ckpt)

    # Merge adapter
    if last_ckpt:
        print("Merging adapter...")
        from peft import PeftModel
        del trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cpu")
        merged = PeftModel.from_pretrained(base, str(last_ckpt))
        merged = merged.merge_and_unload()
        merged_path = PROJECT_ROOT / "models" / "qwen35-9b-lora-merged"
        merged.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        print("Merged to:", merged_path)
        del merged, base; gc.collect()

    print("Completed:", datetime.now().isoformat())

if __name__ == "__main__":
    main()
