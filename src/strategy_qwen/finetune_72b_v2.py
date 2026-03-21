"""Fine-tune Qwen2.5-72B-Instruct-AWQ with LoRA. Uses .venv-72b with compatible autoawq."""
import json, time, torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-72b-lora-v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"

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

    print("Loading 72B-AWQ model on 2 GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    dataset = dataset.map(format_chat)
    print("Dataset:", len(dataset))

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        warmup_steps=50, num_train_epochs=3, learning_rate=1e-4,
        fp16=True, logging_steps=25, save_strategy="epoch",
        seed=42, report_to="none", gradient_checkpointing=True,
        max_seq_length=2048, dataset_text_field="text", packing=False)

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, peft_config=lora_config)

    print("Training: 72B-AWQ, rank=16, epochs=3, lr=1e-4, batch=1x16")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print("Done in", round(elapsed/60, 1), "min")

    checkpoints = sorted((OUTPUT_DIR / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    print("Checkpoint:", checkpoints[-1] if checkpoints else "NONE")
    print("Completed:", datetime.now().isoformat())

if __name__ == "__main__":
    main()
