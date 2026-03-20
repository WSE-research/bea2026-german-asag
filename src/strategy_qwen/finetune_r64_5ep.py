"""Fine-tune with LoRA rank 64 and 5 epochs (train-only) for unbiased comparison."""
import json, time, torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-7b-qlora-r64-5ep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

def prepare_data():
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    print("Train samples:", len(train_data))
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
    return jsonl_path

def finetune(jsonl_path):
    print("\n=== Loading model ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=64, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    dataset = dataset.map(format_chat)
    print("Dataset:", len(dataset))

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        warmup_steps=50, num_train_epochs=5, learning_rate=1e-4,
        bf16=True, logging_steps=25, save_strategy="epoch",
        seed=42, report_to="none", gradient_checkpointing=True,
        max_length=2048, dataset_text_field="text", packing=False,
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, peft_config=lora_config)

    print("\n=== Training (rank 64, 5 epochs, lr=1e-4) ===")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print("\n=== Done in", round(elapsed/60, 1), "min ===")

    # Save last checkpoint path
    checkpoints = sorted((OUTPUT_DIR / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last_ckpt = checkpoints[-1] if checkpoints else None
    print("Last checkpoint:", last_ckpt)
    return last_ckpt

if __name__ == "__main__":
    print("Started:", datetime.now().isoformat())
    jsonl_path = prepare_data()
    ckpt = finetune(jsonl_path)
    print("Checkpoint:", ckpt)
    print("Completed:", datetime.now().isoformat())
