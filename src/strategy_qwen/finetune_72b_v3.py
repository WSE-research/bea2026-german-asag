"""72B AWQ fine-tune using Trainer directly (avoid SFTTrainer multi-GPU bugs)."""
import json, time, torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-72b-lora-v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"

SYSTEM_PROMPT = "Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {\"score\": \"Correct\" | \"Partially correct\" | \"Incorrect\"}"

def main():
    print("Started:", datetime.now().isoformat())
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    print("Train samples:", len(train_data))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare chat texts
    texts = []
    for s in train_data:
        r = s["rubric"]
        user_msg = ("Frage: " + s["question"] + "\n\nBewertungsrubrik:\n" +
                    "- Correct: " + r["Correct"] + "\n" +
                    "- Partially correct: " + r["Partially correct"] + "\n" +
                    "- Incorrect: " + r["Incorrect"] + "\n\n" +
                    "Schuelerantwort: " + s["answer"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps({"score": s["score"]}, ensure_ascii=False)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")

    # removed broken load_dataset
    
    import datasets
    dataset = datasets.Dataset.from_dict({"text": texts})
    tokenized = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=2048, padding=False), remove_columns=["text"])

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training 72B-AWQ with LoRA rank 16...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print("Done in", round(elapsed/60, 1), "min")

    model.save_pretrained(str(OUTPUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
    print("Saved adapter to", OUTPUT_DIR / "adapter")
    print("Completed:", datetime.now().isoformat())

if __name__ == "__main__":
    main()
