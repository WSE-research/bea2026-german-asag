"""72B-AWQ fine-tune: single GPU, LoRA r=8, no grad checkpointing, batch=1."""
import transformers.loss.loss_utils as lu
import torch

# Patch loss for device safety
orig_fce = lu.fixed_cross_entropy
def patched_fce(logits, labels, n, ignore_index=-100, **kw):
    labels = labels.to(logits.device)
    if isinstance(n, torch.Tensor): n = n.to(logits.device)
    return orig_fce(logits, labels, n, ignore_index, **kw)
lu.fixed_cross_entropy = patched_fce

import json, time, datasets
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-72b-lora-final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "3way" / "ALICE_LP_train_3way__v2.json"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'

print("Started:", datetime.now().isoformat())

# Load and prepare data
with open(TRAIN_FILE) as f:
    train_data = json.load(f)
print("Train samples:", len(train_data))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

dataset = datasets.Dataset.from_dict({"text": texts})
tokenized = dataset.map(
    lambda ex: tokenizer(ex["text"], truncation=True, max_length=1024, padding=False),
    remove_columns=["text"]
)
print("Dataset tokenized:", len(tokenized))

# Load model on single GPU
print("Loading 72B-AWQ on GPU 0...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

# LoRA — minimal rank, only q/v projections to save memory
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training — minimal memory footprint
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # effective batch 32
    warmup_steps=20,
    num_train_epochs=2,  # just 2 epochs to save time
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    seed=42,
    report_to="none",
    gradient_checkpointing=False,  # saves memory during backward but uses more compute
    optim="adamw_torch_fused",  # most memory-efficient optimizer
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Training 72B-AWQ: r=8, 2 epochs, lr=5e-5, batch=1x32")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"Done in {elapsed/60:.1f} min")

model.save_pretrained(str(OUTPUT_DIR / "adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
print("Saved to", OUTPUT_DIR / "adapter")
print("Completed:", datetime.now().isoformat())
