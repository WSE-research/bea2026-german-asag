"""
Overnight autonomous fine-tuning: 3 models sequentially.
1. Qwen3.5-9B (quick test, ~90 min)
2. Qwen3.5-27B (main comparison, ~3-4 hrs)
3. Qwen2.5-72B-AWQ (max capacity, overnight)

Each model: train-only (7072 samples) → merge → serve via vLLM → score trial → save results.
Also trains on all data (7899) for the submission model.
"""
import json
import os
import sys
import time
import subprocess
import signal
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import cohen_kappa_score
import httpx

PROJECT_ROOT = Path("/home/jgwozdz/bea26/bea2026-german-asag")
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "3way"
TRAIN_FILE = DATA_DIR / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = DATA_DIR / "ALICE_LP_trial_3way__v2.json"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = 'Du bist ein Bewertungssystem fuer Schuelerantworten. Bewerte die Antwort anhand der Rubrik. Antworte ausschliesslich mit JSON: {"score": "Correct" | "Partially correct" | "Incorrect"}'


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def prepare_jsonl(output_dir, include_trial=False):
    with open(TRAIN_FILE) as f:
        data = json.load(f)
    if include_trial:
        with open(TRIAL_FILE) as f:
            data += json.load(f)
    
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
    
    jsonl_path = output_dir / "train_chat.jsonl"
    with open(jsonl_path, "w") as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return jsonl_path, len(chat_data)


def run_finetune(model_name, output_dir, jsonl_path, num_samples,
                 lora_rank=32, epochs=3, lr=2e-4, batch_size=4, grad_accum=4,
                 use_4bit=True, multi_gpu=False):
    log(f"Loading {model_name} (4bit={use_4bit}, multi_gpu={multi_gpu})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if multi_gpu else {"": 0}

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map=device_map, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        # Full bf16 — for models where 4-bit causes CUDA errors
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map,
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    dataset = dataset.map(format_chat)
    log(f"Dataset: {len(dataset)} samples")
    
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=50, num_train_epochs=epochs, learning_rate=lr,
        bf16=True, logging_steps=25, save_strategy="epoch",
        seed=42, report_to="none", gradient_checkpointing=True,
        max_length=2048, dataset_text_field="text", packing=False,
    )
    
    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, peft_config=lora_config)
    
    log(f"Training: rank={lora_rank}, epochs={epochs}, lr={lr}, batch={batch_size}x{grad_accum}")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log(f"Training done in {elapsed/60:.1f} min")
    
    # Find last checkpoint
    checkpoints = sorted((output_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last_ckpt = checkpoints[-1] if checkpoints else None
    log(f"Last checkpoint: {last_ckpt}")
    
    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    return last_ckpt, elapsed


def merge_adapter(model_name, adapter_path, merged_path):
    log(f"Merging adapter from {adapter_path}...")
    if "AWQ" in model_name or "awq" in model_name.lower():
        # AWQ models can't be cleanly merged — use LoRA serving instead
        log(f"AWQ model detected — skipping merge, will use --enable-lora for serving")
        merged_path.mkdir(parents=True, exist_ok=True)
        # Just copy tokenizer so vLLM can find it
        AutoTokenizer.from_pretrained(model_name).save_pretrained(str(merged_path))
        return "lora"  # signal to use LoRA serving

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_path))
    AutoTokenizer.from_pretrained(model_name).save_pretrained(str(merged_path))
    del model
    import gc; gc.collect()
    log(f"Merged to {merged_path}")
    return "merged"


def start_vllm(merged_path, port=8081, tp=1):
    log(f"Starting vLLM with {merged_path} (tp={tp})...")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(merged_path),
        "--served-model-name", "finetuned",
        "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for health
    for i in range(120):
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
            if r.status_code == 200:
                log(f"vLLM ready after {(i+1)*2}s")
                return proc
        except Exception:
            pass
        time.sleep(2)
    
    log("ERROR: vLLM failed to start within 240s")
    proc.kill()
    return None


def start_vllm_with_lora(base_model, adapter_path, port=8081, tp=1):
    log(f"Starting vLLM with LoRA: base={base_model}, adapter={adapter_path}")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--enable-lora",
        "--lora-modules", f"finetuned={adapter_path}",
        "--served-model-name", "finetuned",
        "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--max-lora-rank", "64",
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(180):
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
            if r.status_code == 200:
                log(f"vLLM+LoRA ready after {(i+1)*2}s")
                return proc
        except Exception:
            pass
        time.sleep(2)

    log("ERROR: vLLM+LoRA failed to start within 360s")
    proc.kill()
    return None


def stop_vllm(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Also kill any GPU-holding processes
    import subprocess as sp
    for dev in ["/dev/nvidia0", "/dev/nvidia1"]:
        try:
            result = sp.run(["fuser", dev], capture_output=True, text=True)
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        except Exception:
            pass
    time.sleep(5)
    log("vLLM stopped, GPUs freed")


def score_trial(model_served_name, output_name, port=8081):
    log(f"Scoring trial set ({output_name})...")
    with open(TRIAL_FILE) as f:
        trial = json.load(f)
    
    golds, preds = [], []
    errors = 0
    start = time.time()
    
    for i, sample in enumerate(trial):
        r = sample["rubric"]
        user_msg = ("Frage: " + sample["question"] + "\n\nBewertungsrubrik:\n" +
                    "- Correct: " + r["Correct"] + "\n" +
                    "- Partially correct: " + r["Partially correct"] + "\n" +
                    "- Incorrect: " + r["Incorrect"] + "\n\n" +
                    "Schuelerantwort: " + sample["answer"])
        
        try:
            resp = httpx.post(f"http://localhost:{port}/v1/chat/completions", json={
                "model": model_served_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ], "max_tokens": 50, "temperature": 0.1,
            }, timeout=60.0)
            content = resp.json()["choices"][0]["message"]["content"].strip()
            try:
                score = json.loads(content).get("score")
            except json.JSONDecodeError:
                brace = content.rfind("{")
                try:
                    score = json.loads(content[brace:]).get("score") if brace >= 0 else None
                except (json.JSONDecodeError, TypeError):
                    score = None
            
            if score in LABELS:
                golds.append(sample["score"])
                preds.append(score)
            else:
                errors += 1
        except Exception:
            errors += 1
        
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            log(f"  [{i+1}/{len(trial)}] {(i+1)/elapsed:.1f} s/s | err={errors}")
    
    elapsed = time.time() - start
    
    if golds:
        g = [LABEL_MAP[x] for x in golds]
        p = [LABEL_MAP[x] for x in preds]
        qwk = cohen_kappa_score(g, p, weights="quadratic")
        acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)
    else:
        qwk, acc = 0, 0
    
    per_class = {}
    for label in LABELS:
        tp = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl == label)
        fp = sum(1 for gl, pl in zip(golds, preds) if gl != label and pl == label)
        fn = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[label] = {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3)}
    
    results = {
        "variant": output_name, "qwk": round(qwk, 4), "accuracy": round(acc, 4),
        "errors": errors, "scored": len(golds), "total": len(trial),
        "elapsed_s": round(elapsed, 1), "per_class": per_class,
    }
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"metrics_{output_name}_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log(f"  {output_name}: QWK={qwk:.4f} Acc={acc:.1%} Err={errors} Time={elapsed:.0f}s")
    return results


def run_experiment(model_name, experiment_name, lora_rank=32, epochs=3, lr=2e-4,
                   batch_size=4, grad_accum=4, tp=1, multi_gpu=False, use_4bit=True):
    """Full pipeline: prepare data → train (train-only) → merge → serve → score → cleanup."""
    log(f"\n{'='*70}")
    log(f"  EXPERIMENT: {experiment_name}")
    log(f"  Model: {model_name}")
    log(f"  Config: rank={lora_rank}, epochs={epochs}, lr={lr}")
    log(f"{'='*70}")
    
    output_dir = PROJECT_ROOT / "models" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = PROJECT_ROOT / "models" / f"{experiment_name}-merged"
    
    # Step 1: Prepare data (train-only for unbiased eval)
    log("Step 1: Preparing training data (train-only)...")
    jsonl_path, num_samples = prepare_jsonl(output_dir, include_trial=False)
    log(f"  {num_samples} samples")
    
    # Step 2: Fine-tune
    log("Step 2: Fine-tuning...")
    checkpoint, train_time = run_finetune(
        model_name, output_dir, jsonl_path, num_samples,
        lora_rank=lora_rank, epochs=epochs, lr=lr,
        batch_size=batch_size, grad_accum=grad_accum,
        multi_gpu=multi_gpu, use_4bit=use_4bit,
    )
    
    if not checkpoint:
        log("ERROR: No checkpoint produced!")
        return None
    
    # Step 3: Merge
    log("Step 3: Merging adapter...")
    merge_mode = merge_adapter(model_name, checkpoint, merged_path)

    # Step 4: Serve via vLLM
    log("Step 4: Starting vLLM...")
    if merge_mode == "lora":
        # AWQ: serve base model with LoRA adapter
        proc = start_vllm_with_lora(model_name, checkpoint, tp=tp)
    else:
        proc = start_vllm(merged_path, tp=tp)
    if not proc:
        return None
    
    # Step 5: Score trial set (out-of-sample)
    log("Step 5: Scoring trial set...")
    results = score_trial("finetuned", f"{experiment_name}_trial")
    results["model"] = model_name
    results["train_time_min"] = round(train_time / 60, 1)
    results["lora_rank"] = lora_rank
    results["epochs"] = epochs
    
    # Step 6: Cleanup
    log("Step 6: Cleanup...")
    stop_vllm(proc)
    
    log(f"\n  RESULT: {experiment_name} → QWK={results.get('qwk', 0):.4f}\n")
    return results


def main():
    log("=" * 70)
    log("  OVERNIGHT FINE-TUNING: 3 MODELS")
    log(f"  Started: {datetime.now().isoformat()}")
    log("=" * 70)
    
    all_results = []
    
    # Experiment 1: Qwen2.5-14B (bf16 LoRA, 1 GPU, ~28GB model + gradient ckpt)
    r1 = run_experiment(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        experiment_name="qwen25-14b-lora",
        lora_rank=32, epochs=3, lr=2e-4,
        batch_size=1, grad_accum=16, tp=1,
        use_4bit=False,
    )
    if r1:
        all_results.append(r1)

    # Experiment 2: Qwen2.5-32B (bf16 LoRA, 2 GPUs, ~64GB model split)
    r2 = run_experiment(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        experiment_name="qwen25-32b-lora",
        lora_rank=32, epochs=3, lr=2e-4,
        batch_size=1, grad_accum=16, tp=2,
        multi_gpu=True, use_4bit=False,
    )
    if r2:
        all_results.append(r2)

    # Experiment 3: Qwen2.5-72B-Instruct-AWQ (AWQ pre-quantized, LoRA on top, 2 GPUs)
    r3 = run_experiment(
        model_name="Qwen/Qwen2.5-72B-Instruct-AWQ",
        experiment_name="qwen25-72b-lora",
        lora_rank=16, epochs=3, lr=1e-4,
        batch_size=1, grad_accum=16, tp=2,
        multi_gpu=True, use_4bit=False,
    )
    if r3:
        all_results.append(r3)
    
    # Final comparison
    log("\n" + "=" * 70)
    log("  FINAL COMPARISON")
    log("=" * 70)
    for r in sorted(all_results, key=lambda x: -x.get("qwk", 0)):
        log(f"  {r['variant']:<35s} QWK={r.get('qwk',0):.4f} Acc={r.get('accuracy',0):.1%} Train={r.get('train_time_min',0):.0f}min")
    log(f"\n  Reference: Qwen2.5-7B R32 3ep      QWK=0.7260")
    log(f"  Reference: Gemini Flash C5c          QWK=0.7480")
    log(f"  Reference: Qwen2.5-7B all-data       QWK=0.7742 (in-sample)")
    log("=" * 70)
    
    # Save comparison
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"overnight_comparison_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    log(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
