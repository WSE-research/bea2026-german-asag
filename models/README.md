# Models Directory — Storage Management

Last cleaned: 2026-03-22

## What is kept (irreplaceable LoRA adapters)

These contain trained LoRA adapter weights that took hours of GPU time to produce.
They CANNOT be re-downloaded — only re-trained.

| Directory | Size | Description |
|-----------|------|-------------|
| `qwen25-32b-lora-alldata/` | 9.1 GB | **Primary submission model.** 32B LoRA, all 7,899 samples, 3 epochs. QWK 0.844 in-sample, 0.769 OOS. |
| `qwen25-32b-lora/` | 9.1 GB | 32B LoRA, train-only (7,072 samples). Best unbiased single-model result: QWK 0.769. |
| `qwen25-14b-lora-alldata/` | 4.7 GB | 14B submission model. QWK 0.828 in-sample, 0.753 OOS. |
| `qwen25-14b-lora/` | 4.7 GB | 14B LoRA, train-only. QWK 0.753. |
| `qwen25-7b-qlora/` | 8.3 GB | 7B QLoRA, all data. QWK 0.774 in-sample, includes 3 epoch checkpoints + R64/5ep sweep. |
| `qwen25-7b-qlora-trainonly/` | 3.1 GB | 7B QLoRA, train-only. QWK 0.726. |
| `qwen35-9b-lora/` | 2.1 GB | Qwen3.5-9B LoRA, train-only. QWK 0.756 (9B beats 14B!). |

**Total kept: ~41 GB of adapters**

## What was deleted (reproducible)

### Merged models (re-create by merging adapter + base model)

| Deleted | Size | How to reproduce |
|---------|------|------------------|
| `qwen25-14b-alldata-merged/` | 28 GB | `PeftModel.from_pretrained(Qwen2.5-14B, qwen25-14b-lora-alldata/checkpoints/checkpoint-1482).merge_and_unload()` |
| `qwen35-9b-lora-merged/` | 17 GB | `PeftModel.from_pretrained(Qwen3.5-9B, qwen35-9b-lora/checkpoints/checkpoint-1326).merge_and_unload()` |

### Failed 72B experiments (not feasible on 2x L40S)

| Deleted | Size | Reason |
|---------|------|--------|
| `qwen25-72b-lora/` | 8.5 MB | Empty — training crashed (dependency issues) |
| `qwen25-72b-lora-v2/` | 8.5 MB | Empty — training crashed (multi-GPU device mismatch) |
| `qwen25-72b-lora-v3/` | 8 KB | Empty — training crashed (gradient error) |
| `qwen25-72b-lora-final/` | 8 KB | Empty — OOM on single GPU |

### HuggingFace cache (re-downloadable)

| Deleted | Size | Reason |
|---------|------|--------|
| `Qwen2.5-72B-Instruct-AWQ` | 39 GB | Never successfully used for training |
| `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit` | 6.7 GB | Unsloth crashed, not used |

### Virtual environments

| Deleted | Size | Reason |
|---------|------|--------|
| `.venv-72b` | 7.7 GB | Created for 72B + Qwen3.5 experiments. No longer needed (3.5-9B training complete, 72B blocked). |

**Total freed: ~99 GB**

## HuggingFace cache kept (needed for inference)

| Model | Size | Used by |
|-------|------|---------|
| `Qwen2.5-32B-Instruct` | 62 GB | 32B PEFT inference (test scoring) |
| `Qwen2.5-14B-Instruct` | 28 GB | 14B merge + vLLM serving |
| `Qwen3.5-9B` | 18 GB | 3.5-9B PEFT inference |
| `Qwen2.5-7B-Instruct` | 15 GB | 7B merge + vLLM serving |

## How to re-merge a model for serving

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Example: re-merge 14B all-data
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, "models/qwen25-14b-lora-alldata/checkpoints/checkpoint-1482")
model = model.merge_and_unload()
model.save_pretrained("models/qwen25-14b-alldata-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct").save_pretrained("models/qwen25-14b-alldata-merged")
```

For 32B: CPU merge requires >48 GB RAM (server has 48 GB). Use direct PEFT inference instead:
```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct", device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "models/qwen25-32b-lora-alldata/checkpoints/checkpoint-1482")
model.eval()
# Use model.generate() directly — no merge needed
```
