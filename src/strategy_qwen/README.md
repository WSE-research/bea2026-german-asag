# Strategy Qwen — BEA 2026 Open-Source Scoring Track

Open-source LLM scoring pipeline for the BEA 2026 Shared Task on
Rubric-based Short Answer Scoring for German (ALICE-LP-1.0 dataset).

## Results Summary

### Fine-Tuning Scaling (out-of-sample, train-only → scored on trial)

| Model | Method | QWK | Acc | Train Time | Hardware |
|-------|--------|-----|-----|------------|----------|
| Qwen2.5-7B | QLoRA 4-bit, r=32, 3ep | 0.726 | 70.9% | 95 min | 1× L40S |
| Qwen2.5-14B | bf16 LoRA, r=32, 3ep | 0.753 | 74.1% | 124 min | 1× L40S |
| **Qwen2.5-32B** | **bf16 LoRA, r=32, 3ep** | **0.769** | **75.7%** | **222 min** | **2× L40S** |
| Stacking (5 models) | LogReg meta-learner | 0.776 | 75.5% | — | CPU |
| Qwen2.5-72B-AWQ | blocked | — | — | — | needs H200 |

### Prompt Engineering (Qwen3.5-27B-FP8, no fine-tuning)

| Variant | QWK (827 trial) | Description |
|---------|----------------|-------------|
| Q26 best-of-breed | 0.721 | German + TF-IDF + rubric-first + adaptive + strict |
| Q1 rubric-only | 0.510 | Bare minimum baseline |

### Comparison with Commercial API

| System | QWK (trial) | Cost |
|--------|-------------|------|
| **Qwen2.5-32B LoRA** | **0.769** | **$0 (local)** |
| Gemini Flash C5c | 0.748 | ~$0.17 |

## Directory Structure

```
src/strategy_qwen/
├── prompting/           # Prompt engineering experiments (Qwen3.5-27B)
│   ├── runner.py        # Core: model calling, score parsing, metrics, variant runner
│   ├── prompts.py       # Round 1: 8 prompt variants (Q1-Q8)
│   ├── iterate.py       # Batch runner for prompt variants
│   ├── round2.py        # Round 2: targeted improvements (Q9-Q15)
│   ├── round3.py        # Round 3: Q12 parameter optimization (Q16-Q22)
│   ├── round4.py        # Round 4: creative experiments (Q23-Q28)
│   ├── round5.py        # Round 5: full trial set confirmation
│   ├── round6.py        # Round 6: temperature sweep + thinking mode
│   ├── round7.py        # Round 7: optimal temp on full trial
│   ├── round8.py        # Round 8: post-processing calibration
│   └── round9.py        # Round 9: BM25 examples, weighted ensemble, per-Q analysis
│
├── finetuning/          # LoRA fine-tuning experiments
│   ├── finetune_7b_qlora.py       # QLoRA 4-bit, Qwen2.5-7B (all data)
│   ├── finetune_7b_trainonly.py   # QLoRA 4-bit, train-only (unbiased eval)
│   ├── finetune_7b_r64_5ep.py     # Hyperparameter sweep: rank 64, 5 epochs
│   ├── finetune_14b_alldata.py    # bf16 LoRA, Qwen2.5-14B (all data)
│   ├── finetune_32b_trainonly.py  # bf16 LoRA, Qwen2.5-32B (train-only)
│   ├── finetune_32b_alldata.py    # bf16 LoRA, Qwen2.5-32B (all data) ← SUBMISSION
│   ├── finetune_72b_awq.py        # AWQ + LoRA attempt (not feasible on 2×L40S)
│   └── overnight_pipeline.py      # Autonomous multi-model pipeline
│
├── evaluation/          # Scoring and analysis
│   ├── score_finetuned.py          # Score any dataset via vLLM-served model
│   ├── score_32b_direct.py         # Score via direct PEFT inference (no vLLM)
│   ├── score_test_set.py           # Simple test set scorer
│   ├── score_test_comprehensive.py # Full pipeline: all models + ensemble + submissions
│   ├── confidence_threshold.py     # Logprob confidence experiment
│   └── stacking_ensemble.py        # LogReg stacking over 5 models
│
└── __init__.py

results/strategy_qwen/    # All metrics + predictions (110 files)
├── metrics_q{N}_*.json           # Prompt variant results
├── metrics_finetune_*.json       # Fine-tuning results
├── predictions_*.json            # Per-sample predictions
├── submission_*.json             # Submission-format files
└── per_question_analysis.json    # Error analysis

models/                   # Adapters and checkpoints (gitignored)
├── qwen25-7b-qlora/              # 7B QLoRA adapter
├── qwen25-7b-qlora-trainonly/    # 7B train-only adapter
├── qwen25-14b-lora/              # 14B train-only adapter
├── qwen25-14b-lora-alldata/      # 14B all-data adapter
├── qwen25-32b-lora/              # 32B train-only adapter
└── qwen25-32b-lora-alldata/      # 32B all-data adapter ← SUBMISSION
```

## Reproduction

### Prerequisites

- GPU: 2× NVIDIA L40S (46 GB each) or equivalent
- CUDA 13.1+, Python 3.10+
- Main venv: `pip install vllm peft trl datasets scikit-learn`
- 72B only: separate venv with `autoawq` + `transformers==4.51.3`

### Step 1: Prompt Engineering (optional, for paper)

```bash
source ~/bea26/.venv/bin/activate
cd ~/bea26/bea2026-german-asag

# Start Qwen3.5-27B-FP8 via vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B-FP8 \
  --tensor-parallel-size 2 --port 8081 \
  --chat-template-kwargs '{"enable_thinking": false}'

# Run prompt iterations
python -m src.strategy_qwen.prompting.iterate --limit 100
python -m src.strategy_qwen.prompting.round2 --limit 100
# ... rounds 3-9
```

### Step 2: Fine-Tuning (main approach)

```bash
# 32B all-data (submission model, ~4 hours on 2×L40S)
python -m src.strategy_qwen.finetuning.finetune_32b_alldata

# 14B all-data (faster alternative, ~2.5 hours on 1×L40S)
CUDA_VISIBLE_DEVICES=0 python -m src.strategy_qwen.finetuning.finetune_14b_alldata
```

### Step 3: Score Test Set

```bash
# Score with fine-tuned 32B via direct PEFT inference
python -m src.strategy_qwen.evaluation.score_32b_direct

# Or comprehensive pipeline (all models + ensemble)
python -m src.strategy_qwen.evaluation.score_test_comprehensive \
  --test-3way data/raw/3way/ALICE_LP_test_3way.json
```

## Key Findings

1. **Prompt-model coupling is checkpoint-specific** — prompts don't transfer across models
2. **Fine-tuning scaling: 7B (0.726) → 14B (0.753) → 32B (0.769)** with diminishing returns
3. **Fine-tuned 14B beats Gemini Flash** (0.753 vs 0.748) on unbiased evaluation
4. **Stacking 5 diverse models** achieves best unbiased QWK (0.776)
5. **Confidence thresholds don't help** on fine-tuned models (>99% confident on all predictions)
6. **72B requires >46 GB VRAM** for LoRA training — not feasible on L40S

## Environment Notes

- `bitsandbytes` 4-bit QLoRA broken on L40S + CUDA 13.1 for models >7B → use bf16 LoRA
- `autoawq` deprecated, incompatible with transformers ≥4.52 → separate venv needed
- 32B merged model can't be created on CPU (needs >48 GB RAM) → use direct PEFT inference
- Qwen3.5 needs transformers 5.x which breaks vLLM → use Qwen2.5 for fine-tuning
