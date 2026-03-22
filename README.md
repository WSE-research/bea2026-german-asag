# BEA 2026 Shared Task — Rubric-based Short Answer Scoring for German

System for the [BEA 2026 Shared Task](https://edutec.science/bea-2026-shared-task/) at ACL 2026.
Predicts rubric-based scores (Correct / Partially correct / Incorrect) for German student answers
on the ALICE-LP-1.0 dataset.

**Authors:** Jonas Gwozdz, Andreas Both — HTWK Leipzig / WSE Research

**Status:** Test data scored and submitted to CodaBench (all 4 tracks). Leaderboards on April 3.

## Results Overview

![Method Comparison](docs/results_comparison.png)

![Fine-Tuning Scaling Curve](docs/scaling_curve.png)

## Submission Results

![Submission Results](docs/submission_results.png)

## Pipeline Architecture

![Pipeline Schematic](docs/pipeline_schematic.png)

## Best Results (827 trial samples)

| System | QWK | Acc | Method | Submitted? |
|--------|-----|-----|--------|------------|
| Qwen2.5-32B LoRA (all data) | 0.844* | 82.6% | bf16 LoRA, r=32, 3 epochs | Yes (all 4 tracks) |
| Qwen2.5-14B LoRA (all data) | 0.828* | 80.9% | bf16 LoRA, r=32, 3 epochs | Yes (all 4 tracks) |
| Stacking ensemble (5 models) | 0.776 | 75.5% | LogReg over 3 FT + Q26 + kNN | — |
| Qwen2.5-32B LoRA (train-only) | 0.769 | 75.7% | Unbiased out-of-sample | — |
| Qwen2.5-14B LoRA (train-only) | 0.753 | 74.1% | Unbiased out-of-sample | — |
| Gemini 3 Flash (C5c prompt) | 0.748 | 73.6% | Commercial API baseline | — |
| Qwen3.5-9B LoRA (train-only) | 0.756 | 74.2% | 9B beats 14B (newer gen!) | — |
| Qwen2.5-7B QLoRA (train-only) | 0.726 | 70.9% | Unbiased out-of-sample | — |
| Qwen3.5-27B (Q26 prompt) | 0.721 | 70.2% | Best prompt-only, open-source | — |
| TF-IDF kNN (k=7) | 0.612 | 64.5% | No LLM needed | — |

*in-sample (trial included in training data, as allowed by task rules)

> **Understanding our evaluation numbers:** The test set has no labels — results are published April 3. To estimate performance, we evaluated two ways on the 827-sample trial set:
>
> - **Out-of-sample (unbiased):** Model trained on 7,072 training samples only, then scored on the 827 trial samples it has *never seen*. This is the honest estimate of real test performance. Our best: **32B at QWK 0.769**.
> - **In-sample (optimistic):** Model trained on *all* 7,899 samples (training + trial), then scored on the same trial samples it was trained on. This inflates the score due to memorization. Our best: **32B at QWK 0.844**.
>
> The task rules allow training on trial data for submissions, so our submitted models use all 7,899 samples for maximum performance. The real test QWK will likely fall between the out-of-sample (0.769) and in-sample (0.844) estimates.


## Test Submissions

Scored 5,094 test samples (2,008 unseen answers + 3,086 unseen questions) with two models:

| Model | Track 1 (3-way seen) | Track 2 (2-way seen) | Track 3 (3-way unseen) | Track 4 (2-way unseen) |
|-------|---------------------|---------------------|----------------------|----------------------|
| 32B LoRA | 2,008 ✅ | 2,008 ✅ | 3,086 ✅ | 3,086 ✅ |
| 14B LoRA | 2,008 ✅ | 2,008 ✅ | 3,086 ✅ | 3,086 ✅ |
| Gemini C5c | 2,008 ✅ | 2,008 ✅ | 3,086 ✅ | 3,086 ✅ |

0 errors across all submissions (12 files: 3 models × 4 tracks, all validated). Results on [CodaBench](https://www.codabench.org/competitions/14622/) after April 3.

## Repository Structure

```
├── data/raw/                     # ALICE-LP-1.0 data (train, trial, test)
├── src/
│   ├── common/                   # Shared utilities (OpenRouter client, data loaders)
│   ├── strategy_a..c6b/          # Gemini prompt engineering (A→C5c)
│   └── strategy_qwen/            # Open-source track
│       ├── prompting/            # Qwen3.5-27B: 33 prompt variants, 8 rounds
│       ├── finetuning/           # LoRA fine-tuning: 7B/14B/32B/72B
│       └── evaluation/           # Scoring, ensemble, confidence threshold
├── results/strategy_qwen/        # 110+ metrics and prediction files
├── submissions/codabench/        # CodaBench-ready zip files
├── models/                       # LoRA adapters (gitignored)
└── docs/                         # Diagrams
```

See [`src/strategy_qwen/README.md`](src/strategy_qwen/README.md) for detailed reproduction steps.

## Approach

### Track 1: Prompt Engineering

- **Gemini 3 Flash** (C5c): 6 iterative strategies, TF-IDF smart examples, adaptive difficulty → QWK 0.748
- **Qwen3.5-27B** (Q26): 33 variants from scratch, German prompts, rubric-first → QWK 0.721

Key finding: **prompt-model coupling is checkpoint-specific** — prompts don't transfer across models.

### Track 2: Fine-Tuning

LoRA fine-tuning of Qwen2.5 models (7B → 14B → 32B):
- Both 14B (0.753) and 32B (0.769) **beat Gemini Flash** on unbiased out-of-sample evaluation
- Diminishing returns: 7B→14B (+0.027), 14B→32B (+0.016)
- 72B blocked by VRAM limits (needs H200 80GB)

### Track 3: Ensemble

Stacking 5 diverse models (3 fine-tuned + 1 prompted + kNN) → QWK 0.776 (LOO-CV).

## Hardware

- 2× NVIDIA L40S (46 GB VRAM each), CUDA 13.1, Ubuntu 22.04
- Fine-tuning: bf16 LoRA (bitsandbytes 4-bit broken on L40S)
- Serving: vLLM for merged models, direct PEFT for 32B

## Links

- Shared task: https://edutec.science/bea-2026-shared-task/
- BEA 2026: https://sig-edu.org/bea/2026
- CodaBench: https://www.codabench.org/competitions/14622/
