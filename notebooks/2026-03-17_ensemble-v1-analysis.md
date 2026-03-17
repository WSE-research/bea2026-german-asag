---
type: experiment-log
date: 2026-03-17
tags: [phd, bea26, shared-task, experiment, ensemble]
parent: "[[bea26-shared-task]]"
---

# Ensemble v1 Analysis — Multi-Model Majority Vote

## Setup

Ran C2 prompt (tuned fewshot, 3 examples/label) across 4 cheap models via OpenRouter, majority vote for final prediction. Full 827-sample trial set.

## Results

### Per-Model Comparison

| Model | QWK | Acc | wF1 | Errors | Notes |
|-------|-----|-----|-----|--------|-------|
| **Gemini 3 Flash** | **0.678** | **68.7%** | **0.690** | 0 | Clear winner |
| Gemma 3 27B | 0.540 | 58.5% | 0.576 | 1 | Distant second |
| Llama 4 Scout | 0.507 | 56.4% | 0.543 | 0 | Weak on German |
| Qwen3 30B MoE | 0.421 | 53.1% | 0.511 | **109** | Unreliable, 13% failure rate |
| **ENSEMBLE (vote)** | **0.605** | **63.5%** | **0.631** | 0 | Worse than Gemini alone! |

### Key Finding: The ensemble is WORSE than the best single model

**Ensemble QWK 0.605 < Gemini alone 0.678.** The weaker models drag down the majority vote. When one model dominates this heavily, the two weaker models can outvote the strong model on cases where Gemini was correct.

### Per-Model Error Patterns

**Llama 4 Scout** has a severe "Partially correct" over-prediction problem:
- PC Recall: 87.5% but Precision: 45.0% — it classifies almost everything as Partially correct
- Correct Recall: only 23.8% — it almost never predicts Correct

**Qwen3 30B** is unreliable:
- 109 parse failures (13%) — returns empty/null responses frequently
- Even on valid responses, QWK is only 0.421
- Heavy Incorrect bias (Incorrect Recall: 82.9%, Correct Recall: 34.8%)

**Gemma 3 27B** is the most balanced of the weak models:
- QWK 0.540, reasonable but still ~0.14 behind Gemini
- More balanced per-class recall than Llama or Qwen

### Agreement Statistics

- Unanimous agreement: **342/827 (41.4%)** — models disagree on 59% of samples
- This confirms the models have very different biases — they're not just adding noise, they're systematically disagreeing

## Conclusions

1. **Ensemble only works when member models are comparable in quality.** With a 0.17+ QWK gap between the best and second-best model, majority vote hurts rather than helps.

2. **Gemini 3 Flash is significantly better at German ASAG** than Llama 4 Scout, Qwen3, and Gemma 3. This may reflect Gemini's stronger multilingual training.

3. **The next ensemble (v2) should use frontier-class models** that are closer to Gemini's quality level: DeepSeek V3.2, MiniMax M2.5, Kimi K2.5. These are all top-20 on academia benchmarks and priced competitively.

4. **C4 (smart examples) with Gemini Flash alone (QWK 0.735) remains our best result** — better than any ensemble configuration tried so far.

## Comparison with All Strategies

| Strategy | QWK | Notes |
|----------|-----|-------|
| **C4 (smart examples, Gemini Flash)** | **0.735** | **Current champion** |
| C2 (tuned, 3ex, Gemini Flash) | 0.709 | Previous best |
| C2 (Gemini Flash in ensemble context) | 0.678 | Lower than standalone C2 (random examples, not smart) |
| Ensemble v1 (4 models, majority vote) | 0.605 | Weak models drag down the vote |
| Gemma 3 alone | 0.540 | |
| Llama 4 Scout alone | 0.507 | |
| Qwen3 30B alone | 0.421 | |
