---
type: experiment-log
date: 2026-03-17
tags: [phd, bea26, shared-task, experiment, prompt-engineering, asag]
parent: "[[bea26-shared-task]]"
---

# BEA 2026 Shared Task — Experiment Log (2026-03-17)

First day of active work. Went from receiving the training data to a working scoring pipeline with 6 strategy variants tested on the full 827-sample trial set.

## Setup

- **Model:** Gemini 3 Flash (`google/gemini-3-flash-preview`) via OpenRouter
- **Dataset:** ALICE-LP-1.0 — 7,072 train + 827 trial, 78 questions, German STEM
- **Metric:** Quadratic Weighted Kappa (QWK)
- **Repo:** https://github.com/WSE-research/bea2026-german-asag

## Strategy Comparison

| Strategy | QWK | Acc | wF1 | Description |
|----------|-----|-----|-----|-------------|
| A (rubric-only) | 0.654 | 68.9% | 0.692 | Minimal prompt, rubric speaks for itself |
| B (rubric+rules) | 0.671 | 70.4% | 0.708 | 7 strict BEWERTUNGSREGELN for boundary decisions |
| C (rubric+fewshot) | 0.718 | 75.3% | 0.754 | 2 examples/label from training set (batch 1 only) |
| C2 (tuned, 2ex) | 0.700 | 70.7% | 0.710 | Tuned "Partially correct" boundary + fewshot |
| **C2 (tuned, 3ex)** | **0.709** | **72.0%** | **0.721** | **Current best — full 827-sample trial set** |
| C2 (3ex, temp=0.1) | 0.710 | 72.1% | 0.722 | Temperature has no measurable effect |
| C2 (4ex) | 0.705 | 70.2% | 0.702 | Rate-limited run (59 errors), inconclusive |
| C3 (structured eval) | 0.692 | 69.7% | 0.700 | Criteria listing — backfired, slower |

## Iteration Methodology

Used a **3-batch iterative approach** on the 827 trial samples:
1. **Batch 1** (samples 0–279): Ran all 3 base strategies (A, B, C), analyzed errors
2. **Batch 2** (samples 280–559): Tested tuned C2 prompt based on batch 1 error analysis
3. **Batch 3** (samples 560–827): Held-out validation of C2

Error analysis after batch 1 identified the #1 error mode: **Incorrect → Partially correct** (27–36% of all errors). The model was too generous, finding "something" in truly incorrect answers. This led to the C2 prompt tuning with explicit boundary rules.

## Key Learnings

### 1. Few-shot examples are the strongest lever

Adding per-question examples from the training set improved QWK by **+5 points** over rubric-only (A: 0.654 → C: 0.718). This is the single most impactful change. The model needs calibration examples to understand the scoring threshold for each question.

### 2. The "Partially correct" boundary is the bottleneck

All strategies over-predict "Partially correct". The confusion matrix pattern across every strategy:
- **Incorrect → Partially correct**: 27–36% of all errors (model too lenient)
- **Correct → Partially correct**: 24–26% of errors (model too cautious)

The model defaults to the middle class when uncertain. Explicit boundary rules in C2 helped but didn't fully solve this.

### 3. More examples = better calibration

| Examples/label | Total examples | QWK |
|---------------|---------------|-----|
| 2 | 6 | 0.700 |
| 3 | 9 | 0.709 |
| 4 | 12 | 0.705 (rate-limited) |

Going from 6 → 9 examples improved every metric. The 4-example run was inconclusive due to rate limiting but suggests diminishing returns.

### 4. Structured reasoning can backfire

Strategy C3 forced the model to list `criteria_met` and `criteria_missed` before scoring. Hypothesis: explicit criterion checking would reduce over-prediction of "Partially correct." Reality: it made the model **more** likely to find something partially met, increasing Partially correct false positives. QWK dropped from 0.700 → 0.692.

**Takeaway:** For classification tasks with an ambiguous middle category, forcing chain-of-thought reasoning can make the model over-think borderline cases and default to the middle.

### 5. Temperature is irrelevant at this scale

Temp 0.1 vs 0.2 produced QWK 0.710 vs 0.709 — within noise. The model's scoring is driven by the prompt and examples, not sampling randomness.

### 6. Confidence scores are useless for filtering

The model reports confidence 0.94–0.97 for both correct and incorrect predictions. The gap between correct (0.96) and wrong (0.94) predictions is only 0.013 — not actionable for any confidence-based filtering or cascading strategy.

### 7. Rate limits are the practical constraint

Running 3 experiments in parallel (12+ concurrent workers total) caused 59 errors per run from OpenRouter 429 responses. The retry logic handles transient 429s but sustained overload causes permanent failures. **Run experiments sequentially** or limit to 3–4 workers when sharing an API key.

## Error Analysis Detail (Strategy C, batch 1)

```
Incorrect → Partially correct:  17 (27%) ← #1 error, model too lenient
Correct → Partially correct:    15 (24%) ← #2 error, model too cautious
Partially correct → Correct:    11 (17%)
Partially correct → Incorrect:  11 (17%)
Correct → Incorrect:             5 (8%)
Incorrect → Correct:             4 (6%)  ← rare, worst for QWK
```

The Correct↔Incorrect confusion (14% combined) is rare but costs 4× more in QWK than adjacent-class errors.

## What Didn't Work

- **Structured criteria evaluation (C3):** Slower (471s vs 307s) and less accurate. The model "over-reasons" into the middle class.
- **Parallel ensemble runs:** Rate limiting destroyed data quality. Need sequential execution.
- **4 examples/label:** Diminishing returns + higher cost + rate limit risk. 3/label is the sweet spot.

## Next Steps

- [ ] Try stronger model (Claude Sonnet or GPT-4) for ceiling estimate
- [ ] Clean ensemble: 3 seeds run sequentially, majority vote
- [ ] Score test set when released (2026-03-21)
- [ ] Submit best results (deadline 2026-03-28)
- [ ] Consider fine-tuning open-source model on 7,899 training samples

## Cost

All experiments today cost < $1.00 on OpenRouter (Gemini Flash pricing). The full trial set (827 samples) with C2-3ex costs approximately $0.05 per run.
