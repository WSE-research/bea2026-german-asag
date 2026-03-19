---
type: plan
date: 2026-03-17
tags: [phd, bea26, shared-task, plan]
parent: "[[bea26-shared-task]]"
---

# BEA 2026 — Iteration Plan (Post Experiment Log)

Based on deep analysis of vault concepts and experimental results from today.

## Current Best: C2 (tuned fewshot, 3ex/label) — QWK 0.709

## Plan: Two Levers

### Lever 1: Smart Example Selection (Strategy C4)

Replace random per-question example selection with two improvements:

**A) Boundary-focused examples:** For each question, select examples that sit near the decision boundaries (Correct↔Partially correct, Partially correct↔Incorrect). These are the hardest-to-distinguish cases that force the model to calibrate exactly where it's weakest.

- Source: [[boundary-focused-exemplar-selection-halves-adjacent-score-errors]] (Chu 2026b, GUIDE framework — 70% reduction in adjacent-score errors)
- Our #1 error mode is Incorrect→Partially correct (27-36% of errors) — boundary examples directly target this

**B) RAG-based retrieval:** For each answer being scored, embed it and retrieve the most similar training answers per label. The model sees examples that *look like* the answer it's scoring, not random ones.

- Source: [[rag-based-example-selection-significantly-outperforms-random-shots]] (Zhao 2025, p<0.001 significance)
- Expected gain: +2-4 QWK points

### Lever 2: Cheap Model Majority Vote

Instead of one expensive model, run multiple cheap models and aggregate via majority vote or soft voting:

- google/gemini-3-flash-preview (current)
- meta-llama/llama-4-scout (latest Llama 4)
- qwen/qwen3-30b-a3b (latest Qwen 3 MoE, very cheap)
- google/gemma-3-27b-it (latest Gemma 3)

Majority vote: 3+ models agree = high confidence. Disagreement = flag for analysis.

- Source: ICALT paper methodology (cross-model agreement), [[probabilistic-annotation-models-outperform-majority-voting]]
- Expected gain: +2-5 QWK points from variance reduction

### NOT doing

- Stronger expensive models (Claude Sonnet, GPT-4) — budget constraint
- Structured criteria listing — proved harmful (C3 experiment)
- Temperature tuning — proved irrelevant
- HITL/confidence routing — no humans in competition
- Fine-tuning — save for after first submission

## Success Criteria

- C4 (smart examples) beats C2 (random examples) on full trial set
- Majority vote of 3+ cheap models beats best single model
- Target: QWK ≥ 0.73 on trial set
