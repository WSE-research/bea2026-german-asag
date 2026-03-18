---
type: experiment-log
date: 2026-03-18
tags: [phd, bea26, shared-task, experiment, ensemble, model-screening]
parent: "[[bea26-shared-task]]"
---

# Model Screening — Frontier-Cheap Models for Ensemble v2

## Motivation

Ensemble v1 failed because the 4 models (Gemini Flash, Llama 4 Scout, Qwen3, Gemma 3) had too large a quality gap — weaker models outvoted the strong one. For v2, we screened 5 newer frontier-cheap models on a 100-sample subset of the trial set to find ensemble candidates closer to Gemini Flash's quality.

## Method

- **Prompt:** C2 (tuned fewshot, 3 examples/label, seed=42)
- **Samples:** First 100 of trial set (35 Correct, 34 Partially correct, 31 Incorrect)
- **Workers:** 3
- **Baseline:** Gemini 3 Flash on the same 100 samples

## Results

### Models That Ran Successfully

| Model | OpenRouter ID | QWK | Acc | wF1 | Errors | Cost/sample |
|-------|--------------|-----|-----|-----|--------|-------------|
| **Gemini 3 Flash** | `google/gemini-3-flash-preview` | **0.670** | 67.0% | 0.671 | 0/100 | ~$0.00005 |
| DeepSeek V3.2 | `deepseek/deepseek-chat` | 0.506 | 57.0% | 0.570 | 0/100 | ~$0.00003 |

### Models Blocked by OpenRouter Privacy Settings

| Model | OpenRouter ID | Issue |
|-------|--------------|-------|
| Mistral Small 4 | `mistralai/mistral-small-2603` | HTTP 404 — "No endpoints available matching your guardrail restrictions and data policy" |
| GPT-5.4 Nano | `openai/gpt-5.4-nano` | Same 404 — privacy policy blocks Mistral and OpenAI providers |

**Fix:** Adjust settings at https://openrouter.ai/settings/privacy to allow all providers.

### Models With Mandatory Reasoning (Unexpected Cost/Behavior)

| Model | OpenRouter ID | Issue | Reasoning tokens | Content |
|-------|--------------|-------|-----------------|---------|
| MiniMax M2.5 | `minimax/minimax-m2.5` | Mandatory reasoning burns token budget | 680/695 tokens | `null` at max_tokens=300 |
| Kimi K2.5 | `moonshotai/kimi-k2.5` | Same — mandatory reasoning, cannot disable | All tokens | `null` at max_tokens=300 |

**Root cause:** Both MiniMax M2.5 and Kimi K2.5 are reasoning models that use internal chain-of-thought before producing output. The reasoning tokens count against `max_tokens`. With the default `max_tokens=300`, the model spends all tokens on reasoning and never produces the JSON response (`content: null`).

**Attempted fix for MiniMax:** Setting `max_tokens=2000` works — the model uses ~680 reasoning tokens + ~15 output tokens = ~695 total. But this makes it **25x more expensive** than Gemini Flash ($0.00125 vs $0.00005 per sample). Attempting to disable reasoning via `{"reasoning": {"effort": "none"}}` returns: *"Reasoning is mandatory for this endpoint and cannot be disabled."*

**MiniMax detailed results (with max_tokens=2000, single sample):**
- Produced valid JSON response: `{"score": "Correct", "confidence": 0.95}`
- Token breakdown: 680 reasoning + 15 output = 695 total completion tokens
- Cost: $0.001253 per sample (vs ~$0.00005 for Gemini Flash)
- Provider: SambaNova (via OpenRouter)

**Kimi K2.5 detailed results:**
- Provider: Inceptron (via OpenRouter)
- Even `max_tokens=50` on a trivial "Say hi" prompt: all tokens consumed by reasoning, `content: null`, `finish_reason: length`
- Reasoning was not disableable

## Key Findings

### 1. Reasoning models are incompatible with cheap batch scoring

MiniMax M2.5 and Kimi K2.5 both force internal chain-of-thought that cannot be disabled via the OpenRouter API. This has two consequences:
- **Cost:** 10–25x more expensive per sample due to reasoning token overhead
- **Reliability:** With standard `max_tokens` settings, responses are `null` (the model never finishes thinking)

This is a significant finding for anyone planning to use these models for batch classification tasks. The advertised $/M token pricing is misleading for reasoning models because most tokens are consumed internally.

### 2. DeepSeek V3.2 underperforms on German ASAG

Despite being marketed as "GPT-5 class reasoning" and performing well on English benchmarks, DeepSeek V3.2 scored only QWK 0.506 on German short answer grading — 0.164 points below Gemini Flash. The main error pattern: massive over-prediction of "Partially correct" (73.5% recall, 45.5% precision). DeepSeek seems to default to the middle class when uncertain about German text, similar to but worse than the pattern seen in Llama 4 and Qwen3 in ensemble v1.

### 3. Gemini 3 Flash remains dominant for German ASAG

Across all models tested (ensemble v1 + v2 screening), Gemini Flash is the only model that achieves competitive QWK on German rubric-based scoring. The gap to the second-best model (Gemma 3 27B at 0.540 from v1, or DeepSeek V3.2 at 0.506 from v2) is massive: +0.13–0.16 QWK.

This likely reflects Gemini's stronger multilingual training and instruction-following on non-English tasks. For the BEA 2026 shared task, the focus should be on **optimizing the single-model strategy (C4 smart examples on Gemini Flash)** rather than pursuing ensembles.

### 4. OpenRouter privacy settings can silently block models

The 404 error for Mistral Small 4 and GPT-5.4 Nano was not a model availability issue but an account-level privacy restriction. The error message ("No endpoints available matching your guardrail restrictions") is not immediately obvious as a privacy setting issue. Worth noting for reproducibility.

## Implications for Ensemble Strategy

The ensemble v2 idea is effectively dead with the current model pool. The only viable path to an ensemble would be:
1. **Unblock Mistral Small 4 / GPT-5.4 Nano** and test whether they're competitive
2. **Multi-seed self-ensemble:** Run Gemini Flash multiple times with different random seeds for example selection, then majority vote
3. **C4 + C2 ensemble:** Use different prompting strategies on Gemini Flash and vote across strategies

## C4 Hyperparameter Sweep (same session)

Also ran a C4 hyperparameter sweep on Gemini Flash to optimize example selection:

| Config | n_boundary | n_similar | ~examples | QWK | Errors |
|--------|-----------|-----------|-----------|-----|--------|
| Baseline (session 1) | 2 | 1 | ~7 | 0.735 | 0 |
| **n_similar=2 (new best)** | **2** | **2** | **~10** | **0.744** | **0** |
| n_boundary=3 | 3 | 1 | ~9 | 0.722 | 71* |
| n_boundary=3, n_similar=2 | 3 | 2 | ~12 | 0.748 | 69* |

*Rate-limited runs (3 parallel experiments), scored ~757/827 samples

**Finding:** More TF-IDF similar examples helps (+0.009 QWK), more boundary examples hurts slightly. The model benefits more from seeing answers that resemble the one being scored than from seeing decision-boundary anchors.

**Best config for test submission:** `--n-boundary 2 --n-similar 2 --seed 42`

## Cost Summary

- Gemini Flash (100 samples): ~$0.005
- DeepSeek V3.2 (100 samples): ~$0.003
- MiniMax M2.5 (5 successful / 95 failed): ~$0.006 (mostly wasted on reasoning tokens of failed attempts)
- Kimi K2.5: ~$0 (all failed before producing output)
- C4 sweep (3 × 827 samples + 1 clean rerun): ~$0.20
- **Total session 2 API cost: < $0.25**
