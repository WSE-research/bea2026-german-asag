# Claude Sonnet Upgrade Plan — BEA 2026 German ASAG

> **Date:** 2026-03-19
> **Goal:** Run best strategy (C5c) with Claude Sonnet via OpenRouter to push QWK beyond 0.748
> **Budget constraint:** Must work on the first try. Use graduated test runs to validate before committing to full test set.

---

## Model Details

| Property | Value |
|----------|-------|
| OpenRouter model ID | `anthropic/claude-sonnet-4.6` |
| Input pricing | $3.00 / 1M tokens |
| Output pricing | $15.00 / 1M tokens |
| Context window | 1,000,000 tokens |
| Reasoning | **Optional** (opt-in only, safe — no MiniMax/Kimi repeat) |
| JSON mode | Supported via `response_format: {"type": "json_object"}` |

**Cost comparison to Gemini Flash:**

| | Gemini 3 Flash | Claude Sonnet 4.6 | Factor |
|-|----------------|-------------------|--------|
| Input | $0.10/M | $3.00/M | 30x |
| Output | $0.40/M | $15.00/M | 37.5x |

## Cost Estimation

Based on Gemini Flash runs (~300 input tokens + ~50 output tokens per sample at C5c):

**But Claude prompts may use more tokens** (different tokenizer, German text). Conservative estimate: ~500 input + 80 output per sample.

| Run | Samples | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|-----|---------|-------------------|--------------------|-----------|
| Smoke test | 10 | 5,000 | 800 | ~$0.03 |
| Mini validation | 50 | 25,000 | 4,000 | ~$0.14 |
| Trial set | 827 | 413,500 | 66,160 | ~$2.23 |
| Test set (5,176) | 5,176 | 2,588,000 | 414,080 | ~$13.97 |

**Total budget for full pipeline (all 4 runs): ~$16.37**

If token counts are higher (e.g., 800 input with more examples from C5c hard questions):

| Run | Samples | High-est Input | High-est Output | High-est Cost |
|-----|---------|----------------|-----------------|---------------|
| Trial set | 827 | 660,000 | 66,160 | ~$2.97 |
| Test set | 5,176 | 4,140,800 | 414,080 | ~$18.64 |

**Worst-case total: ~$22**

---

## Graduated Test Plan

### Phase 1: Smoke Test (10 samples) — validate mechanics
**Purpose:** Confirm Claude Sonnet works through OpenRouter with our pipeline. Zero budget risk.

- [ ] Set `OPENROUTER_MODEL=anthropic/claude-sonnet-4.6`
- [ ] Run: `python -m src.strategy_c5c_adaptive.run --split trial --limit 10 --workers 1 --seed 42`
- [ ] Verify:
  - [ ] No errors (all 10 scored)
  - [ ] Output is valid JSON with `score` and `confidence` keys
  - [ ] Scores are one of the 3 valid labels (exact casing)
  - [ ] No reasoning tokens burned (check response size is small)
  - [ ] Cost per sample is in expected range

**Pass criteria:** 10/10 scored, valid JSON, no reasoning token blowup.
**Abort if:** Errors, malformed output, or cost >10x expected.

### Phase 2: Mini Validation (50 samples) — validate quality
**Purpose:** Check if Claude Sonnet is competitive before committing to full trial run.

- [ ] Run: `python -m src.strategy_c5c_adaptive.run --split trial --limit 50 --workers 2 --seed 42`
- [ ] Compare accuracy to Gemini Flash on same 50 samples (need to extract from existing results)
- [ ] Check per-class distribution: is Claude still over-predicting "Partially correct"?
- [ ] Check cost actuals vs estimates

**Pass criteria:** Accuracy ≥ Gemini Flash on same 50 samples. No systematic errors.
**Abort if:** Accuracy significantly below Gemini Flash (>5pp drop) or cost explosion.

### Phase 3: Full Trial Set (827 samples) — validate QWK
**Purpose:** Get a proper QWK comparison. This is the go/no-go for the test run.

- [ ] Run: `python -m src.strategy_c5c_adaptive.run --split trial --workers 3 --seed 42`
- [ ] Compare QWK to Gemini Flash C5c (0.748)
- [ ] Analyze confusion matrix differences
- [ ] Document per-question improvements/regressions

**Pass criteria:** QWK ≥ 0.748 (matches Gemini) → proceed to test.
**Stretch:** QWK ≥ 0.77 → Claude Sonnet is clearly better.
**Abort if:** QWK < 0.72 → Claude Sonnet is worse, stick with Gemini.

### Phase 4: Test Set Submission (5,176 samples) — final run
**Purpose:** Generate the actual submission files.

- [ ] Run: `python -m src.strategy_c5c_adaptive.run --split test --workers 3 --seed 42`
- [ ] Verify submission files generated (3-way and 2-way)
- [ ] Sanity check: label distribution should be roughly similar to training (29/36/35%)
- [ ] Check for error rate (<1% acceptable)

---

## Potential Issues & Mitigations

### 1. Claude refuses to score (safety guardrails)
- **Risk:** Low. These are educational assessments, not harmful content.
- **Mitigation:** Smoke test catches this immediately.

### 2. Different JSON formatting
- **Risk:** Claude might add explanatory text before/after JSON despite json_mode.
- **Mitigation:** `_parse_json_response()` already strips markdown fences. Smoke test validates.

### 3. German label casing
- **Risk:** Claude might return `"Partially Correct"` (capital C) instead of `"Partially correct"`.
- **Mitigation:** `parse_response()` already does case-insensitive matching. Covered.

### 4. Rate limiting
- **Risk:** Anthropic via OpenRouter may have tighter rate limits than Google.
- **Mitigation:** Start with workers=1 in smoke test, scale to 2–3. Retry logic already handles 429s.

### 5. Token count differences
- **Risk:** Claude's tokenizer produces different token counts than Gemini's.
- **Mitigation:** The cost estimates above are conservative. Smoke test gives actual per-sample cost.

---

## OpenAI Fine-Tuning (Parallel Track)

Andreas offered access to OpenAI fine-tuning platform. This can run in parallel with Claude Sonnet testing.

### Approach
- Fine-tune `gpt-4o-mini` on 7,072 training samples
- Format: chat completions JSONL (system=C2 prompt, user=question+answer+rubric, assistant=label JSON)
- **Key decision:** Include few-shot examples in training data or not?
  - Option A: No examples in training → model learns to score from rubric alone → simpler, cheaper inference
  - Option B: Include examples → model learns the same task it'll see at inference → more consistent but redundant
  - **Recommendation: Option A** — fine-tuning should replace the need for examples

### Cost estimate
- Training: ~7k samples × ~500 tokens × $3/M tokens = ~$10.50
- Inference: much cheaper per-sample than Claude Sonnet

### Timeline
- Prepare JSONL: 1 hour
- Training: 1–4 hours (OpenAI queue)
- Evaluate on trial: 30 min
- **Can be ready before test data arrives (2026-03-21)**

---

## Decision Matrix

After Phase 3 (trial results), choose submission strategy:

| Scenario | Action |
|----------|--------|
| Claude Sonnet QWK > 0.77 | Submit Claude Sonnet as primary |
| Claude Sonnet ≈ Gemini (0.74-0.77) | Submit both, Claude as primary |
| Claude Sonnet < Gemini | Stick with Gemini, try fine-tuned model |
| Fine-tuned model beats both | Submit fine-tuned as primary |
| Ensemble of Claude + fine-tuned > either alone | Submit ensemble |
