# Model & Prompt Screening — 2026-03-19

> **Goal:** Test whether alternative models (Claude Sonnet, GPT-5.4 Mini, MiniMax M2.7) or prompt formats (XML-tagged English) can beat Gemini Flash + C5c (QWK 0.748).
> **Result:** None did. Gemini Flash + C5c German prompt remains the best configuration by a wide margin.

---

## Motivation

With new Anthropic API budget and OpenAI fine-tuning access (via Andreas), we tested whether a stronger model could push QWK beyond 0.748. We also tested whether prompt restructuring (XML tags, English system prompt) following Anthropic's official best practices could improve cross-model performance.

## Infrastructure Changes

### openrouter.py upgrades

1. **Per-request metadata capture** — `call_openrouter()` now returns inline metadata via `_metadata` key:
   - `prompt_tokens`, `completion_tokens`, `total_tokens`
   - `cost` (USD, from OpenRouter's `usage.cost` field)
   - `generation_id`, `model` (actual model version used)
   - `reasoning_tokens`, `cached_tokens`

2. **Generation stats endpoint** — Added `fetch_generation_stats()` for `/api/v1/generation` endpoint (provider name, latency, native tokens). **Finding: consistently returns 404** for all generation IDs tested. Disabled in scorer; inline metadata is sufficient.

3. **Robust JSON parser** — `_parse_json_response()` now handles three cases:
   - Pure JSON (Gemini, GPT)
   - Markdown-fenced JSON
   - **Text + JSON** (Claude Sonnet writes analysis text before the JSON object even in `json_mode`). Parser finds the last `{` in the response and attempts to parse from there.

### New strategy folders

- `src/strategy_c6_claude_sonnet/` — C5c logic with Claude Sonnet, German prompt, full metadata logging
- `src/strategy_c6b_claude_tuned/` — Claude-optimized prompt (XML tags, English system prompt), adaptive difficulty, full metadata logging

Both strategies inherit C4's `SmartExampleSelector` and C5c's adaptive difficulty tiers.

---

## Prompt Variants Tested

### C5c / C6 prompt (German, original)

The existing prompt tuned for Gemini Flash:
- German system prompt: "Du bist ein automatisches Bewertungssystem..."
- Inline decision rules in German (ENTSCHEIDUNGSREGELN)
- Examples as numbered "Beispiel N (label):" blocks
- JSON output format specified in German

### C6b prompt (XML-tagged, English system)

Redesigned following [Anthropic's prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices):
- **English system prompt** — Claude's native language for instruction-following
- **XML tags** for structure: `<question>`, `<rubric>`, `<examples>`, `<example score="...">`, `<student_answer>`
- **Explicit JSON-only instruction**: "Respond with ONLY a JSON object. Do not include any analysis, reasoning, or explanation."
- **`<scoring_levels>` and `<decision_rules>` tags** — same content as C5c but structured for Claude's XML parser
- Added `WICHTIG` reinforcement in C6 scorer (German JSON-only reminder) — partially effective
- C6b moved the entire system prompt to English with XML — fully effective for JSON compliance

### Key prompting lessons from Anthropic docs

1. **XML tags** are parsed unambiguously by Claude — use `<example>` tags, not inline text
2. **Prefilling is deprecated** in Claude 4.6 — cannot force `{` as first token
3. **Structured outputs** (`output_config.format` with JSON schema) guarantee pure JSON — but OpenRouter doesn't expose this parameter
4. **"Tell Claude what to do, not what not to do"** — "Output only JSON" works better than "Don't add analysis"
5. **Few-shot examples in `<examples>` tags** — Claude distinguishes them from instructions

---

## Model Screening Results

### Smoke tests (1 sample, JSON compliance + reasoning check)

| Model | ID | Reasoning | Output tokens (simple) | Content | Verdict |
|-------|----|-----------|----------------------|---------|---------|
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | Optional (opt-in) | 8 (0 reasoning) | Valid JSON | Safe |
| GPT-5.4 Mini | `openai/gpt-5.4-mini` | Optional (configurable) | 12 (0 reasoning) | Valid JSON | Safe |
| MiniMax M2.7 | `minimax/minimax-m2.7` | **Mandatory** | 48 (42 reasoning + 6 visible) | Valid JSON | Risky |

### Scoring prompt tests (1 sample, actual rubric+examples)

| Model | Prompt tokens | Completion tokens | Reasoning tokens | Cost | Score correct? |
|-------|--------------|-------------------|-----------------|------|---------------|
| Claude Sonnet 4.6 | 713 | 19 | 0 | $0.002424 | Yes |
| GPT-5.4 Mini | 555 | 20 | 0 | $0.000506 | Yes |
| MiniMax M2.7 | 626 | 151 | 134 | $0.000369 | Yes |

### Full batch tests (trial set samples)

| Model | Prompt | Samples | QWK | Acc | wF1 | Errors | Cost/sample |
|-------|--------|---------|-----|-----|-----|--------|-------------|
| **Gemini Flash** | **C5c (German)** | **827** | **0.748** | **73.6%** | **0.738** | **0** | **~$0.0002** |
| Gemini Flash | C6b (XML/English) | 50 | 0.435 | 54.0% | 0.541 | 0 | $0.0007 |
| Claude Sonnet 4.6 | C6b (XML/English) | 20 | 0.570 | 60.0% | 0.611 | 0 | $0.0053 |
| Claude Sonnet 4.6 | C6 (German) | 50 | 0.371 | 48.0% | 0.477 | 0 | $0.0061 |
| Claude Sonnet 4.6 | C2 (German) | 20 | ~0.50* | ~50%* | — | 7/20 | ~$0.006 |
| GPT-5.4 Mini | C6b (XML/English) | 50 | 0.488 | 58.0% | 0.582 | 0 | $0.0010 |
| GPT-5.4 Mini | C6 (German) | 50 | 0.419 | 52.0% | 0.518 | 0 | $0.0011 |
| MiniMax M2.7 | C6b (XML/English) | 20 | N/A | N/A | — | **20/20** | — |

*C2+Claude: only 13/20 produced valid output; accuracy on those 13.

---

## Analysis

### Why Gemini Flash dominates

1. **The C5c prompt was co-evolved with Gemini Flash** — every decision rule, boundary definition, and example selection parameter was tuned against this model's behavior. The prompt-model pair is a tightly coupled system.

2. **German language prompts work best for Gemini** — Gemini Flash handles German instructions natively. Switching to English system prompt (C6b) drops QWK by 0.31 on the same model. The XML restructuring adds ambiguity for Gemini's parser.

3. **Gemini Flash is exceptionally cheap** — at ~$0.0002/sample, we can afford full trial runs for every experiment. Other models cost 5–30x more, limiting iteration.

### Why Claude Sonnet underperforms

1. **Conservative grading** — Claude's main error is Correct → Partially correct. It under-awards full marks even when all rubric criteria are met. This is the opposite of Gemini's error pattern (Incorrect → Partially correct, too lenient).

2. **Verbose output in json_mode** — Claude Sonnet adds analysis text before JSON even when `response_format: {"type": "json_object"}` is set. On the German C2 prompt, this caused 7/20 failures where the JSON was truncated by `max_tokens=300`. Fixed by: (a) increasing `max_tokens` to 1024 in C6, (b) robust JSON parser that finds `{` in mixed text, (c) XML prompt in C6b that eliminated preamble text.

3. **XML tags help Claude significantly** — C6b (XML/English) scored 0.570 vs C6 (German) at 0.371 (+0.20 QWK). Anthropic's docs explicitly recommend XML for structured prompts.

### Why GPT-5.4 Mini underperforms

1. **Weaker German comprehension** — GPT-5.4 Mini is a small, efficient model. Its German rubric understanding is less precise than Gemini Flash's.

2. **XML tags help modestly** — C6b scored 0.488 vs C6 at 0.419 (+0.07 QWK). Smaller improvement than Claude.

### Why MiniMax M2.7 fails completely

Same fatal flaw as M2.5: mandatory reasoning (`is_mandatory_reasoning: true`). On simple test prompts, reasoning adds ~42 tokens (manageable). On real scoring prompts with 1000+ token context, reasoning consumes all output budget → `content: null`. The model returns no usable output on 20/20 samples.

### Prompt format is model-specific

| Prompt format | Gemini Flash | Claude Sonnet | GPT-5.4 Mini |
|--------------|-------------|--------------|-------------|
| C5c (German) | **0.748** ✓ | 0.371 | 0.419 |
| C6b (XML/English) | 0.435 | **0.570** ✓ | **0.488** ✓ |

The best prompt format differs by model. There is no universal prompt that works well across all models. The German prompt is optimal for Gemini Flash; the XML/English prompt is better for Claude and GPT but still far below Gemini+German.

---

## Cost Comparison

| Model | Cost/sample | 827 trial | 5,176 test | Factor vs Gemini |
|-------|------------|-----------|------------|------------------|
| Gemini Flash | $0.0002 | $0.17 | $1.03 | 1x |
| GPT-5.4 Mini | $0.0010 | $0.83 | $5.17 | 5x |
| Claude Sonnet 4.6 | $0.0055 | $4.55 | $28.47 | 28x |

---

## Gemini Family Cross-Test

To test whether the C5c prompt is Gemini-family-generic or specific to the exact model, we ran two additional Gemini variants with the same C5c prompt (50 trial samples, seed=42).

| Model | ID | QWK | Acc | Reasoning | Notes |
|-------|----|-----|-----|-----------|-------|
| **Gemini 3 Flash Preview** | `google/gemini-3-flash-preview` | **0.748** (827) | 73.6% | Optional (0) | Tuned model |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | 0.485 | 58.0% | Optional (0) | Previous gen, -0.26 QWK |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` | 0.350 | 48.0% | Optional (0) | Lite variant, worst tested |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | not tested | — | Mandatory (50 on trivial) | Mandatory reasoning |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` | not tested | — | Mandatory | Deprecating 2026-03-26 |
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview-20260219` | not tested | — | Mandatory (170 on trivial) | Too expensive |

**Replication run:** Re-ran C5c on full 827 trial set after all code changes → QWK 0.7415 (original: 0.748, delta 0.006). Results replicate within expected API variance. openrouter.py changes confirmed backward-compatible.

**Finding: The C5c prompt is NOT Gemini-family-generic.** It is calibrated to one specific model checkpoint (`gemini-3-flash-preview-20251217`). Even within the Gemini family, other variants score 0.35–0.49, comparable to or worse than Claude and GPT. The prompt-model coupling is checkpoint-specific, not family-specific.

---

## Conclusions

1. **Gemini 3 Flash Preview + C5c is the optimal configuration** for this task. No model or prompt variant tested comes close.
2. **Prompt-model coupling is checkpoint-specific, not family-specific** — even other Gemini models score 0.35–0.49 with the same prompt. This is the strongest finding from this screening.
3. **XML-tagged prompts help non-Gemini models** but do not close the gap to Gemini+German.
4. **Mandatory reasoning models are incompatible** with cost-effective batch scoring (MiniMax M2.5, M2.7, Gemini Pro variants all confirmed).
5. **The OpenAI fine-tuning track remains the only realistic path** to potentially beating Gemini Flash C5c.

---

## Files Created/Modified

| File | Description |
|------|-------------|
| `src/common/openrouter.py` | Added metadata extraction, generation stats, robust JSON parser |
| `src/strategy_c6_claude_sonnet/` | C5c prompt + Claude Sonnet scorer + full metadata run script |
| `src/strategy_c6b_claude_tuned/` | XML/English prompt + adaptive difficulty + metadata logging |
| `results/strategy_c6/` | Claude Sonnet results (C5c prompt) |
| `results/strategy_c6b/` | Claude-optimized prompt results (various models) |
| `notebooks/2026-03-19_claude-sonnet-plan.md` | Original graduated test plan (partially executed) |
| `notebooks/2026-03-19_model-and-prompt-screening.md` | This document |
