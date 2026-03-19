# C4 Improvement Sprint — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Push QWK from 0.744 to ~0.76-0.78 via four independent improvements on top of Strategy C4.

**Architecture:** Each improvement is a new strategy (C5a, C5b, C5c, C5d) that extends C4. All share the same SmartExampleSelector and Gemini Flash model. Each can be tested independently on the trial set, and the best combined into a final strategy.

**Baseline:** C4 with n_boundary=2, n_similar=2, seed=42 → QWK 0.7436 (827 samples, 0 errors)

---

## Error Analysis (C4 baseline, 222/827 errors)

| Error type | Count | % of errors |
|-----------|-------|-------------|
| Incorrect → Partially correct | 62 | 27.9% |
| Correct → Partially correct | 56 | 25.2% |
| Partially correct → Correct | 48 | 21.6% |
| Partially correct → Incorrect | 42 | 18.9% |
| Incorrect → Correct | 9 | 4.1% |
| Correct → Incorrect | 5 | 2.3% |

Primary target: the 118 middle-class errors (53% of all errors).

---

## Task 1: Strategy C5a — Short-Answer Rules + Anti-Examples

**Idea:** Add post-processing rules for trivially short/empty answers, and inject 2 "anti-examples" into the prompt showing common Incorrect→PC mistakes.

**Files:**
- Create: `src/strategy_c5a_postprocess/prompt.py` — extends C4 prompt with anti-examples section
- Create: `src/strategy_c5a_postprocess/scorer.py` — wraps C4 scorer with post-processing rules
- Create: `src/strategy_c5a_postprocess/run.py` — CLI entry (copy from C4, change imports)

**Post-processing rules:**
1. If `len(answer.strip()) < 15` → force "Incorrect" (covers empty, "?", "Keine Ahnung", single words)
2. If answer matches non-answer pattern (regex: `^[\s?!.\-]*$|(?i)keine ahnung|weiß.*nicht|keine antwort|kein plan`) → force "Incorrect"

**Anti-examples to add to prompt (from actual C4 errors):**
- A vague answer that *looks* partially correct but is actually Incorrect (rephrases question without fachliche Konzepte)
- A short answer that the model over-generously scored as Partially correct

**Run command:**
```bash
OPENROUTER_MODEL="google/gemini-3-flash-preview" python -m src.strategy_c5a_postprocess.run --split trial --workers 3 --n-similar 2 --n-boundary 2 --seed 42
```

---

## Task 2: Strategy C5b — Multi-Seed Majority Vote

**Idea:** Run C4 (n_similar=2, n_boundary=2) three times with different seeds. Majority vote across the three runs.

**Files:**
- Create: `src/strategy_c5b_multiseed/run.py` — runs C4 scorer 3x with seeds 42, 123, 456, majority votes

**Implementation:**
- For each sample, call C4's `score_sample()` three times (reconfiguring seed each time)
- Collect 3 predictions, take majority vote
- If all 3 disagree (rare with 3-way), use seed 42's prediction as tiebreaker
- Track per-seed agreement statistics

**Run command:**
```bash
OPENROUTER_MODEL="google/gemini-3-flash-preview" python -m src.strategy_c5b_multiseed.run --split trial --workers 2
```

**Note:** This costs 3x a normal run (~$0.15). Workers=2 to avoid rate limits with 3 API calls per sample.

---

## Task 3: Strategy C5c — Adaptive Difficulty

**Idea:** Pre-compute question difficulty from training data. Adjust n_boundary and n_similar per question.

**Files:**
- Create: `src/strategy_c5c_adaptive/scorer.py` — computes difficulty, adjusts example params
- Create: `src/strategy_c5c_adaptive/run.py` — CLI entry

**Difficulty tiers (based on training set label distribution):**
- **Easy** (>60% one dominant label): n_boundary=1, n_similar=2 — fewer boundary examples needed
- **Medium** (40-60% dominant): n_boundary=2, n_similar=2 — current default
- **Hard** (<40% dominant, high entropy): n_boundary=3, n_similar=2 — more boundary calibration

**Run command:**
```bash
OPENROUTER_MODEL="google/gemini-3-flash-preview" python -m src.strategy_c5c_adaptive.run --split trial --workers 3
```

---

## Task 4: Strategy C5d — Rubric Decomposition

**Idea:** Instead of asking for a single 3-way score, decompose into binary sub-criteria checks based on the rubric text, then aggregate.

**Files:**
- Create: `src/strategy_c5d_decomposed/prompt.py` — new prompt that asks for binary criterion evaluation
- Create: `src/strategy_c5d_decomposed/scorer.py` — calls LLM, aggregates binary verdicts
- Create: `src/strategy_c5d_decomposed/run.py` — CLI entry

**Prompt approach:**
1. Parse the rubric's "Correct" text to identify key criteria (split by "und"/"," to get 2-4 points)
2. For each criterion, ask: "Does the answer demonstrate [criterion]? (ja/nein)"
3. Aggregate: all yes → Correct; some yes → Partially correct; all no → Incorrect
4. Still include 2-3 examples for calibration

**Run command:**
```bash
OPENROUTER_MODEL="google/gemini-3-flash-preview" python -m src.strategy_c5d_decomposed.run --split trial --limit 100 --workers 3
```

**Note:** Start with 100 samples to validate the approach before full run. This is the most experimental idea.

---

## Execution Plan

1. Spawn 4 subagents in parallel, each implementing one task
2. Run strategies sequentially to avoid rate limits:
   - C5a first (cheapest — same cost as C4 but with post-processing)
   - C5c second (same cost as C4)
   - C5d third (100 samples only — experimental)
   - C5b last (3x cost — most expensive)
3. Document results in this file
4. If any improvement beats 0.744, combine the winning strategies

---

## Results (2026-03-18)

| Strategy | QWK | Acc | wF1 | Errors | Notes |
|----------|-----|-----|-----|--------|-------|
| C4 baseline | 0.744 | 73.2% | 0.734 | 0 | n_similar=2, n_boundary=2 |
| C5a (postprocess + anti-ex) | 0.543 | 65.0% | 0.646 | 1 | **Failed** — 126 overrides too aggressive |
| C5b (multi-seed majority vote) | 0.743 | 73.2% | 0.734 | 0 | Pointless — 98.1% unanimous at temp=0.2 |
| **C5c (adaptive difficulty)** | **0.748** | **73.6%** | **0.738** | **0** | **New best — adaptive n_boundary by question difficulty** |
| C5d (rubric decomposition) | 0.610* | 64.0% | 0.642 | 0 | **Failed** — only 2 criteria avg, too coarse |

*C5d tested on 100 samples only

## Analysis

### C5a: Post-processing killed accuracy
The `len(answer) < 15` rule overrode 126/827 samples (15%). Many legitimate short answers that were Partially correct or Correct got forced to Incorrect. The anti-examples in the prompt may have also made the model more "Incorrect-happy." **Lesson:** Post-processing heuristics on short-answer data are dangerous — German STEM answers can be legitimately short.

### C5b: Multi-seed majority vote is useless at low temperature
98.1% of samples got unanimous agreement across 3 seeds. Only 16/827 had majority vote, 0 had a split. At temperature=0.2, Gemini Flash is nearly deterministic — the seed only affects which random examples fill gaps in the SmartExampleSelector, and those gaps are rare. **Lesson:** Multi-seed majority vote only works if the model has meaningful output variance.

### C5c: Adaptive difficulty works
Easy questions (>60% dominant label, 24/78 questions) benefit from fewer boundary examples (n_boundary=1), achieving 81.5% accuracy vs 70% on medium/hard questions. The improvement is small (+0.004 QWK) but consistent and free. **Lesson:** Over-calibrating easy questions with boundary examples may confuse the model; less is more for obvious cases.

### C5d: Rubric decomposition is too coarse
The model identified only 2.0 criteria per question on average, making the decomposition nearly binary. With so few criteria, the aggregation logic loses the nuance that holistic scoring + examples provides. The approach might work better with rubrics that have 4+ explicit criteria. **Lesson:** Decomposition requires rubrics with enough granularity to decompose into.

## Best Config for Test Submission

**Strategy C5c (adaptive difficulty)** with Gemini Flash:
```bash
OPENROUTER_MODEL="google/gemini-3-flash-preview" python -m src.strategy_c5c_adaptive.run --split test --workers 3 --seed 42
```
