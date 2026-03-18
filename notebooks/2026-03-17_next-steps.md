---
type: plan
date: 2026-03-17
tags: [phd, bea26, shared-task, next-steps]
parent: "[[bea26-shared-task]]"
---

# BEA 2026 — Next Steps (After Session 1)

## Session 1 Results Summary

| Strategy | QWK | Description |
|----------|-----|-------------|
| A (rubric-only) | 0.654 | Baseline |
| B (rubric+rules) | 0.671 | +rules |
| C (rubric+fewshot) | 0.718 | +random examples |
| C2 (tuned, 3ex) | 0.709 | +boundary rules |
| **C4 (smart examples)** | **0.735** | **+boundary-focused + TF-IDF similarity selection** |
| Ensemble v1 | pending | 4 cheap models majority vote |

## Ensemble v2: Frontier-Cheap Models — TESTED 2026-03-18

> **Result: Failed.** None of the candidate models matched Gemini Flash.
> Full analysis: `notebooks/2026-03-18_model-screening.md`

### Screening results (100 samples, C2 prompt)

| Model | QWK | Status |
|-------|-----|--------|
| **Gemini 3 Flash** | **0.670** | baseline |
| DeepSeek V3.2 | 0.506 | eliminated — weak on German |
| MiniMax M2.5 | N/A | eliminated — mandatory reasoning (25x cost) |
| Kimi K2.5 | N/A | eliminated — mandatory reasoning (stalls) |
| Mistral Small 4 | blocked | OpenRouter privacy settings |
| GPT-5.4 Nano | blocked | OpenRouter privacy settings |

### Key learnings

- **Reasoning models (MiniMax, Kimi) are incompatible with cheap batch scoring** — they burn all `max_tokens` on internal chain-of-thought before producing output
- **DeepSeek V3.2 over-predicts "Partially correct" on German** — 73.5% recall but only 45.5% precision
- **Gemini Flash dominates German ASAG** across all 8 models tested (+0.13–0.16 QWK gap to second-best)

## Revised Next Steps (2026-03-18)

### High priority
- [ ] C4 hyperparameter sweep on Gemini Flash:
  - `--n-similar 2 --n-boundary 2`
  - `--n-similar 2 --n-boundary 3`
  - `--n-similar 1 --n-boundary 3`
- [ ] Unblock Mistral Small 4 / GPT-5.4 Nano (OpenRouter privacy settings) and retest
- [ ] Multi-seed self-ensemble: run C4 with seeds 42, 123, 456 → majority vote

### When test data arrives (2026-03-21)
- [ ] Score test set with best C4 config
- [ ] Score test set with self-ensemble if it beats single run
- [ ] Generate submission files (now auto-compiled by run scripts)

### Stretch goals
- [ ] Fine-tune an open-source model if budget allows (Qwen 7B on 7,899 samples)
- [ ] C4 + C2 cross-strategy ensemble on Gemini Flash
