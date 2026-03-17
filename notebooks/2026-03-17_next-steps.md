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

## Next Ensemble: Frontier-Cheap Models (v2)

The first ensemble used somewhat outdated models. The next attempt should use the strongest cheap/mid-tier models available on OpenRouter as of March 2026:

### Models to test:

| Model | OpenRouter ID | Input $/M | Output $/M | Why |
|-------|--------------|-----------|------------|-----|
| **Gemini 3 Flash** | `google/gemini-3-flash-preview` | $0.50 | $3.00 | Current best single model, #1 Academia |
| **DeepSeek V3.2** | `deepseek/deepseek-chat` | $0.26 | $0.38 | GPT-5 class reasoning, very cheap |
| **MiniMax M2.5** | `minimax/minimax-m2.5` | $0.20 | $1.20 | SOTA productivity, cheapest |
| **Kimi K2.5** | `moonshotai/kimi-k2.5` | $0.45 | $2.20 | Strong reasoning + multimodal |

### Run command (when ready):

```bash
cd "C:\Users\jonas.gwozdz\Git Projekte\bea2026-german-asag"

# First verify all models work
python -m src.ensemble.run --split trial --limit 3 --workers 1 \
  --models "google/gemini-3-flash-preview,deepseek/deepseek-chat,minimax/minimax-m2.5,moonshotai/kimi-k2.5"

# If smoke test passes, run full trial
python -m src.ensemble.run --split trial --workers 2 \
  --models "google/gemini-3-flash-preview,deepseek/deepseek-chat,minimax/minimax-m2.5,moonshotai/kimi-k2.5"
```

### Also try: C4 + ensemble v2

Combine the smart example selection from C4 with the ensemble. This would require a small code change to wire C4's SmartExampleSelector into the ensemble runner.

## Other ideas to try next session

- [ ] Run C4 with `--n-similar 2` (more similarity-based examples)
- [ ] Run C4 with `--n-boundary 3` (more boundary examples)
- [ ] Try combining C4 smart examples with ensemble v2 models
- [ ] Fine-tune an open-source model if budget allows (Qwen 7B on 7,899 samples)
- [ ] Score test set when released (2026-03-21)
