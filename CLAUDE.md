# BEA 2026 Shared Task: Rubric-based Short Answer Scoring for German

## Project Overview
Shared task entry for the BEA 2026 Workshop (co-located with ACL 2026).
Task: Given a question, student answer, and textual rubric, predict the score.

## Key Details
- **Metric:** Quadratic Weighted Kappa (QWK)
- **Tracks:** 3-way (Correct / Partially correct / Incorrect) and 2-way (Correct / Incorrect)
- **Labels (exact casing):** `"Correct"`, `"Partially correct"`, `"Incorrect"` — casing matters!
- **Data:** 7,072 train + 827 trial samples, 78 unique questions, German STEM answers
- **Test data released:** 2026-03-21
- **Submission deadline:** 2026-03-28
- **Paper deadline:** 2026-04-24

## Project Structure
```
data/raw/          # Original ALICE-LP-1.0 data (do not modify)
data/processed/    # Cleaned/transformed data
src/               # Source code (scoring pipeline, evaluation, prompts)
notebooks/         # Exploratory analysis
submissions/       # Submission files for each track
paper/             # System description paper (up to 5 pages)
```

## Conventions
- Python 3.11+
- Use `uv` for dependency management if available
- Keep prompts in separate files under `src/prompts/` for version control
- Log all LLM calls with input/output tokens for cost tracking
- Evaluation scripts should output QWK + secondary metrics (P, R, weighted F1)
