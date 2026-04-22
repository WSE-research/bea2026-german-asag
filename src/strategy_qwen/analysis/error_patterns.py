"""Reproduce the §5.3 "Error Patterns" analysis from the BEA 2026 paper.

Produces:
  A. Trial-set confusion analysis on 32B train-only predictions:
     - Adjacent vs skip-a-level error split
     - Correct/PC vs PC/Incorrect boundary share
     - Over- vs under-scoring ratio (leniency)
     - Per-class precision

  B. Test-set cross-model agreement (seen vs unseen):
     - N-way unanimous rate
     - 72B-vs-majority disagreement rate on unseen
     - 72B disagreement direction (PC→Incorrect, Correct→PC)

  C. 72B prediction distribution shift (seen vs unseen training-match)

  D. Short-answer (<20 char) classification rate per track

All inputs are in results/strategy_qwen/ and submissions/. Run from repo root:
    python -m src.strategy_qwen.analysis.error_patterns
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
R = REPO / "results" / "strategy_qwen"
C5C = REPO / "results" / "strategy_c5c"
SUB = REPO / "submissions"
DATA = REPO / "data" / "raw" / "3way"
LABELS = ["Correct", "Partially correct", "Incorrect"]
ORD = {"Correct": 0, "Partially correct": 1, "Incorrect": 2}


def load_preds(path: Path, pred_key: str = "pred", id_key: str = "id") -> dict[str, str]:
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("predictions", [])
    out = {}
    for d in data:
        p = d.get("pred") or d.get("score") or d.get("predicted_score")
        if p in LABELS:
            out[d[id_key]] = p
    return out


def trial_confusion_analysis():
    """Part A — 32B train-only on 827 trial."""
    preds = json.load(open(R / "predictions_confidence_32b_20260321_122110.json"))
    n = len(preds)
    errors = [d for d in preds if d["pred"] != d["gold"]]
    n_err = len(errors)

    # Adjacent vs skip
    adjacent, skip, c_pc, pc_i = 0, 0, 0, 0
    over, under = 0, 0
    for d in errors:
        g, p = ORD[d["gold"]], ORD[d["pred"]]
        dist = abs(g - p)
        if dist == 1:
            adjacent += 1
            if {g, p} == {0, 1}:
                c_pc += 1
            elif {g, p} == {1, 2}:
                pc_i += 1
        else:
            skip += 1
        # Over-scoring = predicted better (lower ordinal) than gold
        if p < g:
            over += 1
        elif p > g:
            under += 1

    # Per-class precision
    conf = defaultdict(lambda: Counter())
    for d in preds:
        conf[d["pred"]][d["gold"]] += 1
    prec = {
        lbl: conf[lbl][lbl] / sum(conf[lbl].values()) if sum(conf[lbl].values()) else 0.0
        for lbl in LABELS
    }

    print("=" * 70)
    print("A. Trial-set error patterns (Qwen2.5-32B train-only, N=827)")
    print("=" * 70)
    print(f"Accuracy: {(n - n_err) / n:.4f}  |  Errors: {n_err} / {n}")
    print(f"  Adjacent:    {adjacent} ({adjacent / n_err:.1%})")
    print(f"    C/PC:      {c_pc} ({c_pc / n_err:.1%})")
    print(f"    PC/I:      {pc_i} ({pc_i / n_err:.1%})")
    print(f"  Skip-level:  {skip} ({skip / n_err:.1%})")
    print(f"  Over-score:  {over}  Under-score: {under}  Ratio: {over / under:.2f}x")
    print("Per-class precision:")
    for lbl in LABELS:
        print(f"  {lbl:<20s}  P = {prec[lbl]:.3f}")


def cross_model_agreement():
    """Part B — test-set agreement across 6 models (no gold needed)."""
    models = {
        "72B":    SUB / "72b" / "72b_ft_unseen_answers_3way.json",
        "32B":    SUB / "32b_ft_unseen_answers_3way.json",
        "14B":    R / "predictions_test_14b_unseen_answers_20260322_150442.json",
        "7B":     R / "predictions_test_7b_unseen_answers_20260325_183938.json",
        "Gemini": C5C / "predictions_test_google_gemini-3-flash-preview_20260322_194325.json",
        "Q26":    R / "predictions_test_q26_unseen_answers_20260325_191918.json",
    }
    models_unseen = {
        "72B":    SUB / "72b" / "72b_ft_unseen_questions_3way.json",
        "32B":    SUB / "32b_ft_unseen_questions_3way.json",
        "14B":    R / "predictions_test_14b_unseen_questions_20260322_152330.json",
        "7B":     R / "predictions_test_7b_unseen_questions_20260325_184557.json",
        "Gemini": C5C / "predictions_test_google_gemini-3-flash-preview_20260322_195931.json",
        "Q26":    R / "predictions_test_q26_unseen_questions_20260325_193358.json",
    }

    print("\n" + "=" * 70)
    print("B. Cross-model agreement on test set")
    print("=" * 70)

    for label, mpaths in [("Seen questions (T1, N=2008)", models),
                          ("Unseen questions (T3, N=3086)", models_unseen)]:
        preds_by_model = {}
        for name, path in mpaths.items():
            try:
                preds_by_model[name] = load_preds(path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  [skip {name}] {e}")
        common = set.intersection(*(set(p.keys()) for p in preds_by_model.values()))
        unanimous = 0
        for sid in common:
            votes = {preds_by_model[m][sid] for m in preds_by_model}
            if len(votes) == 1:
                unanimous += 1
        print(f"  {label}: N={len(common)}, models={list(preds_by_model)}")
        print(f"    Unanimous across {len(preds_by_model)} models: {unanimous} ({unanimous / len(common):.1%})")

        # 72B vs 5-model majority (unseen only)
        if "Unseen" in label and "72B" in preds_by_model:
            others = [m for m in preds_by_model if m != "72B"]
            disagree_72b = 0
            pc_to_i = 0
            c_to_pc = 0
            for sid in common:
                votes5 = Counter(preds_by_model[m][sid] for m in others)
                majority = votes5.most_common(1)[0][0]
                if preds_by_model["72B"][sid] != majority:
                    disagree_72b += 1
                    if majority == "Partially correct" and preds_by_model["72B"][sid] == "Incorrect":
                        pc_to_i += 1
                    if majority == "Correct" and preds_by_model["72B"][sid] == "Partially correct":
                        c_to_pc += 1
            print(f"    72B disagrees with 5-model majority: {disagree_72b} ({disagree_72b / len(common):.1%})")
            print(f"      PC-majority → 72B says Incorrect: {pc_to_i} ({pc_to_i / disagree_72b:.1%} of disagreements)")
            print(f"      C-majority  → 72B says PC:       {c_to_pc} ({c_to_pc / disagree_72b:.1%} of disagreements)")


def distribution_shift_and_length():
    """Part C+D — 72B prediction distribution + short-answer patterns."""
    train = json.load(open(DATA / "ALICE_LP_train_3way__v2.json"))
    seen_data = json.load(open(DATA / "3way_unseen_answers_eval.json"))
    unseen_data = json.load(open(DATA / "3way_unseen_questions_eval.json"))
    seen_preds = load_preds(SUB / "72b" / "72b_ft_unseen_answers_3way.json", pred_key="score")
    unseen_preds = load_preds(SUB / "72b" / "72b_ft_unseen_questions_3way.json", pred_key="score")

    train_dist = {lbl: sum(1 for s in train if s["score"] == lbl) / len(train) for lbl in LABELS}
    seen_dist = {lbl: sum(1 for v in seen_preds.values() if v == lbl) / len(seen_preds) for lbl in LABELS}
    unseen_dist = {lbl: sum(1 for v in unseen_preds.values() if v == lbl) / len(unseen_preds) for lbl in LABELS}

    print("\n" + "=" * 70)
    print("C. Prediction distribution shift (72B)")
    print("=" * 70)
    print(f"{'Label':<20s}  {'Train':>8s}  {'Seen':>8s}  {'Δ train':>10s}  {'Unseen':>8s}  {'Δ train':>10s}")
    for lbl in LABELS:
        print(f"{lbl:<20s}  {train_dist[lbl]:>8.3f}  {seen_dist[lbl]:>8.3f}  "
              f"{seen_dist[lbl] - train_dist[lbl]:>+10.3f}  {unseen_dist[lbl]:>8.3f}  "
              f"{unseen_dist[lbl] - train_dist[lbl]:>+10.3f}")

    print("\n" + "=" * 70)
    print("D. Short-answer (<20 char) classification rate (72B)")
    print("=" * 70)
    for label, data, preds in [("Seen", seen_data, seen_preds),
                                ("Unseen", unseen_data, unseen_preds)]:
        short = [s for s in data if len(s["answer"]) < 20 and s["id"] in preds]
        if not short:
            print(f"  {label}: no short answers")
            continue
        counts = Counter(preds[s["id"]] for s in short)
        print(f"  {label} (N={len(short)}): {dict(counts)}  "
              f"→ Incorrect-rate = {counts.get('Incorrect', 0) / len(short):.1%}")


if __name__ == "__main__":
    trial_confusion_analysis()
    cross_model_agreement()
    distribution_shift_and_length()
