#!/usr/bin/env python3
"""
Comprehensive analysis of the ALICE-LP-1.0 dataset for BEA 2026 Shared Task.
Rubric-based Short Answer Scoring for German.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = r"C:\Users\jonas.gwozdz\Git Projekte\bea2026-german-asag\data\raw"
FIG_DIR = r"C:\Users\jonas.gwozdz\Git Projekte\bea2026-german-asag\notebooks\figures"
os.makedirs(FIG_DIR, exist_ok=True)

PATHS = {
    "3way_train": os.path.join(BASE, "3way", "ALICE_LP_train_3way__v2.json"),
    "3way_trial": os.path.join(BASE, "3way", "ALICE_LP_trial_3way__v2.json"),
    "2way_train": os.path.join(BASE, "2way", "ALICE_LP_train_2way__v2.json"),
    "2way_trial": os.path.join(BASE, "2way", "ALICE_LP_trial_2way___v2.json"),
}

LABELS_3WAY = ["Correct", "Partially correct", "Incorrect"]
LABELS_2WAY = ["Correct", "Incorrect"]
LABEL_COLORS = {"Correct": "#2ecc71", "Partially correct": "#f39c12", "Incorrect": "#e74c3c"}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150,
                      "figure.facecolor": "white", "axes.facecolor": "white"})

# ─── Load data ────────────────────────────────────────────────────────────────
datasets = {}
for key, path in PATHS.items():
    with open(path, "r", encoding="utf-8") as f:
        datasets[key] = json.load(f)

# Convert to DataFrames
dfs = {}
for key, data in datasets.items():
    df = pd.DataFrame(data)
    df["answer_len"] = df["answer"].str.len()
    df["answer_words"] = df["answer"].str.split().str.len()
    dfs[key] = df

train3 = dfs["3way_train"]
trial3 = dfs["3way_trial"]
train2 = dfs["2way_train"]
trial2 = dfs["2way_trial"]

# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  ALICE-LP-1.0 DATASET ANALYSIS — BEA 2026 Shared Task")
print("=" * 80)

# ─── 1. Overall label distribution ───────────────────────────────────────────
print("\n## 1. Overall Label Distribution\n")
for name, df in [("3-way train", train3), ("3-way trial", trial3),
                  ("2-way train", train2), ("2-way trial", trial2)]:
    total = len(df)
    print(f"### {name} (n={total})")
    counts = df["score"].value_counts()
    for label in (LABELS_3WAY if "3-way" in name else LABELS_2WAY):
        c = counts.get(label, 0)
        print(f"  {label:>20s}: {c:5d}  ({100*c/total:5.1f}%)")
    print()

# ─── 2. Per-question analysis (3-way train) ─────────────────────────────────
print("\n## 2. Per-Question Analysis (3-way train)\n")
q_stats = []
for qid, grp in train3.groupby("question_id"):
    n = len(grp)
    vc = grp["score"].value_counts()
    row = {
        "question_id": qid,
        "question_text": grp["question"].iloc[0],
        "n_answers": n,
    }
    for label in LABELS_3WAY:
        c = vc.get(label, 0)
        row[f"n_{label}"] = c
        row[f"pct_{label}"] = 100 * c / n
    q_stats.append(row)

q_df = pd.DataFrame(q_stats)
n_questions = len(q_df)
print(f"Number of unique questions: {n_questions}")
print(f"Answers per question: min={q_df['n_answers'].min()}, max={q_df['n_answers'].max()}, "
      f"mean={q_df['n_answers'].mean():.1f}, median={q_df['n_answers'].median():.0f}")
print()

# Sorted by % Incorrect (difficulty)
q_diff = q_df.sort_values("pct_Incorrect", ascending=False)
print("### Top-3 Hardest Questions (highest % Incorrect):")
for i, (_, r) in enumerate(q_diff.head(3).iterrows()):
    print(f"  {i+1}. qid={r['question_id'][:12]}...  "
          f"Incorrect={r['pct_Incorrect']:.1f}%  Correct={r['pct_Correct']:.1f}%  n={r['n_answers']}")
    print(f"     Q: {r['question_text'][:150]}")
print()

q_easy = q_df.sort_values("pct_Correct", ascending=False)
print("### Top-3 Easiest Questions (highest % Correct):")
for i, (_, r) in enumerate(q_easy.head(3).iterrows()):
    print(f"  {i+1}. qid={r['question_id'][:12]}...  "
          f"Correct={r['pct_Correct']:.1f}%  Incorrect={r['pct_Incorrect']:.1f}%  n={r['n_answers']}")
    print(f"     Q: {r['question_text'][:150]}")
print()

# Full per-question table
print("### Full Per-Question Label Distribution (sorted by % Correct descending):")
print(f"{'QID (short)':>14s} | {'n':>4s} | {'%Corr':>6s} | {'%Part':>6s} | {'%Incr':>6s}")
print("-" * 50)
for _, r in q_easy.iterrows():
    print(f"  {r['question_id'][:12]:>12s} | {r['n_answers']:4d} | {r['pct_Correct']:5.1f}% | "
          f"{r['pct_Partially correct']:5.1f}% | {r['pct_Incorrect']:5.1f}%")

# ─── 3. Answer length analysis (3-way train) ────────────────────────────────
print("\n\n## 3. Answer Length Analysis (3-way train)\n")
al = train3["answer_len"]
print("### Overall character-level statistics:")
print(f"  Mean:   {al.mean():.1f}")
print(f"  Median: {al.median():.0f}")
print(f"  Std:    {al.std():.1f}")
print(f"  Min:    {al.min()}")
print(f"  Max:    {al.max()}")
for p in [5, 25, 75, 95]:
    print(f"  P{p:02d}:    {np.percentile(al, p):.0f}")

aw = train3["answer_words"]
print(f"\n### Overall word-level statistics:")
print(f"  Mean:   {aw.mean():.1f}")
print(f"  Median: {aw.median():.0f}")
print(f"  Std:    {aw.std():.1f}")
print(f"  Min:    {aw.min()}")
print(f"  Max:    {aw.max()}")

print("\n### Per-label answer length (characters):")
for label in LABELS_3WAY:
    subset = train3[train3["score"] == label]["answer_len"]
    print(f"  {label:>20s}: mean={subset.mean():7.1f}, median={subset.median():6.0f}, "
          f"std={subset.std():7.1f}, n={len(subset)}")

print("\n### Per-label answer length (words):")
for label in LABELS_3WAY:
    subset = train3[train3["score"] == label]["answer_words"]
    print(f"  {label:>20s}: mean={subset.mean():6.1f}, median={subset.median():5.0f}, "
          f"std={subset.std():6.1f}")

print("\n### Per-question mean answer length (characters, sorted):")
q_len = train3.groupby("question_id")["answer_len"].mean().sort_values()
for qid, v in q_len.items():
    print(f"  {qid[:12]:>12s}: {v:7.1f} chars")

# Very short answers
short = train3[train3["answer_len"] < 20]
print(f"\n### Very short answers (< 20 chars): {len(short)} total")
print("  Label distribution:")
for label in LABELS_3WAY:
    c = (short["score"] == label).sum()
    print(f"    {label}: {c}")
print("  Examples:")
for _, r in short.head(8).iterrows():
    print(f"    [{r['score'][:4]}] \"{r['answer'][:60]}\" (len={r['answer_len']})")

# Very long answers
long_answers = train3[train3["answer_len"] > 1000]
print(f"\n### Very long answers (> 1000 chars): {len(long_answers)} total")
for label in LABELS_3WAY:
    c = (long_answers["score"] == label).sum()
    print(f"    {label}: {c}")

# ─── 4. Train vs Trial consistency ──────────────────────────────────────────
print("\n\n## 4. Train vs Trial Consistency Check\n")
train_qids = set(train3["question_id"].unique())
trial_qids = set(trial3["question_id"].unique())
print(f"Unique questions in train: {len(train_qids)}")
print(f"Unique questions in trial: {len(trial_qids)}")
print(f"Questions in both: {len(train_qids & trial_qids)}")
print(f"Only in train: {len(train_qids - trial_qids)}")
print(f"Only in trial: {len(trial_qids - train_qids)}")
same_questions = train_qids == trial_qids
print(f"Same set of questions: {same_questions}")

# Per-question label proportion deviation
max_dev = 0
max_dev_qid = ""
max_dev_label = ""
deviations = []
for qid in train_qids & trial_qids:
    tr = train3[train3["question_id"] == qid]
    tl = trial3[trial3["question_id"] == qid]
    for label in LABELS_3WAY:
        p_tr = (tr["score"] == label).mean()
        p_tl = (tl["score"] == label).mean()
        dev = abs(p_tr - p_tl)
        deviations.append({"qid": qid, "label": label, "train_pct": p_tr*100,
                           "trial_pct": p_tl*100, "deviation": dev*100})
        if dev > max_dev:
            max_dev = dev
            max_dev_qid = qid
            max_dev_label = label

dev_df = pd.DataFrame(deviations)
print(f"\nMax label proportion deviation: {max_dev*100:.1f} pp")
print(f"  Question: {max_dev_qid[:12]}...")
print(f"  Label: {max_dev_label}")
print(f"\nDeviation statistics (percentage points):")
print(f"  Mean: {dev_df['deviation'].mean():.2f}")
print(f"  Median: {dev_df['deviation'].median():.2f}")
print(f"  P95: {np.percentile(dev_df['deviation'], 95):.2f}")
print(f"  Max: {dev_df['deviation'].max():.2f}")

# Show top 10 largest deviations
print("\n### Top-10 Largest Train-Trial Deviations:")
top_dev = dev_df.sort_values("deviation", ascending=False).head(10)
for _, r in top_dev.iterrows():
    print(f"  qid={r['qid'][:12]}..  {r['label']:>20s}: "
          f"train={r['train_pct']:5.1f}%  trial={r['trial_pct']:5.1f}%  "
          f"delta={r['deviation']:5.1f}pp")

# ─── 5. 2-way vs 3-way consistency ──────────────────────────────────────────
print("\n\n## 5. 2-way vs 3-way Consistency\n")

# Check same IDs
ids_3_train = set(train3["id"])
ids_2_train = set(train2["id"])
ids_3_trial = set(trial3["id"])
ids_2_trial = set(trial2["id"])
print(f"Train: same IDs in 2-way and 3-way: {ids_3_train == ids_2_train}")
print(f"Trial: same IDs in 2-way and 3-way: {ids_3_trial == ids_2_trial}")

# Verify mapping: Partially correct -> Incorrect
merged_train = train3.merge(train2, on="id", suffixes=("_3", "_2"))
mapping_check = merged_train.apply(
    lambda r: (r["score_3"] == "Correct" and r["score_2"] == "Correct") or
              (r["score_3"] in ["Partially correct", "Incorrect"] and r["score_2"] == "Incorrect"),
    axis=1
)
n_valid = mapping_check.sum()
n_total = len(mapping_check)
print(f"\nMapping verification (train): {n_valid}/{n_total} samples follow the rule")
print(f"  Correct->Correct and (Partially correct|Incorrect)->Incorrect")

# Detailed breakdown
print("\n  Cross-tabulation 3-way score vs 2-way score:")
ct = pd.crosstab(merged_train["score_3"], merged_train["score_2"])
print(ct.to_string())

merged_trial = trial3.merge(trial2, on="id", suffixes=("_3", "_2"))
mapping_check_trial = merged_trial.apply(
    lambda r: (r["score_3"] == "Correct" and r["score_2"] == "Correct") or
              (r["score_3"] in ["Partially correct", "Incorrect"] and r["score_2"] == "Incorrect"),
    axis=1
)
print(f"\nMapping verification (trial): {mapping_check_trial.sum()}/{len(mapping_check_trial)} correct")

# ─── 6. Question text analysis ──────────────────────────────────────────────
print("\n\n## 6. Question Text Analysis\n")

# Get unique questions from train3
unique_q = train3.drop_duplicates("question_id")[["question_id", "question"]].copy()
unique_q["q_len"] = unique_q["question"].str.len()
unique_q["q_words"] = unique_q["question"].str.split().str.len()
q_counts = train3["question_id"].value_counts().to_dict()
unique_q["n_answers"] = unique_q["question_id"].map(q_counts)

print(f"Number of unique question_ids: {len(unique_q)}")
print(f"Number of unique question texts: {unique_q['question'].nunique()}")

# Check if any question_ids share text
text_to_qids = defaultdict(list)
for _, r in unique_q.iterrows():
    text_to_qids[r["question"]].append(r["question_id"])
shared = {k: v for k, v in text_to_qids.items() if len(v) > 1}
if shared:
    print(f"\nQuestion texts shared by multiple question_ids: {len(shared)}")
    for text, qids in shared.items():
        print(f"  Text: \"{text[:80]}...\"  -> {len(qids)} question_ids")
else:
    print("All question_ids have unique question texts.")

print(f"\nQuestion text length (characters):")
print(f"  Min:  {unique_q['q_len'].min()}")
print(f"  Max:  {unique_q['q_len'].max()}")
print(f"  Mean: {unique_q['q_len'].mean():.1f}")
print(f"  Median: {unique_q['q_len'].median():.0f}")

print(f"\nQuestion text length (words):")
print(f"  Min:  {unique_q['q_words'].min()}")
print(f"  Max:  {unique_q['q_words'].max()}")
print(f"  Mean: {unique_q['q_words'].mean():.1f}")

print(f"\n### All 78 Questions (sorted by question_id):")
for i, (_, r) in enumerate(unique_q.sort_values("question_id").iterrows()):
    print(f"  {i+1:2d}. [{r['question_id'][:12]}..] n={r['n_answers']:3d}  "
          f"({r['q_len']:3d} chars)  {r['question'][:150]}")

# ─── 7. Rubric analysis ─────────────────────────────────────────────────────
print("\n\n## 7. Rubric Analysis\n")

# Check rubric consistency per question
rubric_consistent = True
inconsistent_questions = []
for qid, grp in train3.groupby("question_id"):
    rubrics = grp["rubric"].apply(lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if rubrics.nunique() > 1:
        rubric_consistent = False
        inconsistent_questions.append(qid)

print(f"Rubrics consistent within each question: {rubric_consistent}")
if inconsistent_questions:
    print(f"  Inconsistent questions: {len(inconsistent_questions)}")

# Rubric text lengths
rubric_lengths = {"Correct": [], "Partially correct": [], "Incorrect": []}
unique_rubrics = train3.drop_duplicates("question_id")["rubric"]
for rubric in unique_rubrics:
    for key in LABELS_3WAY:
        rubric_lengths[key].append(len(rubric[key]))

print(f"\nRubric text lengths (per category, across {len(unique_rubrics)} unique questions):")
for key in LABELS_3WAY:
    vals = rubric_lengths[key]
    print(f"  {key:>20s}: mean={np.mean(vals):6.1f}, median={np.median(vals):5.0f}, "
          f"min={min(vals):3d}, max={max(vals):4d}")

# Sample 3 rubrics
print("\n### Sample Rubrics (3 different questions):")
sample_rubrics = train3.drop_duplicates("question_id").sample(3, random_state=42)
for _, r in sample_rubrics.iterrows():
    print(f"\n  Question ({r['question_id'][:12]}..): {r['question'][:120]}")
    for key in LABELS_3WAY:
        print(f"    {key}: {r['rubric'][key]}")

# ─── 8. Cross-tabulation ────────────────────────────────────────────────────
print("\n\n## 8. Cross-Tabulation Analysis\n")

# Answer length quartiles x label
train3_copy = train3.copy()
train3_copy["len_quartile"] = pd.qcut(train3_copy["answer_len"], 4,
                                        labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
ct = pd.crosstab(train3_copy["len_quartile"], train3_copy["score"], normalize="index") * 100
print("### Answer Length Quartile x Label (row percentages):")
print(ct[LABELS_3WAY].round(1).to_string())

# Quartile boundaries
print(f"\n  Quartile boundaries (chars):")
for i, q in enumerate([0, 25, 50, 75, 100]):
    print(f"    P{q}: {np.percentile(train3['answer_len'], q):.0f}")

# Questions with highly imbalanced labels
print("\n### Questions with Highly Imbalanced Labels (>60% one class):")
imbalanced = []
for _, r in q_df.iterrows():
    for label in LABELS_3WAY:
        pct = r[f"pct_{label}"]
        if pct > 60:
            imbalanced.append((r["question_id"], label, pct, r["n_answers"],
                              r["question_text"][:100]))
imbalanced.sort(key=lambda x: -x[2])
print(f"  Total: {len(imbalanced)} question-label combinations")
for qid, label, pct, n, qtxt in imbalanced:
    print(f"  {qid[:12]}..  {label:>20s}: {pct:5.1f}%  (n={n})  {qtxt[:80]}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n## 9. Generating Figures\n")

# ─── fig01: Label distribution (3-way train + trial side by side) ────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(LABELS_3WAY))
w = 0.35
train_counts = [train3["score"].value_counts().get(l, 0) for l in LABELS_3WAY]
trial_counts = [trial3["score"].value_counts().get(l, 0) for l in LABELS_3WAY]
train_pcts = [100 * c / len(train3) for c in train_counts]
trial_pcts = [100 * c / len(trial3) for c in trial_counts]

bars1 = ax.bar(x - w/2, train_pcts, w, label=f"Train (n={len(train3)})",
               color=[LABEL_COLORS[l] for l in LABELS_3WAY], alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, trial_pcts, w, label=f"Trial (n={len(trial3)})",
               color=[LABEL_COLORS[l] for l in LABELS_3WAY], alpha=0.5, edgecolor="white",
               hatch="//")

for bar, pct in zip(bars1, train_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10)
for bar, pct in zip(bars2, trial_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(LABELS_3WAY, fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("3-Way Label Distribution: Train vs Trial", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, max(train_pcts + trial_pcts) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig01_label_distribution.png"))
plt.close(fig)
print("  Saved fig01_label_distribution.png")

# ─── fig02: Answers per question (sorted) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
q_sorted = q_df.sort_values("n_answers", ascending=False)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(q_sorted)))
ax.bar(range(len(q_sorted)), q_sorted["n_answers"].values, color=colors, edgecolor="none")
ax.set_xlabel("Question (sorted by answer count)", fontsize=12)
ax.set_ylabel("Number of Answers", fontsize=12)
ax.set_title("Answer Counts per Question (3-way train)", fontsize=14, fontweight="bold")
ax.set_xticks([0, len(q_sorted)//4, len(q_sorted)//2, 3*len(q_sorted)//4, len(q_sorted)-1])
ax.axhline(y=q_df["n_answers"].mean(), color="red", linestyle="--", alpha=0.7,
           label=f"Mean = {q_df['n_answers'].mean():.1f}")
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig02_answers_per_question.png"))
plt.close(fig)
print("  Saved fig02_answers_per_question.png")

# ─── fig03: Answer length by label (box plot) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Characters
data_chars = [train3[train3["score"] == l]["answer_len"].values for l in LABELS_3WAY]
bp = axes[0].boxplot(data_chars, labels=LABELS_3WAY, patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2))
for patch, label in zip(bp["boxes"], LABELS_3WAY):
    patch.set_facecolor(LABEL_COLORS[label])
    patch.set_alpha(0.7)
axes[0].set_ylabel("Answer Length (characters)", fontsize=12)
axes[0].set_title("By Characters", fontsize=12, fontweight="bold")

# Words
data_words = [train3[train3["score"] == l]["answer_words"].values for l in LABELS_3WAY]
bp2 = axes[1].boxplot(data_words, labels=LABELS_3WAY, patch_artist=True, showfliers=False,
                       medianprops=dict(color="black", linewidth=2))
for patch, label in zip(bp2["boxes"], LABELS_3WAY):
    patch.set_facecolor(LABEL_COLORS[label])
    patch.set_alpha(0.7)
axes[1].set_ylabel("Answer Length (words)", fontsize=12)
axes[1].set_title("By Words", fontsize=12, fontweight="bold")

fig.suptitle("Answer Length Distribution by Label (3-way train, no outliers)", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig03_answer_length_by_label.png"), bbox_inches="tight")
plt.close(fig)
print("  Saved fig03_answer_length_by_label.png")

# ─── fig04: Question difficulty (stacked bar, sorted by % correct) ──────────
fig, ax = plt.subplots(figsize=(16, 6))
q_sorted_diff = q_df.sort_values("pct_Correct", ascending=True)
x = range(len(q_sorted_diff))

bottoms = np.zeros(len(q_sorted_diff))
for label in ["Incorrect", "Partially correct", "Correct"]:
    vals = q_sorted_diff[f"pct_{label}"].values
    ax.barh(x, vals, left=bottoms, label=label, color=LABEL_COLORS[label], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    bottoms += vals

ax.set_xlabel("Percentage (%)", fontsize=12)
ax.set_ylabel("Question (sorted by % Correct)", fontsize=12)
ax.set_title("Label Proportions per Question (3-way train)", fontsize=14, fontweight="bold")
ax.set_yticks([])
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim(0, 100)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig04_question_difficulty.png"))
plt.close(fig)
print("  Saved fig04_question_difficulty.png")

# ─── fig05: Answer length distribution (histogram, log scale) ───────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.logspace(np.log10(max(1, train3["answer_len"].min())),
                   np.log10(train3["answer_len"].max()), 50)
for label in LABELS_3WAY:
    subset = train3[train3["score"] == label]["answer_len"]
    ax.hist(subset, bins=bins, alpha=0.6, label=f"{label} (n={len(subset)})",
            color=LABEL_COLORS[label], edgecolor="white", linewidth=0.5)

ax.set_xscale("log")
ax.set_xlabel("Answer Length in Characters (log scale)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Answer Length Distribution by Label (3-way train)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig05_answer_length_distribution.png"))
plt.close(fig)
print("  Saved fig05_answer_length_distribution.png")

# ─── fig06: Label balance heatmap ───────────────────────────────────────────
# Build matrix: questions (sorted by %Correct) x labels
q_sorted_hm = q_df.sort_values("pct_Correct", ascending=False)
hm_data = q_sorted_hm[["pct_Correct", "pct_Partially correct", "pct_Incorrect"]].values
q_short_ids = [qid[:8] for qid in q_sorted_hm["question_id"]]

fig, ax = plt.subplots(figsize=(7, 18))
im = ax.imshow(hm_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Correct", "Partially\ncorrect", "Incorrect"], fontsize=11)
ax.set_yticks(range(len(q_short_ids)))
ax.set_yticklabels(q_short_ids, fontsize=7)
ax.set_ylabel("Question ID (sorted by % Correct)", fontsize=12)
ax.set_title("Label Proportion Heatmap (%)", fontsize=14, fontweight="bold")

# Add text annotations
for i in range(len(hm_data)):
    for j in range(3):
        val = hm_data[i, j]
        color = "white" if val > 65 or val < 15 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.5, label="Percentage (%)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig06_label_balance_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("  Saved fig06_label_balance_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
print("\n\n## Summary Statistics for Quick Reference\n")
print(f"Total samples: train={len(train3)}, trial={len(trial3)}")
print(f"Unique questions: {n_questions}")
print(f"Answers per question: {q_df['n_answers'].min()}-{q_df['n_answers'].max()} "
      f"(mean {q_df['n_answers'].mean():.1f})")
print(f"Answer length (chars): {al.min()}-{al.max()} (mean {al.mean():.1f}, median {al.median():.0f})")
print(f"Answer length (words): {aw.min()}-{aw.max()} (mean {aw.mean():.1f}, median {aw.median():.0f})")
print(f"Rubrics consistent: {rubric_consistent}")
print(f"2-way mapping verified: train={n_valid}/{n_total}, trial={mapping_check_trial.sum()}/{len(mapping_check_trial)}")
print(f"Train-trial same questions: {same_questions}")
print(f"Max train-trial deviation: {max_dev*100:.1f} pp")
print(f"Questions with >60% one class: {len(imbalanced)}")
print(f"\nFigures saved to: {FIG_DIR}")
print("\nDone.")
