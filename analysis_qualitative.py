#!/usr/bin/env python3
"""Qualitative analysis of the ALICE-LP-1.0 dataset for BEA 2026 Shared Task."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import re
from collections import Counter, defaultdict
import textwrap

DATA_PATH = r"data/raw/3way/ALICE_LP_train_3way__v2.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
questions = {}
for s in data:
    qid = s["question_id"]
    if qid not in questions:
        questions[qid] = s["question"]
print(f"Unique questions: {len(questions)}")

# ============================================================
# 1. DOMAIN CLASSIFICATION
# ============================================================
print("\n" + "=" * 80)
print("1. DOMAIN CLASSIFICATION")
print("=" * 80)

# Keyword lists for domain detection (German science terms)
domain_keywords = {
    "Physics": [
        r"[Kk]raft", r"[Ee]nergie", r"[Gg]eschwindigkeit", r"[Ss]trom",
        r"[Ss]pannung", r"[Mm]agnet", r"[Ww]ärme", r"[Ll]icht",
        r"[Bb]ewegung", r"[Bb]eschleunig", r"[Mm]asse", r"[Gg]ewicht",
        r"[Dd]ruck", r"[Tt]emperatur", r"[Ss]chwing", r"[Ww]elle",
        r"[Ff]requenz", r"[Ee]lektr", r"[Gg]ravitation", r"[Rr]eibung",
        r"[Ss]chall", r"[Ss]piegel", r"[Ll]inse", r"[Oo]ptik",
        r"[Nn]ewton", r"[Jj]oule", r"[Ww]att", r"[Vv]olt", r"[Aa]mpere",
        r"[Ss]trahlung", r"[Kk]ern", r"[Aa]tom(?!.*[Bb]indung)",
        r"[Pp]hysik", r"[Mm]echanik", r"[Tt]hermodynamik",
        r"[Ff]eder", r"[Pp]endel", r"[Hh]ebel", r"[Rr]olle",
        r"[Ss]tromkreis", r"[Ww]iderstand", r"[Kk]ondensator",
        r"[Gg]enerator", r"[Tt]ransformator", r"[Mm]otor",
    ],
    "Biology": [
        r"[Zz]elle", r"[Pp]hotosynthese", r"[Vv]erdauung", r"[Ee]volution",
        r"[Öö]kosystem", r"[Pp]flanze", r"[Tt]ier", r"[Oo]rgan",
        r"[Bb]lut", r"[Hh]erz", r"[Ll]unge", r"[Nn]erv", r"[Gg]ehirn",
        r"[Mm]uskel", r"[Kk]nochen", r"[Hh]aut", r"[Aa]uge", r"[Oo]hr",
        r"[Gg]en(?:e|etik)", r"[Dd]NA", r"[Pp]rotein", r"[Ee]nzym",
        r"[Bb]akterie", r"[Vv]irus", r"[Pp]ilz", r"[Aa]lge",
        r"[Bb]iologie", r"[Ll]ebewesen", r"[Oo]rganismus",
        r"[Ff]ortpflanzung", r"[Ww]achstum", r"[Ss]toffwechsel",
        r"[Aa]tmu?ng", r"[Nn]ahrung", r"[Ee]rnährung",
        r"[Bb]iodiversität", r"[Aa]rt(?:en)?(?:vielfalt)?",
        r"[Hh]abitat", r"[Nn]ahrungskette", r"[Nn]ahrungsnetz",
    ],
    "Chemistry": [
        r"[Rr]eaktion", r"[Mm]olekül", r"[Ss]äure", r"[Bb]ase",
        r"[Oo]xidation", r"[Ee]lement", r"[Ss]toff(?:e|en)?",
        r"[Cc]hemie", r"[Cc]hemisch", r"[Vv]erbindung",
        r"[Ll]ösung", r"[Kk]ristall", r"[Gg]as", r"[Ff]lüssig",
        r"[Ff]est(?:stoff)?", r"[Pp]eriodensystem",
        r"[Ii]on(?:en)?", r"[Ee]lektron", r"[Pp]roton", r"[Nn]eutron",
        r"[Bb]indung", r"[Kk]ovalent", r"[Aa]tom(?:.*[Bb]indung)",
        r"[Rr]edox", r"[Kk]atalyse", r"[Kk]atalysator",
        r"[Pp]H", r"[Kk]onzentration", r"[Tt]itration",
        r"[Dd]estillation", r"[Ff]iltration", r"[Ss]ublimation",
        r"[Vv]erdampf", r"[Kk]ondensation", r"[Ss]chmelz",
        r"[Ee]xotherm", r"[Ee]ndotherm", r"[Ee]nthalpie",
        r"[Kk]ollision", r"[Zz]erteilungsgrad", r"[Oo]berfläche.*[Rr]eaktion",
        r"[Vv]erderb", r"[Gg]ärung", r"[Kk]orrosion",
    ],
    "Mathematics": [
        r"[Gg]leichung", r"[Ff]unktion", r"[Gg]raph", r"[Bb]erechn",
        r"[Ww]inkel", r"[Ff]läche", r"[Vv]olumen", r"[Mm]athematik",
        r"[Aa]lgebra", r"[Gg]eometrie", r"[Ss]tatistik",
        r"[Ww]ahrscheinlichkeit", r"[Pp]rozent", r"[Bb]ruch",
        r"[Dd]ezimal", r"[Pp]otenz", r"[Ww]urzel",
        r"[Dd]iagramm", r"[Tt]abelle", r"[Dd]aten",
        r"[Mm]ittelwert", r"[Mm]edian", r"[Ss]tandardabweichung",
    ],
}

def classify_question(question_text):
    """Classify a question into a domain based on keyword matching."""
    scores = {}
    for domain, keywords in domain_keywords.items():
        count = 0
        for kw in keywords:
            if re.search(kw, question_text):
                count += 1
        scores[domain] = count
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "Other/Unclear"
    # If there's a tie or close call, prefer the higher match
    return best_domain

question_domains = {}
for qid, qtext in questions.items():
    domain = classify_question(qtext)
    question_domains[qid] = domain

# Count by domain
domain_q_counts = Counter(question_domains.values())
domain_a_counts = Counter()
for s in data:
    domain_a_counts[question_domains[s["question_id"]]] += 1

print("\n--- Questions per domain ---")
for domain in ["Physics", "Biology", "Chemistry", "Mathematics", "Other/Unclear"]:
    qc = domain_q_counts.get(domain, 0)
    ac = domain_a_counts.get(domain, 0)
    print(f"  {domain:20s}: {qc:3d} questions, {ac:5d} answers")

print(f"\n--- All questions with assigned domain ---")
for i, (qid, qtext) in enumerate(sorted(questions.items(), key=lambda x: question_domains[x[0]]), 1):
    domain = question_domains[qid]
    short_q = qtext[:120].replace("\n", " ")
    if len(qtext) > 120:
        short_q += "..."
    print(f"  [{domain:15s}] Q{i:02d}: {short_q}")

# ============================================================
# 2. RUBRIC PATTERN ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("2. RUBRIC PATTERN ANALYSIS")
print("=" * 80)

# Collect unique rubrics by question
rubrics_by_q = {}
for s in data:
    qid = s["question_id"]
    if qid not in rubrics_by_q:
        rubrics_by_q[qid] = s["rubric"]

# Rubric lengths
correct_lens = []
partial_lens = []
incorrect_lens = []
for qid, rubric in rubrics_by_q.items():
    correct_lens.append(len(rubric.get("Correct", "")))
    partial_lens.append(len(rubric.get("Partially correct", "")))
    incorrect_lens.append(len(rubric.get("Incorrect", "")))

print("\n--- Average rubric length (chars) by score level ---")
print(f"  Correct:            {sum(correct_lens)/len(correct_lens):.1f} chars (min={min(correct_lens)}, max={max(correct_lens)})")
print(f"  Partially correct:  {sum(partial_lens)/len(partial_lens):.1f} chars (min={min(partial_lens)}, max={max(partial_lens)})")
print(f"  Incorrect:          {sum(incorrect_lens)/len(incorrect_lens):.1f} chars (min={min(incorrect_lens)}, max={max(incorrect_lens)})")

# Common phrases in rubrics
print("\n--- Common phrases in rubrics ---")
for level in ["Correct", "Partially correct", "Incorrect"]:
    texts = [rubrics_by_q[qid].get(level, "") for qid in rubrics_by_q]
    # Extract common multi-word phrases
    all_words = " ".join(texts).lower()
    # Find common 3-grams
    words = all_words.split()
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    tc = Counter(trigrams).most_common(10)
    print(f"\n  [{level}] Top trigrams:")
    for tg, cnt in tc:
        print(f"    {cnt:3d}x  '{tg}'")

# Check rubric patterns
print("\n--- Rubric structural patterns ---")
# Are rubrics typically written as "Die Schüler:innen..." or other patterns?
patterns = Counter()
for qid, rubric in rubrics_by_q.items():
    for level, text in rubric.items():
        if text.startswith("Die Schüler"):
            patterns["Starts with 'Die Schüler:innen'"] += 1
        elif text.startswith("Die SuS"):
            patterns["Starts with 'Die SuS'"] += 1
        elif text.startswith("Der/Die"):
            patterns["Starts with 'Der/Die'"] += 1
        else:
            patterns[f"Other start: '{text[:40]}...'"] += 1

print("  Rubric text opening patterns:")
for pat, cnt in patterns.most_common(15):
    print(f"    {cnt:3d}x  {pat}")

# Analyze how "Partially correct" differs from "Correct"
print("\n--- How 'Partially correct' differs from 'Correct' ---")
partial_diff_patterns = Counter()
for qid, rubric in rubrics_by_q.items():
    c = rubric.get("Correct", "").lower()
    p = rubric.get("Partially correct", "").lower()
    if "teilweise" in p:
        partial_diff_patterns["contains 'teilweise'"] += 1
    if "nicht vollständig" in p:
        partial_diff_patterns["contains 'nicht vollständig'"] += 1
    if "unvollständig" in p:
        partial_diff_patterns["contains 'unvollständig'"] += 1
    if "ansatz" in p or "ansätz" in p:
        partial_diff_patterns["contains 'Ansatz/Ansätze'"] += 1
    if "umfassend" in c:
        partial_diff_patterns["Correct uses 'umfassend'"] += 1
    if "vollständig" in c:
        partial_diff_patterns["Correct uses 'vollständig'"] += 1
    if "korrekt" in c:
        partial_diff_patterns["Correct uses 'korrekt'"] += 1
    # Check if partial = correct with "teilweise" substitution
    if c.replace("umfassend", "").replace("vollständig", "").strip() == \
       p.replace("teilweise", "").replace("nicht vollständig", "").strip():
        partial_diff_patterns["Same structure, keyword swap"] += 1

for pat, cnt in partial_diff_patterns.most_common():
    print(f"    {cnt:3d}x  {pat}")

# Print 5 representative rubrics
print("\n--- 5 Representative rubrics (full text) ---")
# Pick from different domains
domains_seen = set()
repr_count = 0
# Sort by domain to get diversity
sorted_qids = sorted(rubrics_by_q.keys(), key=lambda qid: question_domains.get(qid, "ZZZ"))
for qid in sorted_qids:
    domain = question_domains.get(qid, "Other")
    if domain in domains_seen and repr_count >= 3:
        continue
    domains_seen.add(domain)
    rubric = rubrics_by_q[qid]
    q_text = questions[qid]
    print(f"\n  --- Rubric {repr_count+1} (Domain: {domain}) ---")
    print(f"  Question: {q_text[:200]}{'...' if len(q_text)>200 else ''}")
    for level in ["Correct", "Partially correct", "Incorrect"]:
        print(f"  [{level}]: {rubric.get(level, 'N/A')}")
    repr_count += 1
    if repr_count >= 5:
        break

# ============================================================
# 3. ANSWER QUALITY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("3. ANSWER QUALITY ANALYSIS")
print("=" * 80)

# Group samples by score
by_score = defaultdict(list)
for s in data:
    by_score[s["score"]].append(s)

# Print 3 examples per label from different questions
for label in ["Correct", "Partially correct", "Incorrect"]:
    samples = by_score[label]
    print(f"\n--- 3 examples of '{label}' answers ---")
    seen_qids = set()
    count = 0
    for s in samples:
        if s["question_id"] in seen_qids:
            continue
        seen_qids.add(s["question_id"])
        print(f"\n  Example {count+1}:")
        print(f"  Question: {s['question'][:150]}{'...' if len(s['question'])>150 else ''}")
        print(f"  Rubric [{label}]: {s['rubric'][label]}")
        ans_display = s['answer'][:300].replace('\n', ' | ')
        print(f"  Answer: {ans_display}{'...' if len(s['answer'])>300 else ''}")
        count += 1
        if count >= 3:
            break

# Answer length statistics by label
print("\n--- Answer length statistics (chars) by label ---")
for label in ["Correct", "Partially correct", "Incorrect"]:
    lengths = [len(s["answer"]) for s in by_score[label]]
    avg = sum(lengths) / len(lengths) if lengths else 0
    print(f"  {label:20s}: mean={avg:6.1f}, min={min(lengths):4d}, max={max(lengths):5d}, median={sorted(lengths)[len(lengths)//2]:5d}, n={len(lengths)}")

# Word count by label
print("\n--- Answer word count by label ---")
for label in ["Correct", "Partially correct", "Incorrect"]:
    wc = [len(s["answer"].split()) for s in by_score[label]]
    avg = sum(wc) / len(wc) if wc else 0
    print(f"  {label:20s}: mean={avg:5.1f} words, min={min(wc):3d}, max={max(wc):4d}, median={sorted(wc)[len(wc)//2]:4d}")

# ============================================================
# 4. EDGE CASE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("4. EDGE CASE ANALYSIS")
print("=" * 80)

# Shortest 10 answers
all_by_len = sorted(data, key=lambda s: len(s["answer"]))
print("\n--- 10 shortest answers ---")
for i, s in enumerate(all_by_len[:10]):
    print(f"  {i+1}. [{s['score']:20s}] ({len(s['answer']):3d} chars) Answer: '{s['answer'].strip()}'")
    print(f"     Question: {s['question'][:100]}...")

# Longest 5 answers
print("\n--- 5 longest answers ---")
for i, s in enumerate(sorted(data, key=lambda s: len(s["answer"]), reverse=True)[:5]):
    print(f"  {i+1}. [{s['score']:20s}] ({len(s['answer']):5d} chars) Answer: '{s['answer'][:200].strip()}...'")

# Empty or near-empty answers
print("\n--- Empty or near-empty answers (<=5 chars after strip) ---")
empty_count = 0
for s in data:
    stripped = s["answer"].strip()
    if len(stripped) <= 5:
        print(f"  [{s['score']:20s}] Answer: '{stripped}' | Question: {s['question'][:80]}...")
        empty_count += 1
print(f"  Total: {empty_count}")

# Non-answers
print("\n--- Non-answers (?, 'ich weiß nicht', etc.) ---")
non_answer_patterns = [
    r"^\s*\?+\s*$",
    r"(?i)ich\s+wei[ßs]\s+(es\s+)?nicht",
    r"(?i)keine\s+ahnung",
    r"(?i)wei[ßs]\s+ich\s+nicht",
    r"(?i)kein(e)?\s+antwort",
    r"(?i)^-+$",
    r"^\s*\.\s*$",
    r"(?i)^k\.?\s*a\.?\s*$",
]
non_answer_count = 0
for s in data:
    ans = s["answer"].strip()
    for pat in non_answer_patterns:
        if re.match(pat, ans):
            print(f"  [{s['score']:20s}] Answer: '{ans}' | Question: {s['question'][:80]}...")
            non_answer_count += 1
            break
print(f"  Total non-answers found: {non_answer_count}")

# ============================================================
# 5. LANGUAGE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("5. LANGUAGE ANALYSIS")
print("=" * 80)

# Check for non-German content / code-switching
print("\n--- Potential non-German content / code-switching ---")
non_german_patterns = [
    (r"\b(the|and|or|but|is|are|was|were|have|has|this|that|with|from|for)\b", "English words"),
    (r"\b(because|however|therefore|also|which|where|when)\b", "English connectors"),
]
non_german_found = 0
for s in data:
    ans = s["answer"].lower()
    for pat, desc in non_german_patterns:
        matches = re.findall(pat, ans)
        if len(matches) >= 3:  # At least 3 English words to flag
            print(f"  [{s['score']}] {desc}: {matches[:5]} in: '{s['answer'][:100].strip()}...'")
            non_german_found += 1
            break
print(f"  Total samples with potential code-switching: {non_german_found}")

# Special characters and formulas
print("\n--- Answers containing special characters / formulas / numbers ---")
special_patterns = {
    "Contains numbers": r"\d+",
    "Contains mathematical operators (+,-,*,/,=)": r"[+\-*/=]",
    "Contains chemical formulas (e.g., CO2, H2O)": r"[A-Z][a-z]?\d",
    "Contains units (cm, kg, m/s, etc.)": r"\d\s*(cm|mm|km|m|kg|g|mg|l|ml|s|min|h|°C|°|V|A|W|J|N|Pa|Hz)",
    "Contains parentheses with content": r"\([^)]+\)",
    "Contains bullet points or dashes as list": r"(?m)^[\s]*[-•]\s",
}
for desc, pat in special_patterns.items():
    count = sum(1 for s in data if re.search(pat, s["answer"]))
    print(f"  {desc}: {count}/{len(data)} ({100*count/len(data):.1f}%)")

# Common misspellings / student language
print("\n--- Common student language patterns ---")
# Look for common spelling issues
spelling_issues = Counter()
for s in data:
    ans = s["answer"]
    # Lowercase issues at start
    if ans and ans[0].islower() and not ans.startswith(("das", "die", "der", "ein")):
        spelling_issues["Starts with lowercase"] += 1
    # Missing periods
    stripped = ans.strip()
    if stripped and stripped[-1] not in ".!?":
        spelling_issues["No final punctuation"] += 1
    # Colloquial forms
    if re.search(r"(?i)\bweil\b.*\b(ist|hat|wird|kann)\b", ans):
        spelling_issues["'weil' + verb-final (formal)"] += 1
    if re.search(r"(?i)\bweil\b.*,\s", ans):
        spelling_issues["'weil' + comma (maybe V2)"] += 1
    # Abbreviations
    if re.search(r"(?i)\bz\.?\s*b\.?", ans):
        spelling_issues["Uses 'z.B.'"] += 1
    if re.search(r"(?i)\busw\.?", ans):
        spelling_issues["Uses 'usw.'"] += 1
    if re.search(r"(?i)\bbzw\.?", ans):
        spelling_issues["Uses 'bzw.'"] += 1
    # Filler words
    if re.search(r"(?i)\bhalt\b", ans):
        spelling_issues["Uses 'halt' (colloquial)"] += 1
    if re.search(r"(?i)\birgendwie\b", ans):
        spelling_issues["Uses 'irgendwie' (colloquial)"] += 1
    if re.search(r"(?i)\beigentlich\b", ans):
        spelling_issues["Uses 'eigentlich'"] += 1
    # Check for 'ss' vs 'ß' confusion
    if re.search(r"(?i)\bdas+\b", ans) and re.search(r"(?i)\bdass\b", ans):
        spelling_issues["Uses both 'das' and 'dass'"] += 1
    # Comma before 'dass'
    if re.search(r"[^,]\s+dass\b", ans):
        spelling_issues["Missing comma before 'dass'"] += 1

for pat, cnt in spelling_issues.most_common(15):
    print(f"    {cnt:4d}x  {pat}")

# Sentence count distribution
print("\n--- Sentence count per answer ---")
sent_counts = defaultdict(int)
for s in data:
    n_sents = len(re.split(r'[.!?]+', s["answer"].strip()))
    sent_counts[min(n_sents, 10)] += 1  # cap at 10+
for n in sorted(sent_counts.keys()):
    label = f"{n}+" if n == 10 else str(n)
    bar = "#" * (sent_counts[n] // 5)
    print(f"  {label:3s} sentences: {sent_counts[n]:4d} {bar}")

# ============================================================
# 6. MODELING IMPLICATIONS
# ============================================================
print("\n" + "=" * 80)
print("6. MODELING IMPLICATIONS")
print("=" * 80)

# Rubric ambiguity analysis
print("\n--- Rubric ambiguity / vagueness ---")
vague_rubrics = 0
specific_rubrics = 0
for qid, rubric in rubrics_by_q.items():
    total_len = sum(len(v) for v in rubric.values())
    # Check if rubric uses vague quantifiers
    combined = " ".join(rubric.values()).lower()
    if any(w in combined for w in ["umfassend", "teilweise", "nicht"]) and total_len < 300:
        if not any(w in combined for w in ["beispiel", "aspekt", "punkt", "kriterium", "nennt", "erklärt", "beschreibt"]):
            vague_rubrics += 1
        else:
            specific_rubrics += 1
    else:
        specific_rubrics += 1

print(f"  Vague rubrics (rely on umfassend/teilweise without specifics): {vague_rubrics}/{len(rubrics_by_q)}")
print(f"  More specific rubrics (mention criteria, aspects, examples):   {specific_rubrics}/{len(rubrics_by_q)}")

# Label distribution
print("\n--- Label distribution ---")
score_counts = Counter(s["score"] for s in data)
for label, count in score_counts.most_common():
    pct = 100 * count / len(data)
    print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")

# Questions with imbalanced label distribution
print("\n--- Questions with highly imbalanced label distribution ---")
q_label_dist = defaultdict(Counter)
for s in data:
    q_label_dist[s["question_id"]][s["score"]] += 1

for qid, dist in sorted(q_label_dist.items(), key=lambda x: max(x[1].values()) / sum(x[1].values()), reverse=True)[:10]:
    total = sum(dist.values())
    parts = ", ".join(f"{l}={c}" for l, c in dist.most_common())
    dominant_pct = 100 * max(dist.values()) / total
    q_short = questions[qid][:70]
    print(f"  [{question_domains[qid]:10s}] {parts:50s} ({dominant_pct:.0f}% dominant) {q_short}...")

# Answer-rubric alignment challenges
print("\n--- Key modeling challenges ---")
challenges = [
    "1. RUBRIC VAGUENESS: Many rubrics use 'umfassend/teilweise/nicht' without listing specific criteria.",
    "   -> LLM must infer what counts as 'umfassend' from domain knowledge.",
    "2. GERMAN STUDENT LANGUAGE: Answers contain spelling errors, colloquialisms, incomplete sentences.",
    "   -> Model must be robust to noisy German text.",
    "3. DOMAIN BREADTH: Questions span physics, chemistry, biology — model needs broad science knowledge.",
    "4. PARTIAL CREDIT BOUNDARY: The 'Partially correct' category is inherently ambiguous.",
    "   -> Hardest distinction is likely Correct vs Partially correct.",
    "5. SHORT/NON-ANSWERS: Some answers are extremely short or non-answers but still labeled.",
    "   -> Model must handle edge cases gracefully.",
    "6. RUBRIC INTERPRETATION: Rubrics describe expected competencies, not keyword checklists.",
    "   -> Model must do semantic matching, not keyword matching.",
]
for c in challenges:
    print(f"  {c}")

# Check inter-question variance in answer length
print("\n--- Answer length variance across questions ---")
q_lengths = defaultdict(list)
for s in data:
    q_lengths[s["question_id"]].append(len(s["answer"]))

print(f"  {'Domain':12s} {'Mean len':>10s} {'Std':>8s} {'Min':>6s} {'Max':>6s}  Question")
for qid in sorted(q_lengths.keys(), key=lambda x: sum(q_lengths[x])/len(q_lengths[x]), reverse=True)[:10]:
    lens = q_lengths[qid]
    mean_l = sum(lens)/len(lens)
    std_l = (sum((l-mean_l)**2 for l in lens)/len(lens))**0.5
    domain = question_domains[qid]
    q_short = questions[qid][:55]
    print(f"  {domain:12s} {mean_l:10.1f} {std_l:8.1f} {min(lens):6d} {max(lens):6d}  {q_short}...")

# Final summary stats
print("\n--- Summary statistics ---")
print(f"  Total samples:           {len(data)}")
print(f"  Unique questions:        {len(questions)}")
print(f"  Answers per question:    {len(data)/len(questions):.1f} (mean)")
print(f"  Label distribution:      {dict(score_counts)}")
print(f"  Domain distribution:     {dict(domain_q_counts)}")
all_lens = [len(s["answer"]) for s in data]
print(f"  Answer length (chars):   mean={sum(all_lens)/len(all_lens):.1f}, median={sorted(all_lens)[len(all_lens)//2]}")
all_wc = [len(s["answer"].split()) for s in data]
print(f"  Answer length (words):   mean={sum(all_wc)/len(all_wc):.1f}, median={sorted(all_wc)[len(all_wc)//2]}")
