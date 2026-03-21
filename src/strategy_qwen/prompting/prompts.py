"""
Prompt variants for Qwen iteration on BEA26 German ASAG.

Each variant is a function: build_messages(sample, train) -> list[dict]
"""
import json
import random
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ============================================================
# Q1: Minimal rubric-only (German)
# ============================================================
def q1_rubric_only(sample, train):
    """Bare minimum: rubric + answer, German prompt."""
    system = (
        "Du bist ein Bewertungssystem für Schülerantworten. "
        "Bewerte die Antwort anhand der gegebenen Rubrik. "
        "Antworte ausschließlich mit einem JSON-Objekt: "
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q2: Rubric + strict decision rules (German)
# ============================================================
def q2_rubric_rules(sample, train):
    """Add explicit decision rules to reduce 'Partially correct' over-prediction."""
    system = (
        "Du bist ein präzises Bewertungssystem für Schülerantworten. "
        "Bewerte die Antwort STRENG anhand der gegebenen Rubrik.\n\n"
        "ENTSCHEIDUNGSREGELN:\n"
        "1. Vergib 'Correct' wenn die Antwort ALLE Kriterien der Correct-Rubrik erfüllt.\n"
        "2. Vergib 'Incorrect' wenn die Antwort KEINE der Correct- oder Partially-correct-Kriterien erfüllt.\n"
        "3. Vergib 'Partially correct' NUR wenn die Antwort einige aber nicht alle Kriterien erfüllt.\n"
        "4. Im Zweifel zwischen 'Partially correct' und 'Incorrect': wähle 'Incorrect'.\n"
        "5. Im Zweifel zwischen 'Correct' und 'Partially correct': wähle 'Correct'.\n\n"
        "Antworte ausschließlich mit einem JSON-Objekt: "
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    user = (
        f"Frage: {sample['question']}\n\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n\n"
        f"Schülerantwort: {sample['answer']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q3: English XML prompt (structured, clear)
# ============================================================
def q3_english_xml(sample, train):
    """English prompt with XML tags — Qwen handles English well."""
    system = (
        "You are a precise grading system for German student answers. "
        "Grade the student's answer strictly according to the rubric provided.\n\n"
        "DECISION RULES:\n"
        "1. Award 'Correct' only if the answer meets ALL criteria in the Correct rubric.\n"
        "2. Award 'Incorrect' if the answer meets NONE of the Correct or Partially correct criteria.\n"
        "3. Award 'Partially correct' ONLY if the answer meets SOME but not all criteria.\n"
        "4. When uncertain between 'Partially correct' and 'Incorrect', choose 'Incorrect'.\n"
        "5. When uncertain between 'Correct' and 'Partially correct', choose 'Correct'.\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q4: English XML + chain-of-thought
# ============================================================
def q4_english_cot(sample, train):
    """Let the model reason step-by-step before scoring."""
    system = (
        "You are a precise grading system for German student answers. "
        "Grade the student's answer strictly according to the rubric.\n\n"
        "PROCESS:\n"
        "1. Identify the key criteria in the rubric for each score level.\n"
        "2. Check which criteria the student's answer satisfies.\n"
        "3. Determine the appropriate score.\n\n"
        "DECISION RULES:\n"
        "- 'Correct': Answer meets ALL criteria in the Correct rubric.\n"
        "- 'Incorrect': Answer meets NONE of the criteria for Correct or Partially correct.\n"
        "- 'Partially correct': Answer meets SOME but not all criteria.\n"
        "- When uncertain between Partially correct and Incorrect → choose Incorrect.\n"
        "- When uncertain between Correct and Partially correct → choose Correct.\n\n"
        "Respond with your reasoning followed by the score in this exact JSON format on the last line:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Helper: get examples for a question from training data
# ============================================================
_examples_cache = {}

def _get_examples_for_question(question_id, train, n_per_label=1):
    """Get n examples per label for a given question from training data."""
    if question_id not in _examples_cache:
        by_label = defaultdict(list)
        for s in train:
            if s["question_id"] == question_id:
                by_label[s["score"]].append(s)
        _examples_cache[question_id] = by_label

    by_label = _examples_cache[question_id]
    examples = []
    for label in ["Correct", "Partially correct", "Incorrect"]:
        candidates = by_label.get(label, [])
        if candidates:
            selected = random.sample(candidates, min(n_per_label, len(candidates)))
            examples.extend(selected)
    return examples


# ============================================================
# Q5: English XML + few-shot examples (per question)
# ============================================================
def q5_english_fewshot(sample, train):
    """Add 1 example per label from the same question."""
    system = (
        "You are a precise grading system for German student answers. "
        "Grade the student's answer strictly according to the rubric.\n\n"
        "DECISION RULES:\n"
        "1. Award 'Correct' only if the answer meets ALL criteria in the Correct rubric.\n"
        "2. Award 'Incorrect' if the answer meets NONE of the Correct or Partially correct criteria.\n"
        "3. Award 'Partially correct' ONLY if the answer meets SOME but not all criteria.\n"
        "4. When uncertain between 'Partially correct' and 'Incorrect', choose 'Incorrect'.\n"
        "5. When uncertain between 'Correct' and 'Partially correct', choose 'Correct'.\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples_for_question(sample["question_id"], train, n_per_label=1)

    examples_text = ""
    for j, ex in enumerate(examples):
        examples_text += (
            f"<example score=\"{ex['score']}\">\n"
            f"  {ex['answer']}\n"
            f"</example>\n"
        )

    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<scored_examples>\n{examples_text}</scored_examples>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q6: English XML + few-shot + CoT
# ============================================================
def q6_fewshot_cot(sample, train):
    """Combine few-shot examples with chain-of-thought reasoning."""
    system = (
        "You are a precise grading system for German student answers. "
        "Grade the student's answer strictly according to the rubric.\n\n"
        "PROCESS:\n"
        "1. Study the scored examples to calibrate your grading.\n"
        "2. Identify the key criteria the student's answer addresses.\n"
        "3. Compare against the rubric criteria.\n"
        "4. Assign the score.\n\n"
        "DECISION RULES:\n"
        "- 'Correct': Answer meets ALL criteria in the Correct rubric.\n"
        "- 'Incorrect': Answer meets NONE of the criteria.\n"
        "- 'Partially correct': Answer meets SOME but not all criteria.\n"
        "- When uncertain between Partially correct and Incorrect → choose Incorrect.\n\n"
        "First explain your reasoning briefly, then output the score as JSON on the last line:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples_for_question(sample["question_id"], train, n_per_label=1)

    examples_text = ""
    for ex in examples:
        examples_text += (
            f"<example score=\"{ex['score']}\">\n"
            f"  {ex['answer']}\n"
            f"</example>\n"
        )

    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<scored_examples>\n{examples_text}</scored_examples>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q7: English XML + 2 examples per label + CoT
# ============================================================
def q7_more_examples_cot(sample, train):
    """More examples (2 per label = 6 total) + CoT."""
    system = (
        "You are a precise grading system for German student answers. "
        "You will be given a question, a rubric, and scored examples.\n\n"
        "PROCESS:\n"
        "1. Study the scored examples carefully to understand the grading standard.\n"
        "2. Compare the student's answer to the rubric criteria.\n"
        "3. Determine which score level best matches.\n\n"
        "RULES:\n"
        "- 'Correct': ALL criteria met.\n"
        "- 'Incorrect': NONE of the positive criteria met.\n"
        "- 'Partially correct': SOME criteria met.\n"
        "- Bias toward extremes: prefer Correct or Incorrect over Partially correct when borderline.\n\n"
        "Reason briefly, then output JSON on the last line:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_examples_for_question(sample["question_id"], train, n_per_label=2)

    examples_text = ""
    for ex in examples:
        examples_text += (
            f"<example score=\"{ex['score']}\">\n"
            f"  {ex['answer']}\n"
            f"</example>\n"
        )

    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<scored_examples>\n{examples_text}</scored_examples>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Q8: Smart example selection (TF-IDF similarity) + CoT
# ============================================================
_tfidf_cache = {}

def _get_smart_examples(sample, train, n_similar=2, n_boundary=1):
    """Select examples using TF-IDF similarity + boundary examples."""
    qid = sample["question_id"]
    cache_key = qid

    if cache_key not in _tfidf_cache:
        q_samples = [s for s in train if s["question_id"] == qid]
        if not q_samples:
            return []
        answers = [s["answer"] for s in q_samples]
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(answers)
        _tfidf_cache[cache_key] = (q_samples, vectorizer, tfidf_matrix)

    q_samples, vectorizer, tfidf_matrix = _tfidf_cache[cache_key]

    # Find most similar training answers
    query_vec = vectorizer.transform([sample["answer"]])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top-N similar per label
    examples = []
    seen_ids = set()
    for label in LABELS:
        label_indices = [i for i, s in enumerate(q_samples) if s["score"] == label]
        if not label_indices:
            continue
        label_sims = [(i, similarities[i]) for i in label_indices]
        label_sims.sort(key=lambda x: x[1], reverse=True)
        for idx, sim in label_sims[:n_similar]:
            if q_samples[idx]["id"] not in seen_ids:
                examples.append(q_samples[idx])
                seen_ids.add(q_samples[idx]["id"])

    # Add boundary examples (answers near decision boundaries)
    for label in LABELS:
        label_indices = [i for i, s in enumerate(q_samples) if s["score"] == label]
        if not label_indices:
            continue
        # Find answers with medium similarity (boundary region)
        label_sims = [(i, similarities[i]) for i in label_indices]
        label_sims.sort(key=lambda x: x[1])
        mid = len(label_sims) // 2
        for idx, sim in label_sims[max(0, mid - n_boundary):mid + n_boundary]:
            if q_samples[idx]["id"] not in seen_ids:
                examples.append(q_samples[idx])
                seen_ids.add(q_samples[idx]["id"])
                if len(examples) >= n_similar * 3 + n_boundary * 3:
                    break

    return examples


def q8_smart_examples(sample, train):
    """TF-IDF based smart example selection + CoT."""
    system = (
        "You are a precise grading system for German student answers. "
        "You will be given a question, a rubric, and carefully selected scored examples.\n\n"
        "PROCESS:\n"
        "1. Study the scored examples — they are selected to be similar to the answer you must grade.\n"
        "2. Compare the student's answer to both the rubric and the examples.\n"
        "3. Determine the correct score.\n\n"
        "RULES:\n"
        "- 'Correct': ALL criteria met.\n"
        "- 'Incorrect': NONE of the positive criteria met.\n"
        "- 'Partially correct': SOME criteria met.\n"
        "- Prefer clear decisions (Correct/Incorrect) over ambiguous ones (Partially correct).\n\n"
        "Reason briefly, then output JSON on the last line:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect"}'
    )
    rubric = sample["rubric"]
    examples = _get_smart_examples(sample, train, n_similar=2, n_boundary=1)

    examples_text = ""
    for ex in examples:
        examples_text += (
            f"<example score=\"{ex['score']}\">\n"
            f"  {ex['answer']}\n"
            f"</example>\n"
        )

    user = (
        f"<question>{sample['question']}</question>\n\n"
        f"<rubric>\n"
        f"  <correct>{rubric['Correct']}</correct>\n"
        f"  <partially_correct>{rubric['Partially correct']}</partially_correct>\n"
        f"  <incorrect>{rubric['Incorrect']}</incorrect>\n"
        f"</rubric>\n\n"
        f"<scored_examples>\n{examples_text}</scored_examples>\n\n"
        f"<student_answer>{sample['answer']}</student_answer>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Registry
# ============================================================
VARIANTS = {
    "q1_rubric_only": q1_rubric_only,
    "q2_rubric_rules": q2_rubric_rules,
    "q3_english_xml": q3_english_xml,
    "q4_english_cot": q4_english_cot,
    "q5_english_fewshot": q5_english_fewshot,
    "q6_fewshot_cot": q6_fewshot_cot,
    "q7_more_examples_cot": q7_more_examples_cot,
    "q8_smart_examples": q8_smart_examples,
}
