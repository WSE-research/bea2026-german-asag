"""
Prompt construction for Strategy C6b: Claude-optimized prompt.

Key changes from C4/C6 prompt (based on Anthropic prompting best practices):
1. XML tags for structure (Claude parses these unambiguously)
2. Examples wrapped in <examples>/<example> tags
3. Direct, positive instructions ("Output only the JSON" not "Don't add text")
4. Role setting in system prompt
5. Clear separation of rubric, examples, and task
6. English system prompt (Claude's native language) + German content
"""

import logging

logger = logging.getLogger(__name__)

MAX_EXAMPLE_ANSWER_LEN = 500


def build_system_prompt() -> str:
    """Build a Claude-optimized system prompt for German ASAG scoring.

    Uses XML tags and follows Anthropic's prompting best practices for
    classification tasks.
    """
    return """You are a rubric-based scoring system for German STEM student answers. Your task is to classify each student answer into exactly one of three categories.

<scoring_levels>
- "Correct": The answer addresses ALL key criteria from the Correct rubric.
- "Partially correct": The answer addresses AT LEAST ONE specific criterion from the rubric correctly, but misses other essential criteria.
- "Incorrect": The answer addresses NONE of the rubric's key criteria.
</scoring_levels>

<decision_rules>
1. Score based on the rubric AND consistently with the provided example scores.
2. Only evaluate what is written — no charitable assumptions.
3. Incorrect vs Partially correct boundary: Vague statements, everyday knowledge, or rephrasing the question WITHOUT specific content from the rubric criteria are ALWAYS Incorrect. Partially correct requires DEMONSTRABLY at least one concrete criterion mentioned in the rubric.
4. Partially correct vs Correct boundary: Correct requires the answer to FULLY cover the rubric criteria. If the key concepts are correct and complete but minor phrasing imprecision exists, still score Correct. When in doubt, follow the example scores for this question.
5. Empty answers, expressions of ignorance ('?', 'Keine Ahnung'), single words without explanation are ALWAYS Incorrect.
</decision_rules>

<output_format>
Respond with ONLY a JSON object. Do not include any analysis, reasoning, or explanation.
{"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}
</output_format>"""


def _truncate(text: str, max_len: int = MAX_EXAMPLE_ANSWER_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def build_user_prompt(
    question: str,
    answer: str,
    rubric: dict,
    examples: list[dict],
) -> str:
    """Build a Claude-optimized user prompt with XML-tagged structure."""
    parts = [
        f"<question>{question}</question>",
        "",
        "<rubric>",
        f"Correct: {rubric.get('Correct', 'N/A')}",
        f"Partially correct: {rubric.get('Partially correct', 'N/A')}",
        f"Incorrect: {rubric.get('Incorrect', 'N/A')}",
        "</rubric>",
        "",
        "<examples>",
    ]

    for i, ex in enumerate(examples, 1):
        score_label = ex["score"]
        truncated_answer = _truncate(ex["answer"])
        parts.append(f'<example score="{score_label}">')
        parts.append(f"Schülerantwort: {truncated_answer}")
        parts.append(f"Bewertung: {score_label}")
        parts.append("</example>")

    parts.append("</examples>")
    parts.append("")
    parts.append(f"<student_answer>{answer}</student_answer>")

    return "\n".join(parts)


_VALID_SCORES = {"Correct", "Partially correct", "Incorrect"}


def parse_response(response: dict) -> dict:
    raw_score = response.get("score", "")

    score = None
    for valid in _VALID_SCORES:
        if raw_score.strip().lower() == valid.lower():
            score = valid
            break

    if score is None:
        logger.warning("Invalid score '%s', defaulting to 'Incorrect'", raw_score)
        score = "Incorrect"

    confidence = None
    raw_conf = response.get("confidence")
    if raw_conf is not None:
        try:
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = None

    return {"score": score, "confidence": confidence}
