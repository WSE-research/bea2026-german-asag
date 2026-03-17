"""
Prompt construction and response parsing for Strategy C: Rubric + Few-Shot Examples.

Builds German-language scoring prompts that include both the rubric and
concrete example answers (with their scores) from the same question.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Maximum characters for an example answer before truncation
MAX_EXAMPLE_ANSWER_LEN = 500


def build_system_prompt() -> str:
    """System prompt for rubric + few-shot scoring.

    Instructs the LLM to use both the rubric and the provided examples
    to calibrate its scoring.

    Returns:
        German system prompt string.
    """
    return (
        "Du bist ein automatisches Bewertungssystem für Schülerantworten in MINT-Fächern.\n"
        "\n"
        "AUFGABE\n"
        "Bewerte die Antwort eines Schülers anhand der bereitgestellten Bewertungsrubrik "
        "und der Beispielbewertungen.\n"
        "\n"
        "BEWERTUNGSSTUFEN\n"
        '- "Correct": Vollständig korrekte Antwort gemäß Rubrik.\n'
        '- "Partially correct": Teilweise korrekte Antwort — einige Kriterien erfüllt, andere nicht.\n'
        '- "Incorrect": Keine wesentlichen Kriterien erfüllt, falsch, oder ohne Bezug zur Frage.\n'
        "\n"
        "ANLEITUNG\n"
        "1. Lies zuerst die Rubrik, um die Bewertungskriterien zu verstehen.\n"
        "2. Studiere die Beispielbewertungen, um das erwartete Bewertungsniveau zu kalibrieren.\n"
        "3. Bewerte die neue Schülerantwort konsistent mit den Beispielen und der Rubrik.\n"
        "4. Bewerte nur, was tatsächlich geschrieben wurde — keine wohlwollenden Annahmen.\n"
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}'
    )


def _truncate(text: str, max_len: int = MAX_EXAMPLE_ANSWER_LEN) -> str:
    """Truncate text to max_len characters, adding '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def build_user_prompt(
    question: str,
    answer: str,
    rubric: dict,
    examples: list[dict],
) -> str:
    """Build user prompt with question, rubric, few-shot examples, and the answer to score.

    Args:
        question: The question text.
        answer: The student answer to be scored.
        rubric: Dict mapping score labels to rubric descriptions, e.g.
            ``{"Correct": "...", "Partially correct": "...", "Incorrect": "..."}``.
        examples: List of ``{"answer": str, "score": str}`` dicts.
            Should be sorted by score level: Correct first, then
            Partially correct, then Incorrect. Each example shows a
            scored answer from the same question.

    Returns:
        Formatted German user prompt string.
    """
    parts = [
        f"Frage: {question}",
        "",
        "Bewertungsrubrik:",
        f"- Correct: {rubric.get('Correct', 'N/A')}",
        f"- Partially correct: {rubric.get('Partially correct', 'N/A')}",
        f"- Incorrect: {rubric.get('Incorrect', 'N/A')}",
        "",
        "Beispielbewertungen für diese Frage:",
    ]

    for i, ex in enumerate(examples, 1):
        score_label = ex["score"]
        truncated_answer = _truncate(ex["answer"])
        parts.append("")
        parts.append(f"Beispiel {i} ({score_label}):")
        parts.append(f"Schülerantwort: {truncated_answer}")
        parts.append(f"Bewertung: {score_label}")

    parts.append("")
    parts.append("Jetzt bewerte die folgende Antwort:")
    parts.append(f"Schülerantwort: {answer}")

    return "\n".join(parts)


_VALID_SCORES = {"Correct", "Partially correct", "Incorrect"}


def parse_response(response: dict) -> dict:
    """Parse and validate the LLM JSON response.

    Accepts a parsed JSON dict and normalizes the score label
    and confidence value.

    Args:
        response: Parsed JSON dict from the LLM, expected to contain
            ``"score"`` and optionally ``"confidence"``.

    Returns:
        Dict with ``"score"`` (str) and ``"confidence"`` (float or None).
    """
    raw_score = response.get("score", "")

    # Normalize score — try case-insensitive matching
    score = None
    for valid in _VALID_SCORES:
        if raw_score.strip().lower() == valid.lower():
            score = valid
            break

    if score is None:
        logger.warning("Invalid score '%s' in LLM response, defaulting to 'Incorrect'", raw_score)
        score = "Incorrect"

    # Parse confidence
    confidence = None
    raw_conf = response.get("confidence")
    if raw_conf is not None:
        try:
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logger.warning("Invalid confidence '%s', setting to None", raw_conf)
            confidence = None

    return {"score": score, "confidence": confidence}
