"""
Prompt construction and response parsing for Strategy A (Rubric-Only).

Strategy A is the simplest baseline: provide the ALICE rubric verbatim
and ask the LLM to classify the student answer. No additional rules,
no examples. Let the rubric speak for itself.
"""

import logging

logger = logging.getLogger(__name__)

VALID_SCORES = {"Correct", "Partially correct", "Incorrect"}

# Canonical lookup for case-insensitive matching
_SCORE_CANONICAL = {s.lower(): s for s in VALID_SCORES}


def build_system_prompt() -> str:
    """
    Minimal system prompt for rubric-only scoring.
    The rubric is provided in the user prompt — this just sets the task framing.
    """
    return (
        "Du bist ein automatisches Bewertungssystem für Schülerantworten.\n"
        "\n"
        "AUFGABE\n"
        "Bewerte die Antwort eines Schülers anhand der bereitgestellten Bewertungsrubrik.\n"
        "\n"
        "Die Rubrik definiert drei Bewertungsstufen:\n"
        '- "Correct": Die Antwort erfüllt alle Kriterien der Rubrik für eine korrekte Antwort.\n'
        '- "Partially correct": Die Antwort erfüllt einige, aber nicht alle Kriterien.\n'
        '- "Incorrect": Die Antwort erfüllt die Kriterien nicht oder ist falsch.\n'
        "\n"
        "REGELN\n"
        "1. Bewerte ausschließlich anhand der Rubrik — nicht anhand deines eigenen Wissens.\n"
        "2. Bewerte nur, was tatsächlich geschrieben wurde. Keine wohlwollenden Annahmen.\n"
        '3. Leere Antworten, Nichtwissen ("Keine Ahnung", "?") oder themenfremde Antworten sind "Incorrect".\n'
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}'
    )


def build_user_prompt(question: str, answer: str, rubric: dict) -> str:
    """
    Build the user prompt with question, answer, and ALICE rubric.

    Args:
        question: The question text shown to the student.
        answer: The student's answer text.
        rubric: Dict with keys "Correct", "Partially correct", "Incorrect",
                each containing the rubric text for that score level.

    Returns:
        Formatted user prompt string.
    """
    return (
        f"Frage: {question}\n"
        f"\n"
        f"Bewertungsrubrik:\n"
        f"- Correct: {rubric['Correct']}\n"
        f"- Partially correct: {rubric['Partially correct']}\n"
        f"- Incorrect: {rubric['Incorrect']}\n"
        f"\n"
        f"Schülerantwort: {answer}"
    )


def parse_response(response: dict) -> dict:
    """
    Parse LLM JSON response into standardized format.

    Args:
        response: Parsed JSON dict from the LLM, expected to contain
                  "score" and optionally "confidence".

    Returns:
        {"score": str, "confidence": float | None}
        Validates that score is one of the three valid labels.
        Falls back to "Incorrect" if parsing fails.
    """
    # Extract score
    raw_score = response.get("score", "")

    if isinstance(raw_score, str):
        normalized = raw_score.strip()
        # Try case-insensitive match
        canonical = _SCORE_CANONICAL.get(normalized.lower())
        if canonical is not None:
            score = canonical
        else:
            logger.warning("Unknown score value '%s', falling back to 'Incorrect'", raw_score)
            score = "Incorrect"
    else:
        logger.warning("Score is not a string: %s, falling back to 'Incorrect'", type(raw_score))
        score = "Incorrect"

    # Extract confidence
    raw_confidence = response.get("confidence")
    confidence = None

    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logger.warning("Could not parse confidence value: %s", raw_confidence)
            confidence = None

    return {"score": score, "confidence": confidence}
