"""
Strategy C3: Structured Rubric Evaluation with Few-Shot.

Key innovation: Forces the model to explicitly evaluate each rubric criterion
BEFORE assigning a score. This structured reasoning prevents the "Partially
correct" over-prediction by making the model commit to which criteria are met.

The model outputs:
{
  "criteria_met": ["criterion 1 text...", ...],
  "criteria_missed": ["criterion 2 text...", ...],
  "score": "Correct" | "Partially correct" | "Incorrect",
  "confidence": 0.0-1.0
}
"""

import json
import logging

logger = logging.getLogger(__name__)

MAX_EXAMPLE_ANSWER_LEN = 500


def build_system_prompt() -> str:
    return (
        "Du bist ein automatisches Bewertungssystem für Schülerantworten in MINT-Fächern.\n"
        "\n"
        "AUFGABE\n"
        "Bewerte die Antwort eines Schülers anhand der Bewertungsrubrik und der Beispielbewertungen.\n"
        "\n"
        "BEWERTUNGSPROZESS (in dieser Reihenfolge):\n"
        "1. Lies die Rubrik und identifiziere die KERNKRITERIEN für 'Correct'.\n"
        "2. Prüfe die Schülerantwort Kriterium für Kriterium:\n"
        "   - Welche Kernkriterien werden NACHWEISBAR erfüllt?\n"
        "   - Welche Kernkriterien werden NICHT erfüllt oder fehlen?\n"
        "3. Vergib die Bewertung nach dieser Logik:\n"
        '   - "Correct": ALLE Kernkriterien erfüllt.\n'
        '   - "Partially correct": MINDESTENS EIN Kernkriterium nachweisbar erfüllt, '
        'aber andere fehlen.\n'
        '   - "Incorrect": KEIN Kernkriterium nachweisbar erfüllt.\n'
        "4. Orientiere dich an den Beispielbewertungen für das erwartete Niveau.\n"
        "\n"
        "WICHTIG\n"
        "- Bewerte nur, was geschrieben wurde — keine wohlwollenden Annahmen.\n"
        "- Vage Aussagen oder Alltagswissen OHNE fachlichen Rubrik-Bezug = Incorrect.\n"
        "- Kleine sprachliche Ungenauigkeiten bei korrekt dargestellten Konzepten "
        "= trotzdem Correct.\n"
        "- Leere Antworten, '?', 'Keine Ahnung' = IMMER Incorrect.\n"
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        "{\n"
        '  "criteria_met": ["kurze Beschreibung erfüllter Kriterien..."],\n'
        '  "criteria_missed": ["kurze Beschreibung fehlender Kriterien..."],\n'
        '  "score": "Correct" | "Partially correct" | "Incorrect",\n'
        '  "confidence": 0.0-1.0\n'
        "}"
    )


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
    parts.append("Jetzt bewerte die folgende Antwort (mit criteria_met/criteria_missed Analyse):")
    parts.append(f"Schülerantwort: {answer}")

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

    # Extract reasoning (for analysis, not scoring)
    criteria_met = response.get("criteria_met", [])
    criteria_missed = response.get("criteria_missed", [])

    return {
        "score": score,
        "confidence": confidence,
        "criteria_met": criteria_met,
        "criteria_missed": criteria_missed,
    }
