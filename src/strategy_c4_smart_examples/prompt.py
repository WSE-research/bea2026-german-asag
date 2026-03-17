"""
Prompt construction for Strategy C4: Smart Example Selection.

Uses the same prompt as Strategy C2 (Tuned Few-Shot). The improvement in C4
is in example SELECTION (boundary + similarity), not the prompt text itself.

Copied verbatim from Strategy C2.
"""

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
        "BEWERTUNGSSTUFEN\n"
        '- "Correct": Die Antwort trifft ALLE zentralen Punkte der Correct-Rubrik.\n'
        '- "Partially correct": Die Antwort trifft MINDESTENS EINEN zentralen fachlichen Punkt '
        'der Rubrik korrekt, verfehlt aber andere wesentliche Kriterien.\n'
        '- "Incorrect": Die Antwort trifft KEINEN der fachlichen Kernpunkte der Rubrik.\n'
        "\n"
        "ENTSCHEIDUNGSREGELN\n"
        "1. Entscheide anhand der Rubrik UND konsistent mit den Beispielbewertungen.\n"
        "2. Bewerte nur, was geschrieben wurde — keine wohlwollenden Annahmen.\n"
        "3. GRENZE Incorrect vs. Partially correct:\n"
        "   - Vage Aussagen, Alltagswissen oder Umformulierungen der Frage OHNE fachlichen "
        "Inhalt aus der Rubrik sind IMMER Incorrect.\n"
        "   - Partially correct erfordert NACHWEISBAR mindestens einen konkreten fachlichen "
        "Punkt, der in der Rubrik als Kriterium genannt wird.\n"
        "4. GRENZE Partially correct vs. Correct:\n"
        "   - Correct erfordert, dass die Antwort die Rubrik-Kriterien VOLLSTÄNDIG abdeckt.\n"
        "   - Wenn die wesentlichen Konzepte korrekt und vollständig dargelegt sind, kleine "
        "sprachliche Ungenauigkeiten aber vorliegen → trotzdem Correct.\n"
        "   - Im Zweifel: orientiere dich an den Beispielbewertungen für diese Frage.\n"
        "5. Leere Antworten, Nichtwissen ('?', 'Keine Ahnung'), einzelne Wörter ohne "
        "Erklärung → IMMER Incorrect.\n"
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}'
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
    parts.append("Jetzt bewerte die folgende Antwort:")
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

    return {"score": score, "confidence": confidence}
