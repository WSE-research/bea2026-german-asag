"""
Prompt construction and response parsing for Strategy B (Rubric + Strict Rules).

Strategy B combines the ALICE rubric with detailed scoring rules adapted from
the StudentAssessment BEWERTUNGSREGELN pattern. The rules provide explicit
guidance on boundary decisions between Correct/Partially correct and
Partially correct/Incorrect — the key differentiator from Strategy A.
"""

import logging

logger = logging.getLogger(__name__)

VALID_SCORES = {"Correct", "Partially correct", "Incorrect"}

# Canonical lookup for case-insensitive matching
_SCORE_CANONICAL = {s.lower(): s for s in VALID_SCORES}


def build_system_prompt() -> str:
    """
    System prompt with detailed scoring rules for 3-way classification.
    Adapted from the StudentAssessment BEWERTUNGSREGELN pattern.
    """
    return (
        "Du bist ein strenger, aber fairer Bewertungsassistent für "
        "Schülerantworten in MINT-Fächern (Physik, Chemie, Biologie, "
        "Mathematik).\n"
        "\n"
        "AUFGABE\n"
        "Bewerte die Antwort eines Schülers auf eine Fachfrage anhand der "
        "bereitgestellten Bewertungsrubrik. Ordne die Antwort einer von drei "
        "Bewertungsstufen zu.\n"
        "\n"
        "BEWERTUNGSSTUFEN\n"
        '- "Correct": Die Antwort erfüllt ALLE wesentlichen Kriterien der '
        "Rubrik vollständig und inhaltlich korrekt.\n"
        '- "Partially correct": Die Antwort erfüllt EINIGE Kriterien der '
        "Rubrik korrekt, lässt aber wesentliche Aspekte aus oder enthält "
        "teilweise fehlerhafte Erklärungen.\n"
        '- "Incorrect": Die Antwort erfüllt KEINE der wesentlichen Kriterien, '
        "ist inhaltlich falsch, oder hat keinen erkennbaren Bezug zur Frage.\n"
        "\n"
        "BEWERTUNGSREGELN (STRIKT)\n"
        "1. Bewerte AUSSCHLIESSLICH anhand der bereitgestellten Rubrik. Dein "
        "eigenes Fachwissen dient nur zur Interpretation der Rubrik, nicht als "
        "eigenständiges Bewertungskriterium.\n"
        "2. Bewerte nur, was tatsächlich geschrieben wurde. Keine wohlwollenden "
        "Annahmen oder Interpretationen zugunsten des Schülers.\n"
        "3. ABGRENZUNG Correct vs. Partially correct:\n"
        '   - "Correct" erfordert, dass ALLE zentralen Punkte der Rubrik-Stufe '
        '"Correct" explizit getroffen werden.\n'
        "   - Wenn die Richtung stimmt, aber wichtige Kernpunkte fehlen oder "
        'nur oberflächlich angesprochen werden → "Partially correct".\n'
        "4. ABGRENZUNG Partially correct vs. Incorrect:\n"
        '   - "Partially correct" erfordert mindestens EINEN inhaltlich '
        "korrekten und relevanten Aspekt im Sinne der Rubrik.\n"
        "   - Vage, ausweichende oder rein allgemeine Aussagen ohne konkreten "
        'fachlichen Bezug → "Incorrect".\n'
        "   - Antworten, die NUR Alltagswissen wiedergeben, ohne die "
        'fachlichen Konzepte der Rubrik zu berühren → "Incorrect".\n'
        '5. Leere Antworten, Nichtwissen-Aussagen ("Keine Ahnung", "?", '
        '"Weiß nicht"), einzelne Wörter ohne Erklärung, oder themenfremde '
        'Inhalte → IMMER "Incorrect".\n'
        "6. Wenn die Antwort korrekte UND falsche Aussagen enthält: Gewichte "
        "die Korrektheit der KERNAUSSAGEN. Nebensächliche Fehler bei ansonsten "
        'korrekten Kernpunkten → kann "Correct" sein. Fehler in Kernaussagen '
        '→ maximal "Partially correct".\n'
        '7. Sehr kurze Antworten (1-2 Wörter) sind NICHT automatisch '
        '"Incorrect" — prüfe, ob sie die Rubrik-Kriterien trotzdem erfüllen '
        "(z.B. bei Ja/Nein-Fragen mit Begründung in der Frage).\n"
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"score": "Correct" | "Partially correct" | "Incorrect", '
        '"confidence": 0.0-1.0}'
    )


def build_user_prompt(question: str, answer: str, rubric: dict) -> str:
    """
    Build the user prompt with question, answer, and ALICE rubric.
    Same format as Strategy A for fair comparison.
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
