"""
Prompt construction for Strategy C5d: Rubric Decomposition.

Instead of asking the LLM for a single 3-way score, this prompt decomposes the
evaluation into binary sub-criteria checks derived from the "Correct" rubric text.
The LLM evaluates each criterion individually, then the aggregation logic maps
the count of fulfilled criteria to the final 3-way label.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_EXAMPLE_ANSWER_LEN = 500


def build_system_prompt() -> str:
    return (
        "Du bist ein automatisches Bewertungssystem fuer Schuelerantworten in MINT-Faechern.\n"
        "\n"
        "AUFGABE\n"
        "Bewerte die Antwort eines Schuelers, indem du die einzelnen Kriterien der Correct-Rubrik pruefst.\n"
        "\n"
        "VORGEHENSWEISE\n"
        "1. Lies die Correct-Rubrik und identifiziere die zentralen fachlichen Kriterien (2-4 Punkte).\n"
        "2. Pruefe fuer jedes Kriterium: Wird es in der Schuelerantwort erfuellt? (ja/nein)\n"
        "3. Vergib die Gesamtbewertung:\n"
        '   - ALLE Kriterien erfuellt -> "Correct"\n'
        '   - MINDESTENS EINES aber NICHT ALLE erfuellt -> "Partially correct"\n'
        '   - KEIN Kriterium erfuellt -> "Incorrect"\n'
        "\n"
        "REGELN\n"
        "- Bewerte nur, was geschrieben wurde — keine wohlwollenden Annahmen.\n"
        "- Vage Aussagen ohne konkreten Fachbezug erfuellen KEIN Kriterium.\n"
        "- Leere Antworten oder Nichtwissen -> IMMER Incorrect.\n"
        "\n"
        "FORMAT\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"criteria_fulfilled": <Anzahl erfuellter Kriterien>, "criteria_total": <Gesamtanzahl Kriterien>, '
        '"score": "Correct" | "Partially correct" | "Incorrect", "confidence": 0.0-1.0}'
    )


def _truncate(text: str, max_len: int = MAX_EXAMPLE_ANSWER_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def build_user_prompt(
    question: str,
    answer: str,
    rubric: dict,
    examples: list[dict] | None = None,
) -> str:
    """Build the user prompt for rubric decomposition scoring.

    Focuses on the "Correct" rubric text because it defines the full set of
    criteria. Partially correct = subset fulfilled, Incorrect = none fulfilled.

    Optionally includes calibration examples (from SmartExampleSelector).
    """
    parts = [
        f"Frage: {question}",
        "",
        f"Bewertungskriterien (Correct): {rubric.get('Correct', 'N/A')}",
    ]

    if examples:
        parts.append("")
        parts.append("Kalibrierungsbeispiele:")
        for i, ex in enumerate(examples, 1):
            score_label = ex["score"]
            truncated_answer = _truncate(ex["answer"])
            parts.append("")
            parts.append(f"Beispiel {i} ({score_label}):")
            parts.append(f"Schuelerantwort: {truncated_answer}")
            parts.append(f"Bewertung: {score_label}")

    parts.append("")
    parts.append("Jetzt bewerte die folgende Antwort:")
    parts.append(f"Schuelerantwort: {answer}")

    return "\n".join(parts)


_VALID_SCORES = {"Correct", "Partially correct", "Incorrect"}

# Regex to extract JSON from a response that may contain markdown fencing
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_response(response: dict | str) -> dict:
    """Parse the LLM response, extracting score and criteria counts.

    Accepts either:
    - A dict already parsed from JSON
    - A raw string that may contain JSON (with or without markdown fencing)

    Returns dict with keys: score, confidence, criteria_fulfilled, criteria_total.
    """
    # If we got a string, try to parse it as JSON
    if isinstance(response, str):
        text = response.strip()
        # Try direct parse
        try:
            response = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            m = _JSON_BLOCK_RE.search(text)
            if m:
                try:
                    response = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            if isinstance(response, str):
                # Last resort: find first { ... }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end > start:
                    try:
                        response = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        logger.warning("Could not parse response as JSON: %s", text[:200])
                        return {
                            "score": "Incorrect",
                            "confidence": None,
                            "criteria_fulfilled": None,
                            "criteria_total": None,
                        }

    # Now response should be a dict
    if not isinstance(response, dict):
        logger.warning("Response is not a dict: %s", type(response))
        return {
            "score": "Incorrect",
            "confidence": None,
            "criteria_fulfilled": None,
            "criteria_total": None,
        }

    # Extract and validate score
    raw_score = response.get("score", "")
    score = None
    for valid in _VALID_SCORES:
        if str(raw_score).strip().lower() == valid.lower():
            score = valid
            break

    if score is None:
        logger.warning("Invalid score '%s', defaulting to 'Incorrect'", raw_score)
        score = "Incorrect"

    # Extract confidence
    confidence = None
    raw_conf = response.get("confidence")
    if raw_conf is not None:
        try:
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = None

    # Extract criteria counts
    criteria_fulfilled = None
    raw_cf = response.get("criteria_fulfilled")
    if raw_cf is not None:
        try:
            criteria_fulfilled = int(raw_cf)
        except (ValueError, TypeError):
            pass

    criteria_total = None
    raw_ct = response.get("criteria_total")
    if raw_ct is not None:
        try:
            criteria_total = int(raw_ct)
        except (ValueError, TypeError):
            pass

    # Validate consistency: if criteria counts disagree with score, trust the counts
    if criteria_fulfilled is not None and criteria_total is not None and criteria_total > 0:
        if criteria_fulfilled == criteria_total and score != "Correct":
            logger.debug(
                "Score-criteria mismatch: %d/%d but score='%s', overriding to Correct",
                criteria_fulfilled, criteria_total, score,
            )
            score = "Correct"
        elif 0 < criteria_fulfilled < criteria_total and score != "Partially correct":
            logger.debug(
                "Score-criteria mismatch: %d/%d but score='%s', overriding to Partially correct",
                criteria_fulfilled, criteria_total, score,
            )
            score = "Partially correct"
        elif criteria_fulfilled == 0 and score != "Incorrect":
            logger.debug(
                "Score-criteria mismatch: %d/%d but score='%s', overriding to Incorrect",
                criteria_fulfilled, criteria_total, score,
            )
            score = "Incorrect"

    return {
        "score": score,
        "confidence": confidence,
        "criteria_fulfilled": criteria_fulfilled,
        "criteria_total": criteria_total,
    }
