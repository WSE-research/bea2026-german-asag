"""
OpenRouter API client with key rotation and retry logic.

Provides a reliable interface to the OpenRouter chat completions API
with thread-safe round-robin key selection, JSON mode support, and
exponential backoff with jitter on transient failures.

Extended (2026-03-19): Returns full metadata (token counts, cost,
generation ID, model, latency) via ``call_openrouter_full``.
"""

import json
import logging
import os
import random
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key rotation state (thread-safe round-robin)
# ---------------------------------------------------------------------------
_RR_LOCK = threading.Lock()
_RR_INDEX = 0

OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
DEFAULT_MODEL = os.environ.get("OPENROUTER_DEFAULT_MODEL", "google/gemini-3-flash-preview")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_api_keys() -> list[str]:
    """Load API keys from environment variables.

    Checks ``OPENROUTER_API_KEYS`` (comma-separated) first, then falls back
    to the single ``OPENROUTER_API_KEY`` variable.
    When using a local endpoint (OPENROUTER_URL is not openrouter.ai),
    returns a dummy key since no auth is needed.

    Returns:
        List of API key strings.

    Raises:
        EnvironmentError: If neither variable is set or all values are empty
            (only when using OpenRouter).
    """
    multi = os.environ.get("OPENROUTER_API_KEYS", "")
    if multi.strip():
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys

    single = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if single:
        return [single]

    # Local endpoint: no API key needed
    if "openrouter.ai" not in OPENROUTER_URL:
        return ["local-no-key"]

    raise EnvironmentError(
        "No OpenRouter API key found. Set OPENROUTER_API_KEYS (comma-separated) "
        "or OPENROUTER_API_KEY in your environment."
    )


def get_model() -> str:
    """Return the model identifier from ``OPENROUTER_MODEL`` or the default.

    Default: ``google/gemini-3.0-flash``
    """
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_key(keys: list[str]) -> str:
    """Select the next API key via thread-safe round-robin."""
    global _RR_INDEX
    with _RR_LOCK:
        key = keys[_RR_INDEX % len(keys)]
        _RR_INDEX += 1
    return key


def _parse_json_response(text: str) -> dict:
    """Parse JSON from an LLM response.

    Handles multiple formats:
    1. Pure JSON: ``{"score": "Correct", ...}``
    2. Markdown-fenced: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\```
    3. Text + JSON: reasoning text followed by a JSON object
    """
    cleaned = text.strip()

    # Try 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try 2: strip markdown fences
    if cleaned.startswith("```"):
        try:
            first_newline = cleaned.index("\n")
            inner = cleaned[first_newline + 1:]
            if inner.endswith("```"):
                inner = inner[:-3]
            return json.loads(inner.strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3: find last JSON object in the text (handles reasoning + JSON)
    # Search for the last '{' that starts a valid JSON object
    last_brace = cleaned.rfind("{")
    while last_brace >= 0:
        try:
            candidate = cleaned[last_brace:]
            return json.loads(candidate)
        except json.JSONDecodeError:
            last_brace = cleaned.rfind("{", 0, last_brace)

    # Nothing worked — raise with original text for debugging
    raise json.JSONDecodeError(
        f"No valid JSON found in response ({len(text)} chars): {text[:200]}",
        text, 0
    )


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(body: dict) -> dict:
    """Extract all available metadata from an OpenRouter chat completion response.

    Captures token counts, cost, model info, and generation ID for later
    retrieval of detailed stats via ``fetch_generation_stats``.
    """
    usage = body.get("usage", {})
    meta = {
        "generation_id": body.get("id"),
        "model": body.get("model"),
        "created": body.get("created"),
        "system_fingerprint": body.get("system_fingerprint"),
        # Token counts (inline)
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        # Cost (if provided inline)
        "cost": usage.get("cost"),
    }

    # Detailed token breakdowns (if available)
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if prompt_details:
        meta["cached_tokens"] = prompt_details.get("cached_tokens")
    if completion_details:
        meta["reasoning_tokens"] = completion_details.get("reasoning_tokens")

    return meta


def fetch_generation_stats(generation_id: str, timeout: float = 10.0) -> dict | None:
    """Fetch detailed generation stats from OpenRouter's /api/v1/generation endpoint.

    Returns metadata including total_cost, latency, provider_name, native token
    counts, and more. Returns None on failure (non-critical — stats are a bonus).
    """
    if not generation_id:
        return None

    keys = get_api_keys()
    api_key = keys[0]  # Any key works for reading stats

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                GENERATION_URL,
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()

        data = resp.json().get("data", {})
        return {
            "total_cost": data.get("total_cost"),
            "provider_name": data.get("provider_name"),
            "latency_ms": data.get("latency"),
            "generation_time_ms": data.get("generation_time"),
            "tokens_prompt": data.get("tokens_prompt"),
            "tokens_completion": data.get("tokens_completion"),
            "native_tokens_prompt": data.get("native_tokens_prompt"),
            "native_tokens_completion": data.get("native_tokens_completion"),
            "native_tokens_reasoning": data.get("native_tokens_reasoning"),
            "native_tokens_cached": data.get("native_tokens_cached"),
            "finish_reason": data.get("finish_reason"),
            "cache_discount": data.get("cache_discount"),
        }
    except Exception as exc:
        logger.debug("Failed to fetch generation stats for %s: %s", generation_id, exc)
        return None


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 300,
    temperature: float = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2")),
    json_mode: bool = True,
    timeout: float = 30.0,
) -> dict:
    """Call the OpenRouter chat completions API.

    Selects an API key via round-robin, builds the request payload, sends it,
    and returns the parsed JSON from the assistant's response.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_prompt: The user message / scoring prompt.
        model: Model identifier. Defaults to ``get_model()``.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        json_mode: If True, requests ``{"type": "json_object"}`` response format.
        timeout: HTTP request timeout in seconds.

    Returns:
        Parsed JSON dict from the LLM response content.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    keys = get_api_keys()
    api_key = _next_key(keys)
    resolved_model = model or get_model()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bea2026-german-asag",
        "X-Title": "BEA 2026 German ASAG",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload: dict = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    logger.debug(
        "OpenRouter request: model=%s, tokens=%d, temp=%.2f, json=%s",
        resolved_model,
        max_tokens,
        temperature,
        json_mode,
    )

    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()

    body = response.json()
    content = body["choices"][0]["message"]["content"]

    logger.debug("OpenRouter response length: %d chars", len(content))

    parsed = _parse_json_response(content)

    # Extract inline metadata from the response body
    metadata = _extract_metadata(body)

    # Store metadata on the parsed dict under a reserved key
    parsed["_metadata"] = metadata

    return parsed


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    **kwargs,
) -> dict:
    """Call OpenRouter with exponential backoff and jitter on transient errors.

    Retries on HTTP status codes 429, 500, 502, 503, and 504. On each retry
    the round-robin key index advances automatically (via ``call_openrouter``),
    distributing load across available keys.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_prompt: The user message / scoring prompt.
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the backoff delay in seconds.
        **kwargs: Forwarded to ``call_openrouter``.

    Returns:
        Parsed JSON dict from the LLM response content.

    Raises:
        httpx.HTTPStatusError: If all attempts are exhausted.
        json.JSONDecodeError: If the final attempt returns non-JSON.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return call_openrouter(system_prompt, user_prompt, **kwargs)

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_exc = exc

            if status not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                logger.error(
                    "OpenRouter HTTP %d on attempt %d/%d (non-retryable or final)",
                    status,
                    attempt,
                    max_attempts,
                )
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter

            logger.warning(
                "OpenRouter HTTP %d on attempt %d/%d — retrying in %.2fs",
                status,
                attempt,
                max_attempts,
                wait,
            )
            time.sleep(wait)

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc

            if attempt == max_attempts:
                logger.error(
                    "OpenRouter connection error on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter

            logger.warning(
                "OpenRouter connection error on attempt %d/%d — retrying in %.2fs: %s",
                attempt,
                max_attempts,
                wait,
                exc,
            )
            time.sleep(wait)

    # Should not reach here, but satisfy the type checker
    raise last_exc  # type: ignore[misc]
