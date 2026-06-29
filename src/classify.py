"""Classify new regulatory items via the Anthropic API.

One API call per item. The model must return strict JSON (no fences, no prose).
We strip fences defensively and log ERROR on any malformed output — we never
crash and never silently swallow a bad response.

Confidence is the signal we care about most: LOW_CONFIDENCE is logged at warn
whenever confidence falls below the threshold, regardless of the verdict.
"""

import json
import os
import re

import anthropic

from observability import (
    CLASSIFICATION, LOW_CONFIDENCE, ERROR,
    INFO, WARN, CRITICAL,
    log_event,
)

COMPONENT = "classify"

# Configurable via environment variables.
MODEL = os.environ.get("REGMONITOR_MODEL", "claude-haiku-4-5-20251001")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("REGMONITOR_CONFIDENCE_THRESHOLD", "0.70"))
MAX_POLICY_CHARS = 1500  # how much of the policy doc to send per call

VALID_POLICY_AREAS = frozenset(
    {"operational_risk", "aml_cft", "cyber", "icaap", "outsourcing", "ai_governance", "other"}
)

_SYSTEM_PROMPT = (
    "You are a regulatory compliance analyst specialising in GCC banking regulation. "
    "You read incoming regulatory items and classify them for a compliance team. "
    "You return only valid JSON — no explanation, no markdown, no code fences."
)

_USER_TEMPLATE = """\
Classify the following regulatory item.

Title: {title}
Date:  {date}
Source: {source}
URL:   {link}

{policy_section}

Return ONLY a JSON object with exactly these four fields:
{{
  "relevance_score": <float 0.0–1.0  how relevant to a licensed financial institution's compliance obligations>,
  "confidence":      <float 0.0–1.0  your confidence that this classification is correct>,
  "policy_area":     <one of: operational_risk | aml_cft | cyber | icaap | outsourcing | ai_governance | other>,
  "summary":         "<two sentences: sentence 1 describes what the item says; sentence 2 states the specific compliance implication for an FI>"
}}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove optional leading/trailing code fences that models sometimes emit."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_response(raw: str, item: dict) -> dict | None:
    """Parse and validate the model's JSON response. Returns None on failure."""
    cleaned = _strip_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Malformed JSON from model for: {item.get('title', '')[:80]}",
            {
                "json_error": str(exc),
                "raw_excerpt": raw[:400],
                "source": item.get("source", ""),
                "link": item.get("link", ""),
            },
        )
        return None

    required = {"relevance_score", "confidence", "policy_area", "summary"}
    missing = required - data.keys()
    if missing:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Model response missing required fields: {sorted(missing)}",
            {
                "missing_fields": sorted(missing),
                "source": item.get("source", ""),
                "link": item.get("link", ""),
            },
        )
        return None

    # Coerce types defensively — the model sometimes returns strings for numbers.
    try:
        data["relevance_score"] = float(data["relevance_score"])
        data["confidence"] = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Non-numeric score in model response: {exc}",
            {"raw_scores": {k: data.get(k) for k in ("relevance_score", "confidence")}},
        )
        return None

    if data.get("policy_area") not in VALID_POLICY_AREAS:
        log_event(
            ERROR, WARN, COMPONENT,
            f"Unknown policy_area '{data.get('policy_area')}' — defaulting to 'other'",
            {"raw_policy_area": data.get("policy_area"), "link": item.get("link", "")},
        )
        data["policy_area"] = "other"

    return data


# ---------------------------------------------------------------------------
# Per-item classification
# ---------------------------------------------------------------------------

def _classify_one(item: dict, policy_text: str | None, client: anthropic.Anthropic) -> dict | None:
    policy_section = ""
    if policy_text:
        excerpt = policy_text[:MAX_POLICY_CHARS]
        policy_section = f"Policy context (excerpt, {len(policy_text)} chars total):\n{excerpt}\n"

    prompt = _USER_TEMPLATE.format(
        title=item.get("title", "(no title)"),
        date=item.get("date", "unknown"),
        source=item.get("source", "unknown"),
        link=item.get("link", ""),
        policy_section=policy_section,
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Anthropic API error ({exc.status_code}) for: {item.get('title', '')[:80]}",
            {"status_code": exc.status_code, "error": str(exc), "link": item.get("link", "")},
        )
        return None
    except anthropic.APIConnectionError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Anthropic API connection error for: {item.get('title', '')[:80]}",
            {"error": str(exc), "link": item.get("link", "")},
        )
        return None

    raw = response.content[0].text
    result = _parse_response(raw, item)
    if result is None:
        return None

    confidence = result["confidence"]
    ctx = {
        "confidence": confidence,
        "relevance_score": result["relevance_score"],
        "policy_area": result["policy_area"],
        "source": item.get("source", ""),
        "link": item.get("link", ""),
    }

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        log_event(
            LOW_CONFIDENCE, WARN, COMPONENT,
            f"Low confidence ({confidence:.2f}) on: {item.get('title', '')[:80]}",
            ctx,
        )
    else:
        log_event(
            CLASSIFICATION, INFO, COMPONENT,
            f"Classified [{result['policy_area']}] conf={confidence:.2f}: {item.get('title', '')[:80]}",
            ctx,
        )

    return {**item, "classification": result}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_items(items: list[dict], policy_text: str | None = None) -> list[dict]:
    """Classify each item with one Anthropic call. Returns annotated item dicts.

    Items that fail parsing or trigger an API error are dropped with ERROR logged;
    they do NOT propagate as exceptions so the rest of the run continues.
    """
    if not items:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            "ANTHROPIC_API_KEY is not set — classification skipped for all items",
            {"item_count": len(items)},
        )
        return []

    client = anthropic.Anthropic(api_key=api_key)
    results: list[dict] = []

    for item in items:
        classified = _classify_one(item, policy_text, client)
        if classified is not None:
            results.append(classified)

    return results
