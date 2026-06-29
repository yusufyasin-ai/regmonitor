"""
Baseline gap analysis — Phase 2 control condition.

ONE Anthropic call per item: given the regulation item (JSON from Phase 1) and
the full policy document, the model simultaneously extracts every compliance
obligation from the regulation and assesses whether each is already addressed
by the policy.

This single-call approach is the experimental control. Phase 3 will introduce
a multi-step / multi-call version. Do NOT modify run_baseline() when adding
Phase 3 code — add Phase 3 as a separate function alongside it so both remain
independently runnable and comparable.

Standalone usage (control condition, no pipeline required):
    python analyze.py --item <path>.json --policy <path>.md
    python analyze.py --item <path>.json --policy <path>.md --out output/

Output written to output/analysis_baseline_<slug>_<timestamp>.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from observability import (
    BASELINE_DIVERGENCE, LOW_CONFIDENCE, PARSE_OK, HANDOFF, ERROR,
    VALIDATOR_DISAGREEMENT,
    INFO, WARN, CRITICAL,
    log_event,
)

COMPONENT = "analyze"

ANALYZE_MODEL = os.environ.get("REGMONITOR_ANALYZE_MODEL", "claude-sonnet-4-6")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("REGMONITOR_CONFIDENCE_THRESHOLD", "0.70"))
# Fire BASELINE_DIVERGENCE when this fraction of obligations are unaddressed.
DIVERGENCE_THRESHOLD = float(os.environ.get("REGMONITOR_DIVERGENCE_THRESHOLD", "0.40"))

# --- Phase 3 multi-agent configuration ------------------------------------
# The first-pass roles (READER, MAPPER, GAP) share one model. The VALIDATOR is
# deliberately run on a *different* model family to reduce correlated error —
# see _validator_model().
MULTI_FIRSTPASS_MODEL = os.environ.get("REGMONITOR_MULTI_MODEL", "claude-sonnet-4-6")


def _validator_model(firstpass_model: str) -> str:
    """Pick a validator model on a different family than the first-pass roles.

    Engineered independence (requirement a): if the first pass runs on
    claude-sonnet-4-6 the validator runs on claude-opus-4-6, and vice versa.
    An explicit REGMONITOR_VALIDATOR_MODEL overrides the swap.
    """
    override = os.environ.get("REGMONITOR_VALIDATOR_MODEL")
    if override:
        return override
    fp = firstpass_model.lower()
    if "sonnet" in fp:
        return "claude-opus-4-6"
    if "opus" in fp:
        return "claude-sonnet-4-6"
    # First pass is on neither family (e.g. haiku) — fall back to opus so the
    # validator is still a distinct, stronger model.
    return "claude-opus-4-6"

_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _ROOT / "output"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a regulatory compliance gap analyst specialising in GCC banking regulation. "
    "You receive a new regulatory item and an internal policy document. "
    "You extract obligations and assess coverage in a single pass. "
    "You return only valid JSON — no explanation, no markdown, no code fences."
)

_USER = """\
You will perform a two-step gap analysis in a single response.

STEP 1 — Obligation extraction
Identify every distinct compliance obligation this regulation imposes on a licensed financial institution (FI). Focus on concrete requirements ("must", "shall", "is required to", "obligated to") rather than general intent statements. Use the title and summary to infer standard obligations for this regulation type; note any uncertainty in the confidence field.

STEP 2 — Policy coverage assessment
For each obligation, determine whether the policy document already addresses it. An obligation is "addressed" only if the policy contains substantive, relevant coverage — not merely a passing mention.

Regulatory item:
  Title:   {title}
  Date:    {date}
  Source:  {source}
  URL:     {link}
  Summary: {summary}
  Policy area: {policy_area}

Policy document ({policy_chars} characters):
---
{policy_text}
---

Return ONLY a JSON object with this exact structure:
{{
  "obligations": [
    {{
      "id": "OBL-1",
      "description": "<concrete obligation: what the FI must do>",
      "coverage": "<one of: covered | partial | not_covered>",
      "confidence": <float 0.0–1.0 — your confidence that the coverage judgment is correct>,
      "evidence": "<if covered or partial: verbatim phrase or section reference from the policy; if not_covered: empty string>",
      "gap": "<if not_covered or partial: what is specifically absent or incomplete in the policy; if covered: empty string>"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_response(raw: str, item_link: str) -> list[dict] | None:
    """Parse and validate the model's JSON. Returns the obligations list or None."""
    cleaned = _strip_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            "Malformed JSON in baseline analysis response",
            {"json_error": str(exc), "raw_excerpt": raw[:400], "item_link": item_link},
        )
        return None

    if not isinstance(data, dict) or "obligations" not in data:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            "Response missing top-level 'obligations' key",
            {"keys_found": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
             "item_link": item_link},
        )
        return None

    obligations = data["obligations"]
    if not isinstance(obligations, list):
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"'obligations' must be a list, got {type(obligations).__name__}",
            {"item_link": item_link},
        )
        return None

    VALID_COVERAGE = frozenset({"covered", "partial", "not_covered"})

    validated: list[dict] = []
    for i, obl in enumerate(obligations):
        required = {"id", "description", "coverage", "confidence"}
        missing = required - obl.keys()
        if missing:
            log_event(
                ERROR, WARN, COMPONENT,
                f"Obligation at index {i} missing fields: {sorted(missing)} — skipped",
                {"index": i, "missing": sorted(missing), "item_link": item_link},
            )
            continue

        try:
            obl["confidence"] = float(obl["confidence"])
        except (TypeError, ValueError) as exc:
            log_event(
                ERROR, WARN, COMPONENT,
                f"Non-coercible confidence in obligation {obl.get('id', i)}: {exc}",
                {"obligation_id": obl.get("id"), "item_link": item_link},
            )
            continue

        if obl.get("coverage") not in VALID_COVERAGE:
            log_event(
                ERROR, WARN, COMPONENT,
                f"Invalid coverage value '{obl.get('coverage')}' in obligation {obl.get('id', i)} — skipped",
                {"obligation_id": obl.get("id"), "item_link": item_link},
            )
            continue

        obl.setdefault("evidence", "")
        obl.setdefault("gap", "")
        validated.append(obl)

    return validated


# ---------------------------------------------------------------------------
# Core single-call baseline (the control condition — do not modify for Phase 3)
# ---------------------------------------------------------------------------

def run_baseline(
    item: dict,
    policy_text: str,
    client: anthropic.Anthropic,
) -> dict | None:
    """Single Anthropic call: extract obligations and assess policy coverage.

    This is the Phase 2 control condition. It must remain intact after Phase 3
    is added. Do not change its signature, prompt, or output schema.

    Returns a result dict or None if the API call or parse fails.
    """
    cl = item.get("classification") or {}
    summary = cl.get("summary") or "(no summary — run classify first)"
    policy_area = cl.get("policy_area", "unknown")
    item_link = item.get("link", "")

    prompt = _USER.format(
        title=item.get("title", "(no title)"),
        date=item.get("date", "unknown"),
        source=item.get("source", "unknown"),
        link=item_link,
        summary=summary,
        policy_area=policy_area,
        policy_chars=len(policy_text),
        policy_text=policy_text,
    )

    try:
        response = client.messages.create(
            model=ANALYZE_MODEL,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"Anthropic API error ({exc.status_code}) during baseline analysis",
            {"status_code": exc.status_code, "error": str(exc), "item_link": item_link},
        )
        return None
    except anthropic.APIConnectionError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            "Anthropic API connection error during baseline analysis",
            {"error": str(exc), "item_link": item_link},
        )
        return None

    raw = response.content[0].text
    obligations = _parse_response(raw, item_link)
    if obligations is None:
        return None

    log_event(
        PARSE_OK, INFO, COMPONENT,
        f"Baseline parsed {len(obligations)} obligation(s) for: {item.get('title', '')[:80]}",
        {"obligation_count": len(obligations), "item_link": item_link},
    )

    # Per-obligation logging
    for obl in obligations:
        conf = obl["confidence"]
        if conf < LOW_CONFIDENCE_THRESHOLD:
            log_event(
                LOW_CONFIDENCE, WARN, COMPONENT,
                f"Low confidence ({conf:.2f}) on obligation {obl['id']}: {obl['description'][:80]}",
                {
                    "obligation_id": obl["id"],
                    "coverage": obl["coverage"],
                    "confidence": conf,
                    "item_link": item_link,
                },
            )

    # Divergence check: weighted three-state gap ratio
    # not_covered=1.0, partial=0.5, covered=0.0
    n_not_covered = sum(1 for o in obligations if o["coverage"] == "not_covered")
    n_partial     = sum(1 for o in obligations if o["coverage"] == "partial")
    n_covered     = sum(1 for o in obligations if o["coverage"] == "covered")
    gap_ratio = (n_not_covered + 0.5 * n_partial) / len(obligations) if obligations else 0.0

    if gap_ratio >= DIVERGENCE_THRESHOLD:
        gap_ids = [o["id"] for o in obligations if o["coverage"] in ("not_covered", "partial")]
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            (
                f"Policy gap detected: {n_not_covered} not_covered, {n_partial} partial "
                f"(weighted ratio={gap_ratio:.0%}) in: {item.get('title', '')[:80]}"
            ),
            {
                "not_covered": n_not_covered,
                "partial": n_partial,
                "covered": n_covered,
                "total": len(obligations),
                "gap_ratio": round(gap_ratio, 3),
                "item_link": item_link,
                "gap_ids": gap_ids,
            },
        )

    result = {
        "item": {
            "title": item.get("title"),
            "date": item.get("date"),
            "link": item_link,
            "source": item.get("source"),
            "policy_area": policy_area,
        },
        "baseline_model": ANALYZE_MODEL,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "obligations": obligations,
        "summary": {
            "total": len(obligations),
            "covered": n_covered,
            "partial": n_partial,
            "not_covered": n_not_covered,
            "gap_ratio": round(gap_ratio, 3),
            "low_confidence_count": sum(
                1 for o in obligations if o["confidence"] < LOW_CONFIDENCE_THRESHOLD
            ),
        },
    }

    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Baseline complete — {len(obligations)} obligations, ratio={gap_ratio:.0%} "
        f"({n_covered} covered, {n_partial} partial, {n_not_covered} not_covered)",
        {"summary": result["summary"], "item_link": item_link},
    )
    return result


# ===========================================================================
# Phase 3 — multi-agent chain (READER → MAPPER → GAP → VALIDATOR)
#
# This is the experimental condition that runs ALONGSIDE the single-call
# baseline above. run_baseline() is untouched. Each role is a separate
# Anthropic call; every role boundary emits a HANDOFF event recording which
# role produced what and what the next role received. The VALIDATOR runs on a
# different model and re-reads the source texts directly.
# ===========================================================================

_VALID_MULTI_COVERAGE = frozenset({"covered", "partially_covered", "not_covered"})


def _role_call(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    role: str,
    item_link: str,
    max_tokens: int = 3072,
) -> str | None:
    """One Anthropic call for a single role. Returns raw text or None on error."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIStatusError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{role}: Anthropic API error ({exc.status_code}) on model {model}",
            {"role": role, "model": model, "status_code": exc.status_code,
             "error": str(exc), "item_link": item_link},
        )
        return None
    except anthropic.APIConnectionError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{role}: Anthropic API connection error on model {model}",
            {"role": role, "model": model, "error": str(exc), "item_link": item_link},
        )
        return None
    return response.content[0].text


def _parse_json_object(
    raw: str, role: str, item_link: str, required_key: str
) -> dict | None:
    """Parse a role's response into a dict and confirm the expected top-level key."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{role}: malformed JSON in response",
            {"role": role, "json_error": str(exc),
             "raw_excerpt": raw[:400], "item_link": item_link},
        )
        return None
    if not isinstance(data, dict) or required_key not in data:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{role}: response missing top-level '{required_key}' key",
            {"role": role,
             "keys_found": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
             "item_link": item_link},
        )
        return None
    if not isinstance(data[required_key], list):
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{role}: '{required_key}' must be a list, got {type(data[required_key]).__name__}",
            {"role": role, "item_link": item_link},
        )
        return None
    return data


def _log_handoff(
    from_role: str, to_role: str, produced: str, received: str,
    item_link: str, extra: dict | None = None,
) -> None:
    """Record one role boundary: what `from_role` produced and what `to_role` got."""
    ctx = {
        "from": from_role,
        "to": to_role,
        "produced": produced,
        "received": received,
        "item_link": item_link,
    }
    if extra:
        ctx.update(extra)
    log_event(
        HANDOFF, INFO, COMPONENT,
        f"HANDOFF {from_role} → {to_role}: {from_role} produced {produced}; {to_role} received {received}",
        ctx,
    )


def _regulation_text_block(item: dict) -> str:
    """The 'actual' regulation text available for this item (title + official summary).

    The Phase 1 pipeline does not capture the full regulation body; the
    authoritative text we hold is the title and the classification summary.
    The VALIDATOR re-reads THIS rather than the READER's extracted obligations.
    """
    cl = item.get("classification") or {}
    return (
        f"Title:   {item.get('title', '(no title)')}\n"
        f"Date:    {item.get('date', 'unknown')}\n"
        f"Source:  {item.get('source', 'unknown')}\n"
        f"URL:     {item.get('link', '')}\n"
        f"Policy area: {cl.get('policy_area', 'unknown')}\n"
        f"Official summary: {cl.get('summary') or '(no summary — run classify first)'}"
    )


# --- Role 1: READER --------------------------------------------------------

_READER_SYSTEM = (
    "You are the READER, the first role in a multi-agent regulatory gap-analysis chain. "
    "Your only job is to extract every discrete compliance obligation that a new regulatory "
    "item imposes on a licensed financial institution (FI). You do NOT assess policy coverage "
    "— a downstream role does that. Decompose compound requirements into separate obligations. "
    "You return only valid JSON — no explanation, no markdown, no code fences."
)

_READER_USER = """\
Extract every distinct compliance obligation from the regulatory item below. Treat the
title and official summary as the regulation text. Focus on concrete requirements
("must", "shall", "is required to", "obligated to"); split compound requirements into
separate obligations. Where the obligation is inferred (a standard requirement implied
by the regulation type rather than stated verbatim), lower its confidence and say so.

Regulatory item:
{regulation_text}

Return ONLY a JSON object:
{{
  "obligations": [
    {{
      "id": "OBL-1",
      "description": "<one concrete obligation: what the FI must do>",
      "basis": "<the phrase from the title/summary this derives from, or 'inferred from regulation type'>",
      "confidence": <float 0.0-1.0 that this is a genuine, correctly-scoped obligation>
    }}
  ]
}}"""


def _run_reader(client, model, item, item_link):
    raw = _role_call(
        client, model, _READER_SYSTEM,
        _READER_USER.format(regulation_text=_regulation_text_block(item)),
        "READER", item_link,
    )
    if raw is None:
        return None
    data = _parse_json_object(raw, "READER", item_link, "obligations")
    if data is None:
        return None
    obligations = []
    for i, o in enumerate(data["obligations"]):
        if not isinstance(o, dict) or "description" not in o:
            continue
        o.setdefault("id", f"OBL-{i + 1}")
        o.setdefault("basis", "")
        try:
            o["confidence"] = float(o.get("confidence", 0.0))
        except (TypeError, ValueError):
            o["confidence"] = 0.0
        obligations.append(o)
    return obligations


# --- Role 2: MAPPER --------------------------------------------------------

_MAPPER_SYSTEM = (
    "You are the MAPPER in a multi-agent regulatory gap-analysis chain. You receive a list "
    "of obligations extracted by the READER and the full text of an internal policy document. "
    "For each obligation you judge whether the policy already addresses it. The policy is "
    "PROSE, not a structured control register: do not force a clean verdict where the text is "
    "general or ambiguous — capture the ambiguity. You return only valid JSON."
)

_MAPPER_USER = """\
For each obligation below, decide how well the policy document covers it. Use exactly one of:
  - "covered"           : the policy substantively and specifically addresses this obligation.
  - "partially_covered" : the policy touches the area but is silent on key specifics, or only
                          covers it in general terms.
  - "not_covered"       : the policy contains no relevant provision.
The policy is prose and may be vague. When the text is general or open to interpretation,
prefer "partially_covered" with an ambiguity note over an optimistic "covered". Give a
confidence for each decision and quote verbatim policy text as evidence where any exists.

Obligations (from READER):
{obligations_block}

Policy document ({policy_chars} characters):
---
{policy_text}
---

Return ONLY a JSON object:
{{
  "assessments": [
    {{
      "id": "<matching obligation id>",
      "coverage": "<covered | partially_covered | not_covered>",
      "confidence": <float 0.0-1.0 in this coverage decision>,
      "evidence": "<verbatim phrase or section reference from the policy, or empty string>",
      "ambiguity_note": "<where the prose is unclear, general, or open to interpretation; empty if clear>"
    }}
  ]
}}"""


def _run_mapper(client, model, obligations, policy_text, item_link):
    obligations_block = "\n".join(
        f"  {o['id']}: {o['description']}" for o in obligations
    )
    raw = _role_call(
        client, model, _MAPPER_SYSTEM,
        _MAPPER_USER.format(
            obligations_block=obligations_block,
            policy_chars=len(policy_text),
            policy_text=policy_text,
        ),
        "MAPPER", item_link,
    )
    if raw is None:
        return None
    data = _parse_json_object(raw, "MAPPER", item_link, "assessments")
    if data is None:
        return None
    assessments = []
    for a in data["assessments"]:
        if not isinstance(a, dict) or "id" not in a:
            continue
        if a.get("coverage") not in _VALID_MULTI_COVERAGE:
            log_event(
                ERROR, WARN, COMPONENT,
                f"MAPPER: invalid coverage '{a.get('coverage')}' for {a.get('id')} — skipped",
                {"role": "MAPPER", "obligation_id": a.get("id"), "item_link": item_link},
            )
            continue
        try:
            a["confidence"] = float(a.get("confidence", 0.0))
        except (TypeError, ValueError):
            a["confidence"] = 0.0
        a.setdefault("evidence", "")
        a.setdefault("ambiguity_note", "")
        assessments.append(a)
    return assessments


# --- Role 3: GAP -----------------------------------------------------------

_GAP_SYSTEM = (
    "You are the GAP analyst in a multi-agent regulatory gap-analysis chain. You receive the "
    "obligations and the MAPPER's coverage assessments. For every obligation that is NOT "
    "adequately covered (coverage is 'not_covered' or 'partially_covered') you draft a short, "
    "specific findings note describing what the policy is missing and why it matters. "
    "You return only valid JSON."
)

_GAP_USER = """\
Below are the obligations and the MAPPER's coverage assessment for each. For every obligation
whose coverage is "not_covered" or "partially_covered", write one findings note. Do not write
notes for fully "covered" obligations. Keep each note specific and actionable (2-3 sentences).

Obligations and assessments:
{assessment_block}

Return ONLY a JSON object:
{{
  "findings": [
    {{
      "id": "<obligation id>",
      "coverage": "<the MAPPER coverage for this obligation>",
      "severity": "<high | medium | low>",
      "finding": "<what the policy is missing or leaves ambiguous, and why it matters>",
      "recommended_action": "<concrete remediation the policy owner should take>"
    }}
  ]
}}"""


def _run_gap(client, model, obligations, assessments, item_link):
    by_id = {o["id"]: o for o in obligations}
    lines = []
    for a in assessments:
        desc = by_id.get(a["id"], {}).get("description", "(unknown obligation)")
        lines.append(
            f"  {a['id']} [{a['coverage']}, conf={a['confidence']:.2f}]: {desc}\n"
            f"      evidence: {a.get('evidence') or '(none)'}\n"
            f"      ambiguity: {a.get('ambiguity_note') or '(none)'}"
        )
    raw = _role_call(
        client, model, _GAP_SYSTEM,
        _GAP_USER.format(assessment_block="\n".join(lines)),
        "GAP", item_link,
    )
    if raw is None:
        return None
    data = _parse_json_object(raw, "GAP", item_link, "findings")
    if data is None:
        return None
    findings = []
    for f in data["findings"]:
        if not isinstance(f, dict) or "id" not in f:
            continue
        f.setdefault("coverage", "")
        f.setdefault("severity", "")
        f.setdefault("finding", "")
        f.setdefault("recommended_action", "")
        findings.append(f)
    return findings


# --- Role 4: VALIDATOR (engineered independence) ---------------------------

_VALIDATOR_SYSTEM = (
    "You are the independent VALIDATOR — the final role in a multi-agent regulatory "
    "gap-analysis chain. You were deliberately run on a DIFFERENT model from the first-pass "
    "roles to reduce correlated error. Do NOT defer to the upstream obligation list, coverage "
    "assessments, or findings. Re-derive your own judgment FROM SCRATCH using only the actual "
    "regulation text and the actual policy document provided. Only AFTER forming your own view "
    "should you compare it against the upstream positions shown for reference. You return only "
    "valid JSON."
)

_VALIDATOR_USER = """\
Independently re-assess this regulatory item against this policy. Read the SOURCE TEXTS first
and form your own judgment before looking at the upstream positions.

=== ACTUAL REGULATION TEXT (authoritative — read this, not the upstream summaries) ===
{regulation_text}

=== ACTUAL POLICY DOCUMENT ({policy_chars} characters) ===
---
{policy_text}
---

=== UPSTREAM POSITIONS (for comparison ONLY — do not defer to these) ===
{upstream_block}

For EACH upstream obligation, return your independent verdict. Also report any obligation you
find in the regulation that the READER missed. Use coverage values:
"covered" | "partially_covered" | "not_covered".

Return ONLY a JSON object:
{{
  "verdicts": [
    {{
      "id": "<obligation id>",
      "obligation_valid": <true if this is a genuine, correctly-scoped obligation from the regulation; false if the READER hallucinated or misattributed it>,
      "independent_coverage": "<your own coverage verdict, re-derived from the policy text>",
      "validator_confidence": <float 0.0-1.0>,
      "agrees_with_mapper": <true if independent_coverage matches the MAPPER's coverage>,
      "agrees_with_gap": <true if you agree with the GAP analyst's treatment (flagged vs not flagged) of this obligation>,
      "rationale": "<why — cite the policy text you relied on>"
    }}
  ],
  "missed_obligations": [
    {{ "description": "<obligation present in the regulation that the READER did not extract>" }}
  ],
  "overall_assessment": "<one or two sentences on the chain's reliability for this item>"
}}"""


def _run_validator(client, model, item, policy_text, obligations, assessments, findings, item_link):
    cov_by_id = {a["id"]: a for a in assessments}
    flagged_ids = {f["id"] for f in findings}
    by_id = {o["id"]: o for o in obligations}
    lines = []
    for o in obligations:
        a = cov_by_id.get(o["id"], {})
        lines.append(
            f"  {o['id']}: {o['description']}\n"
            f"      MAPPER coverage: {a.get('coverage', '(none)')} (conf={a.get('confidence', 0):.2f})\n"
            f"      GAP flagged as inadequately covered: {'yes' if o['id'] in flagged_ids else 'no'}"
        )
    raw = _role_call(
        client, model, _VALIDATOR_SYSTEM,
        _VALIDATOR_USER.format(
            regulation_text=_regulation_text_block(item),
            policy_chars=len(policy_text),
            policy_text=policy_text,
            upstream_block="\n".join(lines),
        ),
        "VALIDATOR", item_link,
    )
    if raw is None:
        return None
    data = _parse_json_object(raw, "VALIDATOR", item_link, "verdicts")
    if data is None:
        return None
    verdicts = []
    for v in data["verdicts"]:
        if not isinstance(v, dict) or "id" not in v:
            continue
        v.setdefault("obligation_valid", True)
        v.setdefault("independent_coverage", "")
        v.setdefault("agrees_with_mapper", None)
        v.setdefault("agrees_with_gap", None)
        v.setdefault("rationale", "")
        try:
            v["validator_confidence"] = float(v.get("validator_confidence", 0.0))
        except (TypeError, ValueError):
            v["validator_confidence"] = 0.0
        verdicts.append(v)
    data["verdicts"] = verdicts
    data.setdefault("missed_obligations", [])
    data.setdefault("overall_assessment", "")

    # Log a VALIDATOR_DISAGREEMENT for every obligation where the validator's
    # independent position differs from the chain's — with BOTH positions.
    for v in verdicts:
        mapper = cov_by_id.get(v["id"], {})
        mapper_cov = mapper.get("coverage")
        val_cov = v.get("independent_coverage")
        disagrees_coverage = (
            val_cov in _VALID_MULTI_COVERAGE and mapper_cov in _VALID_MULTI_COVERAGE
            and val_cov != mapper_cov
        )
        disagrees_gap = v.get("agrees_with_gap") is False
        invalid_obl = v.get("obligation_valid") is False
        if disagrees_coverage or disagrees_gap or invalid_obl:
            reasons = []
            if disagrees_coverage:
                reasons.append("coverage")
            if disagrees_gap:
                reasons.append("gap-treatment")
            if invalid_obl:
                reasons.append("obligation-validity")
            log_event(
                VALIDATOR_DISAGREEMENT, WARN, COMPONENT,
                f"VALIDATOR disagrees on {v['id']} ({', '.join(reasons)}): "
                f"MAPPER={mapper_cov or 'n/a'} vs VALIDATOR={val_cov or 'n/a'}",
                {
                    "obligation_id": v["id"],
                    "obligation": by_id.get(v["id"], {}).get("description", ""),
                    "disagreement_on": reasons,
                    "first_pass_position": {
                        "coverage": mapper_cov,
                        "confidence": mapper.get("confidence"),
                        "gap_flagged": v["id"] in flagged_ids,
                        "model": MULTI_FIRSTPASS_MODEL,
                    },
                    "validator_position": {
                        "coverage": val_cov,
                        "confidence": v.get("validator_confidence"),
                        "obligation_valid": v.get("obligation_valid"),
                        "rationale": v.get("rationale"),
                        "model": model,
                    },
                    "item_link": item_link,
                },
            )

    # The validator may also flag obligations the READER missed entirely.
    for miss in data["missed_obligations"]:
        desc = miss.get("description") if isinstance(miss, dict) else str(miss)
        if not desc:
            continue
        log_event(
            VALIDATOR_DISAGREEMENT, WARN, COMPONENT,
            f"VALIDATOR found an obligation the READER missed: {desc[:80]}",
            {
                "disagreement_on": ["missed_obligation"],
                "first_pass_position": {"extracted": False, "model": MULTI_FIRSTPASS_MODEL},
                "validator_position": {"obligation": desc, "model": model},
                "item_link": item_link,
            },
        )
    return data


# --- Orchestrator ----------------------------------------------------------

def run_multi_agent(
    item: dict,
    policy_text: str,
    client: anthropic.Anthropic,
) -> dict | None:
    """Phase 3 multi-agent chain: READER → MAPPER → GAP → VALIDATOR.

    Runs alongside run_baseline() (which is untouched). Logs a HANDOFF at every
    role boundary and VALIDATOR_DISAGREEMENT wherever the independent validator
    diverges from the first-pass chain. Returns a result dict or None on failure.
    """
    cl = item.get("classification") or {}
    policy_area = cl.get("policy_area", "unknown")
    item_link = item.get("link", "")
    fp_model = MULTI_FIRSTPASS_MODEL
    val_model = _validator_model(fp_model)

    _log_handoff(
        "INTAKE", "READER",
        "the raw regulation item (title + official summary)",
        "the regulation item to extract obligations from",
        item_link,
        {"first_pass_model": fp_model, "validator_model": val_model,
         "title": item.get("title", "")[:80]},
    )

    # 1. READER
    obligations = _run_reader(client, fp_model, item, item_link)
    if obligations is None:
        return None
    log_event(
        PARSE_OK, INFO, COMPONENT,
        f"READER extracted {len(obligations)} obligation(s) for: {item.get('title', '')[:80]}",
        {"role": "READER", "obligation_count": len(obligations), "item_link": item_link},
    )
    _log_handoff(
        "READER", "MAPPER",
        f"{len(obligations)} extracted obligation(s): {[o['id'] for o in obligations]}",
        f"{len(obligations)} obligation(s) to check against the policy document",
        item_link,
        {"obligation_ids": [o["id"] for o in obligations]},
    )

    # 2. MAPPER
    assessments = _run_mapper(client, fp_model, obligations, policy_text, item_link)
    if assessments is None:
        return None
    n_cov = sum(1 for a in assessments if a["coverage"] == "covered")
    n_part = sum(1 for a in assessments if a["coverage"] == "partially_covered")
    n_not = sum(1 for a in assessments if a["coverage"] == "not_covered")
    _log_handoff(
        "MAPPER", "GAP",
        f"{len(assessments)} coverage assessment(s) "
        f"({n_cov} covered, {n_part} partially_covered, {n_not} not_covered)",
        "the obligations + coverage assessments to draft findings for the inadequately-covered ones",
        item_link,
        {"covered": n_cov, "partially_covered": n_part, "not_covered": n_not},
    )

    # 3. GAP
    findings = _run_gap(client, fp_model, obligations, assessments, item_link)
    if findings is None:
        return None
    _log_handoff(
        "GAP", "VALIDATOR",
        f"{len(findings)} gap finding(s) for inadequately-covered obligation(s): "
        f"{[f['id'] for f in findings]}",
        "the source texts + full upstream chain to independently re-validate on a different model",
        item_link,
        {"finding_ids": [f["id"] for f in findings], "validator_model": val_model},
    )

    # 4. VALIDATOR (different model + re-reads source texts)
    validation = _run_validator(
        client, val_model, item, policy_text, obligations, assessments, findings, item_link
    )
    if validation is None:
        return None
    disagreements = sum(
        1 for v in validation["verdicts"]
        if v.get("agrees_with_mapper") is False
        or v.get("agrees_with_gap") is False
        or v.get("obligation_valid") is False
    )
    _log_handoff(
        "VALIDATOR", "OUTPUT",
        f"independent verdicts on {len(validation['verdicts'])} obligation(s), "
        f"{disagreements} disagreement(s), {len(validation['missed_obligations'])} missed-obligation flag(s)",
        "the assembled multi-agent result",
        item_link,
        {"disagreement_count": disagreements,
         "missed_obligation_count": len(validation["missed_obligations"])},
    )

    # Assemble a merged, per-obligation view (chain decision = MAPPER coverage;
    # the validator is an independent check, not an overrider).
    cov_by_id = {a["id"]: a for a in assessments}
    find_by_id = {f["id"]: f for f in findings}
    val_by_id = {v["id"]: v for v in validation["verdicts"]}
    merged = []
    for o in obligations:
        a = cov_by_id.get(o["id"], {})
        merged.append({
            "id": o["id"],
            "description": o["description"],
            "basis": o.get("basis", ""),
            "coverage": a.get("coverage", "not_covered"),
            "confidence": a.get("confidence", 0.0),
            "evidence": a.get("evidence", ""),
            "ambiguity_note": a.get("ambiguity_note", ""),
            "is_gap": o["id"] in find_by_id,
            "finding": find_by_id.get(o["id"], {}).get("finding", ""),
            "validator": val_by_id.get(o["id"], {}),
        })

    gap_ratio = (n_not + 0.5 * n_part) / len(merged) if merged else 0.0

    result = {
        "item": {
            "title": item.get("title"),
            "date": item.get("date"),
            "link": item_link,
            "source": item.get("source"),
            "policy_area": policy_area,
        },
        "approach": "multi_agent",
        "first_pass_model": fp_model,
        "validator_model": val_model,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "obligations": merged,
        "reader_obligations": obligations,
        "mapper_assessments": assessments,
        "gap_findings": findings,
        "validation": validation,
        "summary": {
            "total": len(merged),
            "covered": n_cov,
            "partially_covered": n_part,
            "not_covered": n_not,
            "gap_count": len(findings),
            "gap_ratio": round(gap_ratio, 3),
            "validator_disagreements": disagreements,
            "missed_obligations": len(validation["missed_obligations"]),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Batch wrapper (used by monitor.py or called directly)
# ---------------------------------------------------------------------------

def analyze_items(items: list[dict], policy_text: str) -> list[dict]:
    """Run run_baseline() over a list of items. Returns only successful results."""
    if not items:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            "ANTHROPIC_API_KEY not set — baseline analysis skipped for all items",
            {"item_count": len(items)},
        )
        return []

    client = anthropic.Anthropic(api_key=api_key)
    results: list[dict] = []

    for item in items:
        result = run_baseline(item, policy_text, client)
        if result is not None:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def save_analysis(result: dict, out_dir: Path | None = None, prefix: str = "baseline") -> Path:
    """Write one analysis result to output/ as a JSON file. Returns the path.

    `prefix` distinguishes the run type (e.g. "baseline" or "multi") in the filename.
    """
    out_dir = out_dir or _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "_", (result["item"].get("source") or "unknown").lower())[:30]
    path = out_dir / f"analysis_{prefix}_{slug}_{ts}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Standalone entrypoint (experimental control — run without the full pipeline)
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Baseline gap analysis: ONE Anthropic call per item. "
            "This is the Phase 2 control condition."
        )
    )
    p.add_argument(
        "--item", "-i", metavar="PATH",
        help="Path to a JSON file containing one classified item (from Phase 1 output). "
             "Pass '-' to read from stdin.",
    )
    p.add_argument(
        "--policy", "-p", metavar="PATH",
        help="Path to the policy .md file. Defaults to the first .md in policy/.",
    )
    p.add_argument(
        "--out", "-o", metavar="DIR",
        help=f"Output directory (default: {_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Print result to stdout only; do not write to output/.",
    )
    p.add_argument(
        "--multi", action="store_true",
        help="Run the Phase 3 multi-agent chain (READER → MAPPER → GAP → VALIDATOR) "
             "instead of the single-call baseline.",
    )
    return p


def _load_item(path_str: str) -> dict:
    if path_str == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _find_policy() -> Path:
    preferred = _ROOT / "policy" / "sample-oprisk-policy.md"
    if preferred.exists():
        return preferred
    candidates = sorted((_ROOT / "policy").glob("*.md"))
    if not candidates:
        raise FileNotFoundError("No .md file found in policy/. Drop one there first.")
    return candidates[0]


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not args.item:
        parser.print_help()
        sys.exit(0)

    try:
        item = _load_item(args.item)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR loading item: {exc}", file=sys.stderr)
        sys.exit(1)

    policy_path = Path(args.policy) if args.policy else _find_policy()
    try:
        policy_text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR loading policy: {exc}", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.multi:
        val_model = _validator_model(MULTI_FIRSTPASS_MODEL)
        print(
            f"Running MULTI-AGENT analysis on: {item.get('title', '(no title)')}\n"
            f"  First-pass model: {MULTI_FIRSTPASS_MODEL} (READER, MAPPER, GAP)\n"
            f"  Validator model:  {val_model} (independent)\n"
            f"  Policy: {policy_path.name} ({len(policy_text):,} chars)\n",
            file=sys.stderr,
        )
        result = run_multi_agent(item, policy_text, client)
        prefix = "multi"
    else:
        print(
            f"Running baseline analysis on: {item.get('title', '(no title)')}\n"
            f"  Model:  {ANALYZE_MODEL}\n"
            f"  Policy: {policy_path.name} ({len(policy_text):,} chars)\n",
            file=sys.stderr,
        )
        result = run_baseline(item, policy_text, client)
        prefix = "baseline"

    if result is None:
        print("Analysis failed — check logs/observation_log.jsonl for details.", file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if not args.no_save:
        out_dir = Path(args.out) if args.out else None
        saved_path = save_analysis(result, out_dir, prefix=prefix)
        print(f"Saved: {saved_path}", file=sys.stderr)

    # Human summary to stderr, full JSON to stdout (pipeline-friendly)
    s = result["summary"]
    if args.multi:
        print(
            f"\nObligations: {s['total']}  |  "
            f"Covered: {s['covered']}  |  "
            f"Partially covered: {s['partially_covered']}  |  "
            f"Not covered: {s['not_covered']}  |  "
            f"Gap ratio: {s['gap_ratio']:.0%}  |  "
            f"Validator disagreements: {s['validator_disagreements']}  |  "
            f"Missed obligations: {s['missed_obligations']}",
            file=sys.stderr,
        )
    else:
        print(
            f"\nObligations: {s['total']}  |  "
            f"Covered: {s['covered']}  |  "
            f"Partial: {s['partial']}  |  "
            f"Not covered: {s['not_covered']}  |  "
            f"Gap ratio: {s['gap_ratio']:.0%}  |  "
            f"Low-confidence: {s['low_confidence_count']}",
            file=sys.stderr,
        )
    print(output_json)


if __name__ == "__main__":
    main()
