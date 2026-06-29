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
    INFO, WARN, CRITICAL,
    log_event,
)

COMPONENT = "analyze"

ANALYZE_MODEL = os.environ.get("REGMONITOR_ANALYZE_MODEL", "claude-haiku-4-5-20251001")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("REGMONITOR_CONFIDENCE_THRESHOLD", "0.70"))
# Fire BASELINE_DIVERGENCE when this fraction of obligations are unaddressed.
DIVERGENCE_THRESHOLD = float(os.environ.get("REGMONITOR_DIVERGENCE_THRESHOLD", "0.40"))

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

def save_analysis(result: dict, out_dir: Path | None = None) -> Path:
    """Write one baseline result to output/ as a JSON file. Returns the path."""
    out_dir = out_dir or _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "_", (result["item"].get("source") or "unknown").lower())[:30]
    path = out_dir / f"analysis_baseline_{slug}_{ts}.json"
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
    return p


def _load_item(path_str: str) -> dict:
    if path_str == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _find_policy() -> Path:
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

    print(
        f"Running baseline analysis on: {item.get('title', '(no title)')}\n"
        f"  Model:  {ANALYZE_MODEL}\n"
        f"  Policy: {policy_path.name} ({len(policy_text):,} chars)\n",
        file=sys.stderr,
    )

    client = anthropic.Anthropic(api_key=api_key)
    result = run_baseline(item, policy_text, client)

    if result is None:
        print("Analysis failed — check logs/observation_log.jsonl for details.", file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if not args.no_save:
        out_dir = Path(args.out) if args.out else None
        saved_path = save_analysis(result, out_dir)
        print(f"Saved: {saved_path}", file=sys.stderr)

    # Human summary to stderr, full JSON to stdout (pipeline-friendly)
    s = result["summary"]
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
