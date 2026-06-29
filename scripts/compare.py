#!/usr/bin/env python3
"""
Comparison harness — single-call baseline vs. multi-agent chain.

Runs ONE regulation item through both approaches in analyze.py:
  - run_baseline()    : the Phase 2 single-call control condition.
  - run_multi_agent() : the Phase 3 READER → MAPPER → GAP → VALIDATOR chain.

It then aligns the two independently-extracted obligation sets and logs a
BASELINE_DIVERGENCE event wherever the approaches disagree on coverage or on
whether an obligation is a gap. Because each approach extracts its own
obligations, alignment is by description similarity (token Jaccard) rather than
by id.

Default input is the saved baseline item:
    "Revised Outsourcing Rulebook Module – Cloud Services"
    (output/baseline_test_input.json)

Usage:
    # Set ANTHROPIC_API_KEY in your environment before running
    python scripts/compare.py
    python scripts/compare.py --item <path>.json --policy <path>.md
    python scripts/compare.py --no-save        # don't write the comparison report
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import anthropic  # noqa: E402

import analyze  # noqa: E402  (src/ is on sys.path)
from observability import BASELINE_DIVERGENCE, HANDOFF, INFO, WARN, log_event  # noqa: E402

COMPONENT = "compare"

_DEFAULT_ITEM = _ROOT / "output" / "baseline_test_input.json"
_OUTPUT_DIR = _ROOT / "output"

# Two obligations are considered "the same" obligation when their descriptions
# overlap at least this much (Jaccard over content words).
_MATCH_THRESHOLD = 0.18

# Normalise the two approaches' coverage vocabularies onto a common scale.
# Baseline uses {covered, partial, not_covered}; multi-agent uses
# {covered, partially_covered, not_covered}.
_COVERAGE_NORM = {
    "covered": "covered",
    "partial": "partial",
    "partially_covered": "partial",
    "not_covered": "not_covered",
}

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "must", "shall", "is", "are", "be", "this", "that", "any", "all", "its",
    "fi", "institution", "institutions", "financial", "licensed", "ensure",
    "required", "require", "requires", "include", "including", "within", "from",
    "their", "which", "as", "at", "each", "every",
})


def _norm_coverage(value: str) -> str:
    return _COVERAGE_NORM.get(value, value or "unknown")


def _is_gap(coverage: str) -> bool:
    """An obligation is a 'gap' when it is not fully covered."""
    return _norm_coverage(coverage) in ("partial", "not_covered")


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_obligations(baseline_obls: list[dict], multi_obls: list[dict]) -> tuple[list, list, list]:
    """Greedily align baseline and multi-agent obligations by description overlap.

    Returns (matched_pairs, baseline_only, multi_only) where matched_pairs is a
    list of (baseline_obl, multi_obl, similarity).
    """
    b_tokens = [(o, _tokens(o.get("description", ""))) for o in baseline_obls]
    m_tokens = [(o, _tokens(o.get("description", ""))) for o in multi_obls]

    candidates = []
    for bi, (_, bt) in enumerate(b_tokens):
        for mi, (_, mt) in enumerate(m_tokens):
            sim = _jaccard(bt, mt)
            if sim >= _MATCH_THRESHOLD:
                candidates.append((sim, bi, mi))
    candidates.sort(reverse=True)  # highest similarity first

    used_b: set[int] = set()
    used_m: set[int] = set()
    matched = []
    for sim, bi, mi in candidates:
        if bi in used_b or mi in used_m:
            continue
        used_b.add(bi)
        used_m.add(mi)
        matched.append((b_tokens[bi][0], m_tokens[mi][0], round(sim, 3)))

    baseline_only = [b_tokens[i][0] for i in range(len(b_tokens)) if i not in used_b]
    multi_only = [m_tokens[i][0] for i in range(len(m_tokens)) if i not in used_m]
    return matched, baseline_only, multi_only


def compare(item: dict, policy_text: str, client: anthropic.Anthropic) -> dict | None:
    """Run both approaches and log BASELINE_DIVERGENCE on every disagreement."""
    item_link = item.get("link", "")
    title = item.get("title", "(no title)")

    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Comparison harness start — running baseline + multi-agent on: {title[:80]}",
        {"item_link": item_link, "title": title[:80]},
    )

    baseline = analyze.run_baseline(item, policy_text, client)
    if baseline is None:
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            "Baseline run failed — cannot compare. See earlier ERROR events.",
            {"item_link": item_link},
        )
        return None

    multi = analyze.run_multi_agent(item, policy_text, client)
    if multi is None:
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            "Multi-agent run failed — cannot compare. See earlier ERROR events.",
            {"item_link": item_link},
        )
        return None

    baseline_obls = baseline["obligations"]
    multi_obls = multi["obligations"]
    matched, baseline_only, multi_only = _match_obligations(baseline_obls, multi_obls)

    divergences: list[dict] = []

    # 1. Matched obligations that disagree on coverage and/or gap status.
    for b, m, sim in matched:
        b_cov = _norm_coverage(b.get("coverage"))
        m_cov = _norm_coverage(m.get("coverage"))
        coverage_diff = b_cov != m_cov
        gap_diff = _is_gap(b.get("coverage")) != _is_gap(m.get("coverage"))
        if not (coverage_diff or gap_diff):
            continue
        kinds = []
        if coverage_diff:
            kinds.append("coverage")
        if gap_diff:
            kinds.append("gap")
        record = {
            "type": "coverage_mismatch",
            "disagreement_on": kinds,
            "similarity": sim,
            "baseline": {
                "id": b.get("id"),
                "description": b.get("description"),
                "coverage": b.get("coverage"),
                "is_gap": _is_gap(b.get("coverage")),
                "confidence": b.get("confidence"),
            },
            "multi_agent": {
                "id": m.get("id"),
                "description": m.get("description"),
                "coverage": m.get("coverage"),
                "is_gap": m.get("is_gap", _is_gap(m.get("coverage"))),
                "confidence": m.get("confidence"),
            },
            "item_link": item_link,
        }
        divergences.append(record)
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            f"Approaches disagree ({'/'.join(kinds)}) on '{b.get('description', '')[:70]}': "
            f"baseline={b.get('coverage')} vs multi-agent={m.get('coverage')}",
            record,
        )

    # 2. Obligations only one approach extracted at all (a coverage/gap
    #    disagreement by omission — one flagged an obligation the other never saw).
    for b in baseline_only:
        record = {
            "type": "baseline_only_obligation",
            "disagreement_on": ["obligation_set", "gap"] if _is_gap(b.get("coverage")) else ["obligation_set"],
            "baseline": {
                "id": b.get("id"),
                "description": b.get("description"),
                "coverage": b.get("coverage"),
                "is_gap": _is_gap(b.get("coverage")),
            },
            "multi_agent": None,
            "item_link": item_link,
        }
        divergences.append(record)
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            f"Obligation found only by BASELINE (multi-agent missed it): "
            f"{b.get('description', '')[:70]} [{b.get('coverage')}]",
            record,
        )

    for m in multi_only:
        record = {
            "type": "multi_agent_only_obligation",
            "disagreement_on": ["obligation_set", "gap"] if m.get("is_gap") else ["obligation_set"],
            "baseline": None,
            "multi_agent": {
                "id": m.get("id"),
                "description": m.get("description"),
                "coverage": m.get("coverage"),
                "is_gap": m.get("is_gap", _is_gap(m.get("coverage"))),
            },
            "item_link": item_link,
        }
        divergences.append(record)
        log_event(
            BASELINE_DIVERGENCE, WARN, COMPONENT,
            f"Obligation found only by MULTI-AGENT (baseline missed it): "
            f"{m.get('description', '')[:70]} [{m.get('coverage')}]",
            record,
        )

    baseline_gaps = sum(1 for o in baseline_obls if _is_gap(o.get("coverage")))
    multi_gaps = sum(1 for o in multi_obls if o.get("is_gap", _is_gap(o.get("coverage"))))

    comparison = {
        "item": {"title": title, "link": item_link, "source": item.get("source")},
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline.get("baseline_model"),
        "multi_first_pass_model": multi.get("first_pass_model"),
        "multi_validator_model": multi.get("validator_model"),
        "baseline_summary": baseline["summary"],
        "multi_summary": multi["summary"],
        "alignment": {
            "matched": len(matched),
            "baseline_only": len(baseline_only),
            "multi_agent_only": len(multi_only),
        },
        "totals": {
            "baseline_obligations": len(baseline_obls),
            "multi_agent_obligations": len(multi_obls),
            "baseline_gaps": baseline_gaps,
            "multi_agent_gaps": multi_gaps,
        },
        "divergence_count": len(divergences),
        "divergences": divergences,
    }

    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Comparison complete — {len(divergences)} divergence(s); "
        f"baseline {len(baseline_obls)} obls/{baseline_gaps} gaps vs "
        f"multi-agent {len(multi_obls)} obls/{multi_gaps} gaps",
        {
            "divergence_count": len(divergences),
            "alignment": comparison["alignment"],
            "totals": comparison["totals"],
            "item_link": item_link,
        },
    )
    return comparison


def _save_comparison(comparison: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "_", (comparison["item"].get("source") or "unknown").lower())[:30]
    path = out_dir / f"comparison_{slug}_{ts}.json"
    path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one regulation item through both the single-call baseline and the "
                    "multi-agent chain, logging BASELINE_DIVERGENCE on every disagreement."
    )
    parser.add_argument(
        "--item", "-i", metavar="PATH", default=str(_DEFAULT_ITEM),
        help=f"Classified item JSON (default: {_DEFAULT_ITEM}).",
    )
    parser.add_argument(
        "--policy", "-p", metavar="PATH",
        help="Policy .md file. Defaults to the first .md in policy/.",
    )
    parser.add_argument(
        "--out", "-o", metavar="DIR",
        help=f"Output directory for the comparison report (default: {_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print the comparison report to stdout only; do not write to output/.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        item = json.loads(Path(args.item).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR loading item: {exc}", file=sys.stderr)
        sys.exit(1)

    policy_path = Path(args.policy) if args.policy else analyze._find_policy()
    try:
        policy_text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR loading policy: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Comparing approaches on: {item.get('title', '(no title)')}\n"
        f"  Baseline model:   {analyze.ANALYZE_MODEL}\n"
        f"  Multi first-pass: {analyze.MULTI_FIRSTPASS_MODEL}\n"
        f"  Multi validator:  {analyze._validator_model(analyze.MULTI_FIRSTPASS_MODEL)}\n"
        f"  Policy: {policy_path.name} ({len(policy_text):,} chars)\n",
        file=sys.stderr,
    )

    client = anthropic.Anthropic(api_key=api_key)
    comparison = compare(item, policy_text, client)
    if comparison is None:
        print("Comparison failed — check logs/observation_log.jsonl for details.", file=sys.stderr)
        sys.exit(1)

    if not args.no_save:
        out_dir = Path(args.out) if args.out else None
        saved = _save_comparison(comparison, out_dir)
        print(f"Saved: {saved}", file=sys.stderr)

    t = comparison["totals"]
    print(
        f"\nBaseline:    {t['baseline_obligations']} obligations, {t['baseline_gaps']} gaps\n"
        f"Multi-agent: {t['multi_agent_obligations']} obligations, {t['multi_agent_gaps']} gaps\n"
        f"Aligned: {comparison['alignment']['matched']} matched, "
        f"{comparison['alignment']['baseline_only']} baseline-only, "
        f"{comparison['alignment']['multi_agent_only']} multi-only\n"
        f"BASELINE_DIVERGENCE events logged: {comparison['divergence_count']}",
        file=sys.stderr,
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
