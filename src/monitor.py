"""Orchestrate fetch → liveness → diff → classify → digest.

Usage:
    cd regmonitor/src
    python monitor.py

The digest is written to output/digest_<ISO-timestamp>.md.
The header of every digest explicitly states which sources were checked
and which failed liveness, so a partial run can never look complete.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from observability import HANDOFF, ERROR, INFO, WARN, CRITICAL, log_event
import fetch as fetch_module
import classify as classify_module

COMPONENT = "monitor"

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "sources.yaml"
_POLICY_DIR = _ROOT / "policy"
_OUTPUT_DIR = _ROOT / "output"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_sources() -> list[dict]:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def _load_policy() -> str | None:
    """Return the first .md file found in policy/, or None with a warning."""
    docs = sorted(_POLICY_DIR.glob("*.md"))
    if not docs:
        log_event(
            ERROR, WARN, COMPONENT,
            "No policy document in policy/ — classification will have no policy context",
            {"policy_dir": str(_POLICY_DIR)},
        )
        return None
    if len(docs) > 1:
        log_event(
            ERROR, WARN, COMPONENT,
            f"Multiple policy docs found; using {docs[0].name}",
            {"chosen": docs[0].name, "all": [d.name for d in docs]},
        )
    return docs[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Digest writer
# ---------------------------------------------------------------------------

def _policy_area_badge(area: str) -> str:
    badges = {
        "operational_risk": "🔧 Operational Risk",
        "aml_cft":          "🔍 AML/CFT",
        "cyber":            "🛡 Cyber",
        "icaap":            "📊 ICAAP",
        "outsourcing":      "🤝 Outsourcing",
        "ai_governance":    "🤖 AI Governance",
        "other":            "📋 Other",
    }
    return badges.get(area, area)


def _write_digest(
    classified_items: list[dict],
    unclassified_new: list[dict],
    healthy_sources: list[str],
    failed_sources: list[str],
    fixture_sources: set[str],
    run_ts: datetime,
) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_file = run_ts.strftime("%Y-%m-%dT%H%M%SZ")
    date_human = run_ts.strftime("%d %B %Y")
    digest_path = _OUTPUT_DIR / f"digest_{ts_file}.md"

    all_new = classified_items + unclassified_new  # unclassified = API key missing / API error

    lines: list[str] = []

    # ---- Title ----
    lines += [
        f"# Regulatory Monitor Digest — {date_human}",
        "",
        f"_Generated: {run_ts.isoformat()}_",
        "",
    ]

    # ---- Coverage block (always comes first, always explicit) ----
    lines += ["## Coverage", ""]

    if healthy_sources:
        lines.append(f"**Sources checked ({len(healthy_sources)}):**")
        for s in healthy_sources:
            tag = " _(fixture)_" if s in fixture_sources else " _(live)_"
            lines.append(f"- {s}{tag}")
    else:
        lines.append("**Sources checked:** none")

    if fixture_sources:
        lines += [
            "",
            "> [!NOTE]",
            f"> **{len(fixture_sources)} source(s) were read from local fixtures, not fetched live.**",
            "> Results reflect the fixture snapshot, not the current state of the live site.",
        ]

    lines.append("")

    if failed_sources:
        lines += [
            "> [!WARNING]",
            f"> **{len(failed_sources)} source(s) failed liveness and are EXCLUDED from this digest:**",
        ]
        for s in failed_sources:
            lines.append(f"> - {s}")
        lines += [
            ">",
            "> **This digest is INCOMPLETE.** Do not treat it as a full regulatory scan.",
            "> Check `logs/observation_log.jsonl` for SOURCE_FAILURE details.",
            "",
        ]
    else:
        lines += [
            "> All configured sources passed liveness checks. Coverage is complete.",
            "",
        ]

    lines += ["---", ""]

    # ---- New items ----
    if not all_new:
        if not failed_sources:
            lines += ["## New Items", "", "No new items found. All sources are current.", ""]
        else:
            lines += [
                "## New Items",
                "",
                "_Cannot determine whether there are new items because one or more sources failed._",
                "",
            ]
    else:
        lines += [f"## New Items ({len(all_new)} total)", ""]

        # Group by source
        by_source: dict[str, list[dict]] = {}
        for item in all_new:
            by_source.setdefault(item.get("source", "Unknown"), []).append(item)

        for src_name, items in by_source.items():
            lines += [f"### {src_name}", ""]

            for item in items:
                title = item.get("title") or "(no title)"
                link = item.get("link", "")
                date = item.get("date", "")
                cl: dict = item.get("classification") or {}

                # Item heading with optional hyperlink
                heading = f"[{title}]({link})" if link else title
                lines += [f"#### {heading}", ""]

                if date:
                    lines.append(f"- **Date:** {date}")

                if cl:
                    area_label = _policy_area_badge(cl.get("policy_area", "other"))
                    rel = cl.get("relevance_score", 0.0)
                    conf = cl.get("confidence", 0.0)
                    conf_flag = " ⚠ low confidence" if conf < classify_module.LOW_CONFIDENCE_THRESHOLD else ""
                    lines += [
                        f"- **Policy area:** {area_label}",
                        f"- **Relevance:** {rel:.2f}",
                        f"- **Confidence:** {conf:.2f}{conf_flag}",
                    ]
                    summary = cl.get("summary", "")
                    if summary:
                        lines += ["", f"> {summary}"]
                else:
                    lines.append("- _Classification unavailable (see logs)_")

                lines.append("")

    # ---- Footer ----
    lines += [
        "---",
        "",
        f"_Run completed: {run_ts.isoformat()}_",
        (
            f"_Sources checked: {len(healthy_sources)} "
            f"({len(fixture_sources)} fixture, {len(healthy_sources) - len(fixture_sources)} live) | "
            f"Sources failed: {len(failed_sources)} | "
            f"New items: {len(all_new)}_"
        ),
    ]

    digest_path.write_text("\n".join(lines), encoding="utf-8")
    return digest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    run_ts = datetime.now(timezone.utc)
    log_event(
        HANDOFF, INFO, COMPONENT,
        "Monitor run started",
        {"timestamp": run_ts.isoformat()},
    )

    # Load config
    try:
        sources = _load_sources()
    except Exception as exc:
        log_event(ERROR, CRITICAL, COMPONENT, f"Failed to load sources.yaml: {exc}", {})
        sys.exit(1)

    policy_text = _load_policy()

    enabled = [s for s in sources if s.get("enabled", True)]
    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Loaded {len(enabled)} enabled source(s) from config",
        {"enabled": [s["name"] for s in enabled]},
    )

    # Fetch, liveness check, diff
    new_items, healthy_sources, failed_sources, fixture_sources = fetch_module.fetch_all(enabled)

    if failed_sources:
        log_event(
            ERROR, CRITICAL, COMPONENT,
            f"{len(failed_sources)} source(s) failed liveness — excluded from digest",
            {"failed": failed_sources},
        )

    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Fetch complete: {len(new_items)} new item(s) across {len(healthy_sources)} healthy source(s)",
        {"new_items": len(new_items), "healthy": len(healthy_sources), "failed": len(failed_sources)},
    )

    # Fetch article text for CBUAE items (static HTML, scrapeable locally)
    for item in new_items:
        if item.get("source_slug") == "cbuae_rulebook":
            article_text = fetch_module.fetch_article_text(
                item.get("link", ""), item.get("source_slug", "")
            )
            if article_text:
                item["article_text"] = article_text

    # Classify
    classified_items = classify_module.classify_items(new_items, policy_text)

    # Items that came back None from classify (API errors) — include unclassified
    classified_links = {it.get("link") for it in classified_items}
    unclassified_new = [it for it in new_items if it.get("link") not in classified_links]

    if unclassified_new:
        log_event(
            ERROR, WARN, COMPONENT,
            f"{len(unclassified_new)} item(s) could not be classified — included in digest without classification",
            {"count": len(unclassified_new)},
        )

    # Write digest
    digest_path = _write_digest(
        classified_items, unclassified_new, healthy_sources, failed_sources, fixture_sources, run_ts
    )

    log_event(
        HANDOFF, INFO, COMPONENT,
        f"Digest written: {digest_path.name}",
        {
            "digest": str(digest_path),
            "new_items": len(new_items),
            "classified": len(classified_items),
            "healthy_sources": len(healthy_sources),
            "failed_sources": len(failed_sources),
        },
    )

    # Human-readable summary to stdout
    print(f"Digest: {digest_path}")
    if failed_sources:
        print(f"WARNING — liveness FAILED for: {', '.join(failed_sources)}")
    print(
        f"Healthy sources: {len(healthy_sources)} | "
        f"New items: {len(new_items)} | "
        f"Classified: {len(classified_items)}"
    )


if __name__ == "__main__":
    run()
