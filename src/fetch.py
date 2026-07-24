"""Fetch regulatory items from configured HTML sources.

Each source is fully described by its config/sources.yaml entry —
no per-source code required. Parser hints (CSS selectors + field extractors)
drive BeautifulSoup. A liveness check gates every source: if the parsed
item count is below min_items we emit SOURCE_FAILURE/critical and exclude
the source from the run. A silent empty parse is never reported as clean.

Fixture support: when a source sets fixture_path, the local HTML file is read
instead of making a network request. The same parser and liveness check run
identically — a fixture that yields zero items fires SOURCE_FAILURE just as a
live source would. The canonical url is still used as the base URL for
resolving relative links inside the fixture.

State lives in state/<slug>.json as a list of short SHA-256 hashes (title+link).
Per-source isolation means a new source starts fresh without touching others.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from observability import (
    SOURCE_FAILURE, PARSE_OK, NEW_ITEM, ERROR,
    INFO, WARN, CRITICAL,
    log_event,
)

COMPONENT = "fetch"

_ROOT = Path(__file__).parent.parent
_STATE_DIR = _ROOT / "state"

_DEFAULT_TIMEOUT = 30
_DEFAULT_MIN_ITEMS = 1
_HEADERS = {
    "User-Agent": (
        "RegMonitor/1.0 (regulatory-compliance monitoring bot; "
        "contact the operator if you have questions)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# State helpers (per-source JSON list of hashes)
# ---------------------------------------------------------------------------

def _slug_from_source(source: dict) -> str:
    raw = source.get("slug") or re.sub(r"[^a-z0-9]+", "_", source["name"].lower()).strip("_")
    return raw


def _state_path(slug: str) -> Path:
    return _STATE_DIR / f"{slug}.json"


def _load_state(slug: str) -> set[str]:
    path = _state_path(slug)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        log_event(ERROR, WARN, COMPONENT, f"Could not read state for {slug}: {exc}", {"slug": slug})
        return set()


def _save_state(slug: str, seen: set[str]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(slug).write_text(
        json.dumps(sorted(seen), indent=2), encoding="utf-8"
    )


def _item_hash(title: str, link: str) -> str:
    key = f"{title.strip()}||{link.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_field(row, field_cfg: dict, base_url: str) -> str:
    """Pull one field value from a BeautifulSoup element using config hints."""
    selector = field_cfg.get("selector", "")
    attr = field_cfg.get("attr", "text")

    target = row.select_one(selector) if selector else row
    if target is None:
        return ""

    if attr == "text":
        return target.get_text(" ", strip=True)

    value = target.get(attr, "") or ""
    if attr == "href" and value and not value.startswith(("http://", "https://")):
        value = urljoin(base_url, value)
    return value.strip()


# ---------------------------------------------------------------------------
# HTML acquisition (live vs fixture)
# ---------------------------------------------------------------------------

def _get_html(source: dict) -> tuple[str, bool]:
    """Acquire raw HTML for a source. Returns (html_text, is_fixture).

    Raises OSError for fixture read failures, requests.RequestException for
    live fetch failures — the caller logs SOURCE_FAILURE and returns early.

    When FETCH_MODE env var is 'live', always fetch from url even if
    fixture_path is configured. This lets GitHub Actions run live while
    local development uses fixtures.
    """
    import os
    fetch_mode = os.environ.get("FETCH_MODE", "fixture")
    fixture_rel = source.get("fixture_path", "")

    if fixture_rel and fetch_mode != "live":
        path = (_ROOT / fixture_rel).resolve()
        return path.read_text(encoding="utf-8"), True

    resp = requests.get(source["url"], headers=_HEADERS, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.text, False

# ---------------------------------------------------------------------------
# Single-source fetch + parse
# ---------------------------------------------------------------------------

def _fetch_source(source: dict) -> tuple[list[dict], bool, bool]:
    """Fetch and parse one source. Returns (items, liveness_ok, is_fixture).

    liveness_ok is False when acquisition fails OR when the parsed count is
    below min_items. SOURCE_FAILURE is logged in both cases. The liveness
    check is identical regardless of whether the source is live or a fixture.
    """
    name = source["name"]
    url = source.get("url", "")
    parser = source.get("parser", {})
    min_items: int = source.get("liveness", {}).get("min_items", _DEFAULT_MIN_ITEMS)
    slug = _slug_from_source(source)

    # --- Acquire HTML ---
    try:
        html_text, is_fixture = _get_html(source)
    except OSError as exc:
        fixture_path = source.get("fixture_path", "")
        log_event(
            SOURCE_FAILURE, CRITICAL, COMPONENT,
            f"Cannot read fixture for {name}: {exc}",
            {"source": name, "fixture_path": fixture_path},
        )
        return [], False, True
    except requests.Timeout:
        log_event(
            SOURCE_FAILURE, CRITICAL, COMPONENT,
            f"Timeout fetching {name} after {_DEFAULT_TIMEOUT}s",
            {"source": name, "url": url},
        )
        return [], False, False
    except requests.RequestException as exc:
        log_event(
            SOURCE_FAILURE, CRITICAL, COMPONENT,
            f"HTTP error fetching {name}: {exc}",
            {"source": name, "url": url, "error": str(exc)},
        )
        return [], False, False

    source_label = f"fixture:{source.get('fixture_path', '')}" if is_fixture else url

    # --- Validate selector config ---
    item_selector = parser.get("item_selector", "")
    fields_cfg = parser.get("fields", {})

    if not item_selector or item_selector == "TODO":
        log_event(
            SOURCE_FAILURE, CRITICAL, COMPONENT,
            f"item_selector not configured for {name} — skipping",
            {"source": name},
        )
        return [], False, is_fixture

    # --- Parse (identical for live and fixture) ---
    soup = BeautifulSoup(html_text, "html.parser")
    rows = soup.select(item_selector)

    # base_url for urljoin: always the canonical url, even for fixtures,
    # so relative hrefs in the fixture resolve to the correct live domain.
    base_url = url

    items: list[dict] = []
    for row in rows:
        title = _extract_field(row, fields_cfg.get("title", {}), base_url)
        date = _extract_field(row, fields_cfg.get("date", {}), base_url)
        link = _extract_field(row, fields_cfg.get("link", {"attr": "href"}), base_url)

        if not title and not link:
            continue  # skip header rows or spacers accidentally matched

        items.append(
            {
                "title": title,
                "date": date,
                "link": link,
                "source": name,
                "source_slug": slug,
                "tags": source.get("tags", []),
            }
        )

    # --- Liveness check (same rule for live and fixture) ---
    if len(items) < min_items:
        log_event(
            SOURCE_FAILURE, CRITICAL, COMPONENT,
            (
                f"Liveness FAILED for {name}: parsed {len(items)} item(s), "
                f"threshold is {min_items}. "
                "Zero items or a broken selector must not be treated as 'no new circulars'."
            ),
            {
                "source": name, "parsed": len(items),
                "min_items": min_items, "source_ref": source_label,
                "is_fixture": is_fixture,
            },
        )
        return items, False, is_fixture

    log_event(
        PARSE_OK, INFO, COMPONENT,
        f"Parsed {len(items)} items from {name}"
        + (" [fixture]" if is_fixture else " [live]"),
        {
            "source": name, "item_count": len(items),
            "source_ref": source_label, "is_fixture": is_fixture,
        },
    )
    return items, True, is_fixture


# ---------------------------------------------------------------------------
# Diff against state
# ---------------------------------------------------------------------------

def _diff_and_commit(items: list[dict], slug: str) -> list[dict]:
    """Return only genuinely new items; persist updated state immediately.

    State is committed before classification so that a classification failure
    on run N does not re-surface the same items on run N+1 (which would cause
    duplicate digest entries). Operators can query LOW_CONFIDENCE/ERROR logs
    to identify items that need manual review.
    """
    seen = _load_state(slug)
    new_items = []

    for item in items:
        h = _item_hash(item.get("title", ""), item.get("link", ""))
        if h not in seen:
            item["_hash"] = h
            new_items.append(item)

    if new_items:
        updated = seen | {it["_hash"] for it in new_items}
        _save_state(slug, updated)

    for item in new_items:
        log_event(
            NEW_ITEM, INFO, COMPONENT,
            f"New item: {item['title'][:100] or item['link']}",
            {
                "source": item["source"],
                "title": item["title"],
                "link": item["link"],
                "date": item.get("date", ""),
            },
        )

    return new_items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all(sources: list[dict]) -> tuple[list[dict], list[str], list[str], set[str]]:
    """Fetch all enabled sources.

    Returns:
        new_items       – flat list of items not seen in previous runs
        healthy         – source names that passed liveness
        failed          – source names that failed liveness (excluded from digest)
        fixture_sources – subset of healthy that were read from local fixtures
    """
    all_new: list[dict] = []
    healthy: list[str] = []
    failed: list[str] = []
    fixture_sources: set[str] = set()

    for source in sources:
        if not source.get("enabled", True):
            continue

        name = source["name"]
        slug = _slug_from_source(source)

        items, liveness_ok, is_fixture = _fetch_source(source)

        if not liveness_ok:
            failed.append(name)
            continue

        healthy.append(name)
        if is_fixture:
            fixture_sources.add(name)

        new_items = _diff_and_commit(items, slug)
        all_new.extend(new_items)

    return all_new, healthy, failed, fixture_sources


# ---------------------------------------------------------------------------
# Article text fetching (CBUAE only — static HTML, text in <td> elements)
# CBB Thomson Reuters pages are JavaScript-rendered and cannot be scraped.
# ---------------------------------------------------------------------------

def fetch_article_text(link: str, source_slug: str) -> str | None:
    """Fetch and extract the main body text from a CBUAE article page.

    Only runs for cbuae_rulebook sources. Returns cleaned plain text or None
    on failure. Never crashes the pipeline — all errors are logged and swallowed.
    """
    if source_slug != "cbuae_rulebook":
        return None

    try:
        resp = requests.get(link, headers=_HEADERS, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log_event(
            ERROR, WARN, COMPONENT,
            f"ARTICLE_FETCH_FAILED: could not fetch {link}: {exc}",
            {"link": link, "source_slug": source_slug, "error": str(exc)},
        )
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, headers, footers, scripts, styles
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Extract text from <td> elements which hold the regulation body
    tds = soup.find_all("td")
    text_parts = []
    for td in tds:
        text = td.get_text(" ", strip=True)
        if len(text) > 50:  # skip short cells (numbers, labels, etc.)
            text_parts.append(text)

    article_text = "\n\n".join(text_parts).strip()

    if not article_text:
        log_event(
            ERROR, WARN, COMPONENT,
            f"ARTICLE_FETCH_FAILED: no text extracted from {link}",
            {"link": link, "source_slug": source_slug},
        )
        return None

    log_event(
        "ARTICLE_FETCH", INFO, COMPONENT,
        f"Fetched article text from {link} ({len(article_text)} chars)",
        {"link": link, "source_slug": source_slug, "char_count": len(article_text)},
    )
    return article_text
