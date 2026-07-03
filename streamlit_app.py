"""
RegMonitor · Agentic Governance Demo

Showcases baseline vs. multi-agent chain on the CBUAE cloud outsourcing item.
Reads pre-computed results from demo_results/ — no API calls.
"""

import json
import re
from pathlib import Path

import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RegMonitor · Agentic Governance",
    page_icon="⚖",
    layout="wide",
)

# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide Streamlit's rainbow decoration bar */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 1px solid #e2e8f0;
}

/* Main content padding */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Section label (small caps above headings) */
.section-label {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1a3a5c;
    margin-bottom: 0.15rem;
}

/* Approach column card */
.approach-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #1a3a5c;
    border-radius: 4px;
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    height: 100%;
}

/* Coverage pills */
.pill {
    display: inline-block;
    padding: 0.13rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}
.pill-covered     { background: #d1fae5; color: #065f46; }
.pill-partial     { background: #fef3c7; color: #92400e; }
.pill-not-covered { background: #fee2e2; color: #991b1b; }

/* Confabulation warning tag */
.confab-tag {
    display: inline-block;
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #f59e0b;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    font-size: 0.68rem;
    font-weight: 700;
    margin-left: 0.4rem;
    vertical-align: middle;
    white-space: nowrap;
}

/* Gap ratio display */
.gap-ratio-wrap { margin: 0.75rem 0 0.5rem 0; }
.gap-ratio-number {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a3a5c;
    line-height: 1;
}
.gap-ratio-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-top: 0.15rem;
}

/* Obligation list item */
.obl-row {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.87rem;
    color: #1e293b;
    line-height: 1.45;
}
.obl-row:last-child { border-bottom: none; }
.obl-pill-wrap { flex-shrink: 0; padding-top: 0.1rem; }

/* Divergence cards */
.div-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #1a3a5c;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.65rem;
}
.div-type {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.div-row {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    font-size: 0.87rem;
    color: #1e293b;
    margin-bottom: 0.35rem;
    line-height: 1.45;
}
.div-row:last-child { margin-bottom: 0; }
.div-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #94a3b8;
    flex-shrink: 0;
    min-width: 4.5rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-top: 0.1rem;
}
.div-empty {
    color: #94a3b8;
    font-style: italic;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    font-size: 0.75rem;
    font-family: monospace;
    color: #334155;
    margin-right: 0.25rem;
}

/* Regulation metadata */
.reg-source {
    color: #64748b;
    font-size: 0.875rem;
    margin: 0.1rem 0 0.75rem 0;
}
.reg-desc {
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
    max-width: 72ch;
    border-left: 3px solid #1a3a5c;
    padding-left: 1rem;
    margin: 0.5rem 0 0 0;
}

/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    color: #94a3b8;
    font-size: 0.8rem;
}

/* Validator note under obligation */
.validator-note {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
    font-style: italic;
}

/* Coverage breakdown row */
.cov-row {
    display: flex;
    gap: 1rem;
    margin: 0.6rem 0 0.25rem 0;
}
.cov-cell { text-align: center; }
.cov-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}
.cov-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-top: 0.1rem;
}
.divider-v {
    width: 1px;
    background: #e2e8f0;
    margin: 0 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "must", "shall", "is", "are", "be", "this", "that", "any", "all", "its",
    "fi", "institution", "institutions", "financial", "licensed", "ensure",
    "required", "require", "requires", "include", "including", "within", "from",
    "their", "which", "as", "at", "each", "every",
})

_COVERAGE_NORM = {
    "covered": "covered",
    "partial": "partial",
    "partially_covered": "partial",
    "not_covered": "not_covered",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _norm(v: str) -> str:
    return _COVERAGE_NORM.get(v, v or "unknown")


def _is_gap(coverage: str) -> bool:
    return _norm(coverage) in ("partial", "not_covered")


def _pill_html(coverage: str) -> str:
    n = _norm(coverage)
    if n == "covered":
        return '<span class="pill pill-covered">Covered</span>'
    if n == "partial":
        return '<span class="pill pill-partial">Partial</span>'
    return '<span class="pill pill-not-covered">Not covered</span>'


def _is_confabulated(obl: dict) -> bool:
    return (obl.get("validator") or {}).get("obligation_valid") is False


def compute_divergences(b_obls: list, m_obls: list, threshold: float = 0.18) -> list:
    b_tok = [(o, _tokens(o.get("description", ""))) for o in b_obls]
    m_tok = [(o, _tokens(o.get("description", ""))) for o in m_obls]

    candidates = []
    for bi, (_, bt) in enumerate(b_tok):
        for mi, (_, mt) in enumerate(m_tok):
            sim = _jaccard(bt, mt)
            if sim >= threshold:
                candidates.append((sim, bi, mi))
    candidates.sort(reverse=True)

    used_b: set = set()
    used_m: set = set()
    divergences = []

    for sim, bi, mi in candidates:
        if bi in used_b or mi in used_m:
            continue
        used_b.add(bi)
        used_m.add(mi)
        b, m = b_tok[bi][0], m_tok[mi][0]
        b_cov = _norm(b.get("coverage", ""))
        m_cov = _norm(m.get("coverage", ""))
        if b_cov != m_cov or _is_gap(b.get("coverage", "")) != _is_gap(m.get("coverage", "")):
            divergences.append({"type": "coverage_mismatch", "baseline": b, "multi_agent": m})

    for i, (o, _) in enumerate(b_tok):
        if i not in used_b:
            divergences.append({"type": "baseline_only", "baseline": o, "multi_agent": None})

    for i, (o, _) in enumerate(m_tok):
        if i not in used_m:
            divergences.append({"type": "multi_only", "baseline": None, "multi_agent": o})

    return divergences


# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    base = Path(__file__).parent / "demo_results"
    b_path = base / "baseline_result.json"
    m_path = base / "multiagent_result.json"
    if not b_path.exists() or not m_path.exists():
        return None, None
    return (
        json.loads(b_path.read_text(encoding="utf-8")),
        json.loads(m_path.read_text(encoding="utf-8")),
    )


baseline, multiagent = load_results()

if baseline is None or multiagent is None:
    st.error(
        "**demo_results/baseline_result.json** or **demo_results/multiagent_result.json** not found. "
        "Populate the `demo_results/` folder before running."
    )
    st.stop()


# ─── Section 1: REGULATION ────────────────────────────────────────────────────

item = baseline.get("item", {})
title = item.get("title", "Untitled regulation")
source = item.get("source", "")
link = item.get("link", "")
description = item.get("description", "")

st.markdown('<span class="section-label">Regulation</span>', unsafe_allow_html=True)
st.markdown(f"## {title}")

source_parts = []
if source:
    source_parts.append(source)
if link:
    source_parts.append(f'<a href="{link}" target="_blank" style="color:#1a3a5c;">View source ↗</a>')
if source_parts:
    st.markdown(
        f'<p class="reg-source">{" · ".join(source_parts)}</p>',
        unsafe_allow_html=True,
    )

if description:
    st.markdown(f'<p class="reg-desc">{description}</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()


# ─── Section 2: SIDE BY SIDE ──────────────────────────────────────────────────

st.markdown('<span class="section-label">Side by Side</span>', unsafe_allow_html=True)
st.markdown("### Approaches compared")

col_left, col_right = st.columns(2, gap="large")


def render_approach(col, label, result, approach):
    summary = result.get("summary", {})
    obls = result.get("obligations", [])

    if approach == "baseline":
        model_html = f'<span class="model-badge">{result.get("baseline_model", "—")}</span>'
        covered = summary.get("covered", 0)
        partial = summary.get("partial", 0)
        not_covered = summary.get("not_covered", 0)
        partial_label = "Partial"
    else:
        fp = result.get("first_pass_model", "—")
        val = result.get("validator_model", "—")
        model_html = (
            f'<span class="model-badge">{fp}</span>'
            f'<span style="color:#94a3b8;font-size:0.75rem;">chain</span>&nbsp;'
            f'<span class="model-badge">{val}</span>'
            f'<span style="color:#94a3b8;font-size:0.75rem;">validator</span>'
        )
        covered = summary.get("covered", 0)
        partial = summary.get("partially_covered", 0)
        not_covered = summary.get("not_covered", 0)
        partial_label = "Partial"

    total = summary.get("total", covered + partial + not_covered)
    gap_ratio = summary.get("gap_ratio", 0.0)

    with col:
        st.markdown(f"#### {label}")
        st.markdown(model_html, unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.85rem;color:#64748b;margin:0.5rem 0 0 0;">'
            f'<strong style="color:#1e293b;">{total}</strong> obligations identified</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="cov-row">'
            f'  <div class="cov-cell">'
            f'    <div class="cov-num" style="color:#065f46;">{covered}</div>'
            f'    <div class="cov-lbl">Covered</div>'
            f'  </div>'
            f'  <div class="divider-v"></div>'
            f'  <div class="cov-cell">'
            f'    <div class="cov-num" style="color:#92400e;">{partial}</div>'
            f'    <div class="cov-lbl">{partial_label}</div>'
            f'  </div>'
            f'  <div class="divider-v"></div>'
            f'  <div class="cov-cell">'
            f'    <div class="cov-num" style="color:#991b1b;">{not_covered}</div>'
            f'    <div class="cov-lbl">Not covered</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="gap-ratio-wrap">'
            f'  <div class="gap-ratio-number">{gap_ratio:.0%}</div>'
            f'  <div class="gap-ratio-label">Gap ratio</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#64748b;margin:1rem 0 0.25rem 0;">Obligations</p>',
            unsafe_allow_html=True,
        )

        rows_html = ""
        for obl in obls:
            confab = _is_confabulated(obl)
            confab_html = (
                '<span class="confab-tag">⚠ Could not be traced to source text.</span>'
                if confab else ""
            )
            rows_html += (
                f'<div class="obl-row">'
                f'  <span class="obl-pill-wrap">{_pill_html(obl.get("coverage", ""))}</span>'
                f'  <span>{obl.get("description", "")} {confab_html}</span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)


render_approach(col_left, "Baseline · Single call", baseline, "baseline")
render_approach(col_right, "Multi-Agent Chain · READER → MAPPER → GAP → VALIDATOR", multiagent, "multi")

st.divider()


# ─── Section 3: DIVERGENCES ───────────────────────────────────────────────────

st.markdown('<span class="section-label">Divergences</span>', unsafe_allow_html=True)
st.markdown("### Where the two approaches disagreed")

divergences = compute_divergences(
    baseline.get("obligations", []),
    multiagent.get("obligations", []),
)

if not divergences:
    st.info("No divergences found — both approaches agreed on all obligations and coverage verdicts.")
else:
    st.markdown(
        f'<p style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">'
        f'{len(divergences)} divergence(s) detected</p>',
        unsafe_allow_html=True,
    )

    _TYPE_LABELS = {
        "coverage_mismatch": "Coverage disagreement",
        "baseline_only": "Obligation found only by baseline",
        "multi_only": "Obligation found only by multi-agent chain",
    }

    for div in divergences:
        dtype = div["type"]
        b = div.get("baseline")
        m = div.get("multi_agent")
        confab = m is not None and _is_confabulated(m)
        confab_tag = (
            '<span class="confab-tag">⚠ Could not be traced to source text.</span>'
            if confab else ""
        )

        if dtype == "coverage_mismatch":
            b_content = (
                f'{_pill_html(b.get("coverage", ""))} '
                f'{b.get("description", "")}'
            )
            m_content = (
                f'{_pill_html(m.get("coverage", ""))} '
                f'{m.get("description", "")} {confab_tag}'
            )
        elif dtype == "baseline_only":
            b_content = (
                f'{_pill_html(b.get("coverage", ""))} '
                f'{b.get("description", "")}'
            )
            m_content = '<span class="div-empty">Did not identify this obligation</span>'
        else:
            b_content = '<span class="div-empty">Did not identify this obligation</span>'
            m_content = (
                f'{_pill_html(m.get("coverage", ""))} '
                f'{m.get("description", "")} {confab_tag}'
            )

        st.markdown(
            f'<div class="div-card">'
            f'  <div class="div-type">{_TYPE_LABELS.get(dtype, dtype)}</div>'
            f'  <div class="div-row">'
            f'    <span class="div-label">Baseline</span>'
            f'    <span>{b_content}</span>'
            f'  </div>'
            f'  <div class="div-row">'
            f'    <span class="div-label">Chain</span>'
            f'    <span>{m_content}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">'
    'Built by Yusuf Yasin · yusuf-yasin.com · '
    'The findings shown are from a practitioner experiment in AI governance, '
    'not a production compliance tool.'
    '</div>',
    unsafe_allow_html=True,
)
