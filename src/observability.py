"""
Observability spine for regmonitor. Every module routes surprising events here.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------
SOURCE_FAILURE = "SOURCE_FAILURE"
PARSE_OK = "PARSE_OK"
NEW_ITEM = "NEW_ITEM"
CLASSIFICATION = "CLASSIFICATION"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
HANDOFF = "HANDOFF"
VALIDATOR_DISAGREEMENT = "VALIDATOR_DISAGREEMENT"
BASELINE_DIVERGENCE = "BASELINE_DIVERGENCE"
ERROR = "ERROR"

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------
INFO = "info"
WARN = "warn"
CRITICAL = "critical"

_VALID_SEVERITIES = {INFO, WARN, CRITICAL}

# ---------------------------------------------------------------------------
# Log path (resolved relative to this file so it works from any cwd)
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "observation_log.jsonl"


def log_event(
    event_type: str,
    severity: str,
    component: str,
    message: str,
    context: dict | None = None,
) -> None:
    """Append one JSON-line record to logs/observation_log.jsonl.

    Args:
        event_type: One of the module-level constants (SOURCE_FAILURE, etc.).
        severity:   "info", "warn", or "critical".
        component:  Name of the calling module / subsystem (e.g. "fetch", "classify").
        message:    Human-readable description of the event.
        context:    Arbitrary key-value pairs for structured querying.
    """
    if severity not in _VALID_SEVERITIES:
        raise ValueError(f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "severity": severity,
        "component": component,
        "message": message,
        "context": context or {},
    }

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
