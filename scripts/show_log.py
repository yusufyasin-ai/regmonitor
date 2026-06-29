#!/usr/bin/env python3
"""Pretty-print observation_log.jsonl, optionally filtered by event_type and/or severity."""

import argparse
import json
import sys
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "logs" / "observation_log.jsonl"

# ANSI colours keyed by severity
_COLOURS = {
    "info": "\033[36m",      # cyan
    "warn": "\033[33m",      # yellow
    "critical": "\033[31m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colour(text: str, severity: str) -> str:
    return f"{_COLOURS.get(severity, '')}{text}{_RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print regmonitor observation log."
    )
    parser.add_argument(
        "--event-type", "-e",
        metavar="TYPE",
        help="Filter to a specific event_type (e.g. SOURCE_FAILURE).",
    )
    parser.add_argument(
        "--severity", "-s",
        metavar="LEVEL",
        choices=["info", "warn", "critical"],
        help="Filter to a specific severity level.",
    )
    parser.add_argument(
        "--no-colour", "-n",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    parser.add_argument(
        "--log", "-l",
        metavar="PATH",
        default=str(_LOG_PATH),
        help=f"Path to log file (default: {_LOG_PATH}).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    use_colour = not args.no_colour and sys.stdout.isatty()
    count = 0

    with log_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[line {lineno}] JSON parse error: {exc}", file=sys.stderr)
                continue

            if args.event_type and rec.get("event_type") != args.event_type:
                continue
            if args.severity and rec.get("severity") != args.severity:
                continue

            sev = rec.get("severity", "")
            ts = rec.get("timestamp", "")
            etype = rec.get("event_type", "")
            comp = rec.get("component", "")
            msg = rec.get("message", "")
            ctx = rec.get("context", {})

            header = f"{ts}  [{etype}]  {comp}  {sev.upper()}"
            if use_colour:
                header = _colour(f"{_BOLD}{header}", sev)

            print(header)
            print(f"  {msg}")
            if ctx:
                for k, v in ctx.items():
                    print(f"    {k}: {v}")
            print()
            count += 1

    if count == 0:
        print("No matching log entries.")


if __name__ == "__main__":
    main()
