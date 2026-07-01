# RegMonitor — Agentic Regulatory Monitoring & Gap Analysis

An agentic regulatory monitoring system built to track publications from GCC regulators, alert on new obligations, and check whether existing policies cover them.

Built as a practitioner experiment in AI governance — not to demonstrate that agentic AI works, but to study where and how it fails.

---

## What it does

RegMonitor runs in two modes, designed to be compared against each other:

**Single-agent baseline** — one agent reads a new regulation and assesses whether existing bank policies cover the obligations it imposes.

**Multi-agent chain** — four specialised agents working in sequence:
1. **Reader** — extracts the discrete obligations from the regulation
2. **Mapper** — checks each obligation against the bank's policy documents
3. **Gap agent** — drafts findings for obligations not adequately covered
4. **Validator** — independently re-reads the source regulation and policy to check the chain's work

A comparison harness runs the same regulation through both systems and logs every divergence — where they disagreed on which obligations exist, and where they reached different coverage verdicts.

---

## What the experiment found

Running the same CBUAE cloud outsourcing regulation through both systems produced materially different outputs, despite using the same underlying AI model.

**The agent design produced the finding, not just the model.** The single-agent and multi-agent systems identified different obligation sets from the same regulation. Three obligations found by one system were missed entirely by the other, and vice versa.

**Confidence and accuracy moved in opposite directions.** The system flagged lowest confidence on a minor remediation deadline. Its most confident calls included two coverage assessments that were wrong.

**A validator agent only adds value if it reads the original source.** The validator in the multi-agent chain had structural independence (a different, more capable model) but not epistemic independence — in some runs it reviewed the chain's summaries rather than the source regulation, and passed through findings it should have caught.

All findings are logged in `logs/observation_log.jsonl` with ISO-8601 timestamps, making them auditable and citable.

---

## Why this matters for AI governance in banking

These findings surface three governance questions that most AI risk frameworks have not yet answered:

1. **Who owns the output when multiple agents produce it?** Accountability cannot be assigned to a model alone when the finding is a product of agent architecture.
2. **What does independence mean for a validator agent?** Structural independence (a different model) is not the same as epistemic independence (reading the original source). The Three Lines of Defense model needs to account for this distinction.
3. **How should banks interpret AI confidence scores?** A high confidence score is not a reliable signal of accuracy in an agentic context. Human review cannot be triaged on confidence alone.

These questions are explored further in a forthcoming paper: *Governing Artificial Intelligence in GCC Banking* (due August 2026) and will be discussed at the MEBIS 2026 panel: *From AI Assistants to AI Agents: Is Autonomous Banking Becoming Reality?*

---

## Architecture

```
regmonitor/
├── config/          # Regulator sources (CBB, CBUAE, extensible to SAMA, QCB, CBO)
├── fixtures/        # Local HTML fixtures for offline/sandbox runs
├── policy/          # Policy documents to check obligations against
├── src/
│   ├── observability.py   # Structured event logger (the evidence spine)
│   ├── fetch.py           # Source fetching with liveness checking
│   ├── classify.py        # Single-step relevance classification
│   ├── analyze.py         # Single-agent baseline + multi-agent chain
│   └── monitor.py         # Main orchestrator
├── scripts/
│   ├── compare.py         # Comparison harness: baseline vs multi-agent
│   └── show_log.py        # Log viewer filtered by event type and severity
└── logs/
    └── observation_log.jsonl   # Dated evidence log
```

---

## Setup

```bash
git clone https://github.com/yusufyasin-ai/regmonitor.git
cd regmonitor
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python3 src/monitor.py
```

To run the comparison harness:

```bash
python3 scripts/compare.py
```

Regulators currently covered via local fixtures: **CBUAE** (rulebook updates) and **CBB** (circulars). The config file is structured to add SAMA, QCB, and CBO without code changes.

---

## Key governance concepts demonstrated

| Concept | Where it appears in the code |
|---|---|
| Silent source failure | `fetch.py` — liveness check separate from content check |
| Architecture-dependent outputs | `scripts/compare.py` — baseline vs chain divergence |
| Confidence-accuracy decoupling | `logs/observation_log.jsonl` — LOW_CONFIDENCE events |
| Epistemic vs structural independence | `analyze.py` — validator role design |
| Accountability gap | Handoff logs at every agent boundary |

---

## Related

- **Site:** [yusuf-yasin.com](https://yusuf-yasin.com)
- **LinkedIn:** [linkedin.com/in/yusufyasin](https://www.linkedin.com/in/yusufyasin/)
- **Paper:** *Governing Artificial Intelligence in GCC Banking* — forthcoming August 2026
