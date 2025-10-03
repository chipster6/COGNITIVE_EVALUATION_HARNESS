# NeoPrompt Lifecycle — **Checkpoint M1: Research & Definition**

Pattern: Master doc + modular sub-artifacts under docs/lifecycle/1-research/

Purpose

- Transform M0 intake into a research-grounded, testable definition with machine-readable hypotheses, evaluation design, baselines, governance mapping, data provenance, uncertainty handling, and review gates. Output is CI-validated and ready for M2 design.

Outcomes

- Literature review and prior-art survey with citations and links
- Falsifiable, measurable hypotheses in YAML with targets and tolerances
- Evaluation design for probe families, metrics, scoring, and reproducibility
- Baseline results in JSON with confidence intervals and environment details
- Governance mapping to Cognitive Pillars and risk domains
- Data provenance, licensing, and compliance notes
- Uncertainty and reliability methods (CIs, power/sample-size)
- Scope boundaries and assumptions to prevent creep
- Review checklist and PR policy for “Ready for M2”

Deliverables (files in this checkpoint)

- docs/lifecycle/1-research/<FEATURE>_research.md — Research Notes
- docs/lifecycle/1-research/<FEATURE>_hypotheses.yaml — Hypotheses
- docs/lifecycle/1-research/<FEATURE>_eval_design.md — Evaluation Design
- docs/lifecycle/1-research/<FEATURE>_baseline.json — Baseline Metrics
- docs/lifecycle/1-research/<FEATURE>_governance.yaml — Pillar/Risk map (+ crosslinks)
- docs/lifecycle/1-research/<FEATURE>_data_provenance.md — Data & licensing
- docs/lifecycle/1-research/<FEATURE>_uncertainty.md — Stats & reliability plan
- docs/lifecycle/1-research/<FEATURE>_scope.md — In/out of scope and assumptions
- docs/lifecycle/1-research/<FEATURE>_review_checklist.md — Reviewer checklist
- docs/lifecycle/1-research/<FEATURE>_decision_log.md — Research decisions & alternatives

Acceptance criteria

- ≥6 references; ≥2 hypotheses with numeric targets; eval design; baselines with 95% CIs and env; governance mapping; provenance; uncertainty plan; scope doc; CI green; PR approved by Research + QA; tag research-<FEATURE>-approved.
