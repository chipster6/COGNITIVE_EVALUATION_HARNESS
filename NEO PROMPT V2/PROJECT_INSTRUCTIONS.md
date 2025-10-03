
Project Instructions — Master Lifecycle Checklist

  

---

  

This document defines the end-to-end process for adding any new feature, pillar, or probe family to NeoPrompt.

It is reusable: replace tokens like <FEATURE>, <PILLAR_ID>, <FAMILY_ID>, <VERSION> with the actual identifiers.


---



0) Intake & Planning

- Define <FEATURE> one-liner (problem, why now, success metric).
- Assign owner & reviewers (Engineering, Research, QA).
- Timebox: start date, code freeze, release.
- Create Linear/GitHub Epic with this checklist pasted in.


---


1) Research
(docs/research/<FEATURE>/)

- Literature scan (3–7 key references) + 1-page summary.
- Competency definition: what this measures vs. doesn’t.
- Failure modes (≥5) + examples; map to detectable signals.
- Task taxonomy: families, bands A–E, inputs/outputs.
- Metrics: point metrics (accuracy/Brier/ECE/etc.), reliability plan (α or KR-20), uncertainty (bootstrap CI).
- Data strategy: synthetic generation plan, seeded RNG, domain shifts, distractors.
- Safety/privacy: PII handling, logging policy (hashes only), red-team notes.
- Determinism plan: seed derivation, seed versioning, reproducibility knobs.
- Sign-off: Engineering + Research.

---

2) Product / Spec:
(docs/specs/<FEATURE>.md)

- Scope & non-goals.
- I/O contracts (strict JSON shapes), examples (valid/invalid).
- Scoring formulas, penalties, weights, aggregation.
- Repair policy: single JSON repair attempt rules.
- Band definitions (A–E parameters).
- Acceptance criteria (what “done” means).
- Telemetry fields (latency, token counts, seed hash).


---

3)Design
docs/design/<FEATURE>.md

- Module diagram (generator → probe → runner → scorer → reports).
- File map (where each file will live).
- Error handling (schema validation, repair, hard fails).
- No-network guarantee (invariant tests + sockets patch).
- Extensibility (new families, more bands).
- Design review sign-off.

  

---

4) Repo Scaffolding / Conventions

- Naming: <PILLAR_ID> lower_snake, <FAMILY_ID> with _v1.
- Version bump plan (SEED_VERSION + package version).
- Schema locations under neoprompt/schemas/.

  

---

5) Schemas & Example

- Response schema → neoprompt/schemas/<FAMILY_ID>.json.
- Suite base → examples/<FEATURE>/suite.yml.
- Expanded suite → examples/<FEATURE>/suite.expanded.json.
- Fixtures → tests/fixtures/corpus/*.txt (if needed).

  

---

  6) Implementation

- Seed utils (derive from derive_probe_seed, hash_seed_for_storage).
- Generator → neoprompt/generators/<FAMILY_ID>.py.
- Probe class → neoprompt/probes/<PILLAR_ID>.py.
- Scorer → neoprompt/scorers/<PILLAR_ID>.py (if separated).
- Aggregator with bootstrap CI.
- Runner hookups (executor, validator, manifest support).
- CLI integration (e.g. examples/run_evaluation.py).


---

7) Tests

- Unit: determinism (test_seed_determinism.py).
- Unit: generator (test_<FAMILY_ID>_generator.py).
- Unit: schema (test_schema_<FAMILY_ID>.py).
- Unit: scorer (test_<PILLAR_ID>_scorer.py).
- Integration: end-to-end (test_<FEATURE>_e2e.py) → generator → executor(stub) → validator → scorer → report.
- Invariants: no-network, seed hash uniqueness.
- Coverage ≥80%.

  
---

8) Reports & Artifacts

- Artifact manifest validated.
- Developer JSON report includes pillar metrics, CIs, counts.
- Enterprise Markdown report includes table rows and status icons.
- Golden outputs saved under examples/<FEATURE>/goldens/.

  

---

9) Docs & Onboarding

- Update README quickstart with <FEATURE> usage.
- Developer guide: how to add a new probe family (reference this checklist).
- CHANGELOG entry.
- Glossary update.

  

---

10) CI/CD & Quality Gates

- GitHub Actions (.github/workflows/ci.yml) runs lint, mypy, tests, coverage gate.
- Optional: release workflow (release.yml) builds wheel, publishes on tag.
- Branch protection: required checks enabled.
- PR template enforces: tests, docs, schema, examples, golden updated.

  


---

11) Release & Versioning

- Bump package version <VERSION>.
- Tag v<VERSION> with release notes.
- Attach sample reports (JSON + Markdown).

  

---

12) Handoff & Ops

- Runbook: how to run locally, regenerate probes, interpret scores.
- Troubleshooting: common failures (invalid JSON, seed drift).
- Ownership: update CODEOWNERS.
- Backfill: run on baseline model(s), commit artifacts under examples/<FEATURE>/runs/.

  
---

13) SaaS Layer Hooks (future) 

- API contract for ingesting manifests/scores.
- Dashboard tile definitions.
- Multi-run comparison views.

  

---

14) Adversarial Extensions (future)

- Attack variants doc + probe packs.
- Safety FSM rules updated.
- Degradation/AUC metrics wired to pillar view.

  


---

Reusable File/Path Template

  

```
docs/

  research/<FEATURE>/

    01_overview.md

    02_failure_modes.md

    03_metrics_reliability.md

    04_data_generation.md

  specs/<FEATURE>.md

  design/<FEATURE>.md

  

neoprompt/

  generators/<FAMILY_ID>.py

  probes/<PILLAR_ID>.py

  scorers/<PILLAR_ID>.py

  schemas/<FAMILY_ID>.json

  

tests/

  unit/test_<FAMILY_ID>_generator.py

  unit/test_schema_<FAMILY_ID>.py

  unit/test_<PILLAR_ID>_scorer.py

  integration/test_<FEATURE>_e2e.py

  invariants/test_no_network.py

  fixtures/corpus/...

  

examples/<FEATURE>/

  suite.yml

  suite.expanded.json

  run_evaluation.py

  goldens/

  runs/


```
  

  

PR Template Snippet

  

  

- Scope & non-goals documented (docs/specs/<FEATURE>.md).
- Schemas added/updated (neoprompt/schemas/<FAMILY_ID>.json).
- Generator/probe/scorer implemented.
- Unit + integration tests added; coverage ≥80%.
- Examples & goldens updated.
- README/docs updated.
- Determinism & no-network tests pass.
- CHANGELOG entry & version bump (if releasing).

  

---
  

Definition of Done (per feature)

- Deterministic probe generation with bands.
- Strict schema validation.
- Scorer + aggregator with bootstrap CIs.
- ≥1 end-to-end integration test.
- No network access.
- Research/spec/design docs committed.
- Examples + golden outputs included.
- CI green with coverage gate.
- Release notes prepared (if tagging).

  

---


This file should live as PROJECT_INSTRUCTIONS.md in the repo root.

