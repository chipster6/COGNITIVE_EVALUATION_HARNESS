Here’s a **re-usable, end-to-end, chronological checklist** you can copy for **any feature/pillar/segment**. Replace tokens like < FEATURE >, <PILLAR_ID>, <FAMILY_ID>, < VERSION >.

---

# **Master Lifecycle Checklist (Reusable)**

## **0) Intake & Planning**

- **Define <FEATURE> one-liner** (problem, why now, success metric).

- **Owner & reviewers assigned** (Eng, Research, QA).

- **Timebox & milestone** (start, code freeze, release).

- **Create Linear/GitHub Epic** with this checklist pasted in.

## **1) Research (add to /docs/research/<FEATURE>.md)

- **Literature scan** (3–7 key references) + 1-page summary.

- **Competency definition**: what this measures vs. doesn’t.

- **Failure modes** (≥5) + examples; mapping to detectable signals.

- **Task taxonomy** (families, bands A–E, inputs/outputs).

- **Metrics**: point metrics (accuracy/Brier/ECE/etc.), **reliability** plan (α or KR-20), **uncertainty** (bootstrap CI).

- **Data strategy**: synthetic generation plan, seeded RNG, domain shifts, distractors.

- **Safety/privacy**: PII handling, logging policy (hashes only), red-team notes.

- **Determinism plan**: seed derivation, seed versioning, reproducibility knobs.

- **Research sign-off** (Eng + Research).

## **2) Product/Spec (add to /docs/specs/<FEATURE>.md**)

- **Scope & non-goals** (bullet list).

- **I/O contracts** (strict JSON shapes), examples (valid/invalid).

- **Scoring spec**: formulas, penalties, weights, aggregation.

- **Repair policy**: single JSON repair attempt rules.

- **Band definitions** (A–E parameters).

- **Acceptance criteria** (what “done” means).

- **Telemetry fields** (latency, token counts, seed hash).

## **3) Design (add to /docs/design/<FEATURE>.md)

- **Module diagram** (generator → probe → runner → scorer → reports).

- **File map** (paths below filled for this feature).

- **Error handling** (schema validation, repair, hard fails).

- **No-network guarantee** (invariant tests + sockets patch).

- **Extensibility** (new families, more bands).

- **Design review sign-off**.

## **4) Repo Scaffolding / Conventions Naming**: <PILLAR_ID> lower_snake, <FAMILY_ID> with_v1

- **Version** bump plan (SEED_VERSION + package version).

- **Schema locations** under neoprompt/schemas/.

## **5) Schemas & Examples**

- **Response schema** neoprompt/schemas/<FAMILY_ID>.json (strict).

- **Suite base** examples/<FEATURE>/suite.yml (cases, bands).

- **Expanded sample** examples/<FEATURE>/suite.expanded.json (golden).

- **Fixtures** (e.g., tests/fixtures/corpus/*.txt if needed).

## **6) Implementation**

- **Seed utils** (if new needs) use existing derive_probe_seed, hash_seed_for_storage.

- **Generator** neoprompt/generators/<FAMILY_ID>.py

  - Deterministic probes, bands, distractors, metadata.

- **Probe class** neoprompt/probes/<PILLAR_ID>.py

  - ProbeGenerator implementation, schema path, prompt assembly.

- **Scorer** neoprompt/probes/<PILLAR_ID>.py or neoprompt/scorers/<PILLAR_ID>.py

  - Per-probe score + pillar aggregator (+ bootstrap CI).

- **Runner hookups** (if new): ensure executor, validator, manifest support family.

- **CLI**: examples/run_evaluation.py can run this family/case.

## **7) Tests (must pass locally)**

- **Unit—determinism** (tests/unit/test_seed_determinism.py add cases for <FEATURE>).

- **Unit—generator** (tests/unit/test_<FAMILY_ID>_generator.py)

  - Same input → same output; band spread; no network.

- **Unit—schema** (tests/unit/test_schema_<FAMILY_ID>.py)

  - Valid/invalid examples; repair once.

- **Unit—scorer** (tests/unit/test_<PILLAR_ID>_scorer.py)

  - Perfect, partial, edge, penalties.

- **Integration E2E** (tests/integration/test_<FEATURE>_e2e.py)

  - generator → executor(stub) → validator → scorer → report; asserts metrics.

- **Invariants** (tests/invariants/test_no_network.py, determinism, seed hash uniqueness).

- **Coverage ≥80%**.

## **8) Reports & Artifacts**

- **Artifact manifest** entries validated (schemas/artifact_manifest.json).

- **Developer JSON report** includes pillar metrics, CIs, counts.

- **Enterprise MD** includes table rows for this pillar, status icons.

- **Golden outputs** saved under examples/<FEATURE>/goldens/.

## **9) Docs & Onboarding**

- Update **README** quickstart with <FEATURE> usage snippet.

- **Developer guide**: “Add a new probe family” steps referencing this checklist.

- **CHANGELOG** entry (Added <FEATURE>).

- **Glossary** update if new terms.

## **10) CI/CD & Quality Gates**

- **GH Actions** (.github/workflows/ci.yml) runs lint, mypy, tests, coverage gate.

- **release.yml** (optional): build wheel, publish on tag.

- **Branch protection**: required checks enabled.

- **PR template** enforces: tests, docs, schema, examples, golden updated.

## **11) Release & Versioning**

- Bump **package version** <VERSION>.

- Tag v<VERSION>; attach release notes (features, schemas added).

- Upload sample **report.dev.json** and **enterprise.md** from a demo run.

## **12) Handoff & Ops**

- **Runbook**: how to run locally, regenerate probes, interpret scores.

- **Troubleshooting**: common failures (invalid JSON, seed drift).

- **Ownership**: CODEOWNERS updated; escalation path.

- **Backfill**: run on at least one baseline model, commit artifacts under examples/<FEATURE>/runs/.

## **13) SaaS Layer Hooks (if applicable later)**

- API contract for ingesting manifests/scores.

- UI dashboard tiles/queries for this pillar.

- Multi-run comparison views + filters.

## **14) Adversarial Extensions (optional)**

- Attack variants doc + probe packs.

- Safety FSM rules updated.

- Degradation/AUC metrics wired to pillar view.

---

## **Reusable File/Path Template (fill tokens)**

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
  scorers/<PILLAR_ID>.py   (if separated)
  schemas/<FAMILY_ID>.json

tests/
  unit/test_<FAMILY_ID>_generator.py
  unit/test_schema_<FAMILY_ID>.py
  unit/test_<PILLAR_ID>_scorer.py
  integration/test_<FEATURE>_e2e.py
  invariants/test_no_network.py
  fixtures/corpus/...(if needed)

examples/<FEATURE>/
  suite.yml
  suite.expanded.json
  run_evaluation.py
  goldens/
  runs/
```

---

## **PR Template Snippet (paste into repo once, reuse per feature)**

- Scope & non-goals documented (docs/specs/<FEATURE>.md)

- Schemas added/updated (neoprompt/schemas/<FAMILY_ID>.json)

- Generator/probe/scorer implemented (neoprompt/...)

- Unit + integration tests added; coverage ≥80%

- Examples & goldens updated (examples/<FEATURE>/)

- README/docs updated

- Determinism & no-network tests pass

- CHANGELOG entry & version bump (if releasing)

---

## **Definition of Done (per feature)**

- Deterministic probe generation with bands, strict schemas, scorer + aggregator with bootstrap CIs, ≥1 E2E test, no network access, docs/spec/design committed, examples/goldens provided, CI green with coverage gate, release notes prepared (if tagging).

---

If you want, I can also output this as a **filled checklist for the MVP pillars** using your pillar IDs and families.
