# Harness Foundation — Intake

## One-liner

Seed, schemas, CI, and governance to make the harness buildable and testable from M0.

## Scope

- Global governance and intake package
- Probe generators, scorers, runner skeleton stubs
- Schemas and CI validation jobs

## Out-of-scope

- Model adapters
- Full probe families
- Reporting UI

## Dependencies

- GitHub Actions availability
- Repository default branch = `main`

## Acceptance criteria

- All 9 intake files present and committed
- CI green on markdownlint, yamllint, ajv, mermaid
- Governance and stakeholders validate against schemas
- Mermaid dependency diagram artifact uploaded
