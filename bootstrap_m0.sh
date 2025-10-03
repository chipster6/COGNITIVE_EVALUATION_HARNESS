#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/cody/COGNITIVE_EVALUATION_HARNESS"
cd "$ROOT"

echo "==> Starting M0 bootstrap..."

# 1) Repo init and branch (skip if already done)
if [ ! -d .git ]; then
  git init
fi
git branch -M main

# 2) Dirs (skip if exist)
mkdir -p docs/lifecycle docs/lifecycle/0-intake docs/templates .github/workflows artifacts

# 3) Bring in canonical M0
if [ -f "NEO PROMPT V2/CHECKLIST/M0.md" ]; then
  cp "NEO PROMPT V2/CHECKLIST/M0.md" docs/lifecycle/Checkpoint-0-Intake-and-Planning.md
  echo "✓ Copied M0.md"
else
  echo "⚠ M0.md source not found; using placeholder"
  cat > docs/lifecycle/Checkpoint-0-Intake-and-Planning.md <<'EOF'
# Checkpoint 0 — Intake and Planning
Placeholder: canonical M0.md not found. Replace with official file.
EOF
fi

# 4) Harness-foundation intake (9 files)
echo "==> Creating harness-foundation intake package..."

cat > docs/lifecycle/0-intake/harness-foundation_intake.md <<'EOF'
# Harness Foundation — Intake

## One-liner
Establish deterministic, offline-first scaffolding, schemas, and CI gates for the evaluation harness.

## Scope
- Global governance and intake assets
- Schemas, minimal CI validation, deterministic project defaults

## Out-of-scope
- Probe generators, scorers, runtime code (later milestones)

## Dependencies
- GitHub Actions availability
- Node toolchain in CI jobs

## Acceptance criteria
- All 9 intake files present and validated in CI
- CI succeeds on markdown, yaml, schema, and mermaid checks
EOF

cat > docs/lifecycle/0-intake/harness-foundation_stakeholders.yaml <<'EOF'
version: "1.0.0"
stakeholders:
  product_owner: Chipster6
  tech_owner: Chipster6
  research_reviewer: Chipster6
  qa_reviewer: Chipster6
EOF

cat > docs/lifecycle/0-intake/harness-foundation_timeline.md <<'EOF'
# Timeline — Harness Foundation

| Phase | Date | Deliverable | Owner |
|-------|------|-------------|-------|
| M0 | 2025-10-03 | Intake scaffolding, CI wiring, schemas committed | Chipster6 |
| M1 | TBD | Research docs and baselines | Chipster6 |
EOF

cat > docs/lifecycle/0-intake/harness-foundation_risks.md <<'EOF'
# Risks — Harness Foundation

- **HF-R1**: Governance or metrics schema mismatch
  - Mitigation: CI ajv checks
  - Owner: Chipster6
  - Status: Open

- **HF-R2**: Mermaid render differences
  - Mitigation: Pin @mermaid-js/mermaid-cli 10.9.1
  - Owner: Chipster6
  - Status: Open
EOF

cat > docs/lifecycle/0-intake/harness-foundation_metrics.json <<'EOF'
{
  "reproducibility_rate": 0.95,
  "latency_p95_ms": 2000,
  "test_coverage": 0.80
}
EOF

cat > docs/lifecycle/0-intake/harness-foundation_dependencies.mmd <<'EOF'
graph TD
  HF[Harness Foundation] --> Schemas
  HF --> Probes
  HF --> Scorers
  CI[CI Validation] --> HF
EOF

cat > docs/lifecycle/0-intake/harness-foundation_governance.yaml <<'EOF'
version: "1.0.0"
owners:
  product_owner: Chipster6
  tech_owner: Chipster6
  research_reviewer: Chipster6
  qa_reviewer: Chipster6
change_control:
  decision_log: ./harness-foundation_decision_log.md
  risk_register: ./harness-foundation_risks.md
  acceptance_gate: "Feature intake complete when all 9 files exist and validate"
EOF

cat > docs/lifecycle/0-intake/harness-foundation_compliance.md <<'EOF'
# Compliance — Harness Foundation

- Strict JSON I/O
- Offline-first; no network egress except GitHub CI
- Deterministic seeds; reproducible artifacts
- Artifacts retained per repo policy
EOF

cat > docs/lifecycle/0-intake/harness-foundation_decision_log.md <<'EOF'
# Decision Log — Harness Foundation

## HF-D1 (2025-10-03): Deterministic defaults

- **Decision**: Deterministic, local-first defaults with strict CI gates
- **Context**: Foundation must be reproducible and auditable
- **Consequences**: All probe generation will use seeded RNG
- **Owner**: Chipster6
EOF

echo "✓ Created 9 harness-foundation files"

# 5) CI workflow
echo "==> Creating CI workflow..."

cat > .github/workflows/m0-intake-validate.yml <<'EOF'
name: M0 Intake Validate
on:
  push:
    branches: [ "main", "master" ]
  pull_request:
permissions:
  contents: read
concurrency:
  group: m0-${{ github.ref }}
  cancel-in-progress: true
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: DavidAnson/markdownlint-cli2-action@v17
        with:
          globs: |
            **/*.md
          config: ./.markdownlint.json

  yamllint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ibiqlik/action-yamllint@v3
        with:
          config_file: ./.yamllint.yml
          format: colored

  json_schema_validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - name: Install ajv
        run: npm i -g ajv-cli@5 ajv-formats@3
      - name: Convert YAML → JSON
        uses: mikefarah/yq@v4
        with:
          cmd: |
            yq -o=json docs/lifecycle/0-intake/harness-foundation_governance.yaml > /tmp/hf_gov.json
            yq -o=json docs/lifecycle/0-intake/harness-foundation_stakeholders.yaml > /tmp/hf_stake.json
            yq -o=json docs/lifecycle/0-intake/governance.yaml > /tmp/global_gov.json
      - name: Validate JSON against schemas
        run: |
          ajv -c ajv-formats -s docs/templates/governance.schema.json   -d /tmp/hf_gov.json
          ajv -c ajv-formats -s docs/templates/stakeholders.schema.json -d /tmp/hf_stake.json
          ajv -c ajv-formats -s docs/templates/governance.schema.json   -d /tmp/global_gov.json
          ajv -c ajv-formats -s docs/templates/metrics.schema.json      -d docs/lifecycle/0-intake/harness-foundation_metrics.json

  mermaid_check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - name: Install mermaid-cli
        run: npm i -g @mermaid-js/mermaid-cli@10.9.1
      - name: Render diagram
        run: |
          mkdir -p artifacts
          mmdc -i docs/lifecycle/0-intake/harness-foundation_dependencies.mmd \
               -o artifacts/hf_deps.svg -b transparent
      - uses: actions/upload-artifact@v4
        with:
          name: mermaid-artifacts
          path: artifacts
EOF

echo "✓ Created CI workflow"

# 6) First commit
echo "==> Creating initial commit..."
git add .
git commit -m "chore(m0): scaffold intake, templates, governance, CI, and docs" || echo "⚠ Commit may already exist"

# 7) Create remote and push
echo "==> Creating GitHub repository and pushing..."
REPO_DESC="$(tr -d '\n' < docs/lifecycle/0-intake/repo_description.md)"

if gh repo view chipster6/COGNITIVE_EVALUATION_HARNESS >/dev/null 2>&1; then
  echo "✓ Repository already exists, adding remote and pushing..."
  git remote add origin https://github.com/chipster6/COGNITIVE_EVALUATION_HARNESS.git 2>/dev/null || true
  git push -u origin main || echo "⚠ Push may have failed, check connection"
else
  echo "✓ Creating new repository..."
  gh repo create COGNITIVE_EVALUATION_HARNESS --public --source=. --remote=origin --push --description "$REPO_DESC"
fi

echo ""
echo "==> M0 Bootstrap Complete!"
echo ""
echo "Next steps:"
echo "  1. Verify CI: gh run list --limit 5"
echo "  2. Watch workflow: gh run watch --exit-status"
echo "  3. View repo: gh repo view --web"
echo ""
