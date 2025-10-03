Restarting at M0. Here is the production-ready artifact.

```markdown
# NeoPrompt Lifecycle — **Checkpoint M0: Intake & Planning**
**File:** `docs/lifecycle/Checkpoint-0-Intake-and-Planning.md`  
**Pattern:** Master doc + modular sub-artifacts under `docs/lifecycle/0-intake/`  
**Note:** Copy embedded templates into the sub-artifact files now; also add copies to `/docs/templates/` for central reuse.

---

## 1) Purpose
Establish a governed, measurable intake for any new **Feature / Pillar / Probe Family** before design or implementation. M0 outputs are machine-validated and PR-reviewed so downstream phases do not re-negotiate scope, success criteria, risks, or dependencies.

---

## 2) Outcomes
- Single-sentence problem statement with “why now.”
- Named owner and reviewers (Research, Engineering, QA).
- Success metrics defined as machine-readable JSON with targets and tolerances.
- Timeline and review cadence set.
- Risk register with likelihood, impact, mitigation, owner.
- Dependency graph in Mermaid.
- CI/CD validation wired for all sub-artifacts.
- Governance links to Cognitive Pillars and risk domains.

---

## 3) Inputs & Constraints
- Project governance docs: pillars list, risk taxonomy, harness roadmap.
- Org conventions: branch model, PR review policy, CODEOWNERS.
- Data sensitivity: prompts, logs, or datasets may be confidential or licensed.

---

## 4) Deliverables (files created in this checkpoint)
1. `docs/lifecycle/0-intake/<FEATURE>_intake.md` — Intake Record
2. `docs/lifecycle/0-intake/<FEATURE>_stakeholders.yaml` — Stakeholders
3. `docs/lifecycle/0-intake/<FEATURE>_timeline.md` — Timeline
4. `docs/lifecycle/0-intake/<FEATURE>_risks.md` — Risk Register
5. `docs/lifecycle/0-intake/<FEATURE>_metrics.json` — Success Metrics
6. `docs/lifecycle/0-intake/<FEATURE>_dependencies.mmd` — Mermaid Graph
7. `docs/lifecycle/0-intake/<FEATURE>_governance.yaml` — Pillar + risk mapping
8. `docs/lifecycle/0-intake/<FEATURE>_compliance.md` — Data/License/PII notes
9. `docs/lifecycle/0-intake/<FEATURE>_decision_log.md` — Decisions & alternatives

> During development, mirror each template to `/docs/templates/` for reuse.

---

## 5) Repository layout

```text
/docs/lifecycle/

├─ Checkpoint-0-Intake-and-Planning.md      # this file

└─ 0-intake/

├─ _intake.md

├─ _stakeholders.yaml

├─ _timeline.md

├─ _risks.md

├─ _metrics.json

├─ _dependencies.mmd

├─ _governance.yaml

├─ _compliance.md

└─ _decision_log.md

/docs/templates/   # keep synced copies of all templates below

```text

---

## 6) Process (atomic)

1. **Create branch**

   ```bash
   git checkout -b intake/<FEATURE>
   mkdir -p docs/lifecycle/0-intake docs/templates

```text

2. **Draft Intake Record** → <FEATURE>_intake.md (use template).
    
3. **Add Stakeholders** → <FEATURE>_stakeholders.yaml (CI-schema enforced).
    
4. **Define Timeline** → <FEATURE>_timeline.md (dates & gates).
    
5. **Fill Risk Register** → <FEATURE>_risks.md (≥3 risks, owners).
    
6. **Encode Metrics** → <FEATURE>_metrics.json (targets, tolerances).
    
7. **Draw Dependencies** → <FEATURE>_dependencies.mmd (Mermaid).
    
8. **Governance Mapping** → <FEATURE>_governance.yaml (pillars, risks).
    
9. **Compliance Notes** → <FEATURE>_compliance.md (data, license, PII).
    
10. **Decision Log** → <FEATURE>_decision_log.md (alternatives, rationale).
    
11. **Wire CI** — ensure GH Action validates YAML/JSON/Mermaid.
    
12. **Open PR** — title intake(<FEATURE>): create M0 pack; attach files.
    
13. **Reviews** — Required: Research, Engineering, QA. CODEOWNERS enforce.
    
14. **Merge & Tag** — tag intake-<FEATURE>-approved; move ticket to “Ready for M1”.
    

---

## **7) Acceptance criteria**

- Problem statement ≤ 280 chars. Clear and atomic.
    
- Success metrics machine-readable with numeric targets and tolerances.
    
- Stakeholders valid emails/handles. YAML passes schema.
    
- Timeline aligns to M1–M14 gates.
    
- Risk register has likelihood, impact, mitigation, owner per row.
    
- Mermaid graph renders.
    
- Governance links at least one Pillar ID and risk domain.
    
- Compliance notes address licensing and data sensitivity.
    
- CI pipeline green; PR approved by all three roles.
    

---

## **8) QA gates**

- **Static checks:** markdownlint, yamllint, jsonlint, Mermaid syntax check.
    
- **Schema checks:** stakeholders.schema.json, metrics.schema.json, governance.schema.json.
    
- **Policy checks:** CODEOWNERS present; PR template completed; required reviewers approved.
    

---

## **9) Security, privacy, compliance (M0 scope)**

- No live data in M0 artifacts. Use placeholders.
    
- Mark any dataset references with license and usage constraints.
    
- Define retention policy for any example logs (default: do not retain).
    
- Redact secrets and API keys from samples.
    

---

## **10) Observability requirements seeded at M0**

- Define **intended** KPIs for later phases: reproducibility, robustness, latency P95, error rate.
    
- Name metric keys now to avoid churn later: reproducibility_rate, latency_p95_ms, robustness_delta, error_rate.
    
- Log format decision: JSON lines with RFC3339 timestamps.
    

---

## **11) Versioning & compatibility**

- Prefix all M0 artifacts with feature identifier.
    
- Include version: 1.0.0 fields in YAML/JSON where applicable.
    
- Changes after merge must bump minor/patch and note in decision log.
    

---

## **12) Embedded templates (copy into sub-files; also store in** 

## **/docs/templates/**

## **)**

  

### **12.1 Intake Record —** 

### **docs/lifecycle/0-intake/<FEATURE>_intake.md**

```text

# Intake Record — <FEATURE>

version: 1.0.0

## One-liner

<≤280 chars problem statement>

## Why Now

<Strategic justification and timing>

## Business/Research Value

- <measurable value or hypothesis impact>

## Success Metrics (machine-readable reference)

See: `./<FEATURE>_metrics.json`

## Timebox

- Start: <YYYY-MM-DD>
- Mid-review: <YYYY-MM-DD>
- End: <YYYY-MM-DD>

## Dependencies

See: `./<FEATURE>_dependencies.mmd`

## Risks

See: `./<FEATURE>_risks.md`

## Stakeholders

See: `./<FEATURE>_stakeholders.yaml`

## Governance

See: `./<FEATURE>_governance.yaml`

## Compliance

See: `./<FEATURE>_compliance.md`

## Decision Log

See: `./<FEATURE>_decision_log.md`

```text

---

### **12.2 Stakeholders YAML —** 

### **docs/lifecycle/0-intake/<FEATURE>_stakeholders.yaml**

```text

version: 1.0.0
feature_id: "<FEATURE>"
owner:
  name: "<Full Name>"
  handle: "<github_or_email>"
reviewers:
  research:
    name: "<Full Name>"
    handle: "<github_or_email>"
  engineering:
    name: "<Full Name>"
    handle: "<github_or_email>"
  qa:
    name: "<Full Name>"
    handle: "<github_or_email>"

```text

**Schema (JSON) for CI** — docs/templates/stakeholders.schema.json

```text

{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["version", "feature_id", "owner", "reviewers"],
  "properties": {
    "version": { "type": "string" },
    "feature_id": { "type": "string", "minLength": 1 },
    "owner": {
      "type": "object",
      "required": ["name", "handle"],
      "properties": {
        "name": { "type": "string", "minLength": 1 },
        "handle": { "type": "string", "minLength": 1 }
      }
    },
    "reviewers": {
      "type": "object",
      "required": ["research", "engineering", "qa"],
      "properties": {
        "research": { "$ref": "#/$defs/person" },
        "engineering": { "$ref": "#/$defs/person" },
        "qa": { "$ref": "#/$defs/person" }
      }
    }
  },
  "$defs": {
    "person": {
      "type": "object",
      "required": ["name", "handle"],
      "properties": {
        "name": { "type": "string", "minLength": 1 },
        "handle": { "type": "string", "minLength": 1 }
      }
    }
  }
}

```text

---

### **12.3 Timeline —** 

### **docs/lifecycle/0-intake/<FEATURE>_timeline.md**

```text

# Timeline — <FEATURE>  (version: 1.0.0)

| Phase | Date        | Gate/Deliverable                       | Owner  |
|------:|-------------|-----------------------------------------|--------|
| M0    | YYYY-MM-DD  | Intake pack merged, tag created         | Owner  |
| M1    | YYYY-MM-DD  | Research docs complete                  | Research Reviewer |
| M2    | YYYY-MM-DD  | Design spec & contracts approved        | Eng Reviewer |
| M3    | YYYY-MM-DD  | Implementation plan complete            | Eng |
| M4    | YYYY-MM-DD  | Prototype build passes unit tests       | Eng |
| ...   | ...         | ...                                     | ...    |

```text

---

### **12.4 Risk Register —** 

### **docs/lifecycle/0-intake/<FEATURE>_risks.md**

```text

# Risk Register — <FEATURE>  (version: 1.0.0)

| ID | Description                                | Likelihood | Impact | Mitigation                               | Owner   | Status  |
|----|--------------------------------------------|------------|--------|-------------------------------------------|---------|---------|
| R1 | Dataset license may restrict redistribution| High       | High   | Use licensed alt / host privately         | Research| Open    |
| R2 | Reviewer bandwidth limited                 | Medium     | Medium | Secondary reviewer pre-assigned           | Owner   | Open    |
| R3 | External API latency affects baselines     | Medium     | Low    | Cache mocks; set timeout/fallback         | Eng     | Open    |

```text

---

### **12.5 Metrics JSON —** 

### **docs/lifecycle/0-intake/<FEATURE>_metrics.json**

```text

{
  "version": "1.0.0",
  "feature_id": "<FEATURE>",
  "metrics": [
    {
      "key": "reproducibility_rate",
      "type": "ratio",
      "target": 0.95,
      "tolerance": 0.02,
      "aggregation": "mean",
      "description": "Fraction of runs yielding identical outputs under fixed seeds."
    },
    {
      "key": "latency_p95_ms",
      "type": "latency",
      "target": 2000,
      "tolerance": 200,
      "aggregation": "p95",
      "description": "End-to-end evaluation latency at P95."
    },
    {
      "key": "robustness_delta",
      "type": "ratio",
      "target": 0.90,
      "tolerance": 0.10,
      "aggregation": "mean",
      "description": "Degradation ratio under adversarial perturbations."
    }
  ]
}

```text

**Schema (JSON) for CI** — docs/templates/metrics.schema.json

```text

{
  "$schema": "<https://json-schema.org/draft/2020-12/schema>",
  "type": "object",
  "required": ["version", "feature_id", "metrics"],
  "properties": {
    "version": { "type": "string" },
    "feature_id": { "type": "string", "minLength": 1 },
    "metrics": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["key", "type", "target", "tolerance", "aggregation", "description"],
        "properties": {
          "key": { "type": "string", "minLength": 1 },
          "type": { "type": "string", "enum": ["ratio", "latency", "count", "throughput"] },
          "target": { "type": ["number", "integer"] },
          "tolerance": { "type": ["number", "integer"] },
          "aggregation": { "type": "string", "minLength": 1 },
          "description": { "type": "string", "minLength": 1 }
        }
      }
    }
  }
}

```text

---

### **12.6 Dependencies Mermaid —** 

### **docs/lifecycle/0-intake/<FEATURE>_dependencies.mmd**

```text

graph TD
  F[<FEATURE>] --> H[Evaluation Harness]
  F --> P[Probe Family: <FAMILY_ID>]
  P --> D[(Dataset: <DATASET_ID>)]
  F --> G[Governance: Pillar <PILLAR_ID>]
  F --> C[CI/CD Validators]

```text

---

### **12.7 Governance Mapping —** 

### **docs/lifecycle/0-intake/<FEATURE>_governance.yaml**

```text

version: 1.0.0
feature_id: "<FEATURE>"
pillars:

- id: "CP-07"
    name: "Tool Use Fidelity"

# add more as needed

risk_domains:

- reliability
- robustness
crosslinks:
  intake_ref: "./<FEATURE>_intake.md"
  risks_ref: "./<FEATURE>_risks.md"
  metrics_ref: "./<FEATURE>_metrics.json"

```text

**Schema (JSON) for CI** — docs/templates/governance.schema.json

```text

{
  "$schema": "<https://json-schema.org/draft/2020-12/schema>",
  "type": "object",
  "required": ["version", "feature_id", "pillars", "risk_domains", "crosslinks"],
  "properties": {
    "version": { "type": "string" },
    "feature_id": { "type": "string" },
    "pillars": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" }
        }
      }
    },
    "risk_domains": {
      "type": "array",
      "items": { "type": "string" }
    },
    "crosslinks": {
      "type": "object",
      "required": ["intake_ref", "risks_ref", "metrics_ref"],
      "properties": {
        "intake_ref": { "type": "string" },
        "risks_ref": { "type": "string" },
        "metrics_ref": { "type": "string" }
      }
    }
  }
}

```text

---

### **12.8 Compliance Notes —** 

### **docs/lifecycle/0-intake/<FEATURE>_compliance.md**

```text

# Compliance Notes — <FEATURE> (version: 1.0.0)

## Data Sources

- <dataset id> — license: <spdx or URL>
- <dataset id> — restricted? yes/no

## PII / Sensitive Data

- Expected? yes/no
- Handling: redact, hash, or exclude; no retention in logs.

## Storage & Retention

- Artifacts: repo only; no raw data.
- Retention: none beyond samples; delete on merge.

## Third-party Services

- <service> — purpose — data sent? yes/no — auth method — region

```text

---

### **12.9 Decision Log —** 

### **docs/lifecycle/0-intake/<FEATURE>_decision_log.md**

```text

# Decision Log — <FEATURE> (version: 1.0.0)

## D-0001: Success metric target levels

- Context: <why>
- Options considered: <A/B/C>
- Decision: <chosen>
- Rationale: <trade-offs>
- Date: <YYYY-MM-DD>
- Owner: <name>

## D-0002: Dependency scope

...

```text

---

## **13) CI/CD wiring (GitHub Actions example)**

  

**File:** .github/workflows/m0-intake-validate.yml

```text

name: M0 Intake Validation
on:
  pull_request:
    paths:
      - "docs/lifecycle/0-intake/**"
      - "docs/templates/**"
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node for schema tools
        uses: actions/setup-node@v4
        with:
          node-version: "22"
      - name: Install validators
        run: |
          npm install -g ajv-cli@5.0.0
          pipx install yamllint
          pipx install markdownlint-cli
      - name: Lint Markdown
        run: markdownlint "docs/**/*.md" || true  # warn only

      - name: Lint YAML
        run: yamllint docs/lifecycle/0-intake || true  # warn only

      - name: Validate Stakeholders YAML
        run: |
          yq -o=json docs/lifecycle/0-intake/*_stakeholders.yaml > /tmp/stakeholders.json
          ajv validate -s docs/templates/stakeholders.schema.json -d /tmp/stakeholders.json

      - name: Validate Metrics JSON
        run: |
          ajv validate -s docs/templates/metrics.schema.json -d docs/lifecycle/0-intake/*_metrics.json

      - name: Validate Governance YAML
        run: |
          yq -o=json docs/lifecycle/0-intake/*_governance.yaml > /tmp/governance.json
          ajv validate -s docs/templates/governance.schema.json -d /tmp/governance.json

      - name: Mermaid syntax check
        run: |
          grep -q "graph " docs/lifecycle/0-intake/*_dependencies.mmd

```text

**PR template** — .github/pull_request_template.md

```text

## M0 Intake Checklist for <FEATURE>

- [ ] Intake record added
- [ ] Stakeholders YAML passes schema
- [ ] Timeline created
- [ ] Risk register with ≥3 risks
- [ ] Metrics JSON passes schema
- [ ] Dependencies Mermaid included
- [ ] Governance mapping passes schema
- [ ] Compliance notes added
- [ ] Decision log created

Reviewers: @research @engineering @qa

```text

**CODEOWNERS** — CODEOWNERS

```text

docs/lifecycle/0-intake/ @research @engineering @qa

```text

---

## **14) Review protocol**

- **Branch naming:** intake/<FEATURE>
    
- **Commit style:** Conventional Commits, e.g., docs(intake): add metrics for <FEATURE>
    
- **Required approvals:** Research, Engineering, QA
    
- **Blocking rules:** CI red = no merge
    

---

## **15) Exit criteria (hard)**

- All nine deliverables present.
    
- All schema validations pass.
    
- Mermaid graph exists and renders.
    
- PR approved by three roles and merged to main.
    
- Git tag intake-<FEATURE>-approved created.
    
- Issue status: **Ready for M1**.
    

---

## **16) Worked example (Tool-Use Fidelity)**

  

**Intake one-liner:** “Baseline and improve deterministic correctness of structured function-calling across controlled and adversarial probes.”

**Metrics:** reproducibility_rate ≥ 0.95 ±0.02, latency_p95_ms ≤ 2000 ±200, robustness_delta ≥ 0.90 ±0.10.

**Risks:** license risk for API mocks; reviewer bandwidth; latency variance.

**Governance:** Pillars CP-07 Tool Use Fidelity, CP-16 Robustness to Noise; domains reliability, robustness.

**Compliance:** no PII; synthetic mocks only.

**Dependencies:** Harness, Probe Family: function-calls, Dataset: mock-apis.

---

## **17) Hand-off to M1**

- Link intake/<FEATURE> PR in M1 Research doc header.
    
- Carry over metrics keys and pillar IDs.
    
- Use M0 decision log as first entries in M1 research notes.
    

---


### **Required state**

- Files exist:
    
    - docs/lifecycle/Checkpoint-0-Intake-and-Planning.md
        
    - docs/lifecycle/0-intake/<FEATURE>_{intake.md,stakeholders.yaml,timeline.md,risks.md,metrics.json,dependencies.mmd,governance.yaml,compliance.md,decision_log.md}
        
    
- Schemas present:
    
    docs/templates/{stakeholders.schema.json,metrics.schema.json,governance.schema.json}
    
- CI added: .github/workflows/m0-intake-validate.yml
    
- PR template has M0 checklist; CODEOWNERS covers docs/lifecycle/0-intake/
    
- Mermaid graph renders; YAML/JSON validate
    
- PR approved by Research, Engineering, QA; merged to main
    
- Tag created: intake-<FEATURE>-approved
    
- Issue moved to “Ready for M1”
    

  

### **Quick self-check (paste in repo root)**

```text

FEATURE=<your_feature_id>

# Files

ls docs/lifecycle/Checkpoint-0-Intake-and-Planning.md \
&& ls docs/lifecycle/0-intake/${FEATURE}_{intake.md,stakeholders.yaml,timeline.md,risks.md,metrics.json,dependencies.mmd,governance.yaml,compliance.md,decision_log.md}

# Schema validation (requires ajv-cli, yq)

yq -o=json docs/lifecycle/0-intake/${FEATURE}_stakeholders.yaml | \
  ajv validate -s docs/templates/stakeholders.schema.json -d /dev/stdin
ajv validate -s docs/templates/metrics.schema.json -d docs/lifecycle/0-intake/${FEATURE}_metrics.json
yq -o=json docs/lifecycle/0-intake/${FEATURE}_governance.yaml | \
  ajv validate -s docs/templates/governance.schema.json -d /dev/stdin

# Mermaid presence

grep -q "graph " docs/lifecycle/0-intake/${FEATURE}_dependencies.mmd && echo "Mermaid OK"

# Tag present

git tag -l "intake-${FEATURE}-approved"

```text

If every command succeeds and the PR is merged with the tag in place, M0 is complete.
