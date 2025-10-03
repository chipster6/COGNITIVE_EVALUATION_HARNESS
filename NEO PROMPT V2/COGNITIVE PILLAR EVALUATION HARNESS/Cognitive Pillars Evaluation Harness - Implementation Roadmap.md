# Cognitive Pillars Evaluation Harness - Implementation Roadmap

**Version:** 1.0.0  
**Total Duration:** 6 weeks (42 dev-days)  
**Parallelization:** 4 weeks with 2 engineers

-----

## Overview

This roadmap delivers a production-ready evaluation harness for 21 cognitive pillars across three phases:

- **Phase 1 (Weeks 1-2):** MVP with 5 core pillars
- **Phase 2 (Weeks 3-4):** 6 high-priority extension pillars
- **Phase 3 (Weeks 5-6):** Remaining 10 pillars + statistical enhancements

-----

## Phase 1: MVP (Weeks 1-2) - 14 Dev-Days

**Goal:** Ship 5 core pillars with full evaluation pipeline

### Week 1: Foundation + Gf + WM

#### Days 1-2: Infrastructure Setup

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Create package structure (`neoprompt/probes/`, `neoprompt/generators/`, `neoprompt/metrics/`, `neoprompt/schemas/`)
- [ ] Implement `seed_utils.py` with `derive_probe_seed()` and `hash_seed_for_storage()`
- [ ] Implement `uncertainty.py` with `bootstrap_ci()`
- [ ] Create base probe classes (`ProbeGenerator`, `ProbeScorer`)
- [ ] Set up pytest framework with test directory structure
- [ ] Create `.gitignore`, `pyproject.toml`, `README.md`

**Deliverables:**

- Package skeleton operational
- Seed derivation deterministic and tested
- Bootstrap CI calculation verified
- Base classes defined with docstrings

**Quality Gates:**

- ✅ `test_seed_determinism()` passes
- ✅ Same seed produces same derived seed
- ✅ Bootstrap CI converges for N=100 samples

-----

#### Days 3-4: Pillar 1 (Abstraction / Gf)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement Raven matrix generator in `generators/raven.py`:
  - [ ] Grid generation (2×2 to 3×3)
  - [ ] Rule grammar (XOR, progression, rotation, mirror, count)
  - [ ] Symbol alphabet generation (disjoint per seed)
  - [ ] Distractor generation with adversarial variants
- [ ] Create `probes/gf.py` with `generate_gf_probes()`
- [ ] Implement `scorers/gf_scorer.py` with exact match + latency tracking
- [ ] Define `schemas/gf_raven_v1.json` (requires `answer` field with pattern `^[A-H]$`)
- [ ] Write unit tests in `tests/unit/test_gf_probe.py`:
  - [ ] Determinism (same seed → same matrix)
  - [ ] Band scaling (A: 4 symbols → E: 12 symbols)
  - [ ] Scoring (exact match, partial credit for similar choices)

**Deliverables:**

- Gf probe generator producing valid matrices
- Scorer with exact matching
- JSON schema validated
- Unit tests passing

**Quality Gates:**

- ✅ 10 variants generated deterministically
- ✅ Band E includes compound rules (2-3 simultaneous)
- ✅ Distractors share surface features but violate deep rules
- ✅ Perfect score = 1.0, wrong answer = 0.0

-----

#### Days 5-7: Pillar 3 (Working Memory)

**Owner:** Engineer B  
**Dev-Days:** 3

**Tasks:**

- [ ] Implement entity tracking generator in `generators/entity_tracking.py`:
  - [ ] Random initial inventories
  - [ ] Move sequence generation (give/transfer operations)
  - [ ] State evolution tracking
  - [ ] Contradiction injection for band E
- [ ] Create `probes/wm.py` with `generate_wm_probes()`
- [ ] Implement narrative formatting (`format_narrative()`)
- [ ] Implement `scorers/wm_scorer.py` with composite formula:
  - [ ] Step accuracy (correct entity-item counts)
  - [ ] Context loss penalty (−0.1 × missing entities)
  - [ ] Contradiction penalty (−0.2 × impossible states)
- [ ] Define `schemas/wm_bags_v1.json`
- [ ] Write unit tests in `tests/unit/test_wm_probe.py`:
  - [ ] Determinism
  - [ ] Band distribution (2 per band A-E)
  - [ ] Perfect score test
  - [ ] Partial credit test
  - [ ] Contradiction detection test

**Deliverables:**

- Working Memory probe generator with natural language narratives
- Scorer with step accuracy + penalties
- JSON schema for final state
- Comprehensive unit tests

**Quality Gates:**

- ✅ Perfect match scores 1.0
- ✅ Missing entity penalized correctly
- ✅ Negative counts detected as contradictions
- ✅ Band E includes contradictory claims

-----

### Week 2: Tool Use + Grounding + Robustness + Integration

#### Days 8-9: Pillar 12 (Tool Use Fidelity)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement virtual tool server in `tools/virtual_server.py`:
  - [ ] Flask app on localhost:7012
  - [ ] Register deterministic tools: `math.add`, `math.multiply`, `search.docs`, `geo.distance`
  - [ ] JSON-RPC `/call` endpoint
  - [ ] Tool catalog endpoint
  - [ ] Argument schema validation
- [ ] Create `probes/tool_use.py` with multi-step chain generator:
  - [ ] Sequential tool calls (e.g., multiply then add)
  - [ ] Type-diverse arguments (primitives, objects, arrays)
  - [ ] Distractor tools for bands B+
- [ ] Implement `scorers/tool_use_scorer.py`:
  - [ ] Correct call rate (required calls present & correct)
  - [ ] Argument precision (type + value exact match)
  - [ ] Unnecessary call penalty (−0.02 per extra call)
- [ ] Define `schemas/tool_chain_v1.json`
- [ ] Write unit tests in `tests/unit/test_tool_use_probe.py`
- [ ] Write integration test with tool server running

**Deliverables:**

- Local tool server with deterministic stubs
- Tool chain probe generator
- Scorer with call validation
- Integration test (server + probe → score)

**Quality Gates:**

- ✅ Tool server runs without network access
- ✅ Correct call sequence scores 1.0
- ✅ Extra calls penalized at −0.02 each
- ✅ Missing required calls score 0.0

-----

#### Days 10-11: Pillar 14 (Evidence Grounding)

**Owner:** Engineer B  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement local corpus manager in `corpus/manager.py`:
  - [ ] Document loading with byte-offset indexing
  - [ ] Span extraction by byte range
  - [ ] Keyword search (deterministic ranking)
- [ ] Create test corpus in `tests/fixtures/corpus/`:
  - [ ] 5-10 text documents (~500 words each)
  - [ ] Diverse topics for multi-document synthesis
- [ ] Create `probes/grounding.py`:
  - [ ] Multi-document question generation
  - [ ] Ground truth span selection
  - [ ] Distractor documents for bands C+
- [ ] Implement `scorers/grounding_scorer.py`:
  - [ ] Precision@k on span boundaries (k=3)
  - [ ] Unsupported claim detection
  - [ ] Composite: precision@3 - unsupported_rate
- [ ] Define `schemas/ground_span_v1.json`
- [ ] Write unit tests

**Deliverables:**

- Local corpus with byte-offset ground truth
- Grounding probe generator
- Scorer with span validation
- Test corpus included

**Quality Gates:**

- ✅ Byte offsets match exactly (no fuzzy matching)
- ✅ Precision@3 computed correctly
- ✅ Unsupported claims penalized
- ✅ Corpus deterministically loaded

-----

#### Days 12-13: Pillar 16 (Robustness)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement noise injection in `generators/noise.py`:
  - [ ] Character-level swaps, deletions, insertions
  - [ ] Noise level ε ∈ [0, 0.4] by band
  - [ ] Deterministic application via seed
- [ ] Create policy FSM in `security/policy_fsm.py`:
  - [ ] Jailbreak pattern matching (regex)
  - [ ] Role leakage detection
  - [ ] Violation severity levels (high/medium/low)
- [ ] Create `probes/robustness.py`:
  - [ ] Noise variant family (`rob_noise_v1`)
  - [ ] Jailbreak variant family (`rob_jailbreak_v1`)
  - [ ] Seeded attack bank
- [ ] Implement `scorers/robustness_scorer.py`:
  - [ ] AUC under accuracy-vs-noise curve
  - [ ] Jailbreak leakage rate
  - [ ] Composite: 0.7×AUC + 0.3×(1 - leakage)
- [ ] Define schemas for both families
- [ ] Write unit tests

**Deliverables:**

- Noise injection generator
- Policy FSM for jailbreak detection
- Robustness probe generator
- Scorer with AUC calculation

**Quality Gates:**

- ✅ Noise level scales from 0 (band A) to 0.4 (band E)
- ✅ AUC computed via trapezoid integration
- ✅ Jailbreak patterns detected correctly
- ✅ Leakage rate = violations / attempts

-----

#### Day 14: Integration & Reporting

**Owner:** Both Engineers  
**Dev-Days:** 1

**Tasks:**

- [ ] Implement probe executor in `runner/executor.py`
- [ ] Implement JSON validator in `runner/validator.py`:
  - [ ] Schema validation (jsonschema library)
  - [ ] Single repair attempt (strip fences, find brackets, fix quotes)
  - [ ] Invalid → score 0
- [ ] Implement report generators in `metrics/reports.py`:
  - [ ] `generate_report_dev_json()` with pillar metrics + CIs
  - [ ] `generate_report_enterprise_md()` with human-readable summary
- [ ] Create `examples/run_evaluation.py`:
  - [ ] Load suite YAML
  - [ ] Expand probes
  - [ ] Execute against stub model
  - [ ] Score and aggregate
  - [ ] Generate reports
- [ ] Write end-to-end integration test in `tests/integration/test_e2e.py`
- [ ] Create example `suite.yml` with all 5 MVP pillars

**Deliverables:**

- Complete pipeline operational
- Example runner producing reports
- Integration test passing
- Documentation updated

**Quality Gates:**

- ✅ End-to-end test runs in <10 seconds
- ✅ Bootstrap CI computed with n=1000 samples
- ✅ Report includes all 5 pillars
- ✅ JSON schema validation working
- ✅ Single repair attempt functional

-----

## Phase 1 Summary

**Total Dev-Days:** 14  
**Pillars Implemented:** 5 (Gf, WM, Tool Use, Grounding, Robustness)  
**Infrastructure:** Complete (seeds, schemas, scorers, tests, reports)

**Phase 1 Quality Gates:**

- ✅ All MVP pillars generate deterministic probes
- ✅ JSON validation with repair working
- ✅ Bootstrap CIs computed correctly (width < 5 points for N=100)
- ✅ End-to-end integration test passes
- ✅ No network calls in any generator (verified via socket blocking test)
- ✅ Seed hashes stored, never plaintext seeds
- ✅ Test coverage > 80% for MVP pillars

-----

## Phase 2: Extension Pillars (Weeks 3-4) - 14 Dev-Days

**Goal:** Add 6 high-priority extension pillars

### Week 3: Planning + Bayesian + Causality

#### Days 15-16: Pillar 11 (Long-Horizon Planning)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement DAG generator in `generators/dag_planning.py`:
  - [ ] Precedence constraint generation
  - [ ] Resource allocation
  - [ ] Topological sort validation
- [ ] Implement critical path algorithm
- [ ] Create `probes/planning.py` with:
  - [ ] Project scheduling tasks
  - [ ] Perturbation variants (task removal/extension)
  - [ ] Replan requirement
- [ ] Implement `scorers/planning_scorer.py`:
  - [ ] DAG validity (0/1 hard gate)
  - [ ] Critical path Jaccard similarity
  - [ ] Makespan error
  - [ ] Replan latency measurement
- [ ] Define `schemas/plan_dag_v1.json`
- [ ] Write unit tests

**Quality Gates:**

- ✅ Cycle detection functional
- ✅ Critical path computed correctly
- ✅ Replan variants include perturbation metadata

-----

#### Days 17-18: Pillar 10 (Bayesian Updating)

**Owner:** Engineer B  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement Beta-Binomial generator in `generators/bayesian.py`
- [ ] Create sequential evidence infrastructure
- [ ] Create `probes/bayesian.py` with:
  - [ ] Prior specification (Beta distribution)
  - [ ] Sequential observations (coin flips, trials)
  - [ ] Posterior computation tasks
- [ ] Implement `scorers/bayesian_scorer.py`:
  - [ ] Log-score calculation
  - [ ] ECE computation (10 fixed bins)
  - [ ] Composite: 0.7×(1 - LS/LS_base) + 0.3×(1 - ECE)
- [ ] Define `schemas/bayes_binom_v1.json`
- [ ] Write unit tests

**Quality Gates:**

- ✅ Posterior parameters computed correctly
- ✅ ECE bins distributed evenly
- ✅ Log-score normalized vs. baseline

-----

#### Days 19-21: Pillar 7 (Causality & Counterfactuals)

**Owner:** Engineer A  
**Dev-Days:** 3

**Tasks:**

- [ ] Implement SCM generator in `generators/scm.py`:
  - [ ] Directed acyclic graph structure
  - [ ] Structural equations (linear/nonlinear)
  - [ ] Exogenous noise variables
- [ ] Implement do-calculus query generator
- [ ] Implement counterfactual twin generator (same U, different X)
- [ ] Create `probes/causality.py`
- [ ] Implement `scorers/causality_scorer.py`:
  - [ ] Counterfactual accuracy (exact match)
  - [ ] Invariance test pass rate
  - [ ] Composite: 0.7×CF_accuracy + 0.3×invariance
- [ ] Define schemas
- [ ] Write unit tests

**Quality Gates:**

- ✅ SCM generation deterministic
- ✅ Counterfactual twins share exogenous noise
- ✅ Invariance tests identify non-causal paths

-----

### Week 4: Stress + Temporal/Spatial + Transfer

#### Days 22-23: Pillar 2 (Stress Decision Quality)

**Owner:** Engineer B  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement timing infrastructure in `runner/timing.py`:
  - [ ] Sub-second precision measurement
  - [ ] Timeout enforcement
  - [ ] Time-to-good-decision (TTGD) tracking
- [ ] Create `probes/stress.py`:
  - [ ] Binary forecasts with tight deadlines
  - [ ] Interrupted arithmetic tasks
  - [ ] Noisy triage scenarios
- [ ] Implement `scorers/stress_scorer.py`:
  - [ ] Brier score normalization
  - [ ] Error rate under constraints
  - [ ] Composite: 0.5×(1 - Brier_norm) + 0.5×(1 - error_rate)
- [ ] Define schemas
- [ ] Write unit tests

**Quality Gates:**

- ✅ Time budgets enforced accurately (±10ms)
- ✅ TTGD distribution computed
- ✅ Brier score normalized to [0,1]

-----

#### Days 24-25: Pillar 15 (Temporal & Spatial Reasoning)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement calendar/timezone generator in `generators/temporal.py`
- [ ] Implement haversine distance calculator
- [ ] Create `probes/temporal_spatial.py`:
  - [ ] Schedule feasibility tasks
  - [ ] Route planning with time windows
  - [ ] Pickup/delivery scenarios
- [ ] Implement `scorers/temporal_spatial_scorer.py`:
  - [ ] Feasibility (0/1 hard gate)
  - [ ] Lateness penalty
  - [ ] Spatial error (meters)
  - [ ] Composite: 0.5×feasibility + 0.25×(1-lateness) + 0.25×(1-spatial_err)
- [ ] Define schemas
- [ ] Write unit tests

**Quality Gates:**

- ✅ Timezone conversions correct
- ✅ Haversine distance ±1% of ground truth
- ✅ Time paradoxes detected

-----

#### Days 26-28: Pillar 5 (Transfer Efficiency)

**Owner:** Engineer B  
**Dev-Days:** 3

**Tasks:**

- [ ] Implement multi-shot infrastructure in `generators/transfer.py`:
  - [ ] Shot management (0, 1, 3 examples)
  - [ ] Disjoint vocabulary generation
  - [ ] Schema transformation
- [ ] Create `probes/transfer.py`:
  - [ ] Schema transfer tasks
  - [ ] Format conversion tasks
  - [ ] Task variant transfer (math↔string)
- [ ] Implement `scorers/transfer_scorer.py`:
  - [ ] AUC of performance vs. shots
  - [ ] Time-to-first-win
  - [ ] Transfer delta (3-shot - 0-shot)
- [ ] Define schemas
- [ ] Write unit tests

**Quality Gates:**

- ✅ AUC computed via trapezoid integration
- ✅ 0-shot, 1-shot, 3-shot variants generated
- ✅ Vocabulary disjoint between source/target

-----

## Phase 2 Summary

**Total Dev-Days:** 14  
**Pillars Implemented:** 6 (Planning, Bayesian, Causality, Stress, Temporal/Spatial, Transfer)  
**Cumulative Pillars:** 11 / 21

**Phase 2 Quality Gates:**

- ✅ All 11 pillars operational
- ✅ Composite scoring with weights implemented
- ✅ No-weak-links penalty functional
- ✅ Reliability metrics (α, KR-20) computed
- ✅ Full test coverage >85%
- ✅ CI gates enforced (JSON valid ≥99.5%, stress error ≤5%)

-----

## Phase 3: Complete + Advanced Features (Weeks 5-6) - 14 Dev-Days

**Goal:** Complete remaining 10 pillars + statistical enhancements

### Week 5: Remaining Pillars

#### Days 29-30: Pillar 13 (Symbolic/Math Reasoning)

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement big-integer arithmetic generator (100-2000 digits)
- [ ] Implement algebra AST comparison
- [ ] Create `probes/symbolic_math.py`
- [ ] Implement scorer with exact matching
- [ ] Write unit tests

-----

#### Days 31-32: Pillar 17 (Ethical/Policy-Constrained)

**Owner:** Engineer B  
**Dev-Days:** 2

**Tasks:**

- [ ] Define policy FSM rules (safe/refuse/offer-alternative)
- [ ] Create `probes/ethical_policy.py`
- [ ] Implement scorer with FSM compliance
- [ ] Write unit tests

-----

#### Days 33-35: Pillars 4, 6, 8, 9, 18-21

**Owner:** Both Engineers  
**Dev-Days:** 3

**Distribute:**

- Engineer A: Throughput (4), Communication (6), Counter-argument (8), Self-Verification (9)
- Engineer B: Memory Consistency (18), Explainability (19), Cost-Aware (20), Cross-Domain Transfer (21)

**Tasks per pillar:**

- [ ] Implement generator
- [ ] Implement scorer
- [ ] Define schema
- [ ] Write unit tests

-----

### Week 6: Statistical Enhancements + Documentation

#### Days 36-37: Statistical Framework

**Owner:** Engineer A  
**Dev-Days:** 2

**Tasks:**

- [ ] Implement IRT 2PL calibration in `metrics/irt.py`:
  - [ ] Item parameter estimation (difficulty, discrimination)
  - [ ] Person ability estimation (θ̂)
  - [ ] Information function computation
- [ ] Implement test-retest infrastructure:
  - [ ] New seed generation for retest
  - [ ] ICC computation (intra-class correlation)
- [ ] Implement factor analysis for construct validity:
  - [ ] Confirmatory factor model
  - [ ] Loadings computation
  - [ ] Fit statistics
- [ ] Write unit tests for statistical methods

**Quality Gates:**

- ✅ IRT convergence for N≥50 items
- ✅ ICC≥0.75 for test-retest
- ✅ Factor loadings interpretable

-----

#### Days 38-40: Documentation & Packaging

**Owner:** Engineer B  
**Dev-Days:** 3

**Tasks:**

- [ ] Complete API documentation (Sphinx):
  - [ ] Docstrings for all public functions
  - [ ] Usage examples
  - [ ] Architecture diagrams
- [ ] Write user guide:
  - [ ] Installation instructions
  - [ ] Quickstart tutorial
  - [ ] Suite authoring guide
  - [ ] Troubleshooting
- [ ] Create example notebooks:
  - [ ] Basic evaluation workflow
  - [ ] Custom probe authoring
  - [ ] Results interpretation
- [ ] Package for PyPI:
  - [ ] `pyproject.toml` finalization
  - [ ] Dependency pinning
  - [ ] Version tagging
  - [ ] Upload to test.pypi.org

**Deliverables:**

- Sphinx documentation site
- User guide (Markdown)
- 3+ example Jupyter notebooks
- PyPI package published

-----

#### Days 41-42: CI/CD Integration

**Owner:** Both Engineers  
**Dev-Days:** 2

**Tasks:**

- [ ] Create GitHub Actions workflow:
  - [ ] Automated test suite on PR
  - [ ] Linting (black, flake8, mypy)
  - [ ] Coverage reporting (codecov)
- [ ] Configure artifact storage:
  - [ ] S3/GCS bucket for artifact retention
  - [ ] Encryption for raw outputs
  - [ ] 7-day default retention
- [ ] Add drift detection checks:
  - [ ] Compare scores vs. baseline
  - [ ] Non-overlapping CI detection
  - [ ] Automated alerts on ≥3 point drops
- [ ] Create release checklist
- [ ] Write deployment guide

**Deliverables:**

- GitHub Actions CI/CD pipeline
- Artifact storage configured
- Drift detection functional
- Release process documented

-----

## Phase 3 Summary

**Total Dev-Days:** 14  
**Pillars Implemented:** 10 (completing all 21)  
**Cumulative Pillars:** 21 / 21

**Phase 3 Quality Gates:**

- ✅ All 21 pillars implemented and tested
- ✅ IRT calibration working (2PL model)
- ✅ Full documentation published
- ✅ CI/CD pipeline functional
- ✅ PyPI package available
- ✅ Test coverage >90%
- ✅ Drift detection automated

-----

## Parallelization Strategy (4-Week Timeline)

**For teams with 2 engineers:**

### Parallel Track 1 (Engineer A)

- **Weeks 1-2:** Gf + Tool Use + Robustness + Integration
- **Weeks 3-4:** Planning + Causality + Stress + Symbolic/Math + CI/CD

### Parallel Track 2 (Engineer B)

- **Weeks 1-2:** WM + Grounding + Reports
- **Weeks 3-4:** Bayesian + Temporal/Spatial + Transfer + Remaining pillars + Documentation

### Sync Points

- **Day 7 (End Week 1):** Review Gf + WM implementations
- **Day 14 (End Week 2):** Integration testing, full MVP validation
- **Day 21 (End Week 3):** Review extension pillars
- **Day 28 (End Week 4):** Final integration, documentation review, release

**Result:** 4 calendar weeks to full system with 2 engineers

-----

## Milestone Summary

|Milestone          |End of Week|Pillars|Cumulative|Dev-Days|
|-------------------|-----------|-------|----------|--------|
|MVP Complete       |2          |5      |5         |14      |
|Extensions Complete|4          |6      |11        |28      |
|Full System        |6          |10     |21        |42      |

-----

## Risk Mitigation

### Technical Risks

**Risk:** Probe generation produces invalid JSON  
**Mitigation:** Schema validation enforced; repair mechanism tested early

**Risk:** Bootstrap CIs too wide (>5 points)  
**Mitigation:** N≥100 probes per pillar; validate CI width in Phase 1

**Risk:** Tool server introduces network dependencies  
**Mitigation:** Socket blocking test enforced; localhost-only binding

**Risk:** IRT calibration doesn’t converge  
**Mitigation:** Fallback to fixed difficulty weights; test with synthetic data

### Schedule Risks

**Risk:** Pillar implementation takes longer than 2-3 days  
**Mitigation:** Stub incomplete pillars; prioritize scoring over edge cases

**Risk:** Integration issues delay end-to-end testing  
**Mitigation:** Daily integration smoke tests; API contracts defined early

**Risk:** Documentation lags implementation  
**Mitigation:** Docstrings mandatory in code reviews; examples written with code

-----

## Success Criteria

### Phase 1 (MVP)

- ✅ 5 pillars generate deterministic probes
- ✅ End-to-end pipeline produces valid reports
- ✅ Bootstrap CIs computed with width <5 points
- ✅ Test coverage ≥80%
- ✅ Example suite runs in <30 seconds

### Phase 2 (Extensions)

- ✅ 11 pillars operational
- ✅ Composite scoring functional with weights
- ✅ No-weak-links penalty working
- ✅ Reliability metrics (α≥0.8) achieved
- ✅ Test coverage ≥85%

### Phase 3 (Complete)

- ✅ All 21 pillars implemented
- ✅ IRT calibration functional
- ✅ Documentation complete and published
- ✅ CI/CD pipeline operational
- ✅ PyPI package available
- ✅ Test coverage ≥90%
- ✅ Drift detection automated

-----

## Post-Launch Roadmap (Optional)

### Months 1-2: Stability & Adoption

- Monitor drift across model updates
- Collect feedback from users
- Expand test corpus for grounding
- Add more tool families (SQL, file I/O)

### Months 3-4: Advanced Features

- Multi-modal probes (image+text)
- Cross-lingual variants (translate narratives)
- Adaptive difficulty (IRT-based item selection)
- Real-time calibration updates

### Months 5-6: Research Extensions

- Causal discovery tasks (structure learning)
- Meta-learning transfer (learn-to-learn)
- Adversarial probe generation (adversarial training)
- Human-in-the-loop validation

-----

## Contact & Support

- **Issues:** GitHub Issues
- **Documentation:** `docs/` directory
- **Examples:** `examples/` directory
- **Community:** Discord / Slack (TBD)

-----

**Version Control:**

- Roadmap v1.0.0 - Initial release
- Update frequency: Quarterly reviews
- Change log: See `CHANGELOG.md`