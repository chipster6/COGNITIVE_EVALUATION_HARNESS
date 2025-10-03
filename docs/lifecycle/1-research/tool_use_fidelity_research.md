# Research Notes — tool_use_fidelity
version: 1.0.0

## Background & Motivation
Structured tool use must be deterministic and robust. Failures include wrong tool selection, malformed arguments, missing calls, duplicate calls, or refusal. We target deterministic function-calling fidelity across controlled and adversarial probes under fixed seeds and stable decoding parameters.

## Literature & Benchmarks (≥6)

1) Schick et al. 2023 — Toolformer: LMs Can Teach Themselves to Use Tools — <https://arxiv.org/abs/2302.04761>  
2) Yao et al. 2023 — ReAct: Synergizing Reasoning and Acting — <https://arxiv.org/abs/2210.03629>  
3) Patel et al. 2023 — Gorilla: Large Language Models Are Strong Tool Learners — <https://arxiv.org/abs/2305.15334>  
4) OpenAI 2023 — Function calling and tool use API update — <https://openai.com/blog/function-calling-and-other-api-updates>  
5) CRFM 2023–2024 — HELM benchmark framework — <https://crfm.stanford.edu/helm/latest/>  
6) Qin et al. 2023 — ToolBench: Making LLMs Masters of APIs — <https://arxiv.org/abs/2307.16789>

## Prior Art & Gaps

- Prior work shows tool-use promise but limited deterministic fidelity reporting, weak CI integration, and inconsistent adversarial coverage. Our gap fill: fixed-seed baseline with confidence intervals, probe taxonomy, JSON schemas, and CI gates.

## Key Insights

- Fidelity depends on constrained prompting and strict schema validation.
- ReAct-style traces help debugging but can mask low-level argument errors.
- Adversarial perturbations degrade argument formatting more than tool choice; both must be tested.

## Dependencies (from M0)

- Metrics keys: reproducibility_rate, latency_p95_ms, robustness_delta.
- Risks: dataset license risk, reviewer bandwidth, provider latency variance.
- Governance: CP-07 Tool Use Fidelity; CP-16 Robustness to Noise.

## Scope Summary

See ./tool_use_fidelity_scope.md

## Risk Annotations

- R1 License risk: we use synthetic data and MIT-licensed assets.
- R3 Latency variance: we collect P95 and run with fixed decoding to reduce noise.

## Decisions & Alternatives

See ./tool_use_fidelity_decision_log.md
