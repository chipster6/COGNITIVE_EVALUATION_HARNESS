# Evaluation Design — tool_use_fidelity
version: 1.0.0

## Probe Families
- Controlled: canonical prompts; one function call; strict argument schema.
- Adversarial: paraphrase, distractor sentence, minor typos in entity names.
- Stress: long context filler (≤ 2k tokens) preserving required tool call.

## Dataset & Splits
- Dataset: mock_api_calls v0.3 (MIT).
- Split: dev (design), test (baseline only).
- Class balance: 50% weather.get_forecast, 50% calendar.create_event.

## Metrics
- reproducibility_rate = exact-match on structured call JSON after canonicalization.
- robustness_delta = score_adv / score_clean per probe id.
- latency_p95_ms from per-run elapsed_ms.

## Controls
- Seeds {1..5}; temperature=0; top_p=1.
- Canonicalize whitespace and key order before comparison.
- JSON Schema validation for arguments.

## Reproducibility
- Pin dataset version + checksum.
- Record env, seeds, commit_sha in run_manifest.json.

## Scoring Pipeline
1. Generate probes → probes/*.jsonl
2. Execute model → runs/<model>/<date>/outputs.jsonl
3. Score → results/<model>/<date>/scores.json
4. Aggregate → tool_use_fidelity_baseline.json

## Error Taxonomy
- tool_choice_error, arg_schema_error, missing_call, extra_call, timeout, refusal, other
