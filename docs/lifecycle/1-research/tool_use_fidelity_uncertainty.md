# Uncertainty & Reliability Plan — tool_use_fidelity
version: 1.0.0

## Confidence Intervals
- For proportions (e.g., reproducibility_rate): Wilson or bootstrap (B=2000).
- For ratios (robustness_delta): bootstrap CI on the ratio distribution.
- Report mean and 95% CI.

## Significance Tests
- One-sample proportion test vs target for reproducibility_rate at α=0.05.
- Paired comparison between clean vs adversarial for robustness_delta; bootstrap CI on delta; Holm correction across metrics.

## Power & Minimum n
- Target power 0.8 for detecting a 3 pp deviation from 0.95 at α=0.05.
- Minimum n per condition (clean/adversarial): 300.
- Seeds: {1..5}. Report per-seed and aggregate.

## Reproducibility
- Fixed seeds and decoding params.
- Record dataset version, checksum, env, commit SHA in run_manifest.json.
- Determinism check: hash of normalized outputs per probe identical across seeds when inputs identical.
