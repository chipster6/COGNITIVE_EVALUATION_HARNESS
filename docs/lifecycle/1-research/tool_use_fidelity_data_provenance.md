# Data Provenance — tool_use_fidelity
version: 1.0.0

## Datasets

| ID             | Version | License | Source URL | Checksum (sha256)                                | Storage Path                     | Access |
|----------------|---------|---------|------------|--------------------------------------------------|----------------------------------|--------|
| mock_api_calls | v0.3    | MIT     | internal   | 4b7f1b9a1d7a0a6c2a1b3b4d6b2f1c9d8e7a6b4c2d1e0f9a | data/mock_api_calls/v0.3/        | team   |

## Generation

- Synthetic prompts with deterministic templates.
- Tool schema: weather.get_forecast(city, date), calendar.create_event(title, date, time).
- Generation script stores split manifests and a global checksum file.

## PII / Sensitive Data

- None. Synthetic only. No real user data.

## Retention & Compliance

- Keep dataset JSONL in repo (MIT).
- No run logs with raw model text committed; only metrics and scores.
- Respect provider ToS for API usage.
