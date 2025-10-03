# Scope & Assumptions — tool_use_fidelity
version: 1.0.0

## In Scope

- Single-turn tool calls with one function per prompt.
- JSON-serializable arguments validated against a JSON Schema.
- Controlled probes (canonical prompts) and adversarial probes (paraphrase, distractor, small typos).
- Local-first execution on macOS; provider calls via OpenRouter.

## Out of Scope (M1)

- Multi-turn tools or tool-chaining.
- Streaming function-calling.
- Multimodal inputs.
- Non-JSON tool protocols (SOAP, XML).
- Real external APIs with side-effects (we use mock APIs).

## Assumptions

- Fixed decoding: temperature=0, top_p=1.
- Seeds in {1,2,3,4,5}; deterministic client.
- Dataset version pinned; manifest includes commit SHA and dataset checksum.
- Tool schema known a priori; responses must pass JSON Schema.
