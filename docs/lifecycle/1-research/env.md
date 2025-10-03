# Hermetic Dev Environment

Use the published dev image for consistent local runs matching CI.

Pull latest image

```bash
docker pull ghcr.io/chipster6/COGNITIVE_EVALUATION_HARNESS/dev:latest
```

Run with bind mount

```bash
docker run --rm -it \
  -v "$(pwd)":/work \
  -w /work \
  ghcr.io/chipster6/COGNITIVE_EVALUATION_HARNESS/dev:latest
```

Inside the container, tools available:

- yq (CLI)
- Python 3.12

Note: Node/ajv/mermaid are installed in CI or on host, not baked into the image.

Optional: bake scripts/Makefile targets to orchestrate baselines in M2.
