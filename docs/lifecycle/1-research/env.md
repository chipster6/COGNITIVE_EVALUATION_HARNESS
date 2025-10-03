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

Inside the container, validators are available:
- markdownlint, yamllint, ajv, yq, node 20, python 3.11

Optional: bake scripts/Makefile targets to orchestrate baselines in M2.
