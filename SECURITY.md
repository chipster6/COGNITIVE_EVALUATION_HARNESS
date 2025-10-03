# Security Scanning & Dev Image Maintenance

## Dev Image Security Policy

This project uses a **CRITICAL CVE gate** on the dev container image to prevent fixable high-severity vulnerabilities from entering the codebase.

### Image Build Strategy

- **Base**: `python:3.12-slim-bookworm` for minimal attack surface and current patches
- **Maintenance**: Weekly automated rebuild (Mondays at 6 AM UTC) to pick up base image security patches
- **Dev Tools**: Node/npm/ajv/mermaid are installed in CI workflows, not baked into the image

### Trivy Scanning Configuration

- **Gate**: Fails builds on any CRITICAL severity vulnerabilities
- **Scope**: OS packages and libraries (`--vuln-type os,library`)
- **Fixed-only**: `--ignore-unfixed true` (only fails on vulnerabilities that have patches available)
- **Timeout**: 5 minutes to prevent workflow flakes
- **Ignore file**: `.trivyignore` for temporary, documented exceptions

### Running Scans Locally

```bash
# Quick local test during development
docker build -t test-dev:local -f .devcontainer/Dockerfile .devcontainer
trivy image --severity CRITICAL --vuln-type os,library test-dev:local

# Full scan matching CI
trivy image --severity CRITICAL --vuln-type os,library --ignore-unfixed --timeout 5m --trivyignore .trivyignore test-dev:local
```

### Managing Exceptions (.trivyignore)

Only use `.trivyignore` for:

- False positives confirmed with upstream
- CVEs unreachable in dev container context
- Short-term workarounds while waiting for base image patches

Required format:

```
# CVE-YYYY-XXXX
# Reason: [brief justification]
# Expires: YYYY-MM-DD
CVE-YYYY-XXXX
```

**Policy**: All ignores must include expiration dates (30-60 days max) to force periodic review.

### CI Workflows

- **Dev Image Build**: `.github/workflows/devimage.yml` - rebuilds on `.devcontainer/**` changes + weekly schedule
- **Image Scan**: `.github/workflows/scout.yml` - scans after successful dev image builds
- **Retrigger**: Use "Re-run jobs" in GitHub Actions or trigger `devimage.yml` via workflow_dispatch

### Troubleshooting

**Scan failures**: Check the Actions log for specific CVE IDs and affected packages. Options:

1. **Preferred**: Update base image or wait for upstream patches
2. **Temporary**: Add targeted `.trivyignore` entry with justification and expiry

**Weekly build failures**: Usually indicate new CVEs in the base image. Review and either:

- Upgrade to a newer base image tag
- Add temporary ignore while tracking upstream fix

### Security Contact

For security-related questions or to report vulnerabilities, please create a GitHub issue with the "security" label.