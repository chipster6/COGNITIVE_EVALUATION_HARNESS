FROM python:3.11-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates jq make bash tini gcc \
 && rm -rf /var/lib/apt/lists/*
# Node toolchain to mirror CI validators
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
# yq + validators used in CI
RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64 -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq
RUN npm i -g ajv-cli@5 ajv-formats@3 @mermaid-js/mermaid-cli@10.9.1
RUN useradd -m -u 1000 dev
USER dev
WORKDIR /work
CMD ["/usr/bin/tini","--","bash"]
