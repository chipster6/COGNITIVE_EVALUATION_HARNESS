ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG BASE_DIGEST
FROM ${BASE_IMAGE}${BASE_DIGEST:+@${BASE_DIGEST}}
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends \
    git curl ca-certificates jq make bash tini gcc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# yq + validators used in CI
RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64 -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq
RUN useradd -m -u 1000 dev
USER dev
WORKDIR /work
CMD ["/usr/bin/tini","--","bash"]
