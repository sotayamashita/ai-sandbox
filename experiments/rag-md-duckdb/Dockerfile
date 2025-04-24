# syntax=docker/dockerfile:1
########################################################################
# Base: Python 3.12 slim (Debian 12 "bookworm")
########################################################################
FROM python:3.12-slim

########################################################################
# 1. Common build tools & prerequisites
########################################################################
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

########################################################################
# 3. Project dependencies
########################################################################
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Expose project virtual-env on PATH
ENV PATH="/app/.venv/bin:${PATH}"

########################################################################
# 4. Copy application source & default entrypoint
########################################################################
COPY src/ ./src/
CMD ["uv", "run", "src/main.py"]
