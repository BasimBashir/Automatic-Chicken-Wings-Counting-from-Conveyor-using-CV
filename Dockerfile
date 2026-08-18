# syntax=docker/dockerfile:1
FROM python:3.12-slim AS compiler
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "cython>=3.0" setuptools
WORKDIR /build
COPY app/ app/
COPY cython_build.py .
RUN python cython_build.py build_ext --inplace \
    && find app -name '*.c' -delete \
    && for so in app/core/*.cpython-*.so app/routers/*.cpython-*.so; do \
        [ -e "$so" ] || continue; rm -f "${so%.cpython-*.so}.py"; \
    done \
    && rm -rf build app/core/__pycache__ app/__pycache__ app/routers/__pycache__

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Ubuntu 24.04 ships Python 3.12 — no PPA needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY --from=compiler /build/app/ app/
COPY best.pt .

RUN mkdir -p app/uploads app/outputs

# Non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd  --uid 1001 --gid 1001 --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5580

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5580/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5580", "--workers", "1"]
