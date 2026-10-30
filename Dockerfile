# Serving image for the churn API (Cloud Run).
# The training image lives in Dockerfile.train and is a separate concern.

# ── Stage 1: build the virtualenv ────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Only the venv crosses the stage boundary — no pip cache, no build metadata.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ /app/src/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

# Cloud Run injects $PORT at runtime, so it must be expanded by a shell.
# exec keeps uvicorn as PID 1 so SIGTERM reaches it on scale-down.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
