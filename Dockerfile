# ---------------------------------------------------------------------------
# Multi-stage build.
#
# WHY TWO STAGES: the builder needs a compiler toolchain to build wheels; the
# runtime does not. Shipping build tools in the final image makes it larger and
# hands an attacker who gets code execution a compiler to work with. We compile
# in stage 1 and copy only the installed packages into stage 2.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 # Clean up in the SAME layer: a separate RUN would leave the files in the
 # previous layer and the image would not shrink at all.
 && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST, install, then copy the source. Docker caches layers
# by content, so this way an ordinary code change does not re-run pip install.
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --prefix=/install -r requirements-postgres.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    # Unbuffered stdout, or your logs appear minutes late (or not at all when
    # the container is killed) because Python buffered them.
    PYTHONUNBUFFERED=1

# Run as a NON-ROOT user. Root in a container is root on the host kernel if
# anything escapes; this is the cheapest meaningful hardening step there is.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /install /usr/local
WORKDIR /app
# WORKDIR creates /app as root; the app user must own it to write local files.
RUN chown appuser:appuser /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

# The orchestrator uses this to decide whether to route traffic here.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/live').status==200 else 1)"

# --workers: one uvicorn process per CPU core, because a single Python process
# uses one core. REMEMBER: total DB connections = workers x pool_size. Four
# workers with pool_size=5 is 20 connections from this container alone.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
