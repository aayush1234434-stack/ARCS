# ARCS demo — multi-stage image with a project-local .venv at /opt/venv
#
# Build:  docker build -t arcs-demo .
# Run:    docker run --rm -p 8000:8000 --env-file .env \
#           -e ARCS_ROUTER_BACKEND=onnx \
#           -e ARCS_DEMO_HOST=0.0.0.0 \
#           -v "$(pwd)/artifacts/router-model:/app/artifacts/router-model:ro" \
#           -v "$(pwd)/logs:/app/logs" arcs-demo
#
# Cloud (Railway/Render/Fly): bake or mount router weights; set secrets via
# platform env vars; prefer ARCS_ROUTER_BACKEND=onnx. See docs/DEPLOY.md.

FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARCS_ROUTER_BACKEND=onnx \
    ARCS_DEMO_HOST=0.0.0.0 \
    PORT=8000

# Application source (see .dockerignore for exclusions)
COPY arcs/ arcs/
COPY scripts/ scripts/
COPY data/ data/
COPY requirements.txt .

# Bake DistilBERT router for cloud deploys (ONNX + tokenizer; no volume required).
# Local Compose still mounts ./artifacts/router-model over this path.
# Requires model.onnx — run: python scripts/export_router_onnx.py
COPY artifacts/router-model/ artifacts/router-model/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1'

CMD ["sh", "-c", "python scripts/run_demo.py --host 0.0.0.0 --port ${PORT:-8000}"]
