# ARCS — Deployment Guide

Production-style deployment for the **demo UI** (`scripts/run_demo.py` → FastAPI on port **8000**). This is not a full eval/training stack — it runs the interactive pipeline for presentations and feedback collection.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose | Local or server |
| Router checkpoint | `artifacts/router-model/` on the **host** (gitignored; train or copy locally) |
| API keys | `GROQ_API_KEY`, `NVIDIA_API_KEY` — **never commit** real values |

---

## Quick start (Docker Compose)

```bash
# From repository root
cp .env.example .env
# Edit .env — paste real keys; do not commit .env

# Router weights must exist on the host (mounted read-only)
ls artifacts/router-model/config.json

# Recommended for containers: ONNX (faster cold start, no PyTorch at inference)
python scripts/export_router_onnx.py
# Set in .env: ARCS_ROUTER_BACKEND=onnx

docker compose up --build
```

Validate config without printing secrets: `docker compose config --quiet`

Open **http://127.0.0.1:8000**

---

## Quick start (docker run)

```bash
docker build -t arcs-demo .

docker run --rm -p 8000:8000 \
  --env-file .env \
  -e ARCS_ROUTER_BACKEND=onnx \
  -e ARCS_DEMO_HOST=0.0.0.0 \
  -v "$(pwd)/artifacts/router-model:/app/artifacts/router-model:ro" \
  -v "$(pwd)/logs:/app/logs" \
  arcs-demo
```

---

## Environment variables

Copy from `.env.example`. **Do not commit `.env`** with real secrets.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (for queries) | — | Generator + spec models |
| `NVIDIA_API_KEY` | Yes (for queries) | — | LLM judge |
| `ARCS_ROUTER_BACKEND` | No | `torch` | **`onnx` recommended in Docker/cloud** |
| `ARCS_ROUTER_CONFIDENCE` | No | `0.75` | Router fallback threshold |
| `ARCS_GENERATOR_MODEL` | No | `llama-3.3-70b-versatile` | Default Groq model |
| `NVIDIA_JUDGE_MODEL` | No | `meta/llama-3.1-8b-instruct` | Judge model |
| `PORT` | No | `8000` | Listen port (Railway/Render/Fly set this) |
| `ARCS_DEMO_PUBLIC` | No | `0` | `1` shows public-demo disclaimer banner |
| `ARCS_DEMO_RATE_LIMIT` | No | `8` | Max `/api/query` requests per IP per window (`0` = off) |
| `ARCS_DEMO_RATE_WINDOW` | No | `60` | Rate-limit window seconds |
| `ARCS_DEMO_FEEDBACK_RATE_LIMIT` | No | `30` | Max `/api/feedback` per IP per window |
| `ARCS_DEMO_PIPELINE_TIMEOUT` | No | `180` | Pipeline wall-clock timeout (seconds) |

`docker-compose.yml` reads `.env` via `env_file`. The template lives in **`.env.example`**.

---

## Health check

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Example response:

```json
{
  "status": "ok",
  "groq_configured": true,
  "nvidia_configured": true,
  "router_backend": "onnx",
  "public_demo": false,
  "disclaimer": null
}
```

- **`status`**: process is up
- **`groq_configured` / `nvidia_configured`**: whether keys are present (not whether quotas are valid)
- **`router_backend`**: active `ARCS_ROUTER_BACKEND` value
- **`public_demo`**: when `ARCS_DEMO_PUBLIC=1`, UI shows `disclaimer`

Legacy alias: `GET /api/health` returns the same payload.

Docker `HEALTHCHECK` and Compose healthcheck both hit `/health`.

---

## Post-deploy smoke (minimal)

After the container is up:

```bash
# 1) Health — keys present, preferred backend onnx
curl -sS http://127.0.0.1:8000/health | python -m json.tool

# 2) One live query (uses Groq + NVIDIA; may take 1–3 minutes)
curl -sS -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is a Python list comprehension?"}' | python -m json.tool
```

Expect HTTP **200** with `query_id` and an `answer`. HTTP **429** means rate limit; **503** means missing keys or pipeline timeout/unavailable — no stack traces in the JSON body.

Optional full 5-query harness (local/CI only; not required for a public demo):

```bash
python scripts/smoke_e2e.py --dry-run
python scripts/smoke_e2e.py --json --quiet
```

---

## Cloud deploy (Railway / Render / Fly)

These platforms build a container, inject secrets as env vars, and expose `PORT`. ARCS is a **single-process demo** — no Postgres, Redis, or auth.

### Shared setup

1. Train/export the router locally:
   ```bash
   python -m arcs.router.train   # if needed
   python scripts/export_router_onnx.py
   ```
2. Set secrets in the platform dashboard (never in git):
   - `GROQ_API_KEY`
   - `NVIDIA_API_KEY`
3. Set non-secret env:
   - `ARCS_ROUTER_BACKEND=onnx`
   - `ARCS_DEMO_PUBLIC=1` (recommended for internet-facing demos)
   - `ARCS_DEMO_RATE_LIMIT=8`
   - `ARCS_DEMO_PIPELINE_TIMEOUT=180`
4. **Router weights:** `artifacts/` is gitignored and excluded from the default Docker build context. For cloud you must either:
   - **Bake at build time:** place `artifacts/router-model/` on the build machine, allow it in `.dockerignore` (`!artifacts/router-model/`), uncomment the `COPY artifacts/router-model/` line in the Dockerfile, build and push the image; or
   - **Mount a volume** (Fly volumes / Render disks) at `/app/artifacts/router-model`.
5. Health check path: `/health` (HTTP). Start period ≥ 60s (ONNX/torch cold start).

### Railway

- New project → Deploy from GitHub (or deploy a pre-built image).
- Variables: paste the secrets/env above. Railway sets `PORT` automatically; `scripts/run_demo.py` respects it.
- If building from git without router weights, push a **pre-built image** that already contains `artifacts/router-model` (ONNX + config).
- Public URL smoke:
  ```bash
  curl -sS https://YOUR_APP.up.railway.app/health
  curl -sS -X POST https://YOUR_APP.up.railway.app/api/query \
    -H 'Content-Type: application/json' \
    -d '{"query":"Hello"}'
  ```

### Render

- New **Web Service** from repo, or deploy a Docker image.
- Docker command is the image `CMD` (no change needed).
- Health check: `/health`.
- Disk (optional): mount at `/app/artifacts/router-model` and `/app/logs`, or bake the router into the image as above.
- Same env vars as Railway.

### Fly.io

```bash
fly launch          # create app from Dockerfile
fly secrets set GROQ_API_KEY=... NVIDIA_API_KEY=...
fly volumes create arcs_router --size 1   # optional persistent router/logs
# mount in fly.toml under [mounts], destination /app/artifacts/router-model
fly deploy
```

Set `ARCS_ROUTER_BACKEND=onnx` and `ARCS_DEMO_PUBLIC=1` in `fly.toml` `[env]` or via secrets/config.

### What not to add

No Postgres, Redis, OAuth, multi-tenant SaaS, or eval/RQ1 jobs on this deploy path. Feedback stays in `logs/requests.jsonl` on the container filesystem (or a mounted volume).

---

## Volumes (Compose / VMs)

| Host path | Container path | Mode | Why |
|---|---|---|---|
| `./artifacts/router-model` | `/app/artifacts/router-model` | ro | DistilBERT router (+ optional `model.onnx`) |
| `./logs` | `/app/logs` | rw | Request logs and feedback for repair loops |

`artifacts/` and `logs/` are excluded from the image via `.dockerignore` — mount or bake them at deploy time.

---

## Router backend in production

1. Export once on a dev machine (requires PyTorch):
   ```bash
   python scripts/export_router_onnx.py
   ```
2. Set `ARCS_ROUTER_BACKEND=onnx` in `.env` / platform env
3. Smoke-test before deploy:
   ```bash
   python scripts/smoke_router.py --backend onnx --query "test query"
   ```

See [README — ONNX router deployment](../README.md#onnx-router-deployment).

---

## Image layout

Multi-stage **Dockerfile**:

1. **builder** — creates `/opt/venv`, installs `requirements.txt`
2. **runtime** — copies venv + `arcs/`, `scripts/`, `data/`; defaults `ARCS_ROUTER_BACKEND=onnx`; exposes 8000; runs:
   ```bash
   python scripts/run_demo.py --host 0.0.0.0 --port ${PORT:-8000}
   ```

---

## Security notes

- Never commit `.env`, API keys, or customer logs
- `.dockerignore` excludes `.env`, `artifacts/`, and `logs/`
- Mount router weights read-only in Compose
- Demo binds to `0.0.0.0` in containers; put a reverse proxy / TLS in front for public internet exposure
- Per-IP in-memory rate limits protect `/api/query` and `/api/feedback` (not a substitute for a CDN/WAF)
- Client responses never include stack traces; failures return generic 500/503 JSON

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `503` API keys not configured | Set keys in `.env` / platform secrets; restart |
| `503` Pipeline timed out | Raise `ARCS_DEMO_PIPELINE_TIMEOUT` or retry; check Groq/NVIDIA status |
| `429` Rate limit exceeded | Wait for the window; raise `ARCS_DEMO_RATE_LIMIT` if needed |
| Router `FileNotFoundError` | Mount or bake `artifacts/router-model` |
| ONNX `model.onnx not found` | Run `export_router_onnx.py` or set `ARCS_ROUTER_BACKEND=torch` |
| Slow container start | Use ONNX backend; PyTorch import is heavy |
| Health check fails during start | Wait for `start_period` (60s); first load can be slow |
