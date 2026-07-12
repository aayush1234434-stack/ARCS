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

## Cloud deploy (Railway recommended; Render alternative)

Use branch **`cursor/production-demo-hardening`** (or merge [PR #7](https://github.com/aayush1234434-stack/ARCS/pull/7) first). That branch respects platform `PORT` and defaults to ONNX in the image.

ARCS is a **single-process demo** — no Postgres, Redis, or auth.

### How the router gets into the container

`artifacts/router-model/` is **gitignored** (do not commit weights). Two options:

| Method | When to use | How |
|---|---|---|
| **Bake (recommended)** | Railway/Render build from your laptop or CI that has the checkpoint | Dockerfile `COPY artifacts/router-model/` — already enabled. Image includes ONNX + tokenizer (~260MB). Torch `model.safetensors` is excluded via `.dockerignore`. |
| **Volume / disk** | Persistent host storage (Compose, Fly volume, Render disk) | Mount host dir → `/app/artifacts/router-model`. Compose does this locally. |

Before any cloud build on your machine:

```bash
ls artifacts/router-model/config.json artifacts/router-model/model.onnx
# if model.onnx missing:
python scripts/export_router_onnx.py
```

Required files for `ARCS_ROUTER_BACKEND=onnx`: `config.json`, `model.onnx`, `tokenizer.json`, `tokenizer_config.json`.

### Required environment variables

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq secret |
| `NVIDIA_API_KEY` | your NVIDIA secret |
| `ARCS_ROUTER_BACKEND` | `onnx` |
| `ARCS_DEMO_PUBLIC` | `1` (recommended for public URL) |

Optional: `ARCS_DEMO_RATE_LIMIT=8`, `ARCS_DEMO_PIPELINE_TIMEOUT=180`.

---

### Railway — exact steps (CLI, bake on your machine)

**Why CLI + local bake:** GitHub builds cannot see gitignored weights. Building from your laptop uploads the Docker context (including `artifacts/router-model/`) so the checkpoint is baked.

1. **Install + login**
   ```bash
   npm i -g @railway/cli    # or: brew install railway
   railway login            # browser OAuth
   ```

2. **Use the deploy branch**
   ```bash
   cd /Users/aayushsingh/Developer/ARCS
   git checkout cursor/production-demo-hardening
   git pull
   ls artifacts/router-model/model.onnx   # must exist
   ```

3. **Create project (click path once)**
   - Open [https://railway.app/new](https://railway.app/new)
   - **Empty Project** → name it `arcs-demo`
   - In the project: **+ Create** → **Empty Service** → name `demo`
   - Or from CLI in the repo:
     ```bash
     railway init    # link / create project
     railway link    # if project already exists
     ```

4. **Set variables (click path)**
   - Project → service **demo** → **Variables** → **+ New Variable** (or Raw Editor):
     ```
     GROQ_API_KEY=...
     NVIDIA_API_KEY=...
     ARCS_ROUTER_BACKEND=onnx
     ARCS_DEMO_PUBLIC=1
     ```
   - Or CLI (paste real keys in your terminal only):
     ```bash
     railway variables set ARCS_ROUTER_BACKEND=onnx ARCS_DEMO_PUBLIC=1
     railway variables set GROQ_API_KEY=YOUR_KEY NVIDIA_API_KEY=YOUR_KEY
     ```

5. **Generate a public URL (click path)**
   - Service → **Settings** → **Networking** → **Generate Domain**
   - Health check path (if asked): `/health`
   - Railway sets `PORT` automatically; the image CMD uses `${PORT:-8000}`.

6. **Deploy (CLI — bakes router)**
   ```bash
   # From repo root; uploads build context including artifacts/router-model/
   railway up --detach
   ```
   Watch the build on the Railway dashboard until **Success**.

7. **Smoke**
   ```bash
   curl -sS https://YOUR_APP.up.railway.app/health | python -m json.tool
   # expect: status=ok, router_backend=onnx, groq_configured=true, nvidia_configured=true

   curl -sS -X POST https://YOUR_APP.up.railway.app/api/query \
     -H 'Content-Type: application/json' \
     -d '{"query":"What is a Python list comprehension?"}' | python -m json.tool
   ```

**Railway + GitHub (no local bake):** only works if you push a **pre-built image** to a registry that already contains `/app/artifacts/router-model`, or you add a private download step in the Dockerfile. Plain “Deploy from GitHub” will **not** include the router.

---

### Render — exact steps (alternative)

Render Git builds also lack gitignored files. Prefer **Deploy an existing image** you built locally, or a Render **Disk** you populate once.

#### Option A — local image → Render (simplest)

```bash
git checkout cursor/production-demo-hardening
docker build -t arcs-demo:onnx .
# Tag + push to Docker Hub or GHCR, e.g.:
docker tag arcs-demo:onnx YOUR_DOCKERHUB_USER/arcs-demo:onnx
docker push YOUR_DOCKERHUB_USER/arcs-demo:onnx
```

Click path:

1. [https://dashboard.render.com](https://dashboard.render.com) → **New +** → **Web Service**
2. **Existing Image** → `YOUR_DOCKERHUB_USER/arcs-demo:onnx`
3. Instance: free/starter is fine for demos
4. **Environment** → add `GROQ_API_KEY`, `NVIDIA_API_KEY`, `ARCS_ROUTER_BACKEND=onnx`, `ARCS_DEMO_PUBLIC=1`
5. Health check path: `/health`
6. **Create Web Service**

#### Option B — GitHub + Disk (no bake in image)

1. **New +** → **Web Service** → connect `aayush1234434-stack/ARCS`
2. Branch: `cursor/production-demo-hardening`
3. Runtime: **Docker**
4. Add a **Disk** mounted at `/app/artifacts/router-model`
5. After first deploy, upload checkpoint onto the disk (Render Shell / one-shot `scp` pattern) — `config.json`, `model.onnx`, tokenizer files
6. Same env vars as above

---

### Fly.io (optional)

```bash
fly launch          # Dockerfile
fly secrets set GROQ_API_KEY=... NVIDIA_API_KEY=...
fly volumes create arcs_router --size 1   # optional
fly deploy          # bake works if router present in build context
```

### What not to add

No Postgres, Redis, OAuth, multi-tenant SaaS, or eval/RQ1 jobs on this deploy path. Feedback stays in `logs/requests.jsonl` on the container filesystem (ephemeral on Railway unless you add a volume).

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
