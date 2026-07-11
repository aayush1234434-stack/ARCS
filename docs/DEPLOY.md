# ARCS — Deployment Guide

Production-style deployment skeleton for the **demo UI** (`scripts/run_demo.py` → FastAPI on port **8000**). This is not a full eval/training stack — it runs the interactive pipeline for presentations and feedback collection.

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

# Optional: ONNX for faster cold start inside the container
python scripts/export_router_onnx.py
# Set in .env: ARCS_ROUTER_BACKEND=onnx

docker compose up --build
```

Open **http://127.0.0.1:8000**

---

## Quick start (docker run)

```bash
docker build -t arcs-demo .

docker run --rm -p 8000:8000 \
  --env-file .env \
  -e ARCS_ROUTER_BACKEND=onnx \
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
| `ARCS_ROUTER_BACKEND` | No | `torch` | `onnx` recommended in Docker (no PyTorch import at inference) |
| `ARCS_ROUTER_CONFIDENCE` | No | `0.75` | Router fallback threshold |
| `ARCS_GENERATOR_MODEL` | No | `llama-3.3-70b-versatile` | Default Groq model |
| `NVIDIA_JUDGE_MODEL` | No | `meta/llama-3.1-8b-instruct` | Judge model |

`docker-compose.yml` reads `.env` via `env_file`. The template lives in **`.env.example`** — copy it to `.env` and fill in keys locally.

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
  "router_backend": "onnx"
}
```

- **`status`**: process is up
- **`groq_configured` / `nvidia_configured`**: whether keys are present (not whether quotas are valid)
- **`router_backend`**: active `ARCS_ROUTER_BACKEND` value

Legacy alias: `GET /api/health` returns the same payload.

Docker `HEALTHCHECK` and Compose healthcheck both hit `/health`.

---

## Post-deploy verification

After the container is up and `/health` shows configured API keys, run the smoke harness. It exercises **5 fixed eval queries** (one per domain plus coding-in-prose `eval-042`) through the full pipeline and requires **zero ERROR rows** — each query must finish with verifier status PASS or FAIL (infra failures count as ERROR).

| Step | Command | Secrets | Expected |
|---|---|---|---|
| Router only | `python scripts/smoke_router.py --backend onnx --query "test query"` | None | Routed domain printed |
| Smoke plan | `python scripts/smoke_e2e.py --dry-run` | None | Lists 5 query ids |
| Live smoke | `python scripts/smoke_e2e.py --json --quiet` | Groq + NVIDIA | `exit_code: 0`, `summary.errors: 0` |

Fixed query ids: `eval-024` (LEGAL), `eval-013` (MEDICAL), `eval-033` (GENERAL), `eval-001` (CODING), `eval-042` (CODING prose).

```bash
# Router-only (no Groq/NVIDIA calls)
python scripts/smoke_router.py --backend onnx --query "test query"

# Full pipeline — plan only (no API)
python scripts/smoke_e2e.py --dry-run

# Live smoke — exit 0 iff 5/5 complete with zero ERROR
python scripts/smoke_e2e.py --json --quiet
```

**Exit codes** (`scripts/smoke_e2e.py`):

| Code | Meaning |
|---:|---|
| `0` | All 5 queries completed (PASS or FAIL); `summary.errors == 0` |
| `1` | One or more ERROR / UNKNOWN rows — inspect `error_class` per row in JSON |
| `2` | Setup failure (missing API keys, router checkpoint, or eval file) |

JSON output includes `exit_code`, per-row `error_class` (`rate_limit`, `judge_parse`, `sandbox`, `empty_code`, `unknown`), and an `error_classes` breakdown when any row fails.

Inside Docker (keys from `.env`, router mounted):

```bash
docker compose exec demo python scripts/smoke_e2e.py --json --quiet
```

**CI (optional):** [`.github/workflows/smoke-e2e.yml`](../.github/workflows/smoke-e2e.yml) runs on `workflow_dispatch` and a nightly schedule. It **always** runs the dry-run step (no secrets). The **live** smoke step runs only when both `GROQ_API_KEY` and `NVIDIA_API_KEY` repository secrets are configured; otherwise the workflow prints a notice and stays green — it is **not** part of the default push CI (`ci.yml`). Forks without secrets therefore pass without failing on missing keys.

---

## Volumes

| Host path | Container path | Mode | Why |
|---|---|---|---|
| `./artifacts/router-model` | `/app/artifacts/router-model` | ro | DistilBERT router (+ optional `model.onnx`) |
| `./logs` | `/app/logs` | rw | Request logs and feedback for repair loops |

`artifacts/` and `logs/` are excluded from the image via `.dockerignore` — mount them at run time.

---

## Router backend in production

1. Export once on a dev machine (requires PyTorch):
   ```bash
   python scripts/export_router_onnx.py
   ```
2. Set `ARCS_ROUTER_BACKEND=onnx` in `.env`
3. Smoke-test before deploy:
   ```bash
   python scripts/smoke_router.py --backend onnx --query "test query"
   ```

See [README — ONNX router deployment](../README.md#onnx-router-deployment).

---

## Image layout

Multi-stage **Dockerfile**:

1. **builder** — creates `/opt/venv`, installs `requirements.txt`
2. **runtime** — copies venv + `arcs/`, `scripts/`, `data/`; exposes 8000; runs:
   ```bash
   python scripts/run_demo.py --host 0.0.0.0 --port 8000
   ```

---

## Security notes

- Never commit `.env`, API keys, or customer logs
- `.dockerignore` excludes `.env`, `artifacts/`, and `logs/`
- Mount router weights read-only in Compose
- This demo binds to `0.0.0.0` inside the container; put a reverse proxy / TLS in front for public internet exposure

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `503 Missing API keys` on `/api/query` | Set keys in `.env`; restart container |
| Router `FileNotFoundError` | Mount `artifacts/router-model` or train router locally |
| ONNX `model.onnx not found` | Run `export_router_onnx.py` or set `ARCS_ROUTER_BACKEND=torch` |
| Slow container start | Use ONNX backend; PyTorch import is heavy |
| Health check fails during start | Wait for `start_period` (60s); first torch load can be slow |
