# AGENTS.md

Instructions for a coding agent working in this repo. Humans: see `README.md`.

## What this app is

A **Databricks App** (FastAPI) that routes chat messages to different LLMs.
The routing *decision* is made by **JEV (TypeSafe System One)**, an external
reasoning service at `https://api.typesafe.ai`. JEV is **not** an LLM and does
**not** generate the chat reply — it only picks *which* model should answer.

Flow per message:

1. The app sends the conversation transcript + the user-defined routing options
   to JEV using its native **`choice`** question type. Criteria are
   `{option_label: description}`. JEV returns `{choice, confidence, probabilities}`.
2. The winning option maps to a **Databricks Foundation Model API (FMAPI)**
   serving endpoint (e.g. `databricks-claude-*`, `databricks-gpt-*`).
3. The app invokes that endpoint (with the app's service principal) to produce
   the actual reply.

So: **JEV = router/judge, Databricks FMAPI = the model that answers.**

## Repo layout

- `src/app.py` — the entire app: UI (`GET /`), `GET /router/models`,
  `POST /router/route`, `POST /router/chat`, `GET /healthz`. Frontend is inline
  HTML/JS in `_INDEX_HTML`.
- `src/app.yaml` — Databricks Apps launch config + env (incl. the JEV secret).
- `src/requirements.txt` — runtime deps.
- Everything deployable lives under `src/`. Keep it a valid Databricks Apps
  bundle: `app.yaml` + `requirements.txt` + `app.py` at the root of `src/`.

## Hard rules

- **Never hardcode `JEV_API_KEY`** (or any token) in code, YAML, README, or
  commits. It is injected at runtime via `app.yaml` `valueFrom` from a
  Databricks secret. `.gitignore` blocks `.env`, `secrets*.json`, keys, etc.
- Don't move files out of `src/` — the deploy step points at `src/`.
- Inference uses raw REST `POST /serving-endpoints/{model}/invocations` via
  `w.api_client.do(...)`, **not** the SDK's typed `serving_endpoints.query()`
  (the typed call has dict-handling quirks on some endpoints). Keep it that way.
- Reasoning models may return text in `message.reasoning_content` instead of
  `message.content`; keep the `reasoning_content` fallback and generous
  `max_tokens`.

## Deploy (Databricks Apps)

Prereqs: Databricks CLI authenticated to the target workspace, and a JEV API key.

1. **Create the JEV secret** (once per workspace):
   ```bash
   databricks secrets create-scope jev
   databricks secrets put-secret jev api_key   # paste the JEV key
   ```
   `app.yaml` references it as `valueFrom: jev-api-key`. Ensure the app has an
   app resource named `jev-api-key` mapped to scope `jev`, key `api_key`.

2. **Sync `src/` to the workspace and create + deploy the app:**
   ```bash
   databricks sync ./src /Workspace/Users/<you>/jev-smart-router
   databricks apps create jev-smart-router
   databricks apps deploy jev-smart-router \
     --source-code-path /Workspace/Users/<you>/jev-smart-router
   ```

3. **Grant the app's service principal `CAN QUERY`** on the FMAPI serving
   endpoints you want to route to. Pay-per-token foundation models are usually
   already queryable; provisioned/custom endpoints may need an explicit grant.

4. **Verify:** open the app URL, or hit `GET /healthz` (returns
   `jev_key_present`) and `GET /router/models` (should list `databricks-*`
   `llm/v1/chat` endpoints). If `/router/models` is empty, the SP lacks endpoint
   access. If routing 502s with a JEV error, check the secret is wired and the
   key is valid.

## Redeploy after edits

Re-run the `databricks sync ...` then `databricks apps deploy ...` steps above
(pointing at the same workspace path). No code lives outside `src/`, so syncing
`src/` is sufficient.

## Local run

```bash
cd src && pip install -r requirements.txt
export JEV_API_KEY=...            # JEV key
export DATABRICKS_HOST=...        # for FMAPI access from your machine
export DATABRICKS_TOKEN=...       # or a CLI profile / OAuth
uvicorn app:app --reload --port 8080
```

## Config (env vars)

`JEV_API_KEY` (required), `JEV_API_URL` (default
`https://api.typesafe.ai/v1/systemone`), `JEV_MODEL` (default `jev-latest`),
`JEV_TIMEOUT` (default `30`), `JEV_LOG_PROMPT` (`1` to log routing decisions to
stdout).
