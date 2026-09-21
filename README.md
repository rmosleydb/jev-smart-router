# 🧭 JEV Smart Router

A [Databricks App](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)
that uses [TypeSafe JEV (System One)](https://typesafe.ai) to decide **which
model** should answer each message, then runs the real inference on the chosen
[Databricks Foundation Model API (FMAPI)](https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html)
endpoint.

You define named **routing options** (e.g. *Coding*, *General*, *Cheap & fast*),
each with a plain-English description and a target model. For every message, JEV
reads the conversation and picks the option whose description best fits — a
model-agnostic, explainable router with confidence scores and a probability
breakdown per option.

![routing UI](docs/screenshot.png)

> The screenshot path is a placeholder — drop a PNG at `docs/screenshot.png` if
> you want it to render.

## How it works

```
                 ┌─────────────────────────────────────────────┐
  user message   │  JEV Smart Router (FastAPI on Databricks App)│
  ──────────────▶│                                              │
                 │  1. POST conversation + option descriptions  │
                 │     to JEV `choice` question type            │
                 │            │                                 │
                 │            ▼                                 │
                 │     JEV returns {choice, confidence,         │
                 │     probabilities}                           │
                 │            │                                 │
                 │            ▼                                 │
                 │  2. map winning option → serving endpoint    │
                 │     run inference via app service principal  │
                 └────────────┬─────────────────────────────────┘
                              ▼
                    Databricks FMAPI endpoint
                    (databricks-claude-*, databricks-gpt-*, …)
                              │
                              ▼
                    assistant reply (+ which model, timings)
```

- **JEV makes the routing decision** using its native `choice` question type:
  the criteria are `{option_label: description}`, and JEV returns the winning
  label plus a full probability distribution.
- **Databricks runs the inference.** The winning option maps to an FMAPI
  serving endpoint, invoked with the app's service principal — no per-user
  token needed.
- The UI shows the routed model, JEV's confidence, a hover tooltip with the
  per-option probability breakdown and JEV routing latency, and the model's
  inference time.

## Endpoints

| Method | Path              | Purpose                                                       |
|--------|-------------------|---------------------------------------------------------------|
| GET    | `/`               | Single-page UI (routing-option manager + chat)                |
| GET    | `/router/models`  | Live list of `databricks-*` `llm/v1/chat` FMAPI endpoints     |
| POST   | `/router/route`   | Ask JEV which option/model to route to (decision only)        |
| POST   | `/router/chat`    | Route via JEV, then run inference on the chosen model         |
| GET    | `/healthz`        | Health probe                                                  |

### Request shape (`/router/route` and `/router/chat`)

```json
{
  "messages": [{"role": "user", "content": "Write a Python quicksort"}],
  "options": [
    {"label": "Coding",  "description": "Programming, code, debugging.",   "model": "databricks-claude-opus-4"},
    {"label": "General", "description": "Everyday questions, chit-chat.",  "model": "databricks-gpt-oss-120b"}
  ]
}
```

### Response (`/router/chat`)

```json
{
  "reply": "def quicksort(a): ...",
  "choice": "Coding",
  "model": "databricks-claude-opus-4",
  "confidence": 1.0,
  "probabilities": {"Coding": 1.0, "General": 0.0},
  "latency_ms": 147.1,
  "inference_ms": 1006.9
}
```

## Configuration

All configuration is via environment variables:

| Variable         | Required | Default                                | Description                                              |
|------------------|----------|----------------------------------------|----------------------------------------------------------|
| `JEV_API_KEY`    | ✅       | —                                      | TypeSafe JEV API key. Inject from a secret — never hardcode. |
| `JEV_API_URL`    |          | `https://api.typesafe.ai/v1/systemone` | JEV System One endpoint.                                 |
| `JEV_MODEL`      |          | `jev-latest`                           | JEV model tag.                                           |
| `JEV_TIMEOUT`    |          | `30`                                   | JEV request timeout (seconds).                           |
| `JEV_LOG_PROMPT` |          | `0`                                    | `1` to log routing decisions to stdout.                  |

## Deploy to Databricks Apps

1. **Store the JEV key in a secret** (never commit it):

   ```bash
   databricks secrets create-scope jev
   databricks secrets put-secret jev api_key   # paste your key
   ```

   The included `app.yaml` references it via `valueFrom: jev-api-key`. Make sure
   your app has an [app resource](https://docs.databricks.com/en/dev-tools/databricks-apps/app-development.html#reference-secrets)
   named `jev-api-key` mapped to that secret (scope `jev`, key `api_key`).

2. **Sync the `src/` folder to your workspace and create the app:**

   ```bash
   databricks sync ./src /Workspace/Users/you@example.com/jev-smart-router
   databricks apps create jev-smart-router
   databricks apps deploy jev-smart-router \
     --source-code-path /Workspace/Users/you@example.com/jev-smart-router
   ```

3. **Grant the app's service principal `CAN QUERY`** on the FMAPI endpoints you
   want to route to (pay-per-token foundation models are covered by default in
   most workspaces).

Open the app URL and start chatting — JEV routes each message to the best model.

## Run locally

```bash
cd src
pip install -r requirements.txt
export JEV_API_KEY=sk-...            # your TypeSafe JEV key
# for FMAPI access from your laptop, authenticate the Databricks SDK, e.g.:
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...      # or use a CLI profile / OAuth
uvicorn app:app --reload --port 8080
```

Then open <http://localhost:8080>.

## Notes

- Inference uses the raw REST `/serving-endpoints/{model}/invocations` call
  rather than the SDK's typed `serving_endpoints.query()`, which has
  dict-handling quirks with some endpoints.
- Reasoning models (which may return text in `reasoning_content` rather than
  `content`) are handled with generous `max_tokens` and a `reasoning_content`
  fallback.
- JEV is an external service (`api.typesafe.ai`); the conversation transcript
  used for the routing decision is sent there. Review your data-handling
  requirements before routing sensitive content.

## License

[MIT](LICENSE)
