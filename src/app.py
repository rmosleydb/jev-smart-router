"""
JEV Smart Router — a Databricks App that uses TypeSafe JEV (System One) to
decide *which* model should answer each message, then runs the real inference
on the chosen Databricks Foundation Model API (FMAPI) endpoint.

    GET  /                 -> single-page UI (routing options manager + chat)
    GET  /router/models    -> live list of databricks-* llm/v1/chat FMAPI endpoints
    POST /router/route     -> given the conversation + configured options, ask JEV
                              (choice question) which model to route to; returns choice
    POST /router/chat      -> route via JEV, then run inference on the chosen model,
                              returns the assistant reply + which model was chosen
    GET  /healthz          -> health probe

How the routing works
---------------------
JEV makes the routing DECISION using its native `choice` question type
(criteria = {option_label: description}). The winning option maps to a
Databricks serving endpoint, which then does the real inference via the app's
service principal against the workspace's FMAPI serving endpoints.

Configuration (env)
-------------------
    JEV_API_KEY   (required)  TypeSafe JEV API key. On Databricks Apps, inject it
                              from a secret via app.yaml `valueFrom` — never hardcode.
    JEV_API_URL   (optional)  default https://api.typesafe.ai/v1/systemone
    JEV_MODEL     (optional)  default "jev-latest"
    JEV_TIMEOUT   (optional)  request timeout in seconds, default 30
    JEV_LOG_PROMPT(optional)  "1" to log routing decisions to stdout
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# --------------------------------------------------------------------------- #
# Config. The JEV key comes from the environment. On Databricks Apps, wire it
# in from a secret via app.yaml `valueFrom` so it is never committed anywhere.
# --------------------------------------------------------------------------- #
JEV_API_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
JEV_TIMEOUT = float(os.environ.get("JEV_TIMEOUT", "30"))
JEV_API_KEY = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")

app = FastAPI(title="JEV Smart Router")
_session = requests.Session()


# --------------------------------------------------------------------------- #
# Databricks SDK (App Service Principal auth, auto-detected from the app env).
# --------------------------------------------------------------------------- #
def _ws():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _flatten(content: Any) -> str:
    """Normalize OpenAI-style content (str or content-parts array) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _conversation_text(messages: list[dict[str, Any]]) -> str:
    """Render the chat so far into a compact transcript for JEV to reason over."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        text = _flatten(m.get("content"))
        if text:
            lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines).strip()


def _jev_choice(state_text: str, criteria: dict[str, str], instructions: str) -> dict[str, Any]:
    """One JEV `choice` question. criteria = {option_label: description}.

    Returns JEV's route answer with an added `latency_ms` measuring the
    round-trip time of the JEV lookup call.
    """
    payload = {
        "model": JEV_MODEL,
        "state": {"conversation": state_text},
        "questions": {
            "route": {"type": "choice", "criteria": criteria, "instructions": instructions}
        },
    }
    t0 = time.perf_counter()
    resp = _session.post(
        JEV_API_URL,
        headers={"Authorization": f"Bearer {JEV_API_KEY}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=JEV_TIMEOUT,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    resp.raise_for_status()
    ans = (resp.json().get("answers") or {}).get("route", {})
    ans["latency_ms"] = latency_ms
    return ans


_ROUTE_INSTRUCTIONS = (
    "Read the conversation and choose the single best destination for handling "
    "the most recent user message, based on each option's description."
)


@app.get("/router/models")
def router_models():
    """Live databricks-* llm/v1/chat FMAPI endpoints the app SP can route to."""
    try:
        w = _ws()
        models = []
        for e in w.serving_endpoints.list():
            task = str(getattr(e, "task", "") or "")
            name = e.name
            if task == "llm/v1/chat" and name.startswith("databricks-"):
                models.append(name)
        models.sort()
        return {"models": models}
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"models": [], "error": str(ex)[:300]}, status_code=200)


@app.post("/router/route")
async def router_route(request: Request):
    """Given messages + options, ask JEV which option (model) to route to."""
    body = await request.json()
    messages = body.get("messages", [])
    options = body.get("options", [])  # [{label, description, model}]
    if not options:
        return JSONResponse({"error": "no routing options configured"}, status_code=400)

    criteria = {o["label"]: (o.get("description") or o["label"]) for o in options}
    convo = _conversation_text(messages)
    ans = _jev_choice(convo, criteria, _ROUTE_INSTRUCTIONS)
    chosen_label = ans.get("choice")
    match = next((o for o in options if o["label"] == chosen_label), None)
    return {
        "choice": chosen_label,
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities"),
        "latency_ms": ans.get("latency_ms"),
        "model": match.get("model") if match else None,
    }


@app.post("/router/chat")
async def router_chat(request: Request):
    """Full turn: JEV routes -> chosen FMAPI model runs inference -> reply."""
    body = await request.json()
    messages = body.get("messages", [])
    options = body.get("options", [])
    if not messages:
        return JSONResponse({"error": "no messages"}, status_code=400)
    if not options:
        return JSONResponse({"error": "no routing options configured"}, status_code=400)

    # 1) JEV makes the routing choice.
    criteria = {o["label"]: (o.get("description") or o["label"]) for o in options}
    convo = _conversation_text(messages)
    try:
        ans = _jev_choice(convo, criteria, _ROUTE_INSTRUCTIONS)
    except Exception as ex:  # noqa: BLE001
        return JSONResponse({"error": f"JEV routing failed: {str(ex)[:200]}"}, status_code=502)

    chosen_label = ans.get("choice")
    match = next((o for o in options if o["label"] == chosen_label), None)
    if not match or not match.get("model"):
        return JSONResponse(
            {"error": f"JEV chose '{chosen_label}' but no model is mapped to it."},
            status_code=422,
        )
    model = match["model"]

    if os.environ.get("JEV_LOG_PROMPT", "0") == "1":
        print(f"[jev-router] choice={chosen_label!r} -> model={model!r} "
              f"conf={ans.get('confidence')}", flush=True)

    # 2) Chosen FMAPI model runs the actual inference (via the app SP).
    #    We use the raw REST /invocations call (the SDK's typed query() has
    #    dict-handling quirks with some endpoints). Reasoning models may put
    #    text in reasoning_content and need generous max_tokens, so we give
    #    headroom and fall back to reasoning_content if content is empty.
    try:
        w = _ws()
        sdk_messages = [
            {"role": m.get("role", "user"), "content": _flatten(m.get("content"))}
            for m in messages
            if _flatten(m.get("content"))
        ]
        _t0 = time.perf_counter()
        raw = w.api_client.do(
            "POST", f"/serving-endpoints/{model}/invocations",
            body={"messages": sdk_messages, "max_tokens": 2048},
        )
        inference_ms = round((time.perf_counter() - _t0) * 1000, 1)
        choice0 = (raw.get("choices") or [{}])[0]
        msg = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        reply = (msg.get("content") or "").strip()
        if not reply:
            reply = (msg.get("reasoning_content") or "").strip()
        if not reply:
            reply = "(model returned no text — it may have hit the token limit while reasoning.)"
    except Exception as ex:  # noqa: BLE001
        return JSONResponse(
            {"error": f"Inference on '{model}' failed: {str(ex)[:250]}",
             "choice": chosen_label, "model": model},
            status_code=502,
        )

    return {
        "reply": reply,
        "choice": chosen_label,
        "model": model,
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities"),
        "latency_ms": ans.get("latency_ms"),
        "inference_ms": inference_ms,
    }


@app.get("/healthz")
def health():
    return {"status": "ok", "service": "jev-smart-router", "jev_key_present": bool(JEV_API_KEY)}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>JEV Smart Router</title>
<style>
  :root {
    --bg:#0e1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
    --muted:#8b949e; --accent:#ff6b35; --accent2:#2f81f7; --ok:#3fb950; --danger:#f85149;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:16px 24px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px;}
  header h1 { font-size:18px; margin:0; font-weight:600; }
  header .tag { font-size:12px; color:var(--muted); border:1px solid var(--border); padding:2px 8px; border-radius:12px;}
  .wrap { display:grid; grid-template-columns:380px 1fr; gap:0; height:calc(100vh - 57px); }
  .side { border-right:1px solid var(--border); padding:18px; overflow-y:auto; }
  .side h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 12px;}
  .opt { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:12px;}
  .opt .row { display:flex; gap:8px; align-items:center; margin-bottom:8px;}
  .opt input[type=text] { flex:1; background:#0d1117; border:1px solid var(--border); color:var(--fg);
        border-radius:8px; padding:8px 10px; font-size:13px;}
  .opt textarea { width:100%; background:#0d1117; border:1px solid var(--border); color:var(--fg);
        border-radius:8px; padding:8px 10px; font-size:13px; resize:vertical; min-height:52px; font-family:inherit;}
  .opt select { width:100%; background:#0d1117; border:1px solid var(--border); color:var(--fg);
        border-radius:8px; padding:8px 10px; font-size:13px; margin-top:8px;}
  .opt label { font-size:11px; color:var(--muted); display:block; margin:8px 0 2px;}
  .icon-btn { background:transparent; border:1px solid var(--border); color:var(--muted); border-radius:8px;
        width:32px; height:32px; cursor:pointer; font-size:15px; line-height:1; }
  .icon-btn:hover { color:var(--fg); border-color:var(--muted); }
  .icon-btn.danger:hover { color:var(--danger); border-color:var(--danger); }
  .add-btn { width:100%; background:var(--panel); border:1px dashed var(--border); color:var(--muted);
        border-radius:10px; padding:10px; cursor:pointer; font-size:13px;}
  .add-btn:hover { color:var(--accent); border-color:var(--accent); }
  .main { display:flex; flex-direction:column; height:100%; }
  .routed { padding:10px 18px; border-bottom:1px solid var(--border); font-size:13px; color:var(--muted);
        display:flex; align-items:center; gap:10px; min-height:44px;}
  .routed b { color:var(--accent); font-weight:600; }
  .routed .conf { color:var(--muted); font-size:12px; }
  .routed .spacer { flex:1; }
  .clear-btn { background:transparent; border:1px solid var(--border); color:var(--muted); border-radius:8px;
        padding:5px 12px; cursor:pointer; font-size:12px; }
  .clear-btn:hover { color:var(--danger); border-color:var(--danger); }
  /* hover tooltip on the orange model name */
  .via { position:relative; cursor:help; border-bottom:1px dotted var(--accent); }
  .via .tip { visibility:hidden; opacity:0; transition:opacity .12s; position:absolute; bottom:135%; left:0;
        z-index:20; background:#0d1117; border:1px solid var(--border); border-radius:10px; padding:10px 12px;
        min-width:230px; box-shadow:0 6px 24px rgba(0,0,0,.5); }
  .via:hover .tip { visibility:visible; opacity:1; }
  .tip .tip-h { font-size:10px; letter-spacing:.05em; color:var(--muted); margin-bottom:6px; }
  .tip .prow { display:flex; align-items:center; gap:8px; margin:3px 0; font-size:12px; }
  .tip .prow .pname { flex:1; color:var(--fg); font-weight:500; }
  .tip .prow.win .pname { color:var(--accent); }
  .tip .prow .pval { color:var(--muted); font-variant-numeric:tabular-nums; }
  .tip .bar { height:5px; background:#21262d; border-radius:3px; overflow:hidden; width:70px; }
  .tip .bar > span { display:block; height:100%; background:var(--muted); }
  .tip .prow.win .bar > span { background:var(--accent); }
  .tip .lat { margin-top:8px; padding-top:8px; border-top:1px solid var(--border); font-size:11px; color:var(--muted); }
  .chat { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px;}
  .msg { max-width:80%; padding:10px 14px; border-radius:12px; font-size:14px; line-height:1.5; white-space:pre-wrap;}
  .msg.user { align-self:flex-end; background:var(--accent2); color:#fff; }
  .msg.assistant { align-self:flex-start; background:var(--panel); border:1px solid var(--border);}
  .msg .who { font-size:10px; opacity:.7; margin-bottom:4px; text-transform:uppercase; letter-spacing:.04em;}
  .msg.assistant .who b { color:var(--accent); }
  .composer { border-top:1px solid var(--border); padding:14px 18px; display:flex; gap:10px;}
  .composer textarea { flex:1; background:#0d1117; border:1px solid var(--border); color:var(--fg);
        border-radius:10px; padding:10px 12px; font-size:14px; resize:none; height:44px; font-family:inherit;}
  .composer button { background:var(--accent); border:none; color:#fff; border-radius:10px; padding:0 20px;
        cursor:pointer; font-size:14px; font-weight:600;}
  .composer button:disabled { opacity:.5; cursor:default;}
  .hint { color:var(--muted); font-size:12px; padding:0 18px 10px;}
  .spin { display:inline-block; width:12px; height:12px; border:2px solid var(--muted); border-top-color:var(--accent);
        border-radius:50%; animation:spin .7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg);} }
</style>
</head>
<body>
<header>
  <h1>🧭 JEV Smart Router</h1>
  <span class="tag">JEV picks the model · Databricks FMAPI runs it</span>
</header>
<div class="wrap">
  <aside class="side">
    <h2>Routing options</h2>
    <div id="options"></div>
    <button class="add-btn" onclick="addOption()">＋ Add routing option</button>
  </aside>
  <section class="main">
    <div class="routed" id="routed">
      <span id="routedText">No inference yet — send a message and JEV will pick a model.</span>
      <span class="spacer"></span>
      <button class="clear-btn" onclick="clearChat()">Clear chat</button>
    </div>
    <div class="chat" id="chat"></div>
    <div class="hint" id="hint"></div>
    <div class="composer">
      <textarea id="input" placeholder="Type a message… JEV will route it to the best model." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}"></textarea>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
  </section>
</div>
<script>
let MODELS = [];
let OPTIONS = [];   // {label, description, model}
let MESSAGES = [];  // {role, content}

async function loadModels() {
  try {
    const r = await fetch('router/models');
    const j = await r.json();
    MODELS = j.models || [];
  } catch(e) { MODELS = []; }
  // seed a couple sensible defaults if empty
  if (OPTIONS.length === 0 && MODELS.length) {
    const pick = (frag) => MODELS.find(m => m.includes(frag)) || MODELS[0];
    OPTIONS = [
      {label:"Coding", description:"Programming, code, debugging, technical/engineering questions.", model:pick("claude")},
      {label:"General", description:"Everyday questions, chit-chat, general knowledge, summaries.", model:pick("gpt")},
    ];
  }
  renderOptions();
}

function renderOptions() {
  const box = document.getElementById('options');
  box.innerHTML = '';
  OPTIONS.forEach((o, i) => {
    const div = document.createElement('div');
    div.className = 'opt';
    const modelOpts = MODELS.map(m => `<option value="${m}" ${m===o.model?'selected':''}>${m}</option>`).join('');
    div.innerHTML = `
      <div class="row">
        <input type="text" value="${escapeHtml(o.label)}" placeholder="Option name"
               oninput="OPTIONS[${i}].label=this.value">
        <button class="icon-btn danger" title="Delete" onclick="delOption(${i})">🗑</button>
      </div>
      <label>Description — when should JEV pick this?</label>
      <textarea oninput="OPTIONS[${i}].description=this.value" placeholder="Describe the kind of request this handles…">${escapeHtml(o.description)}</textarea>
      <label>Model (FMAPI llm/v1/chat)</label>
      <select onchange="OPTIONS[${i}].model=this.value">${modelOpts}</select>
    `;
    box.appendChild(div);
  });
}

function addOption() {
  OPTIONS.push({label:"New option", description:"", model: MODELS[0] || ""});
  renderOptions();
}
function delOption(i) { OPTIONS.splice(i,1); renderOptions(); }

function escapeHtml(s){ return (s||'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function tooltipHtml(details) {
  // details = {model, probabilities:{label:val}, choice, latency_ms}
  const probs = details.probabilities || {};
  const entries = Object.entries(probs).sort((a,b) => b[1]-a[1]);
  let rows = entries.map(([name, val]) => {
    const pct = (val*100);
    const win = (name === details.choice) ? ' win' : '';
    return `<div class="prow${win}">`
      + `<span class="pname">${escapeHtml(name)}</span>`
      + `<span class="bar"><span style="width:${pct.toFixed(1)}%"></span></span>`
      + `<span class="pval">${val.toFixed(3)}</span>`
      + `</div>`;
  }).join('');
  if (!rows) rows = '<div class="prow"><span class="pname">(no breakdown)</span></div>';
  const lat = (details.latency_ms != null)
    ? `<div class="lat">JEV routing latency: <b>${details.latency_ms} ms</b></div>` : '';
  return `<div class="tip"><div class="tip-h">JEV routing decision</div>${rows}${lat}</div>`;
}

function addMsg(role, content, details) {
  MESSAGES.push({role, content});
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  let who;
  if (role === 'assistant') {
    const model = details && details.model;
    const via = model
      ? `· via <span class="via"><b>${escapeHtml(model)}</b>${tooltipHtml(details)}</span>`
      : '';
    who = `<div class="who">Assistant ${via}</div>`;
  } else {
    who = `<div class="who">You</div>`;
  }
  div.innerHTML = who + escapeHtml(content);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function clearChat() {
  MESSAGES = [];
  document.getElementById('chat').innerHTML = '';
  document.getElementById('hint').textContent = '';
  setRouted(null);
}

function setRouted(model, conf) {
  const el = document.getElementById('routedText');
  if (!model) { el.textContent = 'No inference yet — send a message and JEV will pick a model.'; return; }
  const c = (conf!=null) ? `<span class="conf">JEV confidence ${(conf*100).toFixed(0)}%</span>` : '';
  el.innerHTML = `Latest inference routed by JEV to <b>${escapeHtml(model)}</b> ${c}`;
}

async function send() {
  const inp = document.getElementById('input');
  const text = inp.value.trim();
  if (!text) return;
  if (OPTIONS.length === 0) { alert('Add at least one routing option first.'); return; }
  inp.value = '';
  addMsg('user', text);
  const btn = document.getElementById('sendBtn');
  const hint = document.getElementById('hint');
  btn.disabled = true;
  hint.innerHTML = '<span class="spin"></span> JEV is choosing a model…';
  try {
    const r = await fetch('router/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages: MESSAGES, options: OPTIONS})
    });
    const j = await r.json();
    if (j.error) {
      hint.textContent = '⚠ ' + j.error;
      if (j.model) setRouted(j.model, j.confidence);
    } else {
      setRouted(j.model, j.confidence);
      addMsg('assistant', j.reply || '(empty reply)', j);
      const jevT = (j.latency_ms != null) ? ` (${j.latency_ms} ms)` : '';
      const infT = (j.inference_ms != null) ? ` (${(j.inference_ms/1000).toFixed(1)} s)` : '';
      hint.textContent = `JEV chose "${j.choice}"${jevT} → ${j.model}${infT}`;
    }
  } catch(e) {
    hint.textContent = '⚠ ' + e;
  } finally {
    btn.disabled = false;
  }
}

loadModels();
</script>
</body>
</html>
"""
