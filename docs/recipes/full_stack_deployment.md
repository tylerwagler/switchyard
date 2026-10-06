# Full-stack deployment for coding agents

This guide sets up one Switchyard gateway for a team. Claude Code and Codex
point at it. The gateway routes each turn between a small model and a large
model, uses a judge model to decide, serves embeddings and reranking, answers
Claude Code's web searches, and exports metrics. API-key auth and quotas are an
optional add-on.

Read [Getting Started](../getting_started.md) first if you have never run the
server. This guide assumes self-hosted models behind vLLM or another
OpenAI-compatible server. Hosted providers work the same way; swap the client
block.

## What you need

| Service | Purpose | Required? |
|---|---|---|
| Large chat model | The capable tier | Yes |
| Small chat model | The efficient tier, and the judge | Yes |
| Embeddings server | `/v1/embeddings` relay | Optional |
| Rerank server | `/v1/rerank` relay, and web-search re-ranking | Optional |
| SearXNG | Claude Code's `web_search` tool | Optional |
| Valkey or Redis | Web-search cache, and gate quotas | Optional |
| Prometheus or an OTel collector | Metrics and traces | Optional |
| Postgres | Gate API keys | Only with the gate |

The judge is a small model that answers one short question per user turn:
"can the small model handle this?" It can run on the same server as the small
tier. Every model role is an ordinary `[targets.*]` entry, so any role can live
on any upstream.

One machine with 8 GB of RAM runs the gateway comfortably. The gateway does no
model inference itself.

## 1. Install

Build from the fork's `dev` branch. The release on crates.io does not include
the fork's features.

```bash
git clone -b dev https://github.com/tylerwagler/switchyard.git
cd switchyard
cargo install --locked --path crates/switchyard-server
cargo build --release -p switchyard-gate   # only if you want auth and quotas
```

The root `Dockerfile` builds an image that listens on port `4000` and starts
`switchyard-server`. The image also carries `switchyard-gate`; pass
`--entrypoint switchyard-gate` to `docker run` to start the gate instead.

```bash
docker build -t switchyard-server .
docker run --rm -p 4000:4000 -v $PWD/switchyard.toml:/etc/switchyard.toml:ro \
  switchyard-server --config /etc/switchyard.toml
```

## 2. Write the config

Save this as `/etc/switchyard/switchyard.toml`. Replace the host names and
model ids with yours. Every section after `[routes.*]` is optional. Delete the
ones you do not run.

```toml
schema_version = 1

# Any model id that matches no route lands here. Claude Code sends several ids
# (its default Opus, Sonnet and Haiku names); this catches the ones you forget
# to alias.
default_route = "switchyard"

[llm_clients.large]
format = "openai_chat"
base_url = "http://large-box:8000/v1"
timeout_ms = 600000

[llm_clients.small]
format = "openai_chat"
base_url = "http://small-box:8000/v1"
timeout_ms = 600000

[targets.capable]
id = "Qwen/Qwen3-235B-A22B-Instruct"
llm_client = "large"

[targets.efficient]
id = "Qwen/Qwen3-30B-A3B-Instruct"
llm_client = "small"

# The judge. Same model as the efficient tier. A judge target is never an
# answer destination.
[targets.judge]
id = "Qwen/Qwen3-30B-A3B-Instruct"
llm_client = "small"

# The main route. The classifier picks a tier once per user turn. The stage
# router moves between tiers during tool calls.
[routes.switchyard]
id = "switchyard"
type = "composite"
display_name = "Switchyard"
description = "Routes each turn between the small and large model."
context_window = 131072
strip_attribution = true

[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"
response_format_type = "json_object"

[routes.switchyard.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.5

# Fixed routes, so users can pin a tier from Claude Code's /model picker.
[routes.large]
id = "large"
type = "passthrough"
target = "capable"
display_name = "Large only"
context_window = 131072
strip_attribution = true

[routes.small]
id = "small"
type = "passthrough"
target = "efficient"
display_name = "Small only"
context_window = 131072
strip_attribution = true

# Embeddings and rerank relays. POST /v1/embeddings and POST /v1/rerank
# forward the body unchanged to {base_url}/embeddings or {base_url}/rerank.
[embeddings.default]
base_url = "http://embed-box:8001/v1"
model = "BAAI/bge-m3"

[rerank.default]
base_url = "http://embed-box:8002/v1"
model = "BAAI/bge-reranker-v2-m3"

# Claude Code's built-in web_search tool, answered by SearXNG.
[search.searxng]
base_url = "http://searxng:8080"

[cache.valkey]
url = "redis://valkey:6379"

[web_search]
enabled = true
search = "searxng"
rerank = "default"
cache = "valkey"

# Claude Code auto-mode safety checks. Without this section the server answers
# "unsupported" and Claude Code runs its own check.
[safeguards]
judge_route = "small"
```

Notes on the choices:

- **`strip_attribution = true`** removes Claude Code's billing header block
  before the request reaches a non-Anthropic model. Without it, every
  conversation's prompt starts with a different fingerprint, and the prefix
  cache never hits. Leave it off for routes that go to Anthropic's own API.
- **`response_format_type = "json_object"`** is the safe choice for vLLM
  judges. Use the default `json_schema` when the server supports it.
- **`timeout_ms`** is the deadline for the whole response, including streaming.
  Ten minutes is generous for long agent turns.
- **Secrets never go in the file.** A hosted provider client uses
  `api_key_env = "MY_KEY"` and reads the variable at startup. See
  [`[llm_clients.<name>]`](../reference/toml_schema.md#llm_clientsname).
- **`[safeguards]`** sends one short judge request per tool use through the
  named route. Add `shadow_log = "/var/lib/switchyard/safeguards.jsonl"` to
  log the judge's verdicts without acting on them.
- The judge and the efficient tier share a model id, so they must share an
  `llm_client` and the same `extra_body`. To give the judge its own settings,
  such as thinking turned off, serve the model under a second name (vLLM's
  `--served-model-name`) and point the judge at that id.

Validate before starting:

```bash
switchyard-server --config /etc/switchyard/switchyard.toml --dry-run
```

The dry run checks the schema, target references, and that every
`api_key_env` is set. It does not contact any upstream.

## 3. Run it as a service

The server binds `0.0.0.0:4000` by default and does no client authentication.
Put it behind a reverse proxy with TLS, or use `--tls-cert` and `--tls-key`,
and restrict who can reach the port. Section 7 adds API keys.

A system unit, saved as `/etc/systemd/system/switchyard.service`:

```ini
[Unit]
Description=Switchyard LLM gateway
After=network-online.target
Wants=network-online.target

[Service]
DynamicUser=yes
StateDirectory=switchyard
EnvironmentFile=-/etc/switchyard/env
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/switchyard-server \
  --config /etc/switchyard/switchyard.toml \
  --host 0.0.0.0 --port 4000 \
  --routing-log-file /var/lib/switchyard/routing.jsonl
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put provider keys in `/etc/switchyard/env` as `NAME=value` lines, mode `0600`.

```bash
sudo install -m 0755 ~/.cargo/bin/switchyard-server /usr/local/bin/
sudo systemctl daemon-reload
sudo systemctl enable --now switchyard
journalctl -u switchyard -f
```

`--routing-log-file` is optional. It appends one JSON line per routed request
with the route, tier, model and token counts.

## 4. Point Claude Code at it

Claude Code reads its endpoint and model names from the environment. Save this
as `/usr/local/bin/claude-sy` and share it with the team:

```bash
#!/usr/bin/env bash
# Launch Claude Code against the team gateway.
GATEWAY="${SWITCHYARD_URL:-http://gateway.example.internal:4000}"
exec env -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL="$GATEWAY" \
  ANTHROPIC_AUTH_TOKEN="${SWITCHYARD_KEY:-switchyard}" \
  ANTHROPIC_MODEL=switchyard \
  ANTHROPIC_DEFAULT_OPUS_MODEL=switchyard \
  ANTHROPIC_DEFAULT_SONNET_MODEL=switchyard \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=small \
  ANTHROPIC_SMALL_FAST_MODEL=small \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
  CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 \
  claude "$@"
```

What each line does:

- `ANTHROPIC_BASE_URL` is the server root, with no `/v1`.
- `ANTHROPIC_AUTH_TOKEN` must be non-empty. The plain server ignores it. With
  the gate, it is the user's API key.
- `ANTHROPIC_API_KEY` is unset so a personal Anthropic key does not leak to the
  gateway.
- The three `ANTHROPIC_DEFAULT_*_MODEL` values map Claude Code's built-in model
  names onto routes. Background and summary work goes to `small`.
- `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` makes the `/model` picker list
  the routes from `GET /v1/models`, with their `display_name` and
  `description`.
- `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` makes Claude Code label each request
  with its purpose. The gate records that label on usage events.

The same variables can go in the `env` block of Claude Code's `settings.json`
instead of a wrapper script.

Two Claude Code features need an Anthropic-format target in the route:

- `POST /v1/messages/count_tokens` returns 400 `count_tokens_unsupported` when
  every target is `openai_chat`. Claude Code keeps working. Expect this line in
  the logs.
- Server-side `web_search` is answered by the `[web_search]` section instead.
  Without it, vLLM rejects the tool with a 422.

## 5. Point Codex at it

Codex uses a profile, not environment variables. Save this as
`~/.codex/switchyard.config.toml`:

```toml
model = "switchyard"
model_provider = "switchyard"

[model_providers.switchyard]
name = "Switchyard"
base_url = "http://gateway.example.internal:4000/v1"
wire_api = "responses"
```

Run `codex --profile switchyard`. Codex's "Approve for me" reviewer asks for a
model named `codex-auto-review`. Add a passthrough route with that `id`, or the
reviewer gets a 404 and treats it as a denial.

## 6. Check it

```bash
GW=http://gateway.example.internal:4000
curl -s $GW/health
curl -s $GW/v1/upstreams | jq       # TCP reachability of every llm_client
curl -s $GW/v1/models | jq '.data[].id'
curl -s $GW/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"hello"}]}'
curl -s $GW/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":"hello","model":"BAAI/bge-m3"}' | jq '.data[0].embedding | length'
curl -s $GW/v1/stats | jq '{answers: .models, judge: .classifier.models, upstreams}'
```

The last command shows which models answered and which model judged. The
`upstreams` block shows which server actually served, which differs from
`models` whenever a fallback happened. Every answer also carries the header
`x-model-router-selected-model`.

`/health` only says the process is up. Use `/v1/upstreams` or
`GET /v1/models?available=true` as the readiness check.

## 7. Optional: API keys and quotas with the gate

`switchyard-gate` is the same server with an auth and metering layer in front.
It reads the same TOML. It needs two extra services:

- **Postgres** holds the users and keys. The gate's database role calls two
  functions: `gate.lookup_key(hash)` and `gate.lookup_user_by_email(email)`.
  Each returns one row with these columns: `key_id`, `user_id`, `role`,
  `status`, `trusted_forwarder`, `rate_limit_rpm`, `rate_limit_tpm`,
  `hourly_limit`, `daily_limit`, `weekly_limit`, `monthly_limit`, `w_input`,
  `w_cache_read`, `w_cache_write`, `w_output`, `w_reasoning`. Keys are stored as
  lowercase SHA-256 hex of the plain key. A `NULL` limit is unlimited. The
  schema and the functions are not in this repository; your portal owns them.
- **Valkey** holds the quota counters and a `usage:events` stream. Your portal
  reads the stream to bill or report. It can be the same Valkey as the search
  cache.

```bash
GATE_DATABASE_URL=postgres://gate:...@db/portal \
GATE_VALKEY_URL=redis://valkey:6379 \
switchyard-gate --config /etc/switchyard/switchyard.toml --port 4000
```

How requests are treated:

- Clients send `Authorization: Bearer <key>` or `x-api-key: <key>`.
- `/health` is open. The chat endpoints are metered against the user's limits.
  `/metrics`, `/v1/stats`, `/v1/upstreams` and `/v1/decision` need an `admin`
  role, except for `GET` from localhost. Everything else, including embeddings
  and rerank, needs a key but is not metered.
- Over a limit returns 429 with `Retry-After`. Postgres down returns 503 for
  every keyed request. Valkey down returns 503 for metered requests.
- Each usage event records the user, model, token counts by kind, latency,
  Claude Code's request class, and the Claude Code version and entrypoint read
  from the attribution block.

The gate binary has no TLS flags. Terminate TLS in front of it.

## 8. Metrics, traces and logs

- **Prometheus** scrapes `GET /metrics`. `examples/prometheus/` has a scrape
  config and alert rules. Useful series: `switchyard.routing_fallbacks`,
  `switchyard.ttfb_ms`, `switchyard.classifier_fail_open`,
  `switchyard.websearch_queries`, `switchyard.aux_requests`.
- **OpenTelemetry** export turns on when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
  The protocol is OTLP over HTTP. `OTEL_SERVICE_NAME` defaults to
  `switchyard-server`. Set `OTEL_SDK_DISABLED=true` to turn it off. Span and
  attribute names are in the [OpenTelemetry reference](../reference/opentelemetry.md).
- **Logs** go to stderr and follow `RUST_LOG`. The default is
  `info,opentelemetry=warn`.
- **`GET /v1/stats`** is a live in-memory summary. `POST /v1/stats/reset`
  clears it.

## 9. How failures behave

- A failed upstream call is retried `max_retries` times (default 2). After
  that, the client is skipped for `failure_cooldown_ms` (default 5 seconds) and
  the route tries its next target.
- A judge that errors or times out does not fail the request. The composite
  and capability classifiers fail open to the capable tier.
- A context-window overflow on one target falls through to the route's other
  targets. If every target overflows, the client gets 400
  `context_length_exceeded` with a message Claude Code recognizes, so it
  compacts the conversation. See
  [Context-Window Handling](../operations/context_window.md).
- Web search, rerank and the cache all fail open. A dead SearXNG returns a
  tool error to the model, never a 5xx to the user. A dead reranker returns
  results in engine order. A dead cache is skipped.
- Streams that are silent for 15 seconds get a keepalive frame so proxies and
  clients do not drop them.

## Where to go next

- [TOML Schema](../reference/toml_schema.md) for every key.
- [Routing Overview](../routing_algorithms/overview.md) to pick a different
  algorithm. `composite` is a good default. `llm_classifier` with
  `mode = "escalation"` runs the small model first and only escalates when a
  judge sees it struggle.
- [Hosted Web Search](../operations/hosted_web_search.md) for the search
  pipeline in detail.
- [Single-provider coding agents](single_provider_coding_agents.md) to route
  Claude Code or Codex across one hosted provider using the user's own login.
- [pi](../integrations/pi.md) and [Oh My Pi](../integrations/oh_my_pi.md) for
  other agents.
