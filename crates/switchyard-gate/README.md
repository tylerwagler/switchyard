# switchyard-gate

`switchyard-gate` is `switchyard-server` with an access layer in front. It reads the
same TOML file and serves the same endpoints. Before a request reaches the router, the
gate checks the caller's API key, checks the caller's quotas, and records the tokens the
request used. Keys and users live in Postgres. Counters and usage events live in Valkey.

## Run

```bash
cargo build --release -p switchyard-gate
GATE_DATABASE_URL=postgres://switchyard_gate:...@db/portal \
GATE_VALKEY_URL=redis://valkey:6379 \
switchyard-gate --config switchyard.toml --port 4000
```

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--config PATH` | | required | The server TOML file. |
| `--host ADDR` | | `0.0.0.0` | Address to bind. |
| `--port PORT`, `-p` | | `4000` | Port to bind. |
| `--shutdown-timeout DUR` | | server default | How long active requests may finish after shutdown starts. |
| `--database-url URL` | `GATE_DATABASE_URL` | required | Postgres URL for the `switchyard_gate` role. |
| `--valkey-url URL` | `GATE_VALKEY_URL` | required | Valkey URL for counters and the usage stream. |
| `--dry-run` | | off | Check the config, print the served models, and exit. Does not connect to Postgres or Valkey. |

## How a client authenticates

Send the key in `Authorization: Bearer <key>` or in `x-api-key: <key>`. The gate
removes both headers before the router sees the request.

A key marked `trusted_forwarder` may act for another user. When such a key sends
`X-OpenWebUI-User-Email`, the gate looks that email up and bills the request to that
user. If the email has no account, the request is billed to the forwarder's own key.
The gate drops every `x-openwebui-*` header before forwarding. Other keys cannot use
this header.

## Which paths need a key

| Class | Paths | Rule |
|---|---|---|
| Open | `/health` | No key. |
| Metered | `/v1/messages`, `/v1/chat/completions`, `/v1/responses` | Key, quota check, and usage recorded. |
| Admin | `/metrics`, `/v1/stats`, `/v1/stats/reset`, `/v1/upstreams`, `/v1/decision`, `/v1/routing/session-stats` | Key with role `admin`. |
| Keyed | everything else, such as `/v1/models`, `/v1/embeddings`, `/v1/rerank` | Key. Not metered. |

One exception: a `GET` of `/metrics`, `/v1/stats`, `/v1/upstreams` or
`/v1/routing/session-stats` from a loopback address needs no key. This lets a dashboard
in the same container read them.

`GET /v1/models` is rewritten to `GET /v1/models?available=true`, so callers see only
chat routes whose upstream is up.

## Postgres contract

The gate's database role calls two functions and nothing else:

- `gate.lookup_key(hash text)`, where `hash` is the lowercase SHA-256 hex of the plain key.
- `gate.lookup_user_by_email(email text)`, where `email` is lowercased by the gate first.

Each returns zero or one row with these columns, in the types the gate reads:

| Column | Type | Notes |
|---|---|---|
| `key_id` | cast to `text` | NULL when looked up by email. |
| `user_id` | cast to `text` | |
| `role` | `text` | `admin` unlocks admin paths. `pending` is refused with 403. |
| `status` | `text` | Anything but `active` is refused with 403. |
| `trusted_forwarder` | `boolean` | |
| `rate_limit_rpm` | `integer`, nullable | Requests per minute. |
| `rate_limit_tpm` | `bigint`, nullable | Billable tokens per minute. |
| `hourly_limit`, `daily_limit`, `weekly_limit`, `monthly_limit` | `bigint`, nullable | Billable tokens per window. |
| `w_input`, `w_cache_read`, `w_cache_write`, `w_output`, `w_reasoning` | `double precision` | Token weights. |

A `NULL` limit means unlimited. The weights multiply each kind of token to get the
billable count. The usual defaults are `1.0`, `0.1`, `1.0`, `1.0`, `1.0`, so a cached
prefix read counts a tenth of an uncached input token. Results are cached in the process for 30 seconds when found and 10 seconds when
not found, so a revoked key stops working within 30 seconds.

[`schema.sql`](schema.sql) is a minimal schema that satisfies this contract.

## Valkey contract

Quota counters use fixed UTC windows. Weeks start on Monday. Each key holds one integer
and expires after its window can no longer matter:

| Key | Counts | TTL |
|---|---|---|
| `q:<user_id>:rm:<minute>` | requests this minute | 120 s |
| `q:<user_id>:tm:<minute>` | billable tokens this minute | 120 s |
| `q:<user_id>:th:<hour>` | billable tokens this hour | 2 h |
| `q:<user_id>:td:<day>` | billable tokens today | 2 d |
| `q:<user_id>:tw:<week>` | billable tokens this week | 8 d |
| `q:<user_id>:tmo:<month>` | billable tokens this month | 32 d |

`<minute>`, `<hour>` and `<day>` are the Unix time divided by the window length.
`<week>` is shifted so weeks start on Monday. `<month>` is `year * 12 + month - 1`.

When a metered response ends, the gate adds the billable tokens to every window and
appends one entry to the `usage:events` stream with `MAXLEN ~ 1000000`. Fields:

| Field | Meaning |
|---|---|
| `ts` | Unix seconds when the response ended. |
| `user_id` | The billed user. |
| `via` | `key` or `openwebui`. |
| `model` | The target model that answered. |
| `input_tokens`, `output_tokens`, `cached_tokens`, `cache_creation_tokens`, `reasoning_tokens` | Raw counts from the provider. |
| `billable_tokens` | Weighted sum, rounded. This is what quotas count. |
| `latency_ms` | Time to the end of the response. |
| `complete` | `1` if the response finished, `0` if the stream stopped early. |
| `estimated` | `1` when the prompt or output count was estimated because the stream stopped early. |
| `api_key_id` | Only when `via` is `key`. |
| `chat_id` | Only when the forwarder sent `X-OpenWebUI-Chat-Id`. |
| `request_class` | Only when Claude Code sent `x-claude-code-request-class`. |
| `client_version`, `client_entrypoint`, `client_workload` | Only when read from Claude Code's attribution block. |

Something else must read the stream and move the events into a database. That program is
not part of this repository.

## Errors

| Status | When | Body `type` (Anthropic / OpenAI) |
|---|---|---|
| 401 | Missing or unknown key. | `authentication_error` / `invalid_api_key` |
| 403 | Account disabled, pending approval, or not `admin` on an admin path. | `permission_error` / `permission_denied` |
| 429 | A quota is used up. `Retry-After` says how many seconds until the window rolls over. | `rate_limit_error` / `rate_limit_exceeded` |
| 503 | Postgres is unreachable (any keyed path) or Valkey is unreachable (metered paths). | `api_error` / `service_unavailable` |

Requests to `/v1/messages` get Anthropic-shaped error bodies. All others get OpenAI-shaped
bodies.

## Limitations

- There are no TLS flags. Terminate TLS in front of the gate.
- The Postgres connection uses `NoTls`. Keep the database on a private network.
- Embeddings, rerank and web-search calls need a key but are not metered.
- The root `Dockerfile` builds only `switchyard-server`. Build the gate from source.
