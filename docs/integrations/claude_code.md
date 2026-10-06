# Use Switchyard with Claude Code

Claude Code reads its endpoint and model names from environment variables. Point
`ANTHROPIC_BASE_URL` at `switchyard-server` and Claude Code sends every request to
`POST /v1/messages`, where the route picks the target model for each call. This page
lists the variables that matter and the server behaviours that exist for Claude Code.
For a full team setup, with a config file and a service unit, read
[Full-stack deployment for coding agents](../recipes/full_stack_deployment.md). To
route Claude Code across Anthropic's own models with your saved login, read
[Single-provider coding agents](../recipes/single_provider_coding_agents.md#claude-code-with-anthropic).

## Configure

A wrapper script, or the same variables in the `env` block of Claude Code's
`settings.json`:

```bash
#!/usr/bin/env bash
exec env -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL=http://localhost:4000 \
  ANTHROPIC_AUTH_TOKEN=switchyard \
  ANTHROPIC_MODEL=switchyard \
  ANTHROPIC_DEFAULT_OPUS_MODEL=switchyard \
  ANTHROPIC_DEFAULT_SONNET_MODEL=switchyard \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=small \
  ANTHROPIC_SMALL_FAST_MODEL=small \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
  CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 \
  claude "$@"
```

- `ANTHROPIC_BASE_URL` is the server root. Do not add `/v1`. Claude Code adds the path.
- `ANTHROPIC_AUTH_TOKEN` must be non-empty. The plain server ignores the value unless a
  route's LLM client sets `forward_auth = true`, in which case the token goes upstream as
  the API key. With [`switchyard-gate`](../../crates/switchyard-gate/README.md) in
  front, the token is the user's gate API key.
- Unset `ANTHROPIC_API_KEY`, so a personal Anthropic key does not leak to the server.
- `ANTHROPIC_MODEL` is the model Claude Code starts with. The three
  `ANTHROPIC_DEFAULT_*_MODEL` values map Claude Code's built-in names (`opus`, `sonnet`,
  `haiku`) onto route ids. `ANTHROPIC_SMALL_FAST_MODEL` is the model for background work,
  such as summaries. Every value must be a route `id` from your TOML file.
- `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` makes the `/model` picker list the routes
  from `GET /v1/models`. The picker shows each route's `display_name` and `description`
  from [`[routes.<name>]`](../reference/toml_schema.md#routesname).
- `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` makes Claude Code send
  `x-claude-code-request-class` (`main`, `subagent`, `compaction`, and so on) on each
  request. The server does not read it. The gate records it on usage events.

## What the server does for Claude Code

- **Unaliased model ids.** Claude Code sends several model ids of its own, and a new
  release can add one. The top-level `default_route` in the TOML file catches any id that
  matches no route. Without it, an unknown id returns 404.
- **Attribution block.** Claude Code starts every request with a billing header block in
  the first `system` entry. Its fingerprint changes per conversation, so a non-Anthropic
  model never shares a prompt-cache prefix. Set `strip_attribution = true` on routes whose
  targets are not Anthropic's API. See
  [`[routes.<name>]`](../reference/toml_schema.md#routesname).
- **Auto-mode safety checks.** Claude Code asks the server to judge risky tool calls.
  [`[safeguards]`](../reference/toml_schema.md#safeguards) names a route that answers.
  Without the section, the server answers "unsupported" and Claude Code runs its own check.
- **`web_search` tool.** Claude Code's built-in web search expects the server to run the
  search. [`[web_search]`](../reference/toml_schema.md#web_search) answers it with a
  search engine, an optional re-ranker and an optional cache. Without it, vLLM rejects
  the tool with a 422. See
  [Hosted Web Search](../operations/hosted_web_search.md).
- **Keepalive.** A stream that is silent for 15 seconds gets a keepalive frame, so
  proxies and Claude Code do not drop long tool turns.
- **Token counting.** `POST /v1/messages/count_tokens` needs a target with
  `format = "anthropic_messages"` in the route. When the route has none, the server
  returns 400 `count_tokens_unsupported`. Claude Code keeps working. Expect that line in
  the logs.

## Check the routing

```bash
curl -s localhost:4000/v1/stats | jq '{answers: .models, classifier: .classifier.models}'
```

Every response carries the header `x-model-router-selected-model`. With
`--routing-log-file PATH`, the server writes one JSON record per call with the route,
the model and the token counts.
