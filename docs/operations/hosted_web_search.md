# Hosted Web Search

Claude Code supports a server-side `web_search` tool. When present, the client
may emit *dedicated* `/v1/messages` requests that declare only the
`web_search` / `web_search_20250305` tool and instruct the model to "Perform a
web search for the query: …", expecting the *serving layer* to execute the
search and return the results in the Anthropic response (`server_tool_use`
blocks with `web_search_result` content).

`vLLM`'s Anthropic adapter cannot serve these requests: server-side tools carry
no `input_schema`, which vLLM rejects with a 422. Switchyard's **hosted web
search** bridge short-circuits exactly those requests and answers them from a
self-hosted [SearXNG](https://docs.searxng.org) instance, so the native tool
works end to end.

## Configuration

```toml
[web_search]
enabled = true                 # off unless enabled (also the default when absent)
searxng_url = "http://127.0.0.1:8080"   # SearXNG base URL
max_results = 6                # results returned per query (1-20)
timeout_ms = 15000             # per-request timeout
```

The section lives in the deployment's `routes.toml` alongside `[llm_clients]`,
`[targets]`, and `[routes]`. An absent section or `enabled = false` leaves every
request untouched. When enabled, `--dry-run` validates the URL and the
`max_results` range at load time.

Only requests whose *every* declared tool is a web-search tool (or that carry
the single-message "Perform a web search for the query: …" instruction) are
short-circuited; all other traffic is routed exactly as before.

## What the bridge returns

For a short-circuited request the bridge runs SearXNG (`/search?format=json`)
and synthesizes an Anthropic `message` response:

- one `server_tool_use` **web-search result block per result** (`type:
  web_search_result` with `url` / `title` / `content`), and
- a short text block listing the cited results.

Both aggregate and streaming (`SSE`) responses are supported. If the search
fails or returns nothing, the bridge still returns a synthesized response with
an empty result list and a notice — it never falls back to a model backend for
a dedicated web-search request.

Synthetic searches are not model calls: they are excluded from `/v1/stats`
per-model counters and surfaced instead as
`switchyard.websearch_queries` / `switchyard.websearch_duration_seconds`
Prometheus instruments.

## Networking

The bridge calls SearXNG directly from the Switchyard process. It uses the same
client and proxy behavior as the rest of the server, so a loopback SearXNG
(e.g. `http://127.0.0.1:8080`) works out of the box where `NO_PROXY` covers
localhost.

## Relationship to MCP search

Server-side web search is a *client-side* feature of Claude Code: the tool is
declared or omitted by the client, never by the gateway. If your client omits
it (for example, when running with certain auth modes), the bridge simply never
engages — MCP-search servers remain a valid alternative for those setups.
