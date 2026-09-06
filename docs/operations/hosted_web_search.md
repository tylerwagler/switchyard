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
[search.main]                       # named search endpoint (typically SearXNG)
base_url = "http://127.0.0.1:8080"
timeout_ms = 15000
max_results = 20

[rerank.qwen3-vl-rerank]            # optional named rerank backend
base_url = "http://host:8002/v1"
model = "qwen3-vl-rerank"

[cache.valkey]                      # optional named cache backend
url = "redis://valkey.defense.lan:6379"
ttl_s = 3600
key_prefix = "switchyard:websearch"

[web_search]
enabled = true
search = "main"                     # reference the named endpoint
rerank = "qwen3-vl-rerank"          # re-rank candidates before returning
cache = "valkey"                    # memoize raw search results
max_results = 6                     # results returned per query (1-20)
```

The sections live in the deployment's `routes.toml` alongside `[llm_clients]`,
`[targets]`, and `[routes]`. An absent section or `enabled = false` leaves every
request untouched. When enabled, `--dry-run` validates URLs and the
`max_results` range at load time. `search` references a `[search.*]` endpoint;
the legacy inline `searxng_url` key remains a compatibility alias (mutually
exclusive with `search`).

## Re-ranking

When `web_search.rerank` names a `[rerank.*]` backend, the bridge fetches a
surplus of raw candidates (3× `max_results`) and re-ranks them against the query
via the Cohere-shaped `POST /v1/rerank` endpoint (query vs `title\nsnippet`),
returning the top `max_results` best-first. The re-ranker counters the noisy
ordering scraped engines often produce. It is **fail-open**: if the rerank
backend is unreachable or errors, results are returned in raw engine order and
`switchyard.websearch_rerank_errors` increments — a reranker outage never fails
a search.

Only requests whose *every* declared tool is a web-search tool (or that carry
the single-message "Perform a web search for the query: …" instruction) are
short-circuited; all other traffic is routed exactly as before.

## Caching

When `web_search.cache` names a `[cache.*]` backend, raw SearXNG candidates are
memoized under `{key_prefix}:{candidate_count}:{query}` for `ttl_s` seconds.
Caching the *raw* candidates (not the reranked, truncated output) means a
rerank or `max_results` change takes effect without waiting out the TTL.

Only successful searches are stored, so an outage expires with the outage
rather than occupying the cache for a full TTL. Like the reranker the cache is
**fail-open**: an unreachable backend is logged at debug and the search
proceeds against SearXNG. Every round trip is bounded at 250ms, so the cache
can never make a search slower than it would have been without one. The
connection is not written off on first failure — a backend that is down at
startup is retried on the next search.

`switchyard.websearch_cache` counts `hit` and `miss` outcomes.

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

## Serving non-chat backends

The named backends are also served by the gateway itself: `POST /v1/embeddings`
and `POST /v1/rerank` (default = first configured backend, or a `/{name}` path
segment) relay to the `[embeddings.*]` / `[rerank.*]` backends. `GET /v1/models`
advertises a truthful capability listing — chat routes plus `kind: embeddings` /
`kind: rerank` / `kind: search` entries.

## Relationship to MCP search

Server-side web search is a *client-side* feature of Claude Code: the tool is
declared or omitted by the client, never by the gateway. If your client omits
it (for example, when running with certain auth modes), the bridge simply never
engages — MCP-search servers remain a valid alternative for those setups.
