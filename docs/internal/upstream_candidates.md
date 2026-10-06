# Upstream candidates from the `dev` branch

Assessed 2026-10-06 against `upstream/main` at `b9e7ccce`. Upstream is
NVIDIA-NeMo/Switchyard. This page is excluded from the published site.

## How upstream takes contributions

- DCO only, no CLA. Every commit needs `Signed-off-by`.
- Under about 100 lines: open a PR directly. Larger: open an issue first.
- One concern per PR. Squash merge. Conventional Commits title.
- External contributors are merged regularly. The weekly stream is
  `fix(translation)`, libsy algorithm work, metrics fixes, and docs.
- Reviewers prefer allowlists over denylists for anything forwarded, ask for a
  reproduction against the real provider, and push back on config added for
  hypothetical needs.
- The README calls `switchyard-server` a demo. In issue 480 a maintainer
  endorsed "Switchyard owns policy, the gateway owns auth and quota".

## Hygiene before any submission

- Twelve commits have no sign-off and carry `Co-Authored-By` and
  `Claude-Session` trailers. Re-author each candidate as one signed commit.
  The DCO check compares each commit's sign-off to its own author, so the
  work-hours and off-hours emails can both appear in one PR.
- Strip Claude Code wording from commit bodies and comments where the fix is
  generic.
- Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test --workspace`.

## Submit directly (small, generic, tested)

| Commit | Change | Notes |
|---|---|---|
| `42ecd1b5` | Keep Anthropic server tools out of translated tool lists | Submitted 2026-10-06 as upstream PR 917 from branch `fix/translation-server-tools`. |
| `2180f3af` | Treat a `stop_reason` `message_delta` as a completed stream | Cite Anthropic's streaming docs. Name the upstream that omitted `message_stop`. |
| `98cb6fb3` (phrase half) | Add the "configured context size" overflow phrase | Split out. Keep the `capability_rejected: prompt_too_long` marker local. |
| `8c590485` + `c0836d6c` + `acfebe3a` | System blocks stay separate on re-encode; open-list extension fields forwarded | Squash the test fixup. Use a neutral first block in the test, not the billing header. Mention closed issues 508 and 810. |
| `e3f96cc8` | Forward the caller's `anthropic-version` | Changes the `apply_auth` signature in libsy-llm-client. Say so. |
| `5522915d` | SSE keepalive every 15 seconds | Keep the constant, do not add config. Explain why upstream pings do not survive translation. |
| `6c368a3c` (time fields only) | `started_at`, `uptime_s`, `last_request` in `/v1/stats` | Leave the revived fallback counters out. |

## Open an issue first (new public surface)

| Commits | Change | Framing |
|---|---|---|
| `b653aa2a` | `display_name` and `description` on routes | Generic model-list metadata. Drop the `/model` picker wording. |
| `0be1edf8` + `aec74e9e` | `default_route` and `default_model` | "Clients that send unlisted ids". Squash the two. |
| `bd402447` + `0d6cefa8` | Forward `anthropic-beta` verbatim and `context_management` | Both reverse documented upstream decisions. One issue: "Anthropic beta features through an own-key Anthropic backend". Argue that `omit_body_fields` from PR 852 now gives the per-target opt-out. PR only after agreement. |
| `0b78af7a` + `4c12d059` + `2468cc8f` | Per-upstream attribution and TTFB | Touches `protocol::Response` and the Relay plugin. Cite issue 480 and PR 773. Split into attribution plumbing and the TTFB histogram. Keep the zero-series removal out. |
| `600d7051` | `GET /v1/upstreams` | Likely answer: the gateway owns health checks. Accept a no. |

## Superseded

`d7b159a8` + `292b047d` (retry and rate-limit headers on errors) duplicates open
upstream PR 904, which has the same design. Drop ours when 904 merges. The
Retry-After date-to-seconds normalisation is a possible 30-line follow-up.

## Keep local

- `switchyard-gate` (`14cefe57`), attribution metering and request class
  (`d4685a4c`, `a2feb9f1`). Upstream places auth, quota and billing in the
  surrounding gateway. The `strip_attribution` flag alone could be an issue
  later, since it fixes prefix-cache misses for every Claude Code user behind
  a non-Anthropic model.
- Hosted web search, SearXNG, rerank and Valkey cache (12 commits), and the
  `/v1/embeddings` and `/v1/rerank` relay (`1d669768`). New product surface
  with external dependencies. Outside the routing scope.
- Claude Code safeguards (`f28d28c6`, `38e773e9`, `b6316d6e`, `9bb033fe`,
  `29a05943`). About 1,200 lines of Claude Code protocol with a bundled
  prompt. `f28d28c6` alone (answer `unsupported`) could be pitched later.
- The revived fallback counters in `6c368a3c` and the `routing_fallbacks`
  metric (`a058efc4`). Upstream deprecated these on purpose. Ask in an issue
  whether they want a reason-labelled counter before sending code.

## Suggested order

1. `42ecd1b5`, `2180f3af`, overflow phrase, system-block fixes, `anthropic-version`, keepalive, stats time fields.
2. Issues for route metadata, `default_route`, Anthropic betas, attribution and TTFB, `/v1/upstreams`.
3. After PR 904 merges, rebase `dev` and drop our retry-header commits.

## Merge hazard

`dev` carries about 23 upstream PRs re-applied with different hashes. A trial
merge of `upstream/main` (seven commits ahead as of this date) conflicts in
`crates/libsy-llm-client/src/run.rs`. Expect conflicts on each sync until the
duplicated commits are replaced by the upstream versions.

## Open upstream PRs to watch (surveyed 2026-10-06)

Cherry-pick onto `dev` now, small and clean against our hunks:

- 918: Anthropic `model_context_window_exceeded` stop mapped to a token limit.
- 855: Chat and Anthropic 200-with-error bodies become 502, so fallback runs.
- 884: vLLM `abort`, `error`, `repetition` finish reasons become 502.
- 854: connect timeouts fall back to the next target instead of 504 (draft).
- 847: judges see a text projection, so screenshots no longer skip a text-only judge.

Wait for merge, then expect a rebase:

- 904: retry and rate-limit headers on errors. Supersedes our `d7b159a8`. Ask on the PR for `x-should-retry` and Retry-After date normalisation.
- 719: rewrites the `/v1/models` handler and adds an overflow phrase. Land `b653aa2a`, `aec74e9e` and the `98cb6fb3` phrase before it.
- 909: Bedrock Converse. Touches `is_terminal_event`. Land `2180f3af` before it.
- 821: `max_response_bytes` on every client config literal.
- 857, 807, 668, 813, 836: all insert into the client send path where our Anthropic header forwarding sits.
- 910 `judge_deadline_ms`, 791 `judge_char_budget`, 906 Codex read detection: useful routing improvements.
- 858: server defaults to loopback once merged. Update the full-stack guide to pass `--host`.
