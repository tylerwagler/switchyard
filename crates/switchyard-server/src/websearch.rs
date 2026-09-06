// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hosted web search — serves Claude Code's native server-side `web_search` tool.
//!
//! Claude Code (desktop Code tab; the CLI once the `WebSearch(*)` deny is lifted)
//! sometimes issues *dedicated* `/v1/messages` requests that declare only the
//! server-side web-search tool (`web_search` / `web_search_20250305`) and carry an
//! instruction like "Perform a web search for the query: <q>". Server-side search
//! tools carry no `input_schema`, so vLLM's Anthropic adapter rejects them with a
//! 422 — the desktop's WebSearch is dead behind a plain switchyard.
//!
//! This module short-circuits exactly those requests: it detects the dedicated
//! web-search leg, runs SearXNG for the query, and synthesizes an Anthropic
//! `server_tool_use` response (web_search_result blocks plus a short cited
//! answer), aggregate or SSE, through the existing response/SSE plumbing. All
//! other traffic passes through untouched.
//!
//! Configuration lives in the deployment's `[web_search]` section (see
//! `switchyard_runner::WebSearchConfig`); the bridge is off until enabled.

use std::time::Instant;

use axum::Json;
use axum::response::{IntoResponse, Response as AxumResponse};
use opentelemetry::{KeyValue, global};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};
use switchyard_runner::{ResolvedRerank, ResolvedWebSearch};
use switchyard_translation::{LlmStreamError, RawEventStream, WireFormat};

use crate::ServerState;
use crate::sse::frame_stream;

const DEFAULT_MODEL: &str = "claude-fable-5-1";

/// Search-instruction sentence shapes the client prepends to the query.
const QUERY_PREFIXES: &[&str] = &[
    "perform a web search for the query:",
    "perform a web search for query:",
    "web search for the query:",
    "search for the query:",
];

// --- detection ---------------------------------------------------------------

fn is_web_tool_name(value: &str) -> bool {
    let value = value.trim();
    value.eq_ignore_ascii_case("web_search")
        || value.eq_ignore_ascii_case("websearch")
        || value.strip_prefix("web_search_").is_some()
}

fn is_web_tool(tool: &Value) -> bool {
    let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
    let typ = tool.get("type").and_then(Value::as_str).unwrap_or("");
    is_web_tool_name(name) || is_web_tool_name(typ)
}

/// Extracts the search query from Claude Code's instruction sentence.
fn parse_instruction(content: &str) -> Option<String> {
    let lowered = content.to_ascii_lowercase();
    for prefix in QUERY_PREFIXES {
        if let Some(idx) = lowered.find(prefix) {
            let query = content[idx + prefix.len()..].trim();
            if !query.is_empty() {
                return Some(query.to_string());
            }
        }
    }
    None
}

/// Flattens an Anthropic `content` field to plain text so detection and query
/// extraction handle both wire forms: the legacy bare string and the array of
/// content blocks (`[{"type": "text", "text": "…"}, …]`) that real Claude Code
/// clients send. Non-text blocks (thoughts, tool_use, …) are ignored.
fn content_text(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let mut out = String::new();
    if let Some(blocks) = content.as_array() {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text);
            }
        }
    }
    out
}

/// True for the dedicated web-search request shape we short-circuit: every
/// declared tool is a web-search tool, or the single user message is the literal
/// "Perform a web search for the query: …" instruction.
pub(crate) fn is_dedicated_web_search(body: &Value) -> bool {
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() && tools.iter().all(is_web_tool) {
            return true;
        }
    }
    let messages = body.get("messages").and_then(Value::as_array);
    if let Some([message]) = messages.map(|m| m.as_slice()) {
        if let Some(content) = message.get("content") {
            return parse_instruction(&content_text(content)).is_some();
        }
    }
    false
}

fn extract_query(body: &Value) -> String {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages.iter().rev() {
            if let Some(content) = message.get("content") {
                let text = content_text(content);
                if let Some(query) = parse_instruction(&text) {
                    return query;
                }
                if !text.trim().is_empty() {
                    return text.trim().chars().take(500).collect();
                }
            }
        }
    }
    String::new()
}

// --- SearXNG -----------------------------------------------------------------

/// Percent-encodes a string for use as a URL query value (UTF-8 aware).
/// reqwest 0.13 dropped `RequestBuilder::query`, so the query is built by hand.
fn encode_query(value: &str) -> String {
    let mut out = String::new();
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Extra attempts after the first before reporting a search failure. SearXNG's
/// upstream engines suspend briefly (rate limits, CAPTCHAs); a retry usually
/// succeeds without failing the request end-to-end.
const SEARXNG_RETRIES: u32 = 2;
/// Base backoff (ms) between retries, doubled each attempt.
const RETRY_BASE_MS: u64 = 250;
/// Raw candidates fetched per returned result when a rerank backend is configured,
/// so the reranker (not engine ordering) decides which `max_results` win.
const CANDIDATES_PER_RESULT: usize = 3;

async fn search(
    query: &str,
    settings: &ResolvedWebSearch,
    client: &reqwest::Client,
) -> Result<Vec<Value>, String> {
    // With a reranker, fetch a surplus of raw candidates and let it pick; without
    // one, ask the engine for exactly what we return.
    let requested = if settings.rerank.is_some() {
        settings
            .max_results
            .saturating_mul(CANDIDATES_PER_RESULT)
            .clamp(1, 20)
    } else {
        settings.max_results.max(1)
    };
    let url = format!(
        "{}/search?q={}&format=json&pageno=1",
        settings.search_url,
        encode_query(query)
    );
    for attempt in 0..=SEARXNG_RETRIES {
        match search_once(&url, settings.timeout, requested, client).await {
            Ok(results) => return Ok(results),
            Err(error) if attempt < SEARXNG_RETRIES => {
                tracing::debug!(%error, attempt, websearch = true, "search request failed; retrying");
                tokio::time::sleep(std::time::Duration::from_millis(RETRY_BASE_MS << attempt))
                    .await;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("search loop always returns");
}

async fn search_once(
    url: &str,
    timeout: std::time::Duration,
    take: usize,
    client: &reqwest::Client,
) -> Result<Vec<Value>, String> {
    let response = client
        .get(url)
        .timeout(timeout)
        .header("Accept", "application/json")
        .header("User-Agent", "switchyard-websearch/1.0")
        .send()
        .await
        .map_err(|error| format!("search request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("search returned {}", response.status()));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("search body parse failed: {error}"))?;
    let mut out = Vec::new();
    if let Some(results) = body.get("results").and_then(Value::as_array) {
        for item in results.iter().take(take) {
            let url = item.get("url").and_then(Value::as_str).unwrap_or("");
            if url.is_empty() {
                continue;
            }
            out.push(json!({
                "type": "web_search_result",
                "url": url,
                "title": item.get("title").and_then(Value::as_str).unwrap_or(url),
                "content": item.get("content").and_then(Value::as_str).unwrap_or(""),
            }));
        }
    }
    Ok(out)
}

/// Re-ranks `candidates` against `query` via the named Cohere-shaped `/v1/rerank`
/// backend, returning them best-first, or `None` to signal "use raw order" on any
/// failure (the bridge is fail-open: a reranker outage never fails the search).
async fn rerank(
    query: &str,
    candidates: &[Value],
    rerank: &ResolvedRerank,
    client: &reqwest::Client,
) -> Option<Vec<Value>> {
    let documents: Vec<String> = candidates
        .iter()
        .map(|candidate| {
            let title = candidate["title"].as_str().unwrap_or("");
            let content = candidate["content"].as_str().unwrap_or("");
            if title.is_empty() {
                content.to_string()
            } else {
                format!("{title}\n{content}")
            }
        })
        .collect();

    let payload = json!({ "model": rerank.model, "query": query, "documents": documents });
    let response = match client
        .post(format!("{}/rerank", rerank.base_url.trim_end_matches('/')))
        .json(&payload)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            record_rerank_error();
            tracing::warn!(%error, "web search rerank request failed; using raw order");
            return None;
        }
    };
    if !response.status().is_success() {
        record_rerank_error();
        tracing::warn!(
            status = %response.status(),
            "web search rerank returned an error; using raw order"
        );
        return None;
    }
    let body: Value = match response.json().await {
        Ok(body) => body,
        Err(error) => {
            record_rerank_error();
            tracing::warn!(%error, "web search rerank body parse failed; using raw order");
            return None;
        }
    };
    let Some(results) = body.get("results").and_then(Value::as_array) else {
        record_rerank_error();
        tracing::warn!("web search rerank response had no results; using raw order");
        return None;
    };

    let mut scored: Vec<(usize, f64)> = results
        .iter()
        .filter_map(|entry| {
            let index = entry.get("index")?.as_u64()? as usize;
            let score = entry.get("relevance_score")?.as_f64()?;
            Some((index, score))
        })
        .collect();
    // Descending relevance; stable sort keeps original order on ties.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let ranked: Vec<Value> = scored
        .iter()
        .filter_map(|(index, _)| candidates.get(*index).cloned())
        .collect();
    if ranked.is_empty() { None } else { Some(ranked) }
}

fn record_rerank_error() {
    global::meter("switchyard")
        .u64_counter("switchyard.websearch_rerank_errors")
        .build()
        .add(1, &[]);
}

// --- response synthesis ------------------------------------------------------

/// Anthropic's server-side web search returns ONE `server_tool_use` block
/// carrying the query, followed by ONE `web_search_tool_result` block whose
/// `content` holds the whole result array — not one block per result. Claude
/// Code reads that pair; a different shape is dropped rather than rejected,
/// so the search silently returns nothing.
fn build_blocks(query: &str, results: &[Value]) -> (Vec<Value>, String) {
    let tool_id = tool_use_id();
    let answer = if results.is_empty() {
        format!("No web search results found for \u{201c}{query}\u{201d}.")
    } else {
        let lines: Vec<String> = results
            .iter()
            .map(|r| {
                let title = r["title"].as_str().unwrap_or("(untitled)");
                let url = r["url"].as_str().unwrap_or("");
                match r["content"].as_str().unwrap_or("") {
                    "" => format!("- {title} \u{2014} {url}"),
                    snippet => format!("- {title} \u{2014} {url}\n  {snippet}"),
                }
            })
            .collect();
        format!("Web search results for \u{201c}{query}\u{201d}:\n{}", lines.join("\n"))
    };

    let blocks = vec![
        json!({
            "type": "server_tool_use",
            "id": tool_id,
            "name": "web_search",
            "input": { "query": query },
        }),
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": results.iter().map(search_result_block).collect::<Vec<Value>>(),
        }),
        json!({ "type": "text", "text": answer }),
    ];
    (blocks, answer)
}

/// One `web_search_result` entry. `encrypted_content` is opaque to the client
/// and round-tripped back on later turns; upstream it is a ciphertext blob, so
/// the snippet is base64-encoded rather than given an invented plaintext
/// meaning. The readable copy lives in the trailing text block, which is what a
/// self-hosted model actually reads.
fn search_result_block(result: &Value) -> Value {
    let mut block = json!({
        "type": "web_search_result",
        "url": result["url"].as_str().unwrap_or(""),
        "title": result["title"].as_str().unwrap_or("(untitled)"),
        "encrypted_content": BASE64.encode(result["content"].as_str().unwrap_or("")),
    });
    if let Some(age) = result.get("publishedDate").and_then(Value::as_str) {
        block["page_age"] = json!(age);
    }
    block
}

/// Ids must differ per call: a client pairs a `web_search_tool_result` with its
/// `server_tool_use` by id, so a constant id collides as soon as one
/// conversation runs two searches. `Instant`'s `Debug` output opens with a
/// fixed `"Instant { tv_sec"`, so hex-encoding its leading bytes returned the
/// same id every time.
static ID_SEQ: AtomicU64 = AtomicU64::new(0);

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0);
    format!("{nanos:016x}{:08x}", ID_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// `srvtoolu_`-prefixed to match the ids Anthropic issues for server tools.
fn tool_use_id() -> String {
    format!("srvtoolu_{}", unique_suffix())
}

fn message_id() -> String {
    format!("msg_{}", unique_suffix())
}

/// Shown instead of "no results" when SearXNG itself fails, so the client can
/// tell a genuine empty result set from an outage. Uses the documented
/// `web_search_tool_result_error` shape so the client sees a failed search
/// rather than a malformed one.
fn failure_blocks(query: &str, error: &str) -> Vec<Value> {
    let tool_id = tool_use_id();
    vec![
        json!({
            "type": "server_tool_use",
            "id": tool_id,
            "name": "web_search",
            "input": { "query": query },
        }),
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": { "type": "web_search_tool_result_error", "error_code": "unavailable" },
        }),
        json!({
            "type": "text",
            "text": format!(
                "Web search is temporarily unavailable: {error}. No results were returned."
            ),
        }),
    ]
}

fn aggregate_from_content(model: &str, content: &[Value]) -> Value {
    json!({
        "id": message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": { "web_search_requests": 1 },
        },
    })
}

fn sse_from_content(model: &str, content: &[Value]) -> RawEventStream {
    let events: Vec<Value> = {
        let mut events = vec![json!({
            "type": "message_start",
            "message": {
                "id": message_id(),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "server_tool_use": { "web_search_requests": 1 },
                },
            },
        })];
        for (index, block) in content.iter().enumerate() {
            match block["type"].as_str() {
                // Tool-use blocks stream their arguments: `content_block_start`
                // carries an empty `input`, the delta carries the JSON.
                Some("server_tool_use") => {
                    let mut start = block.clone();
                    start["input"] = json!({});
                    events.push(json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": start,
                    }));
                    events.push(json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": serde_json::to_string(&block["input"])
                                .unwrap_or_default(),
                        },
                    }));
                }
                // Result blocks are not streamed as deltas: they arrive whole.
                Some("web_search_tool_result") => {
                    events.push(json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block,
                    }));
                }
                _ => {
                    let mut start = block.clone();
                    start["text"] = json!("");
                    events.push(json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": start,
                    }));
                    events.push(json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": block["text"].as_str().unwrap_or(""),
                        },
                    }));
                }
            }
            events.push(json!({ "type": "content_block_stop", "index": index }));
        }
        events.push(json!({
            "type": "message_delta",
            "delta": { "stop_reason": "end_turn", "stop_sequence": null },
            "usage": { "output_tokens": 0 },
        }));
        events.push(json!({ "type": "message_stop" }));
        events
    };

    let stream = async_stream::stream! {
        for event in events {
            yield Ok::<Value, LlmStreamError>(event);
        }
    };
    Box::pin(stream) as RawEventStream
}


// --- metrics ----------------------------------------------------------------

fn record(outcome: &str, started: Instant) {
    let meter = global::meter("switchyard");
    meter
        .u64_counter("switchyard.websearch_queries")
        .build()
        .add(1, &[KeyValue::new("outcome", outcome.to_string())]);
    meter
        .f64_histogram("switchyard.websearch_duration_seconds")
        .build()
        .record(started.elapsed().as_secs_f64(), &[]);
}

// --- entry point -------------------------------------------------------------

/// Short-circuits dedicated web-search requests with a synthesized response.
/// Everything else returns `None` and the normal routing path runs unchanged.
pub(crate) async fn maybe_short_circuit(
    state: &ServerState,
    wire_format: WireFormat,
    body: &Value,
) -> Option<AxumResponse> {
    if wire_format != WireFormat::AnthropicMessages || !is_dedicated_web_search(body) {
        return None;
    }
    // `state.web_search_config()` is only present when web search is resolved+enabled.
    let settings = state.web_search_config()?;
    let started = Instant::now();
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MODEL)
        .to_string();
    let query = extract_query(body);
    let content = match search(&query, settings, state.web_search_client()).await {
        Ok(candidates) => {
            // Re-rank the surplus of raw candidates when a backend is configured,
            // falling back to raw engine order on any rerank failure (fail-open).
            let ranked = match settings.rerank.as_ref() {
                Some(backend) => {
                    match rerank(&query, &candidates, backend, state.web_search_client()).await {
                        Some(ranked) => {
                            ranked.into_iter().take(settings.max_results).collect::<Vec<_>>()
                        }
                        None => candidates
                            .into_iter()
                            .take(settings.max_results)
                            .collect::<Vec<_>>(),
                    }
                }
                None => candidates.into_iter().take(settings.max_results).collect::<Vec<_>>(),
            };
            record("ok", started);
            build_blocks(&query, &ranked).0
        }
        Err(error) => {
            tracing::warn!(%error, "web search failed");
            record("error", started);
            failure_blocks(&query, &error)
        }
    };

    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if streaming {
        Some(
            frame_stream(
                sse_from_content(&model, &content),
                WireFormat::AnthropicMessages,
            )
            .into_response(),
        )
    } else {
        Some(Json(aggregate_from_content(&model, &content)).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_dedicated_web_tool_request() {
        let body = json!({
            "model": "claude-fable-5-1",
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "Perform a web search for the query: latest AI news"}],
        });
        assert!(is_dedicated_web_search(&body));
        assert_eq!(extract_query(&body), "latest AI news");
    }

    #[test]
    fn ignores_regular_conversations() {
        let body = json!({
            "model": "claude-fable-5-1",
            "tools": [{"name": "Bash"}, {"name": "WebFetch"}],
            "messages": [{"role": "user", "content": "explain the code"}],
        });
        assert!(!is_dedicated_web_search(&body));
    }

    #[test]
    fn parses_instruction_variants() {
        assert_eq!(
            parse_instruction("Perform a web search for the query: foo bar").as_deref(),
            Some("foo bar")
        );
        assert_eq!(
            parse_instruction("Web search for the query: hello").as_deref(),
            Some("hello")
        );
        assert!(parse_instruction("explain this article").is_none());
    }

    #[test]
    fn aggregate_has_search_blocks() {
        let results = vec![json!({
            "type": "web_search_result",
            "url": "https://example.com",
            "title": "Example",
            "content": "snippet",
        })];
        let (content, _) = build_blocks("q", &results);
        let body = aggregate_from_content("claude-fable-5-1", &content);
        let content = body["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "server_tool_use");
        assert_eq!(content[0]["name"], "web_search");
        assert_eq!(content[0]["input"]["query"], "q");
        // the result block references the tool call and carries every result
        assert_eq!(content[1]["type"], "web_search_tool_result");
        assert_eq!(content[1]["tool_use_id"], content[0]["id"]);
        let results = content[1]["content"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["type"], "web_search_result");
        assert_eq!(results[0]["url"], "https://example.com");
        assert_eq!(results[0]["title"], "Example");
        assert!(results[0]["encrypted_content"].as_str().is_some_and(|v| !v.is_empty()));
        // no result is emitted as a bare server_tool_use block
        assert!(content.iter().all(|b| b.get("search_result").is_none()));
        assert_eq!(body["usage"]["server_tool_use"]["web_search_requests"], 1);
        assert_eq!(content[2]["type"], "text");
        assert!(content[2]["text"].as_str().unwrap().contains("Web search results"));
        // the snippet is readable in the text block for a self-hosted model
        assert!(content[2]["text"].as_str().unwrap().contains("snippet"));
    }

    #[test]
    fn ids_are_unique_across_calls() {
        // A conversation can run several searches; colliding ids would make the
        // result blocks unpairable with their tool calls.
        let results = vec![json!({"type":"web_search_result","url":"u","title":"t","content":"c"})];
        let (first, _) = build_blocks("q", &results);
        let (second, _) = build_blocks("q", &results);
        assert_ne!(first[0]["id"], second[0]["id"]);
        assert_eq!(first[1]["tool_use_id"], first[0]["id"]);
        assert_eq!(second[1]["tool_use_id"], second[0]["id"]);
        assert_ne!(message_id(), message_id());
    }
}
