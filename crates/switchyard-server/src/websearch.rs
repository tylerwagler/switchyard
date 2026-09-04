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
        if let Some(content) = message.get("content").and_then(Value::as_str) {
            return parse_instruction(content).is_some();
        }
    }
    false
}

fn extract_query(body: &Value) -> String {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages.iter().rev() {
            let content = message.get("content");
            if let Some(text) = content.and_then(Value::as_str) {
                if let Some(query) = parse_instruction(text) {
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

fn build_blocks(query: &str, results: &[Value]) -> (Vec<Value>, String) {
    let mut blocks = Vec::with_capacity(results.len() + 1);
    for result in results {
        blocks.push(json!({
            "type": "server_tool_use",
            "id": format!("web-{}", hex(result["url"].as_str().unwrap_or(""))),
            "name": "web_search",
            "input": { "query": query },
            "search_result": result,
        }));
    }
    let answer = if results.is_empty() {
        format!("No web search results found for \u{201c}{query}\u{201d}.")
    } else {
        let lines: Vec<String> = results
            .iter()
            .map(|r| {
                format!(
                    "- {} \u{2014} {}",
                    r["title"].as_str().unwrap_or("(untitled)"),
                    r["url"].as_str().unwrap_or("")
                )
            })
            .collect();
        format!("Web search results for \u{201c}{query}\u{201d}:\n{}", lines.join("\n"))
    };
    blocks.push(json!({ "type": "text", "text": answer }));
    (blocks, answer)
}

fn hex(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes().take(16) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn message_id() -> String {
    format!("msg_{}", hex(&format!("{:?}", Instant::now())).chars().take(24).collect::<String>())
}

/// A text-only explanation shown instead of "no results" when SearXNG itself
/// fails, so the client can tell a genuine empty result set from an outage.
fn failure_text(error: &str) -> Vec<Value> {
    vec![json!({
        "type": "text",
        "text": format!(
            "Web search is temporarily unavailable: {error}. No results were returned."
        ),
    })]
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
        "usage": { "input_tokens": 0, "output_tokens": 0 },
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
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            },
        })];
        for (index, block) in content.iter().enumerate() {
            events.push(json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            }));
            let delta = if block["type"] == "server_tool_use" {
                json!({ "type": "input_json_delta", "partial_json": serde_json::to_string(block).unwrap_or_default() })
            } else {
                json!({ "type": "text_delta", "text": block["text"].as_str().unwrap_or("") })
            };
            events.push(json!({ "type": "content_block_delta", "index": index, "delta": delta }));
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
            failure_text(&error)
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
        assert_eq!(content[0]["type"], "server_tool_use");
        assert_eq!(content[0]["search_result"]["url"], "https://example.com");
        assert_eq!(content.last().unwrap()["type"], "text");
        assert!(body["content"].as_array().unwrap().last().unwrap()["text"]
            .as_str()
            .unwrap()
            .contains("Web search results"));
    }
}
