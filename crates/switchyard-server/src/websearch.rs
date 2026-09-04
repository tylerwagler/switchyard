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
use switchyard_runner::WebSearchConfig;
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

async fn search(
    query: &str,
    config: &WebSearchConfig,
    client: &reqwest::Client,
) -> Result<Vec<Value>, String> {
    let url = format!(
        "{}/search?q={}&format=json&pageno=1",
        config.searxng_url(),
        encode_query(query)
    );
    let response = client
        .get(url)
        .timeout(config.timeout())
        .header("Accept", "application/json")
        .header("User-Agent", "switchyard-websearch/1.0")
        .send()
        .await
        .map_err(|error| format!("searxng request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("searxng returned {}", response.status()));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("searxng body parse failed: {error}"))?;
    let mut out = Vec::new();
    if let Some(results) = body.get("results").and_then(Value::as_array) {
        for item in results.iter().take(config.max_results()) {
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

fn aggregate(model: &str, query: &str, results: &[Value]) -> Value {
    let (content, _) = build_blocks(query, results);
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

fn sse_stream(model: &str, query: &str, results: &[Value]) -> RawEventStream {
    let (content, _) = build_blocks(query, results);
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
    let config = state.web_search_config()?;
    if !config.is_enabled() {
        return None;
    }
    let started = Instant::now();
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MODEL)
        .to_string();
    let query = extract_query(body);
    let results = match search(&query, config, state.web_search_client()).await {
        Ok(results) => {
            record("ok", started);
            results
        }
        Err(error) => {
            tracing::warn!(%error, "web search failed");
            record("error", started);
            Vec::new()
        }
    };

    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if streaming {
        Some(
            frame_stream(
                sse_stream(&model, &query, &results),
                WireFormat::AnthropicMessages,
            )
            .into_response(),
        )
    } else {
        Some(Json(aggregate(&model, &query, &results)).into_response())
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
        let body = aggregate("claude-fable-5-1", "q", &results);
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
