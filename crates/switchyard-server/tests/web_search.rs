// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the hosted web search bridge: dedicated `web_search`
//! requests are served from SearXNG rather than passed to a model backend.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use axum::Router;
use axum::Json;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use switchyard_server::{ServerState, build_switchyard_router};
use tokio::net::TcpListener;
use tower::ServiceExt;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

// --- SearXNG stub ------------------------------------------------------------

struct SearxngStub {
    base_url: String,
    queries: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl SearxngStub {
    async fn start(results: Vec<Value>, failures: u32) -> TestResult<Self> {
        let queries = Arc::new(Mutex::new(Vec::new()));
        let state = StubState {
            results: Arc::new(Mutex::new(results)),
            queries: Arc::clone(&queries),
            failures_left: Arc::new(Mutex::new(failures)),
        };
        let app = Router::new()
            .route("/search", get(searxng_search))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            base_url: format!("http://{addr}"),
            queries,
            task,
        })
    }

    async fn recorded_queries(&self) -> Vec<String> {
        self.queries.lock().unwrap().clone()
    }
}

impl Drop for SearxngStub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Clone)]
struct StubState {
    results: Arc<Mutex<Vec<Value>>>,
    queries: Arc<Mutex<Vec<String>>>,
    failures_left: Arc<Mutex<u32>>,
}

async fn searxng_search(
    State(state): State<StubState>,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    state
        .queries
        .lock()
        .unwrap()
        .push(params.get("q").cloned().unwrap_or_default());
    let mut failures_left = state.failures_left.lock().unwrap();
    if *failures_left > 0 {
        *failures_left -= 1;
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "engine outage" })),
        );
    }
    (
        StatusCode::OK,
        Json(json!({ "results": *state.results.lock().unwrap() })),
    )
}

// --- upstream stub (for the feature-off passthrough check) --------------------

async fn upstream_messages(Json(_body): Json<Value>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "id": "msg_upstream",
            "type": "message",
            "role": "assistant",
            "model": "test/main",
            "content": [{"type": "text", "text": "upstream reply"}],
            "stop_reason": "end_turn",
        })),
    )
}

// --- configuration -----------------------------------------------------------

/// Minimal deployment; `web_search_enabled` toggles the `[web_search]` section.
/// `upstream` and `searxng` are the mock HTTP endpoints.
fn deployment(upstream: &str, searxng: &str, web_search_enabled: bool) -> String {
    let web_search = if web_search_enabled {
        format!(
            "\n[web_search]\nenabled = true\nsearxng_url = \"{searxng}\"\nmax_results = 3\n"
        )
    } else {
        String::new()
    };
    format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "anthropic_messages"
base_url = "{upstream}"

[targets.main]
id = "test/main"
llm_client = "upstream"

[routes.main]
id = "test/main"
type = "passthrough"
target = "main"
{web_search}
"#
    )
}

/// Deployment that routes web search through a named `[search.main]` endpoint
/// and a named `[rerank.r]` backend, so the bridge re-ranks candidates.
fn deployment_with_rerank(
    upstream: &str,
    search: &str,
    rerank_url: &str,
    max_results: usize,
) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "anthropic_messages"
base_url = "{upstream}"

[targets.main]
id = "test/main"
llm_client = "upstream"

[routes.main]
id = "test/main"
type = "passthrough"
target = "main"

[search.main]
base_url = "{search}"

[rerank.r]
base_url = "{rerank_url}"
model = "stub-rerank"

[web_search]
enabled = true
search = "main"
rerank = "r"
max_results = {max_results}
"#
    )
}

fn web_search_body() -> Value {
    json!({
        "model": "test/main",
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "Perform a web search for the query: biggest AI news this week"}],
    })
}

/// Real Claude Code clients send `content` as an array of content blocks, not a
/// bare string; the bridge must extract the query from `{"type": "text", …}`
/// blocks, not hand the client an empty query (which SearXNG 400s).
fn web_search_block_content_body() -> Value {
    json!({
        "model": "test/main",
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Perform a web search for the query: biggest AI news this week"},
        ]}],
    })
}

fn sample_results() -> Vec<Value> {
    vec![
        json!({
            "url": "https://example.com/a",
            "title": "Alpha",
            "content": "first snippet",
        }),
        json!({
            "url": "https://example.com/b",
            "title": "Beta",
            "content": "second snippet",
        }),
    ]
}

// --- request helper ----------------------------------------------------------

struct Response {
    status: StatusCode,
    bytes: Vec<u8>,
}

async fn send(app: &Router, method: &str, path: &str, body: Option<Value>) -> TestResult<Response> {
    let mut builder = HttpRequest::builder().method(method).uri(path);
    let request_body = if let Some(body) = body {
        builder = builder.header("content-type", "application/json");
        Body::from(serde_json::to_vec(&body)?)
    } else {
        Body::empty()
    };
    let response = app.clone().oneshot(builder.body(request_body)?).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes().to_vec();
    Ok(Response { status, bytes })
}

async fn started_router(searxng: &str, upstream: &str, enabled: bool) -> TestResult<Router> {
    let state =
        ServerState::from_runner(Runner::from_toml(&deployment(upstream, searxng, enabled))?)?;
    Ok(build_switchyard_router(state))
}

// --- tests -------------------------------------------------------------------

#[tokio::test]
async fn web_search_aggregate_returns_server_tool_use_blocks() -> TestResult {
    let stub = SearxngStub::start(sample_results(), 0).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let content = body["content"].as_array().expect("content array");
    assert_eq!(content[0]["type"], "server_tool_use");
    assert_eq!(content[0]["name"], "web_search");
    assert_eq!(content[1]["type"], "web_search_tool_result");
    assert_eq!(content[1]["tool_use_id"], content[0]["id"]);
    assert_eq!(content[1]["content"][0]["url"], "https://example.com/a");
    assert_eq!(content[1]["content"][0]["type"], "web_search_result");
    assert_eq!(content.last().unwrap()["type"], "text");
    Ok(())
}

#[tokio::test]
async fn web_search_accepts_array_content_blocks() -> TestResult {
    let stub = SearxngStub::start(sample_results(), 0).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_block_content_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let content = body["content"].as_array().expect("content array");
    assert_eq!(content[0]["type"], "server_tool_use");
    // The query must come from the text block, not be empty.
    assert_eq!(stub.recorded_queries().await, vec!["biggest AI news this week"]);
    Ok(())
}

#[tokio::test]
async fn web_search_streaming_emits_anthropic_sse() -> TestResult {
    let stub = SearxngStub::start(sample_results(), 0).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let mut body = web_search_body();
    body["stream"] = json!(true);
    let response = send(&app, "POST", "/v1/messages", Some(body)).await?;
    assert_eq!(response.status, StatusCode::OK);
    let text = String::from_utf8(response.bytes)?;
    assert!(text.contains("message_start"), "missing message_start: {text}");
    assert!(text.contains("content_block_start"), "missing content_block_start");
    assert!(text.contains("server_tool_use"), "missing server_tool_use block");
    assert!(text.contains("message_stop"), "missing message_stop");
    Ok(())
}

#[tokio::test]
async fn web_search_forwards_the_query_to_searxng() -> TestResult {
    let stub = SearxngStub::start(sample_results(), 0).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let queries = stub.recorded_queries().await;
    assert_eq!(queries, vec!["biggest AI news this week"]);
    Ok(())
}

#[tokio::test]
async fn web_search_disabled_does_not_short_circuit() -> TestResult {
    let stub = SearxngStub::start(sample_results(), 0).await?;
    let upstream = UpstreamApp::start().await?;
    // No `[web_search]` section: the request must flow to the upstream backend.
    let app = started_router(&stub.base_url, &upstream.base_url, false).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let text = serde_json::to_string(&body)?;
    assert!(
        !text.contains("server_tool_use"),
        "feature disabled but bridge short-circuited: {text}"
    );
    assert_eq!(body["content"][0]["text"], "upstream reply");
    Ok(())
}

#[tokio::test]
async fn web_search_retries_transient_searxng_failures() -> TestResult {
    // Two 400s (engines suspended), then a 200: the bridge's retry recovers it.
    let stub = SearxngStub::start(sample_results(), 2).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    assert_eq!(body["content"][0]["type"], "server_tool_use");
    assert_eq!(body["content"][1]["content"][0]["url"], "https://example.com/a");
    Ok(())
}

#[tokio::test]
async fn web_search_outage_is_reported_not_masked_as_empty_results() -> TestResult {
    // Unending failures: the bridge answers 200 with a text explanation naming
    // the search error, never a bare "no results" and never a model call.
    let stub = SearxngStub::start(sample_results(), u32::MAX).await?;
    let upstream = UpstreamApp::start().await?;
    let app = started_router(&stub.base_url, &upstream.base_url, true).await?;

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let all = serde_json::to_string(&body)?;
    assert_eq!(
        body["content"][1]["content"]["type"], "web_search_tool_result_error",
        "expected a typed error result on outage: {all}"
    );
    assert_eq!(body["content"][1]["content"]["error_code"], "unavailable");
    assert!(
        body["content"][1]["content"].get("url").is_none(),
        "expected no results on outage: {all}"
    );
    let text = body["content"][2]["text"].as_str().unwrap_or("");
    assert!(text.contains("temporarily unavailable"), "expected outage notice: {text}");
    assert!(
        text.contains("search returned 400"),
        "expected the error detail in the notice: {text}"
    );
    Ok(())
}

#[tokio::test]
async fn web_search_reranks_candidates_best_first() -> TestResult {
    let search = SearxngStub::start(
        vec![
            json!({"url":"https://example.com/a","title":"A","content":"unrelated filler"}),
            json!({"url":"https://example.com/b","title":"B","content":"AI model news today"}),
            json!({"url":"https://example.com/c","title":"C","content":"another on-topic item"}),
        ],
        0,
    )
    .await?;
    // Rerank favors index 1 (B) > 2 (C) > 0 (A).
    let rerank = RerankStub::start(vec![(0, 0.1), (1, 0.9), (2, 0.5)], false).await?;
    let upstream = UpstreamApp::start().await?;
    let state = ServerState::from_runner(Runner::from_toml(&deployment_with_rerank(
        &upstream.base_url,
        &search.base_url,
        &rerank.base_url,
        3,
    ))?)?;
    let app = build_switchyard_router(state);

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let urls: Vec<&str> = body["content"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["url"].as_str())
        .collect();
    assert_eq!(urls, vec!["https://example.com/b", "https://example.com/c", "https://example.com/a"]);
    Ok(())
}

#[tokio::test]
async fn web_search_falls_back_to_raw_order_when_reranker_down() -> TestResult {
    let search = SearxngStub::start(
        vec![
            json!({"url":"https://example.com/a","title":"A","content":"a"}),
            json!({"url":"https://example.com/b","title":"B","content":"b"}),
        ],
        0,
    )
    .await?;
    let rerank = RerankStub::start(vec![], true).await?; // always 500
    let upstream = UpstreamApp::start().await?;
    let state = ServerState::from_runner(Runner::from_toml(&deployment_with_rerank(
        &upstream.base_url,
        &search.base_url,
        &rerank.base_url,
        2,
    ))?)?;
    let app = build_switchyard_router(state);

    let response = send(&app, "POST", "/v1/messages", Some(web_search_body())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.bytes)?;
    let results = body["content"][1]["content"].as_array().expect("results");
    assert_eq!(results[0]["url"], "https://example.com/a");
    assert_eq!(results[1]["url"], "https://example.com/b");
    Ok(())
}

// --- rerank stub -----------------------------------------------------------------

struct RerankStub {
    base_url: String,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct RerankState {
    scores: Arc<Mutex<Vec<(usize, f64)>>>,
    error: bool,
}

impl RerankStub {
    async fn start(scores: Vec<(usize, f64)>, error: bool) -> TestResult<Self> {
        let state = RerankState {
            scores: Arc::new(Mutex::new(scores)),
            error,
        };
        let app = Router::new()
            .route("/rerank", post(rerank_handler))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            base_url: format!("http://{addr}"),
            task,
        })
    }
}

impl Drop for RerankStub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn rerank_handler(
    State(state): State<RerankState>,
    Json(_body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if state.error {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "reranker unavailable" })),
        );
    }
    let results: Vec<Value> = state
        .scores
        .lock()
        .unwrap()
        .iter()
        .map(|(index, score)| json!({ "index": index, "relevance_score": score }))
        .collect();
    (StatusCode::OK, Json(json!({ "results": results })))
}

// --- minimal upstream -----------------------------------------------------------------

struct UpstreamApp {
    base_url: String,
    task: tokio::task::JoinHandle<()>,
}

impl UpstreamApp {
    async fn start() -> TestResult<Self> {
        let app = Router::new().route("/v1/messages", post(upstream_messages));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            base_url: format!("http://{addr}"),
            task,
        })
    }
}

impl Drop for UpstreamApp {
    fn drop(&mut self) {
        self.task.abort();
    }
}
