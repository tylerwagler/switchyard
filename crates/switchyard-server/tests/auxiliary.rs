// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the non-chat relay: `/v1/embeddings` + `/v1/rerank`
//! proxy to configured backends (default/named), and `/v1/models` advertises the
//! embed/rerank/search capabilities.

use std::sync::Arc;
use std::sync::Mutex;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use switchyard_server::{ServerState, build_switchyard_router};
use tokio::net::TcpListener;
use tower::ServiceExt;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

// --- echo stub (both embeddings and rerank backends) --------------------------

struct EchoStub {
    base_url: String,
    received: Arc<Mutex<Vec<Value>>>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct EchoState {
    id: String,
    received: Arc<Mutex<Vec<Value>>>,
}

impl EchoStub {
    async fn start(id: &str, path: &str) -> TestResult<Self> {
        let received = Arc::new(Mutex::new(Vec::new()));
        let state = EchoState {
            id: id.to_string(),
            received: Arc::clone(&received),
        };
        let app = Router::new().route(path, post(echo_handler)).with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            base_url: format!("http://{addr}"),
            received,
            task,
        })
    }

    async fn bodies(&self) -> Vec<Value> {
        self.received.lock().unwrap().clone()
    }
}

impl Drop for EchoStub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn echo_handler(
    State(state): State<EchoState>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    state.received.lock().unwrap().push(body.clone());
    (StatusCode::OK, Json(json!({ "echo_id": state.id, "got": body })))
}

// --- deployment ---------------------------------------------------------------

fn deployment(embed_a: &str, embed_b: &str, rerank: &str, search: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "anthropic_messages"
base_url = "https://upstream.test"

[targets.main]
id = "test/main"
llm_client = "upstream"

[routes.main]
id = "test/main"
type = "passthrough"
target = "main"

[embeddings.e_a]
base_url = "{embed_a}"
model = "m-a"

[embeddings.e_b]
base_url = "{embed_b}"
model = "m-b"

[rerank.r_a]
base_url = "{rerank}"
model = "r-a"

[search.s_a]
base_url = "{search}"
"#
    )
}

// --- request helper -----------------------------------------------------------

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

async fn started(embed_a: &EchoStub, embed_b: &EchoStub, rerank: &EchoStub, search: &EchoStub) -> TestResult<Router> {
    let state = ServerState::from_runner(Runner::from_toml(&deployment(
        &embed_a.base_url,
        &embed_b.base_url,
        &rerank.base_url,
        &search.base_url,
    ))?)?;
    Ok(build_switchyard_router(state))
}

// --- tests -------------------------------------------------------------------

#[tokio::test]
async fn embeddings_default_relays_to_first_backend() -> TestResult {
    let a = EchoStub::start("a", "/embeddings").await?;
    let b = EchoStub::start("b", "/embeddings").await?;
    let r = EchoStub::start("r", "/rerank").await?;
    let s = EchoStub::start("s", "/search").await?;
    let app = started(&a, &b, &r, &s).await?;

    let body = json!({ "input": ["hello world"], "model": "m-a" });
    let response = send(&app, "POST", "/v1/embeddings", Some(body.clone())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let result: Value = serde_json::from_slice(&response.bytes)?;
    assert_eq!(result["echo_id"], "a");
    assert_eq!(a.bodies().await, vec![body]);
    assert!(b.bodies().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn embeddings_named_selects_the_backend() -> TestResult {
    let a = EchoStub::start("a", "/embeddings").await?;
    let b = EchoStub::start("b", "/embeddings").await?;
    let r = EchoStub::start("r", "/rerank").await?;
    let s = EchoStub::start("s", "/search").await?;
    let app = started(&a, &b, &r, &s).await?;

    let response = send(
        &app,
        "POST",
        "/v1/embeddings/e_b",
        Some(json!({ "input": ["x"], "model": "m-b" })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);
    let result: Value = serde_json::from_slice(&response.bytes)?;
    assert_eq!(result["echo_id"], "b");
    assert!(a.bodies().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn embeddings_unknown_name_is_404() -> TestResult {
    let a = EchoStub::start("a", "/embeddings").await?;
    let b = EchoStub::start("b", "/embeddings").await?;
    let r = EchoStub::start("r", "/rerank").await?;
    let s = EchoStub::start("s", "/search").await?;
    let app = started(&a, &b, &r, &s).await?;

    let response = send(&app, "POST", "/v1/embeddings/nope", Some(json!({ "input": [] }))).await?;
    assert_eq!(response.status, StatusCode::NOT_FOUND);
    let result: Value = serde_json::from_slice(&response.bytes)?;
    assert!(result["error"]["message"].as_str().unwrap().contains("embeddings backend named nope"));
    Ok(())
}

#[tokio::test]
async fn rerank_default_relays_the_body() -> TestResult {
    let a = EchoStub::start("a", "/embeddings").await?;
    let b = EchoStub::start("b", "/embeddings").await?;
    let r = EchoStub::start("r", "/rerank").await?;
    let s = EchoStub::start("s", "/search").await?;
    let app = started(&a, &b, &r, &s).await?;

    let body = json!({ "query": "needle", "documents": ["hay", "needle-in-haystack"] });
    let response = send(&app, "POST", "/v1/rerank", Some(body.clone())).await?;
    assert_eq!(response.status, StatusCode::OK);
    let result: Value = serde_json::from_slice(&response.bytes)?;
    assert_eq!(result["echo_id"], "r");
    assert_eq!(r.bodies().await, vec![body]);
    Ok(())
}

#[tokio::test]
async fn models_lists_non_chat_capabilities() -> TestResult {
    let a = EchoStub::start("a", "/embeddings").await?;
    let b = EchoStub::start("b", "/embeddings").await?;
    let r = EchoStub::start("r", "/rerank").await?;
    let s = EchoStub::start("s", "/search").await?;
    let app = started(&a, &b, &r, &s).await?;

    let response = send(&app, "GET", "/v1/models", None).await?;
    assert_eq!(response.status, StatusCode::OK);
    let payload: Value = serde_json::from_slice(&response.bytes)?;
    let kinds: Vec<(String, String)> = payload["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| {
            let id = entry["id"].as_str()?;
            let kind = entry["kind"].as_str()?;
            Some((id.to_string(), kind.to_string()))
        })
        .collect();
    assert!(kinds.contains(&("e_a".into(), "embeddings".into())));
    assert!(kinds.contains(&("e_b".into(), "embeddings".into())));
    assert!(kinds.contains(&("r_a".into(), "rerank".into())));
    assert!(kinds.contains(&("s_a".into(), "search".into())));
    Ok(())
}
